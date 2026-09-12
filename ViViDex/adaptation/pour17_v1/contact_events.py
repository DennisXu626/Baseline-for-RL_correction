"""PhysX pair reports: semantic contacts, including the unchanged static table.

Unlike net force, these reports retain both actor identities. No contact is
inferred from object motion. The frozen G1 force threshold remains separate.
"""
from __future__ import annotations

import re
import numpy as np


class PairContacts:
    def __init__(self, num_envs: int):
        import omni.physx
        from pxr import PhysicsSchemaTools
        self.num_envs = num_envs
        self._decode = PhysicsSchemaTools.intToSdfPath
        self.pairs = {}
        self.callbacks = 0
        self.polls = 0
        self.headers = 0
        self.path_cache = {}
        self.interface = omni.physx.get_physx_simulation_interface()
        self.subscription = self.interface.subscribe_contact_report_events(self._on_report)

    def _path(self, value):
        key = int(value)
        if key not in self.path_cache:
            self.path_cache[key] = str(self._decode(value))
        return self.path_cache[key]

    def _on_report(self, headers, data):
        self.callbacks += 1
        self._consume(headers)

    def poll(self):
        # Installed PhysX API explicitly exposes the current-step report.
        # Manual GPU stepping need not dispatch the Python event subscription.
        report = self.interface.get_contact_report()
        self.polls += 1
        self._consume(report[0])

    def _consume(self, headers):
        from omni.physx.bindings._physx import ContactEventType
        for header in headers:
            self.headers += 1
            a, b = self._path(header.actor0), self._path(header.actor1)
            # Multiple collision shapes may report the same rigid-body pair.
            # A LOST event for one shape must not erase another active shape.
            key = tuple(sorted((int(header.collider0), int(header.collider1))))
            if header.type == ContactEventType.CONTACT_LOST:
                self.pairs.pop(key, None)
            else:
                self.pairs[key] = (a, b)

    def clear(self, env_ids):
        prefixes = tuple(f"/World/envs/env_{int(i)}/" for i in env_ids)
        self.pairs = {k: v for k, v in self.pairs.items()
                      if not any(p.startswith(prefixes) for p in v)}

    def read(self):
        """Returns whole hand, tips, table; side order right/bottle,left/cup."""
        hand = np.zeros((self.num_envs, 2), dtype=bool)
        tips = np.zeros((self.num_envs, 2, 5), dtype=bool)
        table = np.zeros((self.num_envs, 2), dtype=bool)
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        for a, b in self.pairs.values():
            for obj, other in ((a, b), (b, a)):
                match = re.fullmatch(r"/World/envs/env_(\d+)/(Object|Aux)", obj)
                if match is None:
                    continue
                slot = int(match[1])
                side = 0 if match[2] == "Object" else 1
                prefix = f"/World/envs/env_{slot}/"
                if other == prefix + "Table":
                    table[slot, side] = True
                name = other.removeprefix(prefix + "Robot/")
                handed = "right_" if side == 0 else "left_"
                if name.startswith(handed + "hand_") or any(name.startswith(handed + f + "_") for f in fingers):
                    hand[slot, side] = True
                for fi, finger in enumerate(fingers):
                    if name == handed + finger + "_elastomer":
                        tips[slot, side, fi] = True
        return hand, tips, table

    def evidence(self):
        import carb
        settings = carb.settings.get_settings()
        return {"backend": "PhysX actor-pair contact reports; direct post-physics query", "callbacks": self.callbacks,
                "post_physics_queries": self.polls,
                "headers": self.headers, "active_pairs": list(self.pairs.values()),
                "resolved_paths": list(self.path_cache.values()),
                "readback_settings": {k: settings.get(k) for k in (
                    "/physics/disableContactProcessing", "/physics/suppressReadback",
                    "/physics/updateToUsd", "/physics/updateVelocitiesToUsd")}}
