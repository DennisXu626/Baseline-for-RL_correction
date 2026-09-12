"""Validated GPU pair contacts; no collision or rigid-body edits.

The module name preserves the isolated diagnostic's provenance. The same
implementation is used by the environment after its positive/negative checks.
"""
import numpy as np


class GpuPairContacts:
    def __init__(self, raw):
        import omni.physics.tensors as tensors
        self.sim = tensors.create_simulation_view("torch")
        self.sim.set_subspace_roots("/")
        self.dt, self.n = raw.physics_dt, raw.num_envs
        self.fingers = ("thumb", "index", "middle", "ring", "pinky")
        self.views, self.metadata = [], []
        for side_index, side in enumerate(("right", "left")):
            names = [n for n in raw.robot.body_names if n.startswith(side + "_hand_")
                     or any(n.startswith(side + "_" + f + "_") for f in self.fingers)]
            paths = [f"/World/envs/env_{i}/Robot/{n}" for i in range(self.n) for n in names]
            filters = [[f"/World/envs/env_{i}/" + ("Object" if side_index == 0 else "Aux")]
                       for i in range(self.n) for _ in names]
            view = self.sim.create_rigid_contact_view(paths, filters, max_contact_data_count=4096*self.n)
            actual = list(view.sensor_paths)
            assert set(actual) == set(paths), ("hand sensor paths mismatch", actual, paths)
            expected_filters = dict(zip(paths, filters))
            assert list(view.filter_paths) == [expected_filters[p] for p in actual], "hand filter slot mismatch"
            self.views.append(("hand", side_index, view, actual))
            self.metadata.append({"kind": "hand", "side": side, "sensors": actual, "filters": view.filter_paths})
        # GPU sensors must be dynamic. Resolve the actual static collider for
        # the FILTER, not the Table container Xform used by the old adapter.
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        table_colliders = []
        for i in range(self.n):
            root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
            assert not root.HasAPI(UsdPhysics.RigidBodyAPI), "table must remain static"
            colliders = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
            assert len(colliders) == 1, ("expected registered cuboid's one static collider", colliders)
            table_colliders.extend(colliders)
        paths = [f"/World/envs/env_{i}/{name}" for i in range(self.n) for name in ("Object", "Aux")]
        filters = [[table_colliders[i]] for i in range(self.n) for _ in range(2)]
        table = self.sim.create_rigid_contact_view(paths, filters, max_contact_data_count=1024*self.n)
        assert set(table.sensor_paths) == set(paths), ("static table sensors unavailable", table.sensor_paths)
        expected_filters = dict(zip(paths, filters))
        assert list(table.filter_paths) == [expected_filters[p] for p in table.sensor_paths], "table filter slot mismatch"
        self.views.append(("table", None, table, list(table.sensor_paths)))
        self.metadata.append({"kind": "table", "sensors": table.sensor_paths, "filters": table.filter_paths})
        self.latest = []

    def clear(self, env_ids):
        # No cached contact latch: each read queries the current physical step.
        self.latest = []

    def read(self):
        hand = np.zeros((self.n, 2), bool)
        tips = np.zeros((self.n, 2, 5), bool)
        table = np.zeros((self.n, 2), bool)
        self.latest = []
        for kind, si, view, paths in self.views:
            _, _, _, _, counts, starts = view.get_contact_data(self.dt)
            values = counts.cpu().numpy()
            assert int(values.sum()) < view.max_contact_data_count, "contact buffer capacity reached"
            self.latest.append({"kind": kind, "side": si, "counts": values.tolist()})
            for row, path in enumerate(paths):
                env_id = int(path.split("/env_")[1].split("/")[0])
                if kind == "table":
                    assert values.shape[1] == 1
                    table[env_id, 0 if path.endswith("/Object") else 1] = values[row, 0] > 0
                elif values[row, 0] > 0:
                    hand[env_id, si] = True
                    for fi, f in enumerate(self.fingers):
                        if path.endswith(f"/{('right', 'left')[si]}_{f}_elastomer"):
                            tips[env_id, si, fi] = True
        return hand, tips, table

    def evidence(self):
        return {"backend": "GPU detailed sensor/filter contact counts; actual static table collider as filter",
                "views": self.metadata, "latest_counts": self.latest, "no_scene_physics_edits": True}
