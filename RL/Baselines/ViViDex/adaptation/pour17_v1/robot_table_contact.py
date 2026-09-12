"""Opt-in GPU arm-link/table contact measurement; no scene or physics edits."""
import numpy as np


class RobotTableContacts:
    def __init__(self, raw):
        import omni.physics.tensors as tensors
        self.view = tensors.create_simulation_view("torch")
        self.view.set_subspace_roots("/")
        self.num_envs, self.dt = raw.num_envs, raw.physics_dt
        self.arm_names = [name for name in raw.robot.body_names if "_arm_l" in name]
        if not self.arm_names:
            raise RuntimeError("no registered arm links for arm/table measurement")
        paths = [f"/World/envs/env_{i}/Robot/{name}"
                 for i in range(self.num_envs) for name in self.arm_names]
        filters = [[f"/World/envs/env_{i}/Table/geometry/mesh"]
                   for i in range(self.num_envs) for _ in self.arm_names]
        self.pair = self.view.create_rigid_contact_view(
            paths, filters, max_contact_data_count=32768)
        if set(self.pair.sensor_paths) != set(paths):
            raise RuntimeError("arm/table sensor paths incomplete")
        expected = dict(zip(paths, filters))
        if list(self.pair.filter_paths) != [expected[path] for path in self.pair.sensor_paths]:
            raise RuntimeError("arm/table filter mapping mismatch")
        self.latest_by_env = [{} for _ in range(self.num_envs)]

    def read(self):
        counts = self.pair.get_contact_data(self.dt)[4].cpu().numpy()
        flags = np.zeros(self.num_envs, dtype=bool)
        latest = [{} for _ in range(self.num_envs)]
        for row, path in enumerate(self.pair.sensor_paths):
            value = int(counts[row, 0])
            if value:
                env_id = int(path.split("/env_")[1].split("/")[0])
                latest[env_id][path.rsplit("/", 1)[-1]] = value
                flags[env_id] = True
        self.latest_by_env = latest
        return flags

    def evidence(self):
        return {"backend": "GPU detailed rigid contact view",
                "sensors": list(self.pair.sensor_paths),
                "filters": list(self.pair.filter_paths),
                "body_rule": "runtime body name contains _arm_l",
                "no_scene_or_physics_edits": True}
