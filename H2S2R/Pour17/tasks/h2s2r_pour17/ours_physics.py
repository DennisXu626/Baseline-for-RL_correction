"""Uniform physics alignment for both single-hand H2S2R diagnostics.

No upstream reward, observation, reset or controller method is imported.
Values are the supplied rounded readbacks; no missing calibration is invented.
"""
from pathlib import Path
import hashlib
import json

SPEC_PATH = Path(__file__).with_name('ours_physics_spec.json')


def physics_record():
    content = SPEC_PATH.read_bytes()
    return {**json.loads(content), 'sha256': hashlib.sha256(content).hexdigest(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def configure_physics(cfg):
    spec = physics_record()
    cfg.bottle_mass_kg = cfg.cup_mass_kg = spec['object_mass_kg']
    # Parent scene uses this for SuperGrip on the pads; object bindings are
    # independently changed to average by apply_object_physics below.
    cfg.bottle_friction = cfg.cup_friction = spec['object_friction']
    cfg.body_friction = spec['body_friction']
    cfg.table_friction = spec['table_friction']
    cfg.ours_physics_record = spec
    for group, indices in (('arm_shoulder', (0, 1)), ('arm_elbow', (2, 3)),
                           ('arm_wrist', (4, 5, 6))):
        actuator = cfg.robot_cfg.actuators[group]
        actuator.stiffness = {f'{side}_arm_j{i+1}': spec['arm_kp'][i]
                              for side in ('R', 'L') for i in indices}
        actuator.damping = {f'{side}_arm_j{i+1}': spec['arm_kd'][i]
                            for side in ('R', 'L') for i in indices}
        actuator.effort_limit_sim = spec['arm_effort_nm'][indices[0]]
    hands = cfg.robot_cfg.actuators['hands']
    hands.stiffness, hands.damping = spec['hand_kp'], spec['hand_kd']
    hands.effort_limit_sim = spec['hand_effort_nm']
    for actuator in cfg.robot_cfg.actuators.values():
        actuator.armature = 0.0
        actuator.friction = 0.0
    return cfg


def apply_object_physics(cfg):
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.bottle_friction, dynamic_friction=cfg.bottle_friction,
        restitution=0.0, friction_combine_mode='average', restitution_combine_mode='average')
    material.func('/World/Materials/H2S2RUniformObject', material)
    for name in ('Object', 'Aux'):
        for path in sim_utils.find_matching_prim_paths(f'/World/envs/env_.*/{name}'):
            UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateMassAttr(cfg.bottle_mass_kg)
            sim_utils.bind_physics_material(path, '/World/Materials/H2S2RUniformObject',
                                            stronger_than_descendants=True)


def verify_runtime(env):
    """Fail before PPO if effective gains, object/pad materials or masses differ."""
    import torch
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    spec = env.cfg.ours_physics_record
    view = env.hand.root_physx_view
    kp, kd = view.get_dof_stiffnesses(), view.get_dof_dampings()
    effort = view.get_dof_max_forces()
    rows = []
    for index, name in enumerate(env.hand.joint_names):
        if '_arm_j' in name:
            j = int(name[-1]) - 1
            expected = (spec['arm_kp'][j], spec['arm_kd'][j], spec['arm_effort_nm'][j])
        else:
            expected = (spec['hand_kp'], spec['hand_kd'], spec['hand_effort_nm'])
        for tensor, target in zip((kp, kd, effort), expected):
            if not torch.allclose(tensor[:, index], torch.full_like(tensor[:, index], target), atol=0.01, rtol=1e-6):
                raise RuntimeError(f'Uniform effective joint physics mismatch: {name}, expected {expected}')
        rows.append({'name': name, 'stiffness': float(kp[0, index]),
                     'damping': float(kd[0, index]), 'effort': float(effort[0, index])})
    assert torch.count_nonzero(view.get_dof_armatures()) == 0
    assert torch.count_nonzero(view.get_dof_friction_properties()) == 0
    objects = {}
    for name, asset in (('bottle', env.object), ('cup', env.aux)):
        masses = asset.root_physx_view.get_masses()
        materials = asset.root_physx_view.get_material_properties()
        assert torch.allclose(masses, torch.full_like(masses, spec['object_mass_kg']), atol=1e-6)
        target = torch.tensor(
            [spec['object_friction'], spec['object_friction'], spec['restitution']],
            device=materials.device,
        )
        assert torch.allclose(materials, target.expand_as(materials), atol=1e-6)
        objects[name] = {'masses_env0': masses[0].tolist(), 'materials_env0': materials[0].tolist(),
                         'inertias_env0': asset.root_physx_view.get_inertias()[0].tolist()}
    pads = []
    for sensor in env._contact_sensors:
        materials = sensor.body_physx_view.get_material_properties()
        target = torch.tensor(
            [spec['pad_friction'], spec['pad_friction'], spec['restitution']],
            device=materials.device,
        )
        assert torch.allclose(materials, target.expand_as(materials), atol=1e-6)
        pads.append({'path': sensor.cfg.prim_path, 'materials_first_body': materials[0].tolist()})
    robot_materials = view.get_material_properties()
    robot_target = torch.tensor(
        [spec['body_friction'], spec['body_friction'], spec['restitution']],
        device=robot_materials.device,
    )
    assert torch.allclose(robot_materials, robot_target.expand_as(robot_materials), atol=1e-6)
    stage = omni.usd.get_context().get_stage()
    resolved = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0')):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if not mat:
            continue
        name = str(mat.GetPath())
        mode = str(PhysxSchema.PhysxMaterialAPI(mat.GetPrim()).GetFrictionCombineModeAttr().Get())
        if name.endswith('H2S2RUniformObject'):
            assert mode == 'average'
        elif name.endswith(('SuperGrip', 'LowGrip')):
            assert mode == 'multiply'
        if name.endswith('/Table/geometry/material'):
            material_api = UsdPhysics.MaterialAPI(mat.GetPrim())
            static_friction = float(material_api.GetStaticFrictionAttr().Get())
            dynamic_friction = float(material_api.GetDynamicFrictionAttr().Get())
            assert abs(static_friction - spec['table_friction']) <= 1e-6
            assert abs(dynamic_friction - spec['table_friction']) <= 1e-6
        resolved.append({'shape': str(prim.GetPath()), 'material': name, 'friction_combine': mode})
    return {'status': 'passed', 'scope': 'all-env tensor checks; env0 USD combine-mode resolution',
            'objects': objects, 'pads': pads, 'joints': rows, 'bindings_env0': resolved,
            'robot_materials_env0': robot_materials[0].tolist(),
            'velocity_limits_env0': view.get_dof_max_velocities()[0].tolist(),
            'spec': spec}
