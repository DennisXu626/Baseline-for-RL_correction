"""WP3 two-slot reset/hold readback. No training, root edits or reference solve.

Mesh coordinates use the live PhysX body transform, not stale GPU USD poses.
The table-box scene query is against cooked PhysX geometry; mesh clipping is
separate evidence about the authored visual surface, never an AABB verdict.
"""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np


def plain(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def clipped_triangle(triangle, half):
    polygon = list(triangle)
    for axis in range(3):
        for sign in (-1, 1):
            result = []
            if not polygon:
                return []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = sign*a[axis]-half[axis], sign*b[axis]-half[axis]
                if da <= 0:
                    result.append(a)
                if (da < 0 < db) or (db < 0 < da):
                    result.append(a+(b-a)*da/(da-db))
            polygon = result
    return polygon


def main():
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    for key in ("bundle-root", "robot-urdf", "reference", "output-dir"):
        ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--control-steps",type=int,choices=(1,40),default=40)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    if not args.enable_cameras:
        ap.error("camera evidence required")
    if args.output_dir.exists():
        raise RuntimeError("new versioned output directory required")
    args.output_dir.mkdir(parents=True)
    app = AppLauncher(args).app
    import torch
    import cv2
    import carb
    import omni.usd
    import omni.physx
    import omni.physics.tensors as tensors
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from isaaclab.sensors import Camera, CameraCfg
    import isaaclab.sim as sim_utils
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config
    from .geometry import sha256, quat_to_matrix_wxyz

    class ReviewCamera(Camera):
        def reset(self, env_ids=None):
            # Each global review camera has one render product, not two env
            # entries. Scene reset ids must not index its single timestamp.
            super().reset(None)

    class ReviewEnv(Pour17JointEnv):
        def _setup_scene(self):
            super()._setup_scene()
            self.cameras = []
            for i in range(4):
                cam = ReviewCamera(CameraCfg(prim_path=f"/World/WP3Camera{i}", width=640, height=480,
                    data_types=["rgb"], update_period=0., spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24., horizontal_aperture=36., clipping_range=(.05, 20.))))
                self.cameras.append(cam)
                self.scene.sensors[f"wp3_camera{i}"] = cam

    cfg = isaac_scene_config(args.bundle_root, num_envs=2, seed=1701)
    cfg.sim.device, cfg.sim.log_dir = args.device, str(args.output_dir/"isaaclab_logs")
    # Diagnostic query structure only; no collider/filter/solver change.
    cfg.sim.enable_scene_query_support = True
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.external_evaluator_controls_termination, cfg.curriculum_stage = True, 0
    raw, writer = None, None
    rows, geometry, camera_frames = [], [], []
    def save(name, value):
        (args.output_dir/name).write_text(json.dumps(value, indent=2, default=plain)+"\n")
    try:
        raw = ReviewEnv(cfg)
        stage = omni.usd.get_context().get_stage()
        body_names = list(raw.robot.body_names)
        origins = raw.scene.env_origins.detach().cpu().numpy()
        home_action = (2*(raw.reset_q-raw.lower)/(raw.upper-raw.lower)-1)[None].repeat(2, 1)
        assert bool(((home_action >= -1) & (home_action <= 1)).all())
        poses = []
        for slot in range(2):
            for vi, (eye, target) in enumerate((((0., -3., 1.05), (0., 0., .9)),
                                               ((.001, 0., 3.8), (0., 0., .85)))):
                eye, target = np.array(eye)+origins[slot], np.array(target)+origins[slot]
                raw.cameras[slot*2+vi].set_world_poses_from_view(
                    eyes=torch.tensor(eye[None], device=raw.device, dtype=torch.float32),
                    targets=torch.tensor(target[None], device=raw.device, dtype=torch.float32))
                poses.append(dict(slot=slot, view=("side", "top")[vi], eye=eye, target=target))
        for _ in range(8):
            raw.sim.render()  # no physical/control warmup
        writer = cv2.VideoWriter(str(args.output_dir/"reset_hold.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20., (1280, 960))
        assert writer.isOpened()
        view = tensors.create_simulation_view("torch")
        paths = [f"/World/envs/env_{i}/Robot/{n}" for i in range(2) for n in body_names]
        filters = [[f"/World/envs/env_{i}/Table/geometry/mesh"] for i in range(2) for _ in body_names]
        pair = view.create_rigid_contact_view(paths, filters, max_contact_data_count=8192)
        assert set(pair.sensor_paths) == set(paths)
        expected_filters = dict(zip(paths, filters))
        assert list(pair.filter_paths) == [expected_filters[p] for p in pair.sensor_paths]

        def matrix(prim):
            return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

        def live_transforms():
            value = raw.robot.root_physx_view.get_link_transforms().detach().cpu().numpy()
            result = np.tile(np.eye(4), (2, len(body_names), 1, 1))
            for i in range(2):
                for j in range(len(body_names)):
                    result[i,j,:3,:3] = quat_to_matrix_wxyz(value[i,j,[6,3,4,5]])
                    result[i,j,:3,3] = value[i,j,:3]
            return value, result

        def geometry_at(label):
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
            _, transforms = live_transforms()
            record = dict(label=label, slots=[])
            for slot in range(2):
                table = stage.GetPrimAtPath(f"/World/envs/env_{slot}/Table/geometry/mesh")
                table_world = matrix(table)
                # Actual table is Cube(size2) with local scaling in its transform.
                bound = cache.ComputeWorldBound(table).ComputeAlignedRange()
                lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
                center, half = (lo+hi)/2, (hi-lo)/2
                assert np.max(np.abs(table_world[:3,:3]-np.diag(np.diag(table_world[:3,:3])))) < 1e-8
                hits_by_inset = {}
                for inset in (0., 1e-5, .001):
                    hits = []
                    def hit(h):
                        hits.append(dict(collision=str(h.collision), rigid_body=str(h.rigid_body)))
                        return True
                    count = omni.physx.get_physx_scene_query_interface().overlap_box(
                        carb.Float3(*(half-inset)), carb.Float3(*center), carb.Float4(0,0,0,1), hit, False)
                    hits_by_inset[str(inset)] = dict(count=count, hits=hits)
                assert any(h["collision"].startswith(f"/World/envs/env_{slot}/Table")
                    for h in hits_by_inset["0.0"]["hits"]), "query lacks table self-hit; zero results are not valid evidence"
                meshes = []
                robot = stage.GetPrimAtPath(f"/World/envs/env_{slot}/Robot")
                colliders = [dict(path=str(p.GetPath()),type=p.GetTypeName(),schemas=p.GetAppliedSchemas(),
                    enabled=p.GetAttribute("physics:collisionEnabled").Get(),
                    approximation=p.GetAttribute("physics:approximation").Get(),
                    attributes={a.GetName():a.Get() for a in p.GetAttributes() if a.GetName().startswith(("physics:","physxCollision:","radius","height","size","axis"))},
                    relationships={r.GetName():[str(t) for t in r.GetTargets()] for r in p.GetRelationships()})
                    for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.CollisionAPI)]
                for prim in Usd.PrimRange(robot):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    mesh = UsdGeom.Mesh(prim)
                    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                    if not len(points):
                        continue
                    relative = str(prim.GetPath()).split("/Robot/")[-1]
                    body = relative.split("/")[0]
                    world = matrix(prim)
                    if body in body_names:
                        body_prim = stage.GetPrimAtPath(str(robot.GetPath())+"/"+body)
                        world = transforms[slot,body_names.index(body)] @ np.linalg.inv(matrix(body_prim)) @ world
                    points_w = points @ world[:3,:3].T + world[:3,3]
                    pmin, pmax = points_w.min(0), points_w.max(0)
                    candidate = bool(np.all(pmax > lo) and np.all(pmin < hi))
                    collision = prim.HasAPI(UsdPhysics.CollisionAPI)
                    row = dict(path=str(prim.GetPath()), body=body, collision_api=collision,
                        collision_enabled=prim.GetAttribute("physics:collisionEnabled").Get() if collision else None,
                        approximation=prim.GetAttribute("physics:approximation").Get(),
                        visibility=str(mesh.ComputeVisibility()), purpose=str(mesh.ComputePurpose()),
                        world_aabb=[pmin, pmax], aabb_candidate=candidate,
                        matrix_world_live=world, surface_triangle_intersections=0, witness=None)
                    if candidate:
                        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
                        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
                        offset = 0
                        for count in counts:
                            face = indices[offset:offset+count]
                            offset += count
                            for j in range(1, count-1):
                                tri = points_w[face[[0,j,j+1]]] - center
                                if np.any(tri.min(0) > half) or np.any(tri.max(0) < -half):
                                    continue
                                clipped = clipped_triangle(tri, half-1e-6)
                                if len(clipped) >= 3:
                                    row["surface_triangle_intersections"] += 1
                                    if row["witness"] is None:
                                        row["witness"] = np.array(clipped)+center
                    meshes.append(row)
                record["slots"].append(dict(slot=slot, table_transform=table_world,
                    table_world_bounds=[lo,hi], table_env_bounds=[lo-origins[slot],hi-origins[slot]],
                    physx_cooked_overlap_box=hits_by_inset, meshes=meshes, all_collision_shapes=colliders))
            geometry.append(record)
            save("scene_geometry.json", geometry)

        def snapshot(label, completed, do_geometry=False):
            link, _ = live_transforms()
            counts = pair.get_contact_data(raw.physics_dt)[4].detach().cpu().numpy()
            q = raw.robot.root_physx_view.get_dof_positions()[:,raw.joint_ids]
            qd = raw.robot.root_physx_view.get_dof_velocities()[:,raw.joint_ids]
            targets = raw.robot.root_physx_view.get_dof_position_targets()[:,raw.joint_ids]
            record = dict(label=label, completed_global_steps=completed,
                episode_completed_steps=raw.episode_length_buf.clone(), reference_index=raw._reference_index().clone(),
                action_rule="normalized canonical q; not normalized zeros", home_action=home_action,
                robot_body_names=body_names, joint_names=raw.joint_names,
                q=q.clone(), qd=qd.clone(), physx_joint_targets=targets.clone(),
                cached_joint_targets=raw.robot.data.joint_pos_target[:,raw.joint_ids].clone(),
                root_transforms_xyzw=raw.robot.root_physx_view.get_root_transforms().clone(),
                body_transforms_world_xyzw=link,
                body_transforms_env_xyz=(link[...,:3]-origins[:,None,:]),
                env_origins=origins, canonical=raw.reset_state(),
                robot_root_prim_world=[matrix(stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")) for i in range(2)],
                robot_table_contact_counts=dict(zip(pair.sensor_paths, counts.tolist())),
                contact_limit="Zero count on fixed/non-reporting shapes does not prove no geometric overlap.")
            rows.append(record)
            save("reset_world_readback.json", rows)
            if do_geometry:
                geometry_at(label)

        def frame(label, completed):
            raw.sim.render()
            images = []
            for i, camera in enumerate(raw.cameras):
                camera.update(0., force_recompute=True)
                rgb = camera.data.output["rgb"][0,:,:,:3].cpu().numpy().astype(np.uint8)
                assert rgb.std() > 1
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image,(0,0),(640,30),(20,20,20),-1)
                cv2.putText(image, f"slot{i//2} {('SIDE','TOP')[i%2]} | {label} | total step{completed}", (8,22),
                            cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
                images.append(image)
            tiled = np.concatenate((np.concatenate(images[:2],1),np.concatenate(images[2:],1)),0)
            writer.write(tiled)
            index = len(camera_frames)
            camera_frames.append(dict(video_frame=index, label=label, completed_global_steps=completed,
                episode_step=raw.episode_length_buf.clone().cpu().tolist(), reference_index=raw._reference_index().cpu().tolist(),
                action=home_action.cpu().tolist(), timing="readback after write/step, then render-only; no added physical step"))
            if completed in (0,20,40):
                cv2.imwrite(str(args.output_dir/f"frame_{index:03}_{label}.png"),tiled)

        # Authored source and composed stage facts before explicit reset.
        authored = Usd.Stage.Open(str(args.bundle_root/"world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"))
        def inventory(st):
            facts = []
            for p in st.Traverse():
                if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint) or p == st.GetDefaultPrim():
                    facts.append(dict(path=str(p.GetPath()), type=p.GetTypeName(), schemas=p.GetAppliedSchemas(),
                        attributes={a.GetName(): a.Get() for a in p.GetAttributes() if a.GetName().startswith(("xformOp", "physics:", "physxRigidBody:", "physxArticulation:"))},
                        relationships={r.GetName(): [str(x) for x in r.GetTargets()] for r in p.GetRelationships()}))
            return facts
        save("authored_and_runtime_scene.json", dict(authored=inventory(authored), runtime=inventory(stage),
            cameras=poses, cfg_robot_init_state=vars(cfg.robot.init_state),
            source_inputs={str(p):sha256(p) for p in (args.reference,args.robot_urdf,
                args.bundle_root/"world/world_manifest.json",args.bundle_root/"world/canonical_reset_v1.json",
                args.bundle_root/"world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd",Path(__file__))}))
        snapshot("constructor_reset_before_explicit_reset",0,True)
        raw.reset(seed=1701)
        snapshot("first_reset_before_write_data",0)
        raw.scene.write_data_to_sim()
        snapshot("first_reset_after_write_data",0,True)
        frame("first_reset",0)
        for step in range(1,args.control_steps+1):
            raw.step(home_action)
            snapshot("hold",step, step in (1,20,21,40))
            frame("hold",step)
            if step == 20:
                raw.reset(seed=1701)
                snapshot("repeat_reset_before_write_data",step)
                raw.scene.write_data_to_sim()
                snapshot("repeat_reset_after_write_data",step,True)
                frame("repeat_reset",step)
        writer.release()
        writer = None
        save("camera_frames.json",dict(cameras=poses,frames=camera_frames,fps=20,render_only_initial_frames=8))
        save("result.json",dict(status="COMPLETED_WP3_RESET_HOLD_NOT_PHYSICS_PASS",num_envs=2,
            actual_control_steps=args.control_steps, slot_transitions=2*args.control_steps, video_frames=len(camera_frames),
            scene_query_support_enabled_for_diagnostics_only=True,
            no_root_pose_write=True, no_reference_execution=True, no_training=True))
    except BaseException:
        (args.output_dir/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if writer is not None:
            writer.release()
        if raw is not None:
            raw.close()
        app.close()


if __name__ == "__main__":
    main()
