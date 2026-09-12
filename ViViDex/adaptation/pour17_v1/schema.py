"""WP2 state-only schema, env-frame metres/radians; right then left."""
FIELDS = (
    ("joint_q_rad", 58), ("joint_qd_rad_s", 58),
    ("wrist_pose_velocity", 26), ("object_pose_velocity", 26),
    ("target_wrist_pose", 14), ("target_object_pose", 14),
    ("target_ten_landmarks", 60), ("wrist_relative_object", 14),
    ("target_wrist_relative_object", 14),
    ("actual_palm_five_tips", 36), ("actual_hand_rot_velocity", 120), ("hand_points_minus_object", 36),
    ("final_object_goal_xyz", 6), ("palm_minus_final_goal", 6),
    ("object_minus_final_goal", 6),
    ("future_object_goals_1_5_10_quat_then_xyz", 42),
    ("phase_sin_cos_1_4_6_8", 8),
)
OBSERVATION_DIM = sum(width for _, width in FIELDS)


def schema():
    offset = 0
    fields = []
    for name, width in FIELDS:
        fields.append({"name": name, "start": offset, "stop": offset + width})
        offset += width
    return {"schema": "pour17_state_v2", "observation_dim": offset, "action_dim": 58,
            "fields": fields, "coordinates": "per-env fixed world frame; xyz metres, q radians; quaternions wxyz",
            "side_order": ["right/bottle", "left/cup"],
            "future": "reference index offsets 1,5,10; clamp to last; per offset side then wxyz,xyz",
            "clipping": "no observation clipping; finite required; actions clipped [-1,1]",
            "images_in_policy": False}
