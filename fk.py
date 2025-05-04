import numpy as np
import re

def dh_to_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics_dh(joint_angles, dh_params):
    T = np.eye(4, dtype=np.float32)
    for i in range(6):
        theta = joint_angles[i] + dh_params[i, 3]
        d = dh_params[i, 0]
        a = dh_params[i, 1]
        alpha = dh_params[i, 2]
        T_i = dh_to_matrix(theta, d, a, alpha)
        T = T @ T_i
    return T

# Modified DH parameters for Aubo
dh_true = np.array([
    [0.1632, 0., 0.5 * np.pi, 0.],
    [0., 0.647, np.pi, 0.5 * np.pi],
    [0., 0.6005, np.pi, 0.],
    [0.2013, 0., -0.5 * np.pi, -0.5 * np.pi],
    [0.1025, 0., 0.5 * np.pi, 0.],
    [0.094, 0., 0., 0.]
], dtype=np.float32)

# 读取前10组关节角
joint_list = []
filename = "aubo_record_2025-05-01_01-41-25.txt"
with open(filename, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if "Joints:" in line and i + 1 < len(lines):
            joint_line = lines[i + 1].strip()
            try:
                joint_vals = [float(x.strip()) for x in joint_line.strip("[]").split(',')]
                print(f"[Joint {len(joint_list)}] {joint_vals}")
                joint_list.append(joint_vals)
            except Exception as e:
                print(f"✗ Failed to parse joint line: {joint_line} — {e}")
        if len(joint_list) >= 10:
            break

if not joint_list:
    print("✗ 未找到任何 JointPos 数据，请检查文件格式和正则表达式。")
    exit()

print(f"\n✅ 成功提取 {len(joint_list)} 组关节角，开始正向运动学计算...\n")

# 执行 FK 并打印结果
for idx, q in enumerate(joint_list):
    T = forward_kinematics_dh(np.array(q), dh_true)
    pos = T[:3, 3]
    print(f"[{idx}] Joint: {np.round(q, 4)} → Flange Position: {np.round(pos, 6)}")