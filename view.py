import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import csv
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- Standard DH 正向运动学 --------------------
def dh_to_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(joint_angles, dh_params):
    T = np.eye(4)
    for i in range(len(joint_angles)):
        theta = joint_angles[i] + dh_params[i][3]
        d = dh_params[i][0]
        a = dh_params[i][1]
        alpha = dh_params[i][2]
        T = T @ dh_to_matrix(theta, d, a, alpha)
    return T

def forward_fk(dh_params: np.ndarray, joint_list: np.ndarray, p3: np.ndarray):
    xyz_list = []
    for joints in joint_list:
        T = forward_kinematics(joints, dh_params)
        p = T[:3, :3] @ p3 + T[:3, 3]
        xyz_list.append(p)
    return np.array(xyz_list)

def apply_mirror(xyz, mirror_axes: str):
    mirror = [1, 1, 1]
    if 'x' in mirror_axes: mirror[0] = -1
    if 'y' in mirror_axes: mirror[1] = -1
    if 'z' in mirror_axes: mirror[2] = -1
    mirror_tensor = torch.diag(torch.tensor(mirror, dtype=torch.float32))
    xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
    return torch.matmul(xyz_tensor, mirror_tensor).numpy()

def compare_trajectories(xyz_fk: np.ndarray, xyz_meas: np.ndarray, title_suffix=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_fk[:, 0], xyz_fk[:, 1], xyz_fk[:, 2], label='FK Trajectory (DH)', color='r')
    # ax.plot(xyz_meas[:, 0], xyz_meas[:, 1], xyz_meas[:, 2], label='Laser Measured', color='b')

    ax.scatter(0, 0, 0, color='k', s=50, marker='o', label='Base Point')
    ax.text(0, 0, 0, 'Base', color='k')

    ax.set_title(f"FK vs Laser Trajectory {title_suffix}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def compute_alignment_loss(a: np.ndarray, b: np.ndarray):
    return np.mean((a - b) ** 2)

def visualize_all_mirror_combinations(fk_xyz, laser_xyz):
    axes = ['x', 'y', 'z']
    combinations = [''.join(c) for i in range(4) for c in itertools.combinations(axes, i)]
    for code in combinations:
        mirrored = apply_mirror(fk_xyz, code)
        loss = compute_alignment_loss(mirrored, laser_xyz)
        print(f"组合: {code or 'none':>3s}    MSE误差: {loss:.6f}")
        compare_trajectories(mirrored, laser_xyz, title_suffix=f"(mirror: {code or 'none'})")

def parse_aubo_joints(path_txt):
    joints = []
    with open(path_txt, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if "Joints:" in line:
            match = re.search(r'\[([^\]]+)\]', lines[idx+1])
            if match:
                joints.append([float(x.strip()) for x in match.group(1).split(',')])
    return np.array(joints)

def parse_laser_xyz(path_csv):
    xyz_list = []
    with open(path_csv, 'r', encoding='gbk') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            try:
                x = float(row[1].strip()) / 1000.0
                y = float(row[2].strip()) / 1000.0
                z = float(row[3].strip()) / 1000.0
                xyz_list.append([x, y, z])
            except:
                continue
    return np.array(xyz_list)

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    aubo_txt = "aubo_record_2025-05-01_01-41-25.txt"
    laser_csv = "laser.csv"

    dh_final = np.array([
        [0.0,     0.1634,  np.pi/2,  0.0],
        [0.647,   0.0,     0.0,      np.pi],
        [0.6005,  0.0,     0.0,      np.pi],
        [0.0,     0.2013, -np.pi/2, -np.pi/2],
        [0.0,     0.1025,  np.pi/2,  0.0],
        [0.0,     0.094,   0.0,      0.0],
    ], dtype=np.float32)

    p3_final = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    joint_list = parse_aubo_joints(aubo_txt)
    xyz_list = parse_laser_xyz(laser_csv)

    min_len = min(len(joint_list), len(xyz_list))
    joint_list = joint_list[:min_len]
    xyz_list = xyz_list[:min_len]

    xyz_fk = forward_fk(dh_final, joint_list, p3_final)

    xyz_fk = apply_mirror(xyz_fk, 'xy')

    # compare_trajectories(xyz_fk, xyz_list, title_suffix=f"(mirror: x)")

    # -------------------- 外参优化（SO(3) + 平移） --------------------
    # 转为 torch tensor
    xyz_fk_tensor = torch.tensor(xyz_fk, dtype=torch.float32)
    xyz_list_tensor = torch.tensor(xyz_list, dtype=torch.float32)

    # 优化变量：旋转向量（李代数）和位移向量
    rot_vec = torch.zeros(3, requires_grad=True)
    trans = torch.zeros(3, requires_grad=True)

    optimizer = optim.Adam([rot_vec, trans], lr=1e-2)

    def so3_exp(rotvec):
        theta = torch.norm(rotvec)
        if theta.item() < 1e-5:
            return torch.eye(3)
        k = rotvec / theta
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    for step in range(500):
        R = so3_exp(rot_vec)
        pred = (R @ xyz_fk_tensor.T).T + trans
        loss = nn.MSELoss()(pred, xyz_list_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    print("优化后的旋转向量 rot_vec:", rot_vec.detach().numpy())
    print("优化后的平移向量 trans:", trans.detach().numpy())
