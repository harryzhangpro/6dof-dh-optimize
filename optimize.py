import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import math
import numpy as np
import argparse
import csv
import re
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import transform_from, plot_transform
import pytransform3d.plot_utils as p3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(suppress=True)

# 激光外参
laser_angles_deg = torch.tensor([0., 0., 50.], dtype=torch.float32, device=device)
laser_omega = laser_angles_deg * (np.pi/180)  # axis-angle vector: [ωx, ωy, ωz]
laser_t = torch.tensor([1.1, 1.4, 0.], dtype=torch.float32, device=device)

dh_true = np.array([
    [0.1632, 0., 0.5 * np.pi, 0.],
    [0., 0.647, np.pi, 0.5 * np.pi],
    [0., 0.6005, np.pi, 0.],
    [0.2013, 0., -0.5 * np.pi, -0.5 * np.pi],
    [0.1025, 0., 0.5 * np.pi, 0.],
    [0.094, 0., 0., 0.]
], dtype=np.float32)
# True sphere center in flange frame (meters)
p3_true = torch.tensor([1., 1., 1.], dtype=torch.float32, device=device)

def dh_transform_batch(theta, d, a, alpha):
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    B = theta.shape[0]
    T = torch.zeros((B, 4, 4), dtype=theta.dtype, device=theta.device)

    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct

    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st

    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d

    T[:, 3, 3] = 1.0
    return T

def forward_kinematics_batch(joint_angles_batch, dh_params):
    B = joint_angles_batch.shape[0]
    T = torch.eye(4, device=joint_angles_batch.device).unsqueeze(0).repeat(B, 1, 1)
    for i in range(6):
        theta = joint_angles_batch[:, i] + dh_params[i, 3]
        d = dh_params[i, 0]
        a = dh_params[i, 1]
        alpha = dh_params[i, 2]
        T_i = dh_transform_batch(theta, d, a, alpha)
        T = torch.bmm(T, T_i)
    return T

def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(omega)
    axis = omega / (theta + 1e-8)  # 避免除 0

    x, y, z = axis
    K = torch.zeros((3, 3), dtype=omega.dtype, device=omega.device)
    K[0, 1] = -z
    K[0, 2] = y
    K[1, 0] = z
    K[1, 2] = -x
    K[2, 0] = -y
    K[2, 1] = x

    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    R = eye + sin_theta * K + (1 - cos_theta) * (K @ K)

    return R

def generate_data_batch(mode: str = 'single', steps: int = 20, N_multi: int = 120):
    if mode == 'single':
        samples = []
        for axis in range(6):
            for val in np.linspace(-np.pi/4, np.pi/4, steps, dtype=np.float32):
                q = np.zeros(6, dtype=np.float32)
                q[axis] = val
                samples.append(q)
        joint_array = np.array(samples)
    else:
        t = np.linspace(0, 1, N_multi, dtype=np.float32)
        joint_array = np.stack([
            np.sin(t * np.pi),
            np.sin(t * np.pi * 0.8),
            np.cos(t * np.pi),
            np.cos(t * np.pi * 1.2),
            np.sin(t * np.pi * 1.5),
            np.cos(t * np.pi * 0.5)
        ], axis=1) * (np.pi/4)

    joint_tensor = torch.tensor(joint_array, dtype=torch.float32, device=device)
    dh_tensor = torch.tensor(dh_true, dtype=torch.float32, device=device)
    T_batch = forward_kinematics_batch(joint_tensor, dh_tensor)  # (N,4,4)

    # 提取 R 和 t
    R_base_flange = T_batch[:, :3, :3]        # (N,3,3)
    p_base_flange = T_batch[:, :3, 3]         # (N,3)

    p_flange_laser = p3_true[None, :].expand(R_base_flange.shape[0], -1)  # (N, 3)
    p_laser_base = torch.bmm(R_base_flange, p_flange_laser.unsqueeze(-1)).squeeze(-1) + p_base_flange
    
    # 激光器外参变换（从 base 到激光坐标系）
    R_laser = so3_exp_map(laser_omega)       # (3,3)
    R_laser_T = R_laser.T
    p_laser = torch.matmul(p_laser_base - laser_t, R_laser_T)  # (N,3)

    return joint_array, p_laser.cpu().numpy()

def compute_loss_batch(dh_params, p3, omega, joint_list, xyz_list, neighbor_pairs, focus_axis=None):
    """
    Compute MSE loss over neighbor pairs using SO(3) for laser rotation.
    dh_params: (6,4) tensor
    p3: (3,) tensor
    omega: (3,) tensor axis-angle for laser
    joint_list: (N,6) numpy array
    xyz_list: (N,3) numpy array
    neighbor_pairs: list of (i,j) index pairs
    """
    idx_i = [i for i,j in neighbor_pairs]
    idx_j = [j for i,j in neighbor_pairs]

    # compute rotation matrix via exponential map
    R_laser = so3_exp_map(omega)

    # measurements
    qi = torch.tensor(joint_list[idx_i], dtype=torch.float32, device=device)
    qj = torch.tensor(joint_list[idx_j], dtype=torch.float32, device=device)
    pi_meas = torch.tensor(xyz_list[idx_i], dtype=torch.float32, device=device)
    pj_meas = torch.tensor(xyz_list[idx_j], dtype=torch.float32, device=device)

    # forward kinematics
    Ti = forward_kinematics_batch(qi, dh_params)
    Tj = forward_kinematics_batch(qj, dh_params)

    B = Ti.size(0)
    p3_batch = p3.unsqueeze(0).expand(B, -1)

    pi_fk = torch.bmm(Ti[:, :3, :3], p3_batch.unsqueeze(-1)).squeeze(-1) + Ti[:, :3, 3]
    pj_fk = torch.bmm(Tj[:, :3, :3], p3_batch.unsqueeze(-1)).squeeze(-1) + Tj[:, :3, 3]

    # rotate into laser frame
    # R_laser_inv = R_laser.T
    # pi_rot = (R_laser_inv @ pi_fk.T).T
    # pj_rot = (R_laser_inv @ pj_fk.T).T
    pi_rot = (R_laser @ pi_fk.T).T
    pj_rot = (R_laser @ pj_fk.T).T

    delta_fk = pj_rot - pi_rot
    delta_meas = pj_meas - pi_meas

    if focus_axis is not None:
        delta_theta = torch.abs(qj - qi)
        w = delta_theta[:, focus_axis] / (torch.sum(delta_theta, dim=1) + 1e-6)
        loss = torch.sum(w * torch.sum((delta_fk - delta_meas)**2, dim=1))
    else:
        loss = torch.sum((delta_fk - delta_meas)**2)

    return loss

def optimize(dh, joint_list, xyz_list, lr):
    pairs = [(i, i+1) for i in range(len(joint_list)-1)]
    dh_mask = (torch.tensor(dh_true, dtype=torch.float32, device=device) != 0).float()
    dh_params_raw = torch.tensor(dh, dtype=torch.float32, device=device, requires_grad=True)
    p3 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
    # omega = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    omega = (torch.tensor([0.1, 0.1, 0.1]) * np.pi / 180).to(device).detach().clone().requires_grad_(True)
    dh_init = (dh_mask * dh_params_raw + (1 - dh_mask) * torch.tensor(dh_true, dtype=torch.float32, device=device)).detach().cpu().numpy().copy()
    p3_init = p3.detach().cpu().numpy().copy()
    print("\nomega + p3...")
    optimizer1 = torch.optim.Adam([omega, p3], lr=1e-1)
    for epoch in range(2000):
        optimizer1.zero_grad()
        dh_params = dh_mask * dh_params_raw + (1 - dh_mask) * torch.tensor(dh_true, dtype=torch.float32, device=device)
        loss = compute_loss_batch(dh_params, p3, omega, joint_list, xyz_list, pairs, focus_axis=None)
        loss.backward()
        optimizer1.step()
        print(f"[omega + p3] Epoch {epoch}: Loss={loss.detach().item():.10f}", end='\r', flush=True)

    print("omega_est (deg):", np.degrees(omega.detach().cpu().numpy()))
    print(p3.detach().cpu().numpy())

    print("\nomega + mdh + p3...")
    optimizer2 = torch.optim.Adam([dh_params_raw, p3, omega], lr=lr)
    for axis in range(6):
        for epoch in range(1000):
            optimizer2.zero_grad()
            dh_params = dh_mask * dh_params_raw + (1 - dh_mask) * torch.tensor(dh_true, dtype=torch.float32, device=device)
            loss = compute_loss_batch(dh_params, p3, omega, joint_list, xyz_list, pairs, focus_axis=axis)
            loss.backward()
            optimizer2.step()
            print(f"[Axis{axis+1}] Epoch {epoch}: Loss={loss.detach().item():.10f}", end='\r', flush=True)
        print("\r")

    print("\n联合微调...")
    for epoch in range(2000):
        optimizer2.zero_grad()
        dh_params = dh_mask * dh_params_raw + (1 - dh_mask) * torch.tensor(dh_true, dtype=torch.float32, device=device)
        loss = compute_loss_batch(dh_params, p3, omega, joint_list, xyz_list, pairs, focus_axis=None)
        loss.backward()
        optimizer2.step()
        print(f"[all Joint] Epoch {epoch}: Loss={loss.detach().item():.10f}", end='\r', flush=True)

    dh_final = (dh_mask * dh_params_raw + (1 - dh_mask) * torch.tensor(dh_true, dtype=torch.float32, device=device)).detach().cpu().numpy()
    p3_final = p3.detach().cpu().numpy()
    omega_final = omega.detach().cpu().numpy()

    return dh_final, p3_final, dh_init, p3_init, omega_final

def load_joint_and_xyz(aubo_txt_path, laser_csv_path):
    joint_list = []
    xyz_list = []

    with open(aubo_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].strip().startswith("Joints:") and (i + 1 < len(lines)):
                next_line = lines[i + 1].strip()
                match = re.search(r'\[([^\]]+)\]', next_line)
                if match:
                    joint_values = [float(x.strip()) for x in match.group(1).split(',')]
                    if len(joint_values) == 6:
                        joint_list.append(joint_values)

    with open(laser_csv_path, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 4:
                try:
                    x = float(row[1]) / 1000.0
                    y = float(row[2]) / 1000.0
                    z = float(row[3]) / 1000.0
                    xyz_list.append([x, y, z])
                except ValueError:
                    continue  # 跳过非法行

    min_len = min(len(joint_list), len(xyz_list))
    joint_array = np.array(joint_list[:min_len], dtype=np.float32)
    xyz_array = np.array(xyz_list[:min_len], dtype=np.float32)

    assert joint_array.ndim == 2 and joint_array.shape[1] == 6, \
        f"joint_array shape 异常: {joint_array.shape}, 应为 (N,6)"
    assert xyz_array.ndim == 2 and xyz_array.shape[1] == 3, \
        f"xyz_array shape 异常: {xyz_array.shape}, 应为 (N,3)"

    print(f"[INFO] 成功读取 joint={joint_array.shape}, xyz={xyz_array.shape} (单位: 米)")
    return joint_array, xyz_array

def visualize_generated_p3_positions(joint_list, xyz_list):
    R_laser = so3_exp_map(laser_omega).cpu().numpy()
    t_laser = laser_t.cpu().numpy()

    flange_points = []
    base_points = []
    xyz_points = np.array(xyz_list)
    flange_transforms = []

    dh_tensor = torch.tensor(dh_true, dtype=torch.float32, device=device)

    for q in joint_list:
        q_torch = torch.tensor(q, dtype=torch.float32, device=device)
        T = forward_kinematics_batch(q_torch.unsqueeze(0), dh_tensor)[0].cpu().numpy()
        R_flange_base = T[:3, :3]
        p_flange_base = T[:3, 3]

        # base → laser 坐标变换
        p_flange_laser = (p_flange_base - t_laser) @ R_laser.T
        p_base_laser = (-t_laser) @ R_laser.T

        R_flange_laser = R_flange_base @ R_laser.T
        T_flange_laser = transform_from(R=R_flange_laser, p=p_flange_laser)

        flange_points.append(p_flange_laser)
        base_points.append(p_base_laser)
        flange_transforms.append(T_flange_laser)

    flange_points = np.array(flange_points)
    base_points = np.array(base_points)

    # 自适应范围
    all_pts = np.concatenate([flange_points, base_points, xyz_points], axis=0)
    center = np.mean(all_pts, axis=0)
    max_range = np.max(np.ptp(all_pts, axis=0)) * 0.6

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Flange Poses in Laser Frame")
    ax.set_xlabel("X (laser)")
    ax.set_ylabel("Y (laser)")
    ax.set_zlabel("Z (laser)")
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # 点绘制
    ax.scatter(0, 0, 0, c='red', s=80, marker='*', label='Laser (origin)')
    ax.scatter(base_points[:, 0], base_points[:, 1], base_points[:, 2], c='blue', s=30, label='Base (in laser frame)')
    ax.scatter(flange_points[:, 0], flange_points[:, 1], flange_points[:, 2], c='green', s=20, label='Flange (in laser frame)')
    ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], c='orange', s=20, alpha=0.6, label='Laser hit (xyz_list)')

    # 连线
    for base, flange, hit in zip(base_points, flange_points, xyz_points):
        ax.plot([base[0], flange[0]], [base[1], flange[1]], [base[2], flange[2]], 'k--', alpha=0.2)
        ax.plot([flange[0], hit[0]], [flange[1], hit[1]], [flange[2], hit[2]], 'g-', alpha=0.4)

    # 所有 flange 坐标系箭头
    for T in flange_transforms:
        plot_transform(ax=ax, A2B=T, s=1)

    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_arm_and_laser_points(joint_list, xyz_list, dh_params, num_points=10):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("visualize_arm_and_laser_points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(min(num_points, len(joint_list))):
        q = torch.tensor(joint_list[i], dtype=torch.float32, device=device).unsqueeze(0)
        T = forward_kinematics_batch(q, dh_params)[0]

        flange_in_base = T[:3, 3].detach().cpu().numpy()
        p3 = p3_true.detach().cpu().numpy()
        hit_base = T[:3, :3].detach().cpu().numpy() @ p3 + flange_in_base
        hit_laser = xyz_list[i]

        ax.scatter(*hit_base, c='green', s=40)
        ax.scatter(*hit_laser, c='orange', s=40)

        offset = 0.05
        ax.text(*hit_base + offset, f"F{i}", fontsize=9, color='green', ha='center', va='center')
        ax.text(*hit_laser + offset, f"L{i}", fontsize=9, color='orange', ha='center', va='center')

    plt.tight_layout()
    plt.show()

def mirror_pose_data(xyz_array: np.ndarray, mode: str = 'x') -> np.ndarray:
    """
    对 xyz 数据做镜像变换：
    - mode = 'x'：沿 x 轴镜像（x → -x）
    - mode = 'y'：沿 y 轴镜像
    - mode = 'z'：沿 z 轴镜像
    - mode = 'xyz': 同时镜像所有轴

    Returns: 镜像后的 xyz_array
    """
    mirrored = xyz_array.copy()
    if mode == 'x':
        mirrored[:, 0] *= -1
    elif mode == 'y':
        mirrored[:, 1] *= -1
    elif mode == 'z':
        mirrored[:, 2] *= -1
    elif mode == 'xyz':
        mirrored *= -1
    else:
        raise ValueError(f"不支持的镜像模式: {mode}")
    return mirrored

def run_sim(mode="single"):
    joint_list, xyz_list = generate_data_batch(mode=mode, steps=20, N_multi= 120)

    visualize_arm_and_laser_points(joint_list, xyz_list, 
                                  torch.tensor(dh_true, dtype=torch.float32, device=device),
                                  num_points=10)

    visualize_generated_p3_positions(joint_list, xyz_list)

    dh_sim =dh_true + np.random.normal(scale=0.001, size=(6,4))
    dh_final, p3_final, dh_init, p3_init, omega = optimize(dh_sim, joint_list, xyz_list, 1e-3)

    print(dh_init)
    print(dh_final)
    print(p3_init)
    print(p3_final)
    print("omega_est (deg):", np.degrees(omega))

def run_true(aubo_txt_path, laser_csv_path):
    joint_list, xyz_list = load_joint_and_xyz(aubo_txt_path, laser_csv_path)

    visualize_arm_and_laser_points(joint_list, xyz_list, 
                                  torch.tensor(dh_true, dtype=torch.float32, device=device),
                                  num_points=10)

    # xyz_list = mirror_pose_data(xyz_list, 'z')
    dh_final, p3_final, dh_init, p3_init, omega = optimize(dh_true, joint_list, xyz_list, 1e-3)

    print(dh_init)
    print(dh_final)
    print(p3_init)
    print(p3_final)
    print("omega_est (deg):", np.degrees(omega))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help='使用模拟数据进行DH运算')
    parser.add_argument('--true', nargs=2, metavar=('AUBO_TXT', 'LASER_CSV'), help='使用真实数据进行DH运算')

    args = parser.parse_args()

    if args.sim:
        print("使用模拟数据运行")
        run_sim("single")
    elif args.true:
        aubo_path, laser_path = args.true
        print(f"使用真实数据运行：{aubo_path} 和 {laser_path}")
        run_true(aubo_path, laser_path)  
    else:
        print("请指定 --sim 或 --true aubo.txt laser.csv")