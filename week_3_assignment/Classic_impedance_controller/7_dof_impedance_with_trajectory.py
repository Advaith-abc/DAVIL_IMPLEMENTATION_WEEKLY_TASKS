import numpy as np
import os
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "panda", "panda.xml")
end_effector_name = "hand"
TRAJ_SPEED_SCALE = 2.0   

# params for controller (Now using Impedance terminology)
K_POS = 200.0
B_POS = 30.0
K_ORI = 200.0
B_ORI = 30.0
NULLSPACE_DAMPING = 10.0 # Keeps the elbow stable

# for now I have taken hardcoded waypoints, but can be updated to some sort of planner in the future
WAYPOINTS_JOINT = [
    np.array([0.0,  0.0,   0.0,  -1.5708,  0.0,   1.5708,  0.0   ]),  
    np.array([0.4, -0.3,   0.3,  -1.2,     0.2,   1.8,     0.3   ]),  
    np.array([0.8, -0.5,   0.5,  -1.0,     0.4,   2.0,     0.5   ]),  
]

# use forward kinematics to get the end effector pose for each joint space wps
def get_ee_pose(q, model, data):
    data.qpos[:7] = q
    mujoco.mj_fwdPosition(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_name)
    pos  = data.xpos[ee_id].copy()
    quat = data.xquat[ee_id].copy()  
    return pos, quat

# havent read this part from the tb yet as it goes into IK, but this is necessary to convert motion in task space to joint torques
def get_jacobian(model, data):
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_name)
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, Jp, Jr, ee_id)
    return np.vstack([Jp[:, :7],
                      Jr[:, :7]])  # (6, 7)

# due to different conventions in mujoco (wxyz) and scipy (xyzw)
def wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

# allocate a time duration for each wp, its based on euclidean distance between the wps
def allocate_times(waypoints_task):
    durations = []
    for i in range(len(waypoints_task) - 1):
        dist = np.linalg.norm(waypoints_task[i+1][0] - waypoints_task[i][0])
        durations.append(max(dist * TRAJ_SPEED_SCALE, 0.5))
    print(f"Segment durations: {[f'{d:.3f}s' for d in durations]}")
    print(f"Total trajectory time: {sum(durations):.3f}s")
    return durations

#given the boundary conditions, we can solve for the coeffs of the quintic polynomial of each segment
def solve_quintic(p0, v0, a0, p1, v1, a1, T):
    A = np.array([
        [1,  0,    0,     0,      0,       0      ],
        [0,  1,    0,     0,      0,       0      ],
        [0,  0,    2,     0,      0,       0      ],
        [1,  T,    T**2,  T**3,   T**4,    T**5   ],
        [0,  1,    2*T,   3*T**2, 4*T**3,  5*T**4 ],
        [0,  0,    2,     6*T,    12*T**2, 20*T**3],
    ])
    b = np.array([p0, v0, a0, p1, v1, a1])
    return np.linalg.solve(A, b)

# given the coeffs of the quintic polynomial, we can eval poistion,velocity,accelaration at any time t
def eval_quintic(coeffs, t):
    c = coeffs
    pos = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

    vel = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4

    acc = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
    return pos, vel, acc

# Apply the boundary conditions, imp thing is that since I have split into segments, I need the boundary conditions at the intermediate ones as well
def build_quintic_trajectory(waypoints_task, durations):
    n         = len(waypoints_task)
    positions = np.array([wp[0] for wp in waypoints_task])  # (n, 3)

    # Estimate velocities at interior waypoints
    vels = np.zeros_like(positions)
    for i in range(1, n - 1):
        dt_prev  = durations[i - 1]
        dt_next  = durations[i]
        vels[i]  = (0.5 * (positions[i]   - positions[i-1]) / dt_prev + 0.5 * (positions[i+1] - positions[i])   / dt_next)

    accs    = np.zeros_like(positions)  
    segs    = []
    t_start = 0.0

    for i in range(n - 1):
        T      = durations[i]
        coeffs = np.zeros((3, 6))
        for ax in range(3):
            coeffs[ax] = solve_quintic(positions[i][ax], vels[i][ax], accs[i][ax], positions[i+1][ax], vels[i+1][ax], accs[i+1][ax], T)
        segs.append({'coeffs': coeffs, 'T': T, 't_start': t_start})
        t_start += T

    print(f"Built quintic trajectory: {len(segs)} segments")
    return segs

# declare the total time of the trajectory, finds the current segment based on the absolute time, then eval the local time in that segment to get the desired pos,vel,acc
def eval_quintic_trajectory(segs, t_abs):
    total_T = segs[-1]['t_start'] + segs[-1]['T']
    t_abs   = np.clip(t_abs, 0.0, total_T)

    seg = segs[-1]
    for s in segs:
        if t_abs <= s['t_start'] + s['T'] + 1e-9:
            seg = s
            break

    t_local = t_abs - seg['t_start']
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    for ax in range(3):
        pos[ax], vel[ax], acc[ax] = eval_quintic(seg['coeffs'][ax], t_local)
    return pos, vel, acc

# enforces sign consistency of quaternions, builds the segmnents
def build_slerp_trajectory(waypoints_task, durations):

    quats = [wp[1].copy() for wp in waypoints_task]  # wxyz

    # Enforce global sign consistency
    for i in range(1, len(quats)):
        q0 = wxyz_to_xyzw(quats[i - 1])
        q1 = wxyz_to_xyzw(quats[i])
        if np.dot(q0, q1) < 0.0:
            quats[i] = -quats[i]

    segs    = []
    t_start = 0.0
    for i in range(len(quats) - 1):
        segs.append({
            'q0':      quats[i],
            'q1':      quats[i + 1],
            'T':       durations[i],
            't_start': t_start
        })
        t_start += durations[i]

    print(f"Built SLERP trajectory: {len(segs)} segments")
    return segs

# find the segment based on time, then use the interpolation and accquire w
def eval_slerp_trajectory(segs, t_abs):
    total_T = segs[-1]['t_start'] + segs[-1]['T']
    t_abs   = np.clip(t_abs, 0.0, total_T)

    seg = segs[-1]
    for s in segs:
        if t_abs <= s['t_start'] + s['T'] + 1e-9:
            seg = s
            break

    t_local = t_abs - seg['t_start']
    t_norm  = t_local / seg['T']

    # using scipy's slerp
    q0 = wxyz_to_xyzw(seg['q0'])
    q1 = wxyz_to_xyzw(seg['q1'])
    key_rots = R.from_quat(np.array([q0, q1]))
    slerp_fn = Slerp([0.0, 1.0], key_rots)
    quat_interp = xyzw_to_wxyz(slerp_fn([t_norm]).as_quat()[0])

    # constant w
    r0    = R.from_quat(q0)
    r1    = R.from_quat(q1)
    omega = (r1 * r0.inv()).as_rotvec() / seg['T']  # (3,)

    return quat_interp, omega

# orientation error required by the controller
def orientation_error(q_des_wxyz, q_cur_wxyz):
    q_des = wxyz_to_xyzw(q_des_wxyz)
    q_cur = wxyz_to_xyzw(q_cur_wxyz)
    if np.dot(q_des, q_cur) < 0.0:
        q_des = -q_des
    delta_r = R.from_quat(q_des) * R.from_quat(q_cur).inv()
    return delta_r.as_rotvec()  # (3,)

# Reads task space motion and computes the joint torques using Impedance Control
def calc_impedance_torques(model, data, pos_des, vel_des, quat_des, omega_des):

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_name)

    # current state
    pos_cur  = data.xpos[ee_id].copy()
    quat_cur = data.xquat[ee_id].copy()
    J = get_jacobian(model, data)       
    xdot_cur = J @ data.qvel[:7]    
    vel_cur  = xdot_cur[:3]
    omg_cur  = xdot_cur[3:]

    # compute errors
    pos_err = pos_des  - pos_cur
    vel_err = vel_des  - vel_cur
    ori_err = orientation_error(quat_des, quat_cur)
    omg_err = omega_des - omg_cur

    error_6d = np.concatenate([pos_err, ori_err])
    vel_err_6d = np.concatenate([vel_err, omg_err])

    # Construct the 6x6 Stiffness and Damping matrices
    K_matrix = np.diag([K_POS]*3 + [K_ORI]*3)
    B_matrix = np.diag([B_POS]*3 + [B_ORI]*3)

    # Task space wrench (virtual spring-damper behavior)
    F = K_matrix @ error_6d + B_matrix @ vel_err_6d

    # 1. Primary Task Torques
    tau_task = J.T @ F

    # 2. Nullspace Torques (Damping internal elbow motions)
    J_T = J.T
    J_T_pinv = np.linalg.pinv(J_T)
    null_projector = np.eye(7) - (J_T @ J_T_pinv)
    tau_null = null_projector @ (-NULLSPACE_DAMPING * data.qvel[:7])

    # convert using jacobian and add gravity compensation
    tau = tau_task + tau_null + data.qfrc_bias[:7]
    return tau  # (7,)

def plot_trajectory_results(log_t, log_pos_des, log_pos_cur, log_err_pos, log_err_ori, log_tau):
    t = np.array(log_t)
    p_des = np.array(log_pos_des)
    p_cur = np.array(log_pos_cur)
    p_err = np.array(log_err_pos)

    # --- Figure 1: Position Tracking ---
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(t, p_des[:, 0], 'r--', alpha=0.7, label='X Desired')
    plt.plot(t, p_cur[:, 0], 'r', label='X Actual')
    plt.plot(t, p_des[:, 1], 'g--', alpha=0.7, label='Y Desired')
    plt.plot(t, p_cur[:, 1], 'g', label='Y Actual')
    plt.plot(t, p_des[:, 2], 'b--', alpha=0.7, label='Z Desired')
    plt.plot(t, p_cur[:, 2], 'b', label='Z Actual')
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.title('Task Space Position Tracking')
    plt.legend(loc='upper right', ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_1 = "position_tracking.png"
    plt.savefig(save_path_1, dpi=150, bbox_inches="tight")
    print(f"\nSaved tracking plot to -> {save_path_1}")

    # --- Figure 2: Position Error ---
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(t, p_err[:, 0], 'r', label='X Error')
    plt.plot(t, p_err[:, 1], 'g', label='Y Error')
    plt.plot(t, p_err[:, 2], 'b', label='Z Error')
    plt.ylabel('Error (m)')
    plt.xlabel('Time (s)')
    plt.title('Cartesian Position Error')
    plt.legend(loc='upper right', ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_2 = "position_error.png"
    plt.savefig(save_path_2, dpi=150, bbox_inches="tight")
    print(f"Saved error plot to -> {save_path_2}")

    plt.show()

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # converts the joint space waypoints to task space
    waypoints_task = []
    for i, q in enumerate(WAYPOINTS_JOINT):
        pos, quat = get_ee_pose(q, model, data)
        waypoints_task.append((pos, quat))
        print(f"  Waypoint {i}: pos={np.round(pos,3)}, quat(wxyz)={np.round(quat,3)}")

    # allocate the times to the segments, build the trajectories, and compute total time
    durations  = allocate_times(waypoints_task)
    quintic_segs = build_quintic_trajectory(waypoints_task, durations)
    slerp_segs   = build_slerp_trajectory(waypoints_task, durations)
    total_time   = sum(durations)

    mujoco.mj_resetData(model, data)
    data.qpos[:7] = WAYPOINTS_JOINT[0]
    mujoco.mj_forward(model, data)
    
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_name)
    
    log_t = []
    log_pos_des = []
    log_pos_cur = []
    log_err_pos = []
    log_err_ori = []
    log_tau = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        while viewer.is_running():
            # use the trajecotry
            pos_des, vel_des, _  = eval_quintic_trajectory(quintic_segs, t)
            quat_des, omega_des  = eval_slerp_trajectory(slerp_segs, t)

            # apply torques (Swapped to Impedance formulation)
            tau = calc_impedance_torques(model, data, pos_des, vel_des, quat_des, omega_des)
            data.ctrl[:7] = tau
            data.ctrl[7:] = 0.0  
            
            log_t.append(t)
            log_pos_des.append(pos_des.copy())
            pos_cur = data.xpos[ee_id].copy()
            log_pos_cur.append(pos_cur)
            log_err_pos.append(pos_des - pos_cur)
            
            ori_err_vec = orientation_error(quat_des, data.xquat[ee_id].copy())
            log_err_ori.append(np.linalg.norm(ori_err_vec))
            log_tau.append(tau.copy())

            mujoco.mj_step(model, data)
            viewer.sync()

            t += model.opt.timestep
            if t > total_time + 1.0:  
                print("Trajectory complete.")
                break
                
    plot_trajectory_results(log_t, log_pos_des, log_pos_cur, log_err_pos, log_err_ori, log_tau)

if __name__ == "__main__":
    main()