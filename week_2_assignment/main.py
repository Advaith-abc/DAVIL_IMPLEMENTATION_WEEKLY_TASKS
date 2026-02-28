import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R, Slerp

MODEL_PATH    = "assets/panda/panda.xml"
end_effector_name = "hand"
TRAJ_SPEED_SCALE = 2.0   

# params for controller
KP_POS = 200.0
KD_POS = 30.0
KP_ORI = 200.0
KD_ORI = 30.0

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

# reads task space motion and computes the joint torques, then adds gravity compensation
def calc_torques(model, data, pos_des, vel_des, quat_des, omega_des):

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

    # task space wrench
    F = np.concatenate([
        KP_POS * pos_err + KD_POS * vel_err,
        KP_ORI * ori_err + KD_ORI * omg_err,
    ])  # (6,)

    # conert using jacobian and add gracity compensation
    tau = J.T @ F + data.qfrc_bias[:7]
    return tau  # (7,)

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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        while viewer.is_running():
            # use the trajecotry
            pos_des, vel_des, _  = eval_quintic_trajectory(quintic_segs, t)
            quat_des, omega_des  = eval_slerp_trajectory(slerp_segs, t)

            # apply torques
            tau = calc_torques(model, data, pos_des, vel_des, quat_des, omega_des)
            data.ctrl[:7] = tau
            data.ctrl[7:] = 0.0  

            mujoco.mj_step(model, data)
            viewer.sync()

            t += model.opt.timestep
            if t > total_time + 1.0:  
                print("Trajectory complete.")
                break

if __name__ == "__main__":
    main()