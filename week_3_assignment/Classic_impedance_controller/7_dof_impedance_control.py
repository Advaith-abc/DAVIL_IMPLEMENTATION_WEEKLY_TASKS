import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "panda", "panda.xml")

# Cartesian Stiffness (N/m) and Damping (N.s/m)
DEFAULT_K_POS = 1000.0
DEFAULT_B_POS = 50.0

# Rotational Stiffness (Nm/rad) and Damping (Nm.s/rad)
DEFAULT_K_ORI = 50.0
DEFAULT_B_ORI = 5.0

SIM_DURATION       = 5.0
CONVERGENCE_THRESH = 0.005 # Meters
SEED               = 42


def sample_cartesian_setpoint(model, data, body_id, rng):

    lo = model.jnt_range[:7, 0]
    hi = model.jnt_range[:7, 1]
    q_target = rng.uniform(lo, hi)
    
    data.qpos[:7] = q_target
    mujoco.mj_kinematics(model, data)
    
    x_target    = np.copy(data.xpos[body_id])
    quat_target = np.copy(data.xquat[body_id])
    
    return x_target, quat_target, q_target


def run_impedance(model, q_home, x_target, quat_target, K_pos, B_pos, body_id):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = q_home
    mujoco.mj_forward(model, data)

    dt      = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    times   = np.zeros(n_steps)
    
    # Pre-allocate arrays to log BOTH states
    pos_log = np.zeros((n_steps, 3))
    q_log   = np.zeros((n_steps, 7))

    K_matrix = np.diag([K_pos]*3 + [DEFAULT_K_ORI]*3)
    B_matrix = np.diag([B_pos]*3 + [DEFAULT_B_ORI]*3)

    print(f"\n  (K_pos={K_pos}, B_pos={B_pos})")
    converged = False

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    ori_err = np.zeros(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                break

            with viewer.lock():
                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                J = np.vstack([jacp[:, :7], jacr[:, :7]]) 
                
                pos_err = x_target - data.xpos[body_id]
                mujoco.mju_subQuat(ori_err, quat_target, data.xquat[body_id]) 
                error_6d = np.hstack([pos_err, ori_err])
                
                x_dot = J @ data.qvel[:7]
                F_cart = K_matrix @ error_6d - B_matrix @ x_dot
                tau_task = J.T @ F_cart
                
                # Nullspace Damping
                J_T = J.T
                J_T_pinv = np.linalg.pinv(J_T)
                null_projector = np.eye(7) - (J_T @ J_T_pinv)
                tau_null = null_projector @ (-10.0 * data.qvel[:7])
                
                data.ctrl[:7] = tau_task + tau_null + data.qfrc_bias[:7]
                data.ctrl[7:] = 0.0 
                
                mujoco.mj_step(model, data)

            viewer.sync()
            try:
                time.sleep(dt)
            except KeyboardInterrupt:
                raise

            times[step] = step * dt
            # Log both Cartesian Position and Joint Angles
            pos_log[step] = data.xpos[body_id].copy()
            q_log[step]   = data.qpos[:7].copy() 

            err_mag = np.linalg.norm(pos_err)
            if not converged and err_mag < CONVERGENCE_THRESH:
                converged = True
                print(f"   Converged to Cartesian target at t = {step*dt:.3f}s")

    if not converged:
        print(f"   Did not converge (final Cartesian error = {err_mag*1000:.1f} mm)")

    return times, pos_log, q_log


def plot_errors(times_t, pos_log_t, x_target, q_log_t, q_target, K_pos, B_pos):
    plt.close("all") 
    
    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig1.canvas.manager.set_window_title('Cartesian Errors')
    fig1.suptitle(
        f"Cartesian Position Error (Target - Actual)\n"
        f"K_pos = {K_pos}  |  B_pos = {B_pos}",
        fontsize=12, fontweight="bold"
    )
    
    pos_error = x_target - pos_log_t
    labels_xyz = ['X Error (m)', 'Y Error (m)', 'Z Error (m)']
    colors_xyz = ['#d62728', '#2ca02c', '#1f77b4']

    for i in range(3):
        ax = axes1[i]
        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", label="Zero Error")
        ax.plot(times_t, pos_error[:, i], color=colors_xyz[i], linewidth=2.0, label=labels_xyz[i])
        ax.set_ylabel(labels_xyz[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=10, loc="upper right")
    
    axes1[-1].set_xlabel("Time (s)", fontsize=11)
    fig1.tight_layout()
    
    fig1_filename = f"7_dof_cartesian_err_K{int(K_pos)}_B{int(B_pos)}.png"
    fig1.savefig(fig1_filename, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {fig1_filename}")

    fig2, axes2 = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
    fig2.canvas.manager.set_window_title('Joint Errors')
    fig2.suptitle(
        f"Joint Angle Errors (Target - Actual)\n"
        f"Notice how errors may not reach 0° due to 7-DOF Kinematic Redundancy!\n"
        f"K_pos = {K_pos}  |  B_pos = {B_pos}",
        fontsize=12, fontweight="bold"
    )
 
    q_error = q_target - q_log_t 

    for j in range(7):
        ax = axes2[j]
        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", label="Zero Error")
        ax.plot(times_t, np.degrees(q_error[:, j]), color="#ff7f0e", linewidth=2.0, label=f"Joint {j+1} Error")
        ax.set_ylabel(f"J{j+1} (deg)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=10, loc="upper right")
 
    axes2[-1].set_xlabel("Time (s)", fontsize=12)
    fig2.tight_layout()
    
    fig2_filename = f"7_dof_joint_err_K{int(K_pos)}_B{int(B_pos)}.png"
    fig2.savefig(fig2_filename, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {fig2_filename}")

    fig1.show()
    fig2.show()
    plt.pause(0.1)


def prompt_gains(current_k, current_b):
    print(f"\n{'─'*50}")
    print(f"  Current gains: K_pos = {current_k}   B_pos = {current_b}")
    try:
        k_in = input(f"  New K_pos [{current_k}]: ").strip()
        b_in = input(f"  New B_pos [{current_b}]: ").strip()
        K_pos = float(k_in) if k_in else current_k
        B_pos = float(b_in) if b_in else current_b
    except ValueError:
        print("  Invalid input — keeping current gains")
        K_pos, B_pos = current_k, current_b
    return K_pos, B_pos


def main():
    rng   = np.random.default_rng(seed=SEED)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
    q_home = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0])
    
    data = mujoco.MjData(model)
    x_target, quat_target, q_target = sample_cartesian_setpoint(model, data, body_id, rng)

    print(f"  Cartesian Target (m): X={x_target[0]:.3f}, Y={x_target[1]:.3f}, Z={x_target[2]:.3f}")

    K_pos, B_pos = DEFAULT_K_POS, DEFAULT_B_POS

    while True:
        try:
            times_t, pos_log_t, q_log_t = run_impedance(model, q_home, x_target, quat_target, K_pos, B_pos, body_id)
            
            plot_errors(times_t, pos_log_t, x_target, q_log_t, q_target, K_pos, B_pos)

            print("\n  Enter different Cartesian gain values")
            K_pos, B_pos = prompt_gains(K_pos, B_pos)
            
        except KeyboardInterrupt:
            print("\nExiting program...")
            plt.close('all') 
            sys.exit(0)      
    plt.show()

if __name__ == "__main__":
    main()