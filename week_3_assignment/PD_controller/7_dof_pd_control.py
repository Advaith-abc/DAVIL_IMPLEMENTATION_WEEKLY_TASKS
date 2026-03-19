import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "panda", "panda.xml")

DEFAULT_KP = 200.0
DEFAULT_KD = 30.0

# Scale factors to balance the contribution of each joint to the overall error and torque magnitudes
JOINT_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25])

SIM_DURATION       = 5.0
CONVERGENCE_THRESH = 0.01
SEED               = 42


def sample_setpoint(model: mujoco.MjModel, rng: np.random.Generator) -> np.ndarray:
    lo = model.jnt_range[:7, 0]
    hi = model.jnt_range[:7, 1]
    return rng.uniform(lo, hi)


def run_torque(model, q_home, q_target, kp, kd):
    kp_vec = kp * JOINT_SCALE
    kd_vec = kd * JOINT_SCALE

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = q_home
    mujoco.mj_forward(model, data)

    dt      = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    times   = np.zeros(n_steps)
    q_log   = np.zeros((n_steps, 7))

    print(f"\n  (Kp={kp}, Kd={kd})")
    converged = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                break

            with viewer.lock():
                err           = q_target - data.qpos[:7]
                tau           = kp_vec * err - kd_vec * data.qvel[:7] + data.qfrc_bias[:7]
                data.ctrl[:7] = tau
                data.ctrl[7:] = 0.0
                mujoco.mj_step(model, data)

            viewer.sync()
            try:
                time.sleep(dt)
            except KeyboardInterrupt:
                raise

            times[step] = step * dt
            q_log[step] = data.qpos[:7].copy()

            if not converged and np.max(np.abs(q_target - data.qpos[:7])) < CONVERGENCE_THRESH:
                converged = True
                print(f"   Converged at t = {step*dt:.3f}s")

    if not converged:
        print(f"   Did not converge (final max|err| = "
              f"{np.max(np.abs(q_target - q_log[-1]))*1000:.1f} mrad)")

    return times, q_log

def plot_joint_angles(times_t, q_log_t, q_target, kp, kd):
    plt.close("all")
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        f"Joint angles — Torque control\n"
        f"Kp = {kp}  |  Kd = {kd}",
        fontsize=11, fontweight="normal"
    )
 
    for j in range(7):
        ax = axes[j]
 
        ax.axhline(np.degrees(q_target[j]),
                   color="black", linewidth=2.0, linestyle="-", label="target")
 
        ax.plot(times_t, np.degrees(q_log_t[:, j]),
                color="#378ADD", linewidth=1.4, label="Joint angle")
 
        ax.set_ylabel(f"J{j+1} (deg)", fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)
 
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
 
    out_path = os.path.join(os.path.dirname(__file__), f"pd_7dof_joint_angles_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)

def prompt_gains(current_kp, current_kd):
    print(f"\n{'─'*50}")
    print(f"  Current gains: Kp = {current_kp}   Kd = {current_kd}")
    try:
        kp_in = input(f"  New Kp [{current_kp}]: ").strip()
        kd_in = input(f"  New Kd [{current_kd}]: ").strip()
        kp = float(kp_in) if kp_in else current_kp
        kd = float(kd_in) if kd_in else current_kd
    except ValueError:
        print("  Invalid input — keeping current gains")
        kp, kd = current_kp, current_kd
    return kp, kd

def main():
    rng   = np.random.default_rng(seed=SEED)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    q_home   = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0])
    q_target = sample_setpoint(model, rng)

    print(f"  Target (deg): {np.round(np.degrees(q_target), 1)}")

    kp, kd = DEFAULT_KP, DEFAULT_KD

    while True:
        try:
            # Run simulation
            times_t, q_log_t = run_torque(model, q_home, q_target, kp, kd)
            
            # Plot the results
            plot_joint_angles(times_t, q_log_t, q_target, kp, kd)

            # Ask for new inputs
            print("\n  Enter different gain values")
            kp, kd = prompt_gains(kp, kd)
            
        except KeyboardInterrupt:
            print("\nExiting program...")
            plt.close('all') 
            sys.exit(0)      
    plt.show()

if __name__ == "__main__":
    main()