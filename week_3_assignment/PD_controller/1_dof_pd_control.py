import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time 
from dataclasses import dataclass

XML = """
<mujoco model="1dof_pd_ablation">
  <compiler angle="radian"/> <option gravity="0 0 -9.81" timestep="0.002"/>
 
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="4 4 0.1" rgba="0.9 0.9 0.9 1"/>
 
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"
             limited="true" range="-3.14159 3.14159"/>
      <geom type="capsule" fromto="0 0 0  1 0 0"
            size="0.05" rgba="0.8 0.2 0.2 1" mass="1.0"/>
      <site name="tip" pos="1 0 0" size="0.06" rgba="0 1 1 1"/>
    </body>
  </worldbody>
 
  <actuator>
    <motor name="act_joint1" joint="joint1" gear="1"/>
  </actuator>
</mujoco>
"""
 


PROFILES = [
    ("underdamped  (kp=300, kv=1)",   300.0,  1.0),
    ("nominal      (kp=100, kv=10)",  100.0, 10.0),
    ("overdamped   (kp=100, kv=80)",  100.0, 80.0),
    ("high-gain    (kp=500, kv=40)",  500.0, 40.0),
    ("low-gain     (kp=20,  kv=5)",    20.0,  5.0),
]


Q_START  = 0.0          # starting joint angle
Q_TARGET = np.pi / 2    # target  joint angle  

SIM_DURATION       = 4.0   # seconds per profile
CONVERGENCE_THRESH = 0.01  

def compute_torque(data: mujoco.MjData, kp: float, kv: float) -> float:
    error = Q_TARGET - data.qpos[0]
    return kp * error - kv * data.qvel[0]


@dataclass
class RunResult:
    label:   str
    times:   np.ndarray   # (n_steps,)
    error:   np.ndarray   # q_target - q  at each step  (rad)
    torque:  np.ndarray   # actuator force at each step (N·m)


def run_profile(label: str, kp: float, kv: float) -> RunResult:

    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)


    data.qpos[0] = Q_START
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    dt      = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    times   = np.zeros(n_steps)
    errors  = np.zeros(n_steps)
    torques = np.zeros(n_steps)

    print(f"  Profile : {label}")
    print(f"  kp = {kp:.1f}   kv = {kv:.1f}")

    converged = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                print(f"  Viewer closed at step {step}  (t = {step*dt:.3f}s)")
                break

            tau = compute_torque(data, kp, kv)
            data.ctrl[0] = tau
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            t              = step * dt
            err            = Q_TARGET - data.qpos[0]
            times[step]    = t
            errors[step]   = err
            torques[step]  = data.actuator_force[0]

            if step == 10: 
                print(f"  step 10 | tau={tau:.2f}  q={data.qpos[0]:.4f}  err={err:.4f}")

            if not converged and abs(err) < CONVERGENCE_THRESH:
                converged = True
                print(f"  Converged at t = {t:.3f}s")

    if not converged:
        print(f"  Did NOT converge — final error = {errors[-1]*1000:.1f} mrad")

    return RunResult(label=label, times=times, error=errors, torque=torques)


def plot_results(results: list[RunResult]) -> None:
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"1-DOF arm — PD gain ablation   |   target = {np.degrees(Q_TARGET):.1f}°",
        fontsize=11, fontweight="normal"
    )

    # Plot 1: joint error
    ax1 = axes[0]
    for res, c in zip(results, colors):
        ax1.plot(res.times, np.degrees(res.error),
                 label=res.label, color=c, linewidth=1.8)
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Position error (deg)")
    ax1.set_title("Joint error  (q_target − q)")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # Plot 2: actuator torque
    ax2 = axes[1]
    for res, c in zip(results, colors):
        ax2.plot(res.times, res.torque,
                 label=res.label, color=c, linewidth=1.8)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (N·m)")
    ax2.set_title("Actuator torque")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("pd_1dof_ablation.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → pd_1dof_ablation.png")
    plt.show()


def print_summary(results: list[RunResult]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {'Profile':<38} {'Settle (s)':>10}  {'Final err (mrad)':>15}")
    print(f"{'─'*60}")
    for res in results:
        # settled = first index where error stays below threshold for 100 steps
        settled_idx = next(
            (i for i in range(len(res.error) - 100)
             if np.all(np.abs(res.error[i:i+100]) < CONVERGENCE_THRESH)),
            None
        )
        settle_str    = f"{res.times[settled_idx]:.3f}" if settled_idx else f">{SIM_DURATION:.1f}"
        final_err_str = f"{res.error[-1]*1000:.1f}"
        print(f"  {res.label:<38} {settle_str:>10}  {final_err_str:>15}")

def main():

    results = []
    for label, kp, kv in PROFILES:
        res = run_profile(label, kp, kv)
        results.append(res)

    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()