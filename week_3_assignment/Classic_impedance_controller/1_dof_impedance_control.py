import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time 
from dataclasses import dataclass

XML = """
<mujoco model="1dof_impedance_control">
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

# stiffness and dampness values
IMPEDANCE_PROFILES = [
    ("stiff_spring   (K=500, B=10)",  500.0, 10.0),
    ("soft_spring    (K=50,  B=2)",    50.0,  2.0),
    ("overdamped     (K=200, B=50)",  200.0, 50.0),
]

Q_START  = 0.0          
Q_TARGET = np.pi / 2    

SIM_DURATION = 15.0     

@dataclass
class RunResult:
    label:   str
    times:   np.ndarray   
    q_error: np.ndarray   
    torque:  np.ndarray   

def get_cartesian_target(model, data, site_id, target_q):
    q_orig = data.qpos[0]
    
    data.qpos[0] = target_q
    mujoco.mj_kinematics(model, data) 
    x_target = np.copy(data.site_xpos[site_id])
    
    # Revert state
    data.qpos[0] = q_orig
    mujoco.mj_kinematics(model, data)
    return x_target

def compute_impedance_torque(model, data, site_id, x_target, K_cart, B_cart) -> float:
    # Get current Cartesian position of the tip
    x_current = data.site_xpos[site_id]
    
    # Get the Jacobian for the tip (3x1 matrix for 1-DOF translation)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    
    # Calculate Cartesian velocity: x_dot = J * q_dot
    x_dot = jacp @ data.qvel
    
    # Impedance Control Law (Cartesian Force)
    # F = K(x_target - x_current) - B(x_dot)
    F_cart = K_cart * (x_target - x_current) - B_cart * x_dot
    
    # Map Cartesian force to joint torque (tau = J^T * F)
    tau_impedance = jacp.T @ F_cart
    
    # we need to add compensation for non-linear dynamics 
    # data.qfrc_bias contains Coriolis, centrifugal, and gravity forces.
    tau_total = tau_impedance[0] + data.qfrc_bias[0]
    
    return tau_total

def run_profile(label: str, K_cart: float, B_cart: float) -> RunResult:
    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip')
    x_target = get_cartesian_target(model, data, site_id, Q_TARGET)

    data.qpos[0] = Q_START
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    dt      = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    times   = np.zeros(n_steps)
    q_errors= np.zeros(n_steps)
    torques = np.zeros(n_steps)

    print(f"\nProfile : {label}")
    print(f"K (Stiffness) = {K_cart:.1f} N/m, B (Damping) = {B_cart:.1f} N·s/m")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                print(f"Viewer closed at step {step}  (t = {step*dt:.3f}s)")
                break

            tau = compute_impedance_torque(model, data, site_id, x_target, K_cart, B_cart)
            data.ctrl[0] = tau
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            t = step * dt
            err_q = Q_TARGET - data.qpos[0] 
            
            times[step]    = t
            q_errors[step] = err_q
            torques[step]  = data.actuator_force[0]

    return RunResult(label=label, times=times, q_error=q_errors, torque=torques)

def plot_results(results: list[RunResult]) -> None:
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"1-DOF arm — Cartesian Impedance Control | Target = {np.degrees(Q_TARGET):.1f}°",
        fontsize=11, fontweight="bold"
    )

    ax1 = axes[0]
    for res, c in zip(results, colors):
        ax1.plot(res.times, np.degrees(res.q_error),
                 label=res.label, color=c, linewidth=1.8)
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Joint Position Error (deg)")
    ax1.set_title("Joint Error (Target - Actual)")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    for res, c in zip(results, colors):
        ax2.plot(res.times, res.torque,
                 label=res.label, color=c, linewidth=1.8)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Actuator Torque (N·m)")
    ax2.set_title("Compensating Actuator Torque")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("impedance_1dof_joint_angles.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():
    results = []
    for label, K, B in IMPEDANCE_PROFILES:
        res = run_profile(label, K, B)
        results.append(res)

    plot_results(results)

if __name__ == "__main__":
    main()