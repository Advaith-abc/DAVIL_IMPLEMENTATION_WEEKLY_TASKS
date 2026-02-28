import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import expm

# 3 dof planar arm
XML = """
<mujoco model="3R_planar_robot_Linear_Trajectory">
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="4 4 0.1" rgba=".9 .9 .9 1"/>
    
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" rgba="0.8 0.2 0.2 1"/>
      
      <body name="link2" pos="1 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" rgba="0.2 0.8 0.2 1"/>
        
        <body name="link3" pos="1 0 0">
          <joint name="joint3" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" rgba="0.2 0.2 0.8 1"/>
          
          <site name="end_effector" pos="1 0 0" size="0.051" rgba="0 1 1 1"/>
        </body>
      </body>
    </body>
    
    <body name="poe_marker" pos="0 0 0" mocap="true">
      <geom type="sphere" size="0.07" rgba="1 1 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def twist_to_matrix(S):
    T = np.zeros((4, 4))
    T[:3, :3] = skew_symmetric(S[:3])
    T[:3, 3] = S[3:]
    return T

# Screw axes and Home Configuration
S1 = np.array([0, 0, 1, 0, 0, 0])
S2 = np.array([0, 0, 1, 0, -1, 0])
S3 = np.array([0, 0, 1, 0, -2, 0])

M = np.array([
    [1, 0, 0, 3.0],
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.1],
    [0, 0, 0, 1.0]
])

def calculate_poe_fk(q):
    T1 = expm(twist_to_matrix(S1) * q[0])
    T2 = expm(twist_to_matrix(S2) * q[1])
    T3 = expm(twist_to_matrix(S3) * q[2])
    return T1 @ T2 @ T3 @ M

def linear_interp(start_val, end_val, t, duration):
    t_norm = np.clip(t / duration, 0.0, 1.0)
    return start_val + (end_val - start_val) * t_norm

# 3. MuJoCo Simulation Loop
def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    
    q_home = np.array([0.0, 0.0, 0.0])
    q_goal = np.array([np.pi/3, np.pi/4, np.pi/4]) 
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = time.time() - start_time
            
            if t <= 1.0:
                # Move to goal at constant speed
                q_current = linear_interp(q_home, q_goal, t, 1.0)
                
            elif t <= 2.0:
                # Wait at goal
                q_current = q_goal
                
            elif t <= 3.0:
                # Return to home at constant speed
                t_return = t - 2.0 
                q_current = linear_interp(q_goal, q_home, t_return, 1.0)
                
            else:
                # Stop the simulation
                print("Sequence complete. Closing viewer.")
                break 
                
            # Apply the calculated joint angles
            data.qpos[:] = q_current
            
            # Visualization 
            mujoco.mj_kinematics(model, data)
            
            T_curr = calculate_poe_fk(q_current)
            data.mocap_pos[0] = T_curr[:3, 3]
            
            rot_mat = T_curr[:3, :3].flatten()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot_mat)
            data.mocap_quat[0] = quat
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()