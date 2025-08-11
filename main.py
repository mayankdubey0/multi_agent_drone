import mujoco
import mujoco.viewer
import numpy as np
import quaternion  # pip install numpy-quaternion
from PD_control import PD_control
import time
import matplotlib.pyplot as plt

def slow_sim(speed):
    match speed:
        case 0:
            time.sleep(0.4)
        case 1:
            time.sleep(0.02)
        case 2:
            time.sleep(0.01)
        case 3:
            time.sleep(0)

def plot_position_time(data, data_name):
    """
    Plots x, y, z position vs time from a list of [x, y, z] points.

    Parameters:
        data (list of list): [[x, y, z], [x, y, z], ...]
                             Each index is a timestep in order.
    """
    data = list(map(list, data))  # ensure list of lists
    time_steps = range(len(data))

    # Separate x, y, z
    x_vals = [p[0] for p in data]
    y_vals = [p[1] for p in data]
    z_vals = [p[2] for p in data]


    plt.plot(time_steps, x_vals, label="X position")
    plt.plot(time_steps, y_vals, label="Y position")
    plt.plot(time_steps, z_vals, label="Z position")
    
    plt.xlabel("Time step")
    plt.ylabel(f"{data_name}")
    plt.title(f"{data_name} vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# Load the model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\multi_agent_drone.xml")
data = mujoco.MjData(model)

Kp = np.diag([10, 10, 10])
Kd = np.diag([5, 5, 5])

Kp_att = np.diag([50, 50, 50])
Kd_att = np.diag([5, 5, 5])

# Set desired state [x, y, z, dx, dy, dz, q0, q1, q2, q3, dtheta_x, dtheta_y, dtheta_z]
des_drone1_state = np.array([ 0.5,      0.866,      1.25,       
                                0,          0,         0,      
                                1,          0,         0,      0,      
                                0,          0,         0])

des_drone2_state = np.array([  -1,          0,      1.25,
                                0,          0,         0,
                                1,          0,         0,      0,
                                0,          0,         0])

des_drone3_state = np.array([ 0.5,     -0.866,      1.25, 
                                0,          0,         0,
                                1,          0,         0,      0, 
                                0,          0,         0])

drone1_cont = PD_control(des_drone1_state, Kp, Kd, Kp_att, Kd_att)
drone2_cont = PD_control(des_drone2_state, Kp, Kd, Kp_att, Kd_att)
drone3_cont = PD_control(des_drone3_state, Kp, Kd, Kp_att, Kd_att)

sim_speed = 2;

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ctrl = np.zeros(model.nu)
        
        # obtaining observed state information
        ball_pos = data.qpos[:3]
        ball_attitude = data.qpos[3:7]          # [w, x, y, z]
        ball_vel = data.qvel[:3]
        ball_ang_vel = data.qvel[3:6]           # [wx, wy, wz]
        ball_state = np.concatenate([ball_pos, ball_vel, ball_attitude, ball_ang_vel])

        drone1_pos = data.qpos[7:10]
        drone1_attitude = data.qpos[10:14]      # [w, x, y, z]
        drone1_vel = data.qvel[6:9]
        drone1_ang_vel = data.qvel[9:12]        # [wx, wy, wz]
        drone1_state = np.concatenate([drone1_pos, drone1_vel, drone1_attitude, drone1_ang_vel])

        drone2_pos = data.qpos[14:17]
        drone2_attitude = data.qpos[17:21]      # [w, x, y, z]
        drone2_vel = data.qvel[12:15]
        drone2_ang_vel = data.qvel[15:18]       # [wx, wy, wz]
        drone2_state = np.concatenate([drone2_pos, drone2_vel, drone2_attitude, drone2_ang_vel])

        drone3_pos = data.qpos[21:24]
        drone3_attitude = data.qpos[24:28]      # [w, x, y, z]
        drone3_vel = data.qvel[18:21]
        drone3_ang_vel = data.qvel[21:24]       # [wx, wy, wz]
        drone3_state = np.concatenate([drone3_pos, drone3_vel, drone3_attitude, drone3_ang_vel])

        drone1_cont.curr_state = drone1_state  
        drone2_cont.curr_state = drone2_state  
        drone3_cont.curr_state = drone3_state
        data.ctrl = np.concatenate([drone1_cont.run_controller(), drone2_cont.run_controller(), drone3_cont.run_controller()])

        # err_data.append(controller.curr_state[:3] - des_state[:3])
        # data.ctrl = controller.run_controller()
        # # qvel_curr  = data.qvel     # np.array([wx, wy, wz])
        # pos_data.append(position.copy())
        # step+=1
        # ang_pos_data.append(attitude.copy())
        
        mujoco.mj_step(model, data)

        slow_sim(2)
        viewer.sync()

      
        viewer.sync()