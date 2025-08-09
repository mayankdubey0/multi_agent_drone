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


# Load model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\x2.xml")
data = mujoco.MjData(model)

pos_data = []
ang_pos_data = []
err_data = []

# Set desired state [x, y, z, dx, dy, dz, q0, q1, q2, q3, dtheta_x, dtheta_y, dtheta_z]
des_state = np.array([0.24, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

Kp = np.diag([10, 10, 10])
Kd = np.diag([5, 5, 5])

Kp_att = np.diag([1, 1, 1])
Kd_att = np.diag([0.1, 0.1, 0.1])

Kp_att = np.diag([50, 50, 50])
Kd_att = np.diag([5, 5, 5])

controller = PD_control(des_state, Kp, Kd, Kp_att, Kd_att)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        position = data.qpos[:3]
        attitude = data.qpos[3:7]       # [w, x, y, z]
        velocity = data.qvel[:3]        
        angular_vel = data.qvel[3:6]    # [wx, wy, wz]
        state = np.concatenate([position, velocity, attitude, angular_vel])

        controller.curr_state = state  
        err_data.append(controller.curr_state[:3] - des_state[:3])
        data.ctrl = controller.run_controller()
        # qvel_curr  = data.qvel     # np.array([wx, wy, wz])
        pos_data.append(position.copy())
        step+=1
        ang_pos_data.append(attitude.copy())
        
        mujoco.mj_step(model, data)

        slow_sim(2)
        viewer.sync()

plot_position_time(pos_data, "Position")
plot_position_time(err_data, "Pos. Error")