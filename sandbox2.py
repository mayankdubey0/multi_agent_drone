import mujoco
import mujoco.viewer
import numpy as np
import quaternion  # pip install numpy-quaternion
from PD_control import PD_control
import time

# Set desired state [x, y, z, dx, dy, dz, q0, q1, q2, q3, dtheta_x, dtheta_y, dtheta_z]
des_state = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
controller = PD_control(des_state)


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


# Load model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\x2.xml")
data = mujoco.MjData(model)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        position = data.qpos[:3]
        attitude = data.qpos[3:7]       # [w, x, y, z]
        velocity = data.qvel[:3]        
        angular_vel = data.qvel[3:6]    # [wx, wy, wz]
        state = np.concatenate([position, velocity, attitude, angular_vel])

        controller.curr_state = state  
        data.ctrl = controller.run_controller()
        # qvel_curr  = data.qvel     # np.array([wx, wy, wz])
        
        mujoco.mj_step(model, data)

        slow_sim(2)
        viewer.sync()