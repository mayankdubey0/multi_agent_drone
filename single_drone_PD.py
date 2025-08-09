import mujoco
import mujoco.viewer
import numpy as np
import quaternion  # pip install numpy-quaternion

# Load model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\x2.xml")
data = mujoco.MjData(model)

# Drone parameters
mass = 1.325               # in kg
g = 9.81                 # gravity
thrust_hover = mass * g  # force to hover

L = 0.1
b = 0.4

M = np.array([
    [ 1,  1,  1,  1],
    [ L, -L, -L,  L],
    [-L, -L,  L,  L],
    [-b,  b, -b,  b],
])
M_inv = np.linalg.inv(M)

# Control gains
Kp = np.diag([10, 10, 10])
Kd = np.diag([1, 1, 1])

# Target state
qpos_des = np.array([1.0, 1.0, 1.5, 1, 0, 0, 0])
qvel_des = np.array([0, 0, 0, 0, 0, 0])



def attitude_control(qpos_des, qvel_des, qpos_curr, qvel_curr, Kp, Kd):

    pos_des = qpos_des[:3]
    vel_des = qvel_des[:3]
    quat_des = quaternion.from_float_array(qpos_des[3:7])
    omega_des = qvel_des[3:6]

    pos_curr = qpos_curr[:3]
    vel_curr = qvel_curr[:3]
    quat_curr = quaternion.from_float_array(qpos_curr[3:7])
    quat_curr_inv = np.quaternion(quat_curr.w, -quat_curr.x, -quat_curr.y, -quat_curr.z)
    omega_curr = qvel_curr[3:6]

    pos_err = pos_des - pos_curr
    
    x_err = pos_des[0] - pos_curr[0]
    dx_err = vel_des[0] - vel_curr[0]

    y_err = pos_des[1] - pos_curr[1]
    dy_err = vel_des[1] - vel_des[1]

    z_err = pos_des[2] - pos_curr[2]
    dz_err = vel_des[2] - vel_curr[2]

    des_ax = 10*x_err + 1*dx_err
    des_ay = 10*y_err + 1*dy_err

    thrust_vec = quat_curr * np.quaternion(0, 0, 1) * quat_curr_inv

    

    quat_err = quat_des * quat_curr_inv
    e_v = np.array([quat_err.x, quat_err.y, quat_err.z])
    # print("Error quat:", quat_err, "Error z:", z_err)
    return (50*z_err + 5*dz_err + thrust_hover, -Kp@e_v.T*np.sign(quat_err.w) - Kd@(omega_des - omega_curr))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Get state
        qpos_curr = data.qpos    # quaternion.from_float_array([w, x, y, z])
        qvel_curr  = data.qvel     # np.array([wx, wy, wz])

        # your gains (tweak for responsiveness vs. damping)
        torque_des = attitude_control(qpos_des, qvel_des, qpos_curr, qvel_curr, Kp, Kd)
        torque_des = np.array([torque_des[0], torque_des[1][0], torque_des[1][1], torque_des[1][2]])
        
        ctrl = M_inv@torque_des
        ctrl[ctrl < 0] = 0
        data.ctrl = ctrl
        print(ctrl)


        mujoco.mj_step(model, data)
 
        for i in range(3000):
            a = 1

        viewer.sync()