import mujoco
import mujoco.viewer
import numpy as np
import quaternion  # pip install numpy-quaternion

# Load model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\x2 - Copy.xml")
data = mujoco.MjData(model)

# Drone parameters
mass = 1.0               # in kg
g = 9.81                 # gravity
thrust_hover = mass * g  # force to hover

L = 0.1
b = 0.4

M = np.array([
    [ 1,  1,  1,  1],
    [ L,  -L, -L, L],
    [-L,  -L,  L,  L],
    [ -b, b,  -b, b],
])
M_inv = np.linalg.inv(M)

# Control gains
Kp_pos = np.array([5, 5, 10])
Kd_pos = np.array([3, 3, 5])

Kp_att = np.array([60, 60, 60])
Kd_att = np.array([10, 10, 10])

Kp = np.diag([100, 100, 100])
Kd = np.diag([50, 50, 50])

# Target state
target_pos = np.array([1.0, 1.0, 1.5])
target_quat = np.quaternion(1, 0, 0, 0)



def attitude_control(quat_curr, omega_curr, quat_des, omega_des, Kp, Kd):
    quat_err = quat_des * np.quaternion(quat_curr.w, -quat_curr.x, -quat_curr.y, -quat_curr.z)
    e_v = np.array([quat_err.x, quat_err.y, quat_err.z])
    print("Error:", e_v)
    return -Kp@e_v.T*np.sign(quat_err.w) - Kd@(omega_des - omega_curr)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # # Get state

        q_curr = quaternion.from_float_array(data.qpos)    # quaternion.from_float_array([w, x, y, z])
        omega  = data.qvel      # np.array([wx, wy, wz])

        # desired hover orientation (level, yaw=0):
        q_des = np.quaternion(1, 0, 0, 0)

        # your gains (tweak for responsiveness vs. damping)
        torque_des = attitude_control(q_curr, omega, q_des, np.array([0,0,0]), Kp, Kd)
        torque_des = np.array([5, torque_des[0], torque_des[1], torque_des[2]])
        data.ctrl = M_inv@torque_des
        print(torque_des)
        # wrench = np.array([2, ])


      
        # then map τx,τy,τz into your motor commands as usual
        # print("Body‐frame torques:", tau_body)
        
        # print(data.qpos)
        # pos = data.qpos[:3]
        # vel = data.qvel[:3]
        # quat = np.quaternion(*data.qpos[3:7])
        # omega = data.qvel[3:6]

        # # --- Position Control ---
        # pos_err = target_pos - pos
        # vel_err = -vel
        # accel_cmd = Kp_pos * pos_err + Kd_pos * vel_err + np.array([0, 0, g])
        # thrust = mass * accel_cmd  # world-frame force

        # # Convert thrust to body frame (get rotation matrix from quaternion)
        # R = quaternion.as_rotation_matrix(quat)
        # thrust_body = R.T @ thrust  # rotate into body frame

        # # Total thrust is in body-z
        # u1 = thrust_body[2]

        # # --- Attitude Control ---
        # # Desired orientation: align body-z with world thrust vector
        # thrust_dir_world = thrust / np.linalg.norm(thrust)
        # z_body_desired = thrust_dir_world
        # x_c = np.cross([0, 1, 0], z_body_desired)
        # if np.linalg.norm(x_c) < 1e-6:  # handle singularity
        #     x_c = np.array([1, 0, 0])
        # x_body_desired = x_c / np.linalg.norm(x_c)
        # y_body_desired = np.cross(z_body_desired, x_body_desired)

        # # Construct desired rotation matrix
        # R_desired = np.stack([x_body_desired, y_body_desired, z_body_desired], axis=1)

        # # Desired quaternion
        # qd = quaternion.from_rotation_matrix(R_desired)

        # # Quaternion error
        # q_err = qd * quat.inverse()
        # if q_err.w < 0: q_err = -q_err
        # axis_angle = 2 * np.arccos(q_err.w)
        # if axis_angle < 1e-6:
        #     rot_vec = np.zeros(3)
        # else:
        #     rot_axis = np.array([q_err.x, q_err.y, q_err.z]) / np.sin(axis_angle / 2)
        #     rot_vec = axis_angle * rot_axis

        # torque_cmd = Kp_att * rot_vec - Kd_att * omega

        # # Map [thrust, torque] → motor commands
        # # You must define your motor mixing matrix (B_inv)
        # # For now, assume:
        # # ctrl = [m0, m1, m2, m3] = f(u1, tau)
        # B_inv = np.array([
        #     [ 1,  1,  -1, 1],
        #     [ 1, -1, -1, -1],
        #     [ -1,  1, -1, -1],
        #     [ 1,  1,  1, -1],
        # ]) * 0.1

        # # B_inv = np.linalg.inv(B_inv)
       
        # u = np.hstack((u1, torque_cmd))
        # motor_cmds = B_inv @ u
        # data.ctrl[:] = np.clip(motor_cmds, 0, 20)
        # print(data.ctrl)

        mujoco.mj_step(model, data)

        for i in range(300000):
            a = 1

        viewer.sync()