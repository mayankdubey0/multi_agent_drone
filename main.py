import mujoco
import mujoco.viewer
import numpy as np
import quaternion

# Load the model and data
model = mujoco.MjModel.from_xml_path(r"c:\mujoco-3.3.2-windows-x86_64\model\skydio_x2\multi_agent_drone.xml")
data = mujoco.MjData(model)

sim_speed = 1;

# Controller gains
Kp = np.array([50, -50, 50, -50])
Kd = np.array([30, -30, 30, -30])

target_quat_drone1 = np.quaternion(1, 0, 0, 0)
target_pos_drone1 = [0.5, 0.866, 1.25]

target_quat_drone2 = np.quaternion(1, 0, 0, 0)
target_pos_drone1 = [-1, 0, 1.25]

target_quat_drone3 = np.quaternion(1, 0, 0, 0)
target_pos_drone3 = [0.5, -0.866, 1.25]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        ctrl = np.zeros(model.nu)
        
        # obtaining observed state information
        ball_pos = data.qpos[:3]
        ball_quaternion = data.qpos[3:7]
        ball_quaternion = quaternion.as_float_array(ball_quaternion)
        ball_vel = data.qvel[:3]
        ball_omega = data.qvel[3:6]

        drone1_pos = data.qpos[7:10]
        drone1_quaternion = data.qpos[10:14]
        drone1_quaternion = quaternion.as_quat_array(drone1_quaternion)
        drone1_vel = data.qvel[6:9]
        drone1_omega = data.qvel[9:12]

        drone2_pos = data.qpos[14:17]
        drone2_quaternion = data.qpos[17:21]
        drone2_quaternion = quaternion.as_quat_array(drone2_quaternion)
        drone2_vel = data.qvel[12:15]
        drone2_omega = data.qvel[15:18]

        drone3_pos = data.qpos[21:24]
        drone3_quaternion = data.qpos[24:28]
        drone3_quaternion = quaternion.as_quat_array(drone3_quaternion)
        drone3_vel = data.qvel[18:21]
        drone3_omega = data.qvel[21:24]

        # calculating error in state
        pos_err_drone1 = target_pos_drone1 - drone1_pos
        # vel_err_drone1 = -vel_err_drone1
        quat_err_drone1 = target_quat_drone1 * drone1_quaternion.inverse()
        if drone1_quaternion.w < 0:
            quat_err_drone1 = -quat_err_drone1
        phi = 2 * np.arccos(quat_err_drone1.w)
        
        sin_h = np.sqrt(max(0, 1 - quat_err_drone1.w**2))
        if sin_h < 1e-6:
            e_rot = np.zeros(3)
        else:
            axis = np.array([quat_err_drone1.x, quat_err_drone1.y, quat_err_drone1.z]) / sin_h
            e_rot = np.array(phi * axis)

        # 5) PD torque
        tau = Kp * e_rot + Kd * (-drone2_omega) + 9.81
        ctrl[:4] = tau
        print(ctrl)


        

        
        quat_err_drone2 = target_quat_drone2 * drone2_quaternion.inverse()
        quat_err_drone3 = target_quat_drone3 * drone3_quaternion.inverse()
        


        # for drone_idx in range(num_drones):
        #     qpos_i = drone_idx * qpos_per_drone
        #     qvel_i = drone_idx * qvel_per_drone
        #     ctrl_i = drone_idx * motors_per_drone

        #     z = data.qpos[qpos_i + 2]
        #     z_vel = data.qvel[qvel_i + 2]
        #     u = Kp * (z_target - z) - Kd * z_vel
        #     u = np.clip(u, 0, 20)

        #     for j in range(motors_per_drone):
        #         ctrl[ctrl_i + j] = u / motors_per_drone

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        
        match sim_speed:
            case 0:
                delay = 30000000;
            case 1:
                delay = 2000000;
            case 2:
                delay = 300000;
            case 3:
                delay = 0;
        for i in range(delay): 
            a = 1
        viewer.sync()
