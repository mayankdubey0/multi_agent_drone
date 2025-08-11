import numpy as np
import quaternion 
from scipy.spatial.transform import Rotation as R

class PD_control:
    '''Class for PD + Feedforward controller'''
    def __init__(self, des_state, Kp=np.eye(3,3), Kd=np.eye(3,3), Kp_att=np.eye(3,3), Kd_att=np.eye(3,3)):

        # System Params:
        self.m = 1.325
        self.g = np.array([0, 0, -9.81])
        L = 0.1
        b = 0.4 
        M = np.array([
            [ 1,  1,  1,  1],
            [ L, -L, -L,  L],
            [-L, -L,  L,  L],
            [-b,  b, -b,  b],
        ])
        self.M_inv = np.linalg.inv(M)

        self.add_sensor_noise = False
        self.add_disturbance = False

        self.Kp = Kp            # 6x6 for x, y, z
        self.Kd = Kd            # 3x3 for dx, dy, dz
        self.Kp_quat = Kp_att   # 4x4 for q0, q1, q2, q3
        self.Kd_quat = Kd_att   # 3x3 for dtheta_x, dtheta_y, dtheta_z

        '''
        States are represented as:
            [x, y, z, dx, dy, dz, q0, q1, q2, q3, dtheta_x, dtheta_y, dtheta_z]
        '''

        self.curr_state = np.zeros(13) # current state of drone from simulator
        self.des_state = des_state # desired state of the drone
        self.state_err = np.zeros(13) # state error (the quaternion error on this is kind of weird)
        self.des_att_subloop = np.zeros(4) 
        self.des_force = np.zeros(3)
        self.des_moments = np.zeros(3)


    def calc_state_err(self):
        '''
        calculates the position error (xyz)
        '''
        self.state_err[:6] = self.des_state[:6] - self.curr_state[:6]


    def calc_des_quat(self):
        '''
        Performs out loop PD + feedforward term
        Then identifies required attitude to obtain acceleration
        '''
        self.des_force = self.m*(self.Kp @ self.state_err[:3] + self.Kd @ self.state_err[3:6] - self.g) # desired 

        n = np.linalg.norm(self.des_force)
        if n < 1e-8: 
            return

        z_b_des = self.des_force / n

        psi = 0.0
        x_c = np.array([np.cos(psi), np.sin(psi), 0.0])
        # Guard collinearity
        if np.linalg.norm(np.cross(z_b_des, x_c)) < 1e-8:
            x_c = np.array([1.0, 0.0, 0.0])

        y_b_des = np.cross(z_b_des, x_c); y_b_des /= np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, z_b_des)

        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        quat_xyzw = R.from_matrix(R_des).as_quat()
        self.des_att_subloop = np.roll(quat_xyzw, 1)  # [w,x,y,z]
        if self.des_att_subloop[0] < 0:  # shortest hemisphere
            self.des_att_subloop = -self.des_att_subloop



    def calc_att_err(self):
        # Use the attitude computed by the subloop
        q_des = quaternion.from_float_array(self.des_att_subloop)
        q_curr = quaternion.from_float_array(self.curr_state[6:10])
        q_err = q_des * q_curr.conj()
        if q_err.w < 0: q_err = -q_err

        self.state_err[6:10]   = np.array([q_err.w, q_err.x, q_err.y, q_err.z])
        self.state_err[10:13]  = self.des_state[10:13] - self.curr_state[10:13]  # rate error (body)

        


    def calc_des_motor_outputs(self):
        # PD on quaternion vector part, signed by scalar part
        e_vec = self.state_err[7:10]
        sgn   = 1.0 if self.state_err[6] >= 0 else -1.0
        e_w   = self.state_err[10:13] 
        self.des_moments = -(self.Kp_quat @ (e_vec * sgn) + self.Kd_quat @ e_w)

        # Thrust = projection of desired force (world) onto current body z
        q_curr = quaternion.from_float_array(self.curr_state[6:10])
        z_body = q_curr * np.quaternion(0,0,0,1) * q_curr.conj()
        z_body_vec = np.array([z_body.x, z_body.y, z_body.z])
        thrust = float(self.des_force @ z_body_vec)

        force_and_moments = np.insert(self.des_moments, 0, thrust)
        ctrl = self.M_inv @ force_and_moments
        ctrl[ctrl < 0] = 0
        return ctrl


    
    def run_controller(self):

        if self.add_sensor_noise == True:
            self.current_state += (np.random.rand(13)/3)
        
        self.calc_state_err()         
        # print(self.state_err[:3])
        self.calc_des_quat()
        self.calc_att_err()
        ctrl = self.calc_des_motor_outputs()

        if self.add_disturbance == True:
            ctrl += (np.random.rand(13)/3)
        
        return ctrl