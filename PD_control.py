import numpy as np
import quaternion 
from scipy.spatial.transform import Rotation as R

class PD_control:
    '''Class for PD + Feedforward controller'''
    def __init__(self, des_state,
                 Kp=np.eye(3,3), Kd=np.eye(3,3), Kp_att=np.eye(3,3), Kd_att=np.eye(3,3), 
                 des_f=np.zeros(3)):

        # System Params:
        self.m = 1.325
        self.g = np.array([0, 0, -9.81])

        self.Kp = Kp            # 6x6 for x, y, z
        self.Kd = Kd            # 3x3 for dx, dy, dz
        self.Kp_quat = Kp_att   # 4x4 for q0, q1, q2, q3
        self.Kd_quat = Kd_att   # 3x3 for dtheta_x, dtheta_y, dtheta_z

        '''
        States are represented as:
            [x, y, z, dx, dy, dz, q0, q1, q2, q3, dtheta_x, dtheta_y, dtheta_z]
        '''

        self.curr_state = np.zeros(13)
        self.des_state = des_state
        self.state_err = np.zeros(13)

        self.des_thrust = des_f


    def calc_state_err(self):
        self.state_err[:6] = self.des_state[:6] - self.curr_state[:6]


    def calc_des_quat(self):
        self.des_f = self.m*(self.Kp@self.state_err[:3] + self.Kd@self.state_err[3:6])
        des_body_norm = self.des_f / np.linalg.norm(self.des_f)

        psi = 0 #desired_yaw 
        x_c = np.array([np.cos(psi), np.sin(psi), 0])

        y_b_des = np.cross(des_body_norm, x_c)
        y_b_des /= np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, des_body_norm)

        R_des = np.column_stack((x_b_des, y_b_des, des_body_norm))
        quat = R.from_matrix(R_des).as_quat()  # [x, y, z, w]
        self.des_state[6:10] = np.roll(quat, 1)


    def calc_att_err(self):
        q_des = quaternion.from_float_array(self.des_state[6:10])
        q_curr = quaternion.from_float_array(self.curr_state[6:10])
        q_err = q_des * q_curr.conj()
        print(q_des)

        if q_err.w < 0:
            q_err = -q_err

        self.state_err[6:10] = np.array([q_err.w, q_err.x, q_err.y, q_err.z])
        self.state_err[10:13] = self.des_state[10:13] - self.curr_state[10:13]


    def calc_des_motor_outputs(self):
        attitude_f = -self.Kp_quat@self.state_err[7:10].T*np.sign(self.state_err[6]) - self.Kd_quat@self.des_state[10:13] 
        self.des_f += attitude_f   

    
    def run_controller(self):

        self.calc_state_err()
        self.calc_des_quat()
        self.calc_att_err()
        ctrl = self.calc_des_motor_outputs()
        
        return ctrl







