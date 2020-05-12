from __future__ import print_function
import torch
import numpy as np
from Config import Dynamics_Config
import math

PI = 3.1415926

class Reference(Dynamics_Config):
    def reference_trajectory_sin(self, x, length, k=1 / 5):
        """
        Generate reference trajectory of sin curves.
        Assume no lateral speed.

        :param x:
        :param length:
        :return:
        """
        reference_trajectory = np.zeros([length, 2])
        psi = np.arctan(k * np.cos(k * x))
        for i in range(length):
            x = x + self.Ts * (self.u * np.cos(psi))
            y = np.sin(k * x)
            psi = np.arctan(k * np.cos(k * x))
            reference_trajectory[i, :] = np.array([y, psi])
        return reference_trajectory

class Dynamic_Model(Dynamics_Config):

    def __init__(self, linearity = False):
        self._state = np.zeros([self.BATCH_SIZE, 5])
        self._reset_index = np.zeros(self.BATCH_SIZE)
        self._random_init()
        self.linearity = linearity
        # super(StateModel, self).__init__()

    def _random_init(self):
        self._state[:, 0] = np.random.normal(0.0, self.y_range , self.BATCH_SIZE)
        self._state[:, 1] =  np.random.normal(0.0, 0.15 , self.BATCH_SIZE)
        self._state[:, 2] =  np.random.normal(0.0, 0.1 , self.BATCH_SIZE)
        self._state[:, 3] =  np.random.normal(0.0, 0.05 , self.BATCH_SIZE)
        self._state[:, 4] = 0 * PI * np.random.rand(self.BATCH_SIZE)
        init_state = self._state
        self.init_state = init_state

    def _reset_state(self):
        for i in range(self.BATCH_SIZE):
            if self._reset_index[i] == 1:
                self._state[i, :] = self.init_state[i, :]

    def set_zero_state(self):
        self._state = np.array([0., 0., 0., 0., 0.])[np.newaxis, :]

    def check_done(self):
        threshold = np.kron(np.ones([self.BATCH_SIZE, 1]), np.array([self.y_range, self.psi_range]))
        check_state = self._state[:, [0, 2]]
        sign_error = np.sign(np.abs(check_state) - threshold) # if abs state is over threshold, sign_error = 1
        self._reset_index = np.max(sign_error, axis=1) # if one state is over threshold, _reset_index = 1
        self._reset_state()

    def _state_function(self, state2d, control):
        """
        non-linear model of the vehicle
        Parameters
        ----------
        state2d : np.array
            shape: [batch, 2], state of the state function
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        gradient of the states
        """

        if len(control.shape) == 1:
            control = control.reshape(1, -1)

        # input state
        u_lat = state2d[:, 0]
        beta = u_lat / self.u
        omega_r = state2d[:, 1]

        # control
        delta = control[:, 0]

        # alpha_1 = -delta + np.arctan(beta + self.a * omega_r / self.u)
        # alpha_2 = np.arctan(beta - self.b * omega_r / self.u)
        # when alpha_2 is small
        alpha_1 = -delta + beta + self.a * omega_r / self.u
        alpha_2 = beta - self.b * omega_r / self.u

        if self.linearity == True:
            # Fiala tyre model
            F_y1 = -self.D * self.F_z1 * np.sin(self.C * np.arctan(self.B * alpha_1))
            F_y2 = -self.D * self.F_z2 * np.sin(self.C * np.arctan(self.B * alpha_2))
        else:
            # linear tyre model
            F_y1 = self.k1 * alpha_1
            F_y2 = self.k2 * alpha_2

        deri_u_lat = (np.multiply(F_y1, np.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_omega_r = (np.multiply(self.a * F_y1, np.cos(delta)) - self.b * F_y2) / self.I_zz

        # when delta is small
        # deri_u_lat = (F_y1 + F_y2) / (self.m * self.u) - omega_r
        # deri_omega_r = (self.a * F_y1 - self.b * F_y2) / self.I_zz

        deri_state = np.concatenate((deri_u_lat[np.newaxis, :], deri_omega_r[np.newaxis, :]), 0)

        return deri_state.transpose(), F_y1, F_y2, alpha_1, alpha_2

    def _sf_with_axis_transform(self, control):
        """
        state function with the axis transform, the true model of ADP problem
        Parameters
        ----------
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        state_dot ： np.array
            shape: [batch, 4], the gradient of the state
        """
        # state [\y, \psi, \beta, \omega, \x]
        # x is not a state variable, only for plotting assist
        assert len(self._state.shape) == 2

        y = self._state[:, 0]           # lateral position of vehicle
        v_lateral = self._state[:, 1]   # lateral speed
        psi = self._state[:, 2]         # yaw angle
        omega = self._state[:, 3]       # yaw rate
        x = self._state[:, 4]           # longitudinal position

        dot_y = self.u * np.sin(psi) + v_lateral * np.cos(psi)
        dot_psi = omega
        dot_x = self.u * np.cos(psi) - v_lateral * np.sin(psi)
        state2d = self._state[:, [1, 3]]  # .reshape(2, -1)
        state2d_dot, F_y1, F_y2, _, _ = self._state_function(state2d, control)
        dot_u_lat = state2d_dot[:, 0]
        dot_omega = state2d_dot[:, 1]
        state_dot = np.concatenate([dot_y[:, np.newaxis],
                                    dot_u_lat[:, np.newaxis],
                                    dot_psi[:, np.newaxis],
                                    dot_omega[:, np.newaxis],
                                    dot_x[:, np.newaxis]], axis=1)

        return state_dot, F_y1, F_y2

    def step(self, action):
        """
        The environment will transform to the next state
        Parameters
        ----------
        action : np.array
            shape: [batch, 1]

        Returns
        -------

        """
        # state
        state_dot, F_y1, F_y2 = self._sf_with_axis_transform(action)
        self._state = self._state + state_dot * self.Ts  # TODO: 此处写+=会发生init_state也变化的现象 why?

        # cost
        l = self._utility(action)

        # state_derivative
        f_xu = state_dot[:,[0, 1, 2, 3]]

        # x is not a state
        s = self._state[:, [0, 1, 2, 3]]

        # x is used to help plotting
        position = self._state[:, 4]

        return s, l, f_xu, position, F_y1, F_y2

    def _utility(self, control):
        """
        Output the utility/cost of the step
        Parameters
        ----------
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        utility/cost
        """
        l = 0
        l += 20 * np.power(self._state[:, 0], 2)[:, np.newaxis]
        l += 0.2 * np.power(self._state[:, 2], 2)[:, np.newaxis]
        l += 10 * np.power(control, 2)
        return l

    def get_state(self):
        s = self._state[:, [0, 1, 2, 3]]
        return s

    def get_all_state(self):
        s = self._state
        return s

    def set_state(self, origin_state):
        self._state[:, [0, 1, 2, 3]] = origin_state

    def set_real_state(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1,-1)
        self._state = state


    def get_PIM_deri(self, control):
        p_l_u = 2 * 2 * control
        # approximate partial derivative of f(x,u) with respect to u
        # control_ = control + 1e-3
        # f_xu = self._sf_with_axis_transform(state, control)
        # f_xu_ = self._sf_with_axis_transform(state, control_)
        # p_f_u = self.Ts * 1000. * (f_xu_ - f_xu)[:, [0, 2, 3, 4]]

        beta = self._state[:, 3][:, np.newaxis]
        omega = self._state[:, 4][:, np.newaxis]
        delta = control
        shape = control.shape
        # alpha_1 = -delta + np.arctan(beta + self.a * omega / self.u)
        alpha_1 = -delta + beta + self.a * omega / self.u

        para_u_lat = - self.D * self.g * self.b / self.L
        para_omega = - self.D * self.a * self.b * self.m * self.g / self.I_zz / self.L
        temp1 = np.cos(self.C * np.arctan(self.B * alpha_1))
        temp2 = np.multiply(-self.C * self.B * np.reciprocal(1 + (self.B * alpha_1) ** 2), np.cos(delta))
        deri = np.multiply(temp1, temp2) - np.multiply(np.sin(self.C * np.arctan(self.B * alpha_1)), np.sin(delta))

        # Nonlinear tyre model
        partial_deri_u_lat = para_u_lat * deri
        partial_deri_omega = para_omega * deri

        # # linear tyre model
        # partial_deri_u_lat = - self.k1 / (self.m * self.u) * (np.cos(delta) + np.multiply(alpha_1, np.sin(delta)))
        # partial_deri_omega = - self.a * self.k1 / self.I_zz * (np.cos(delta) + np.multiply(alpha_1, np.sin(delta)))

        partial_deri_y = np.zeros(shape)
        partial_deri_psi = np.zeros(shape)
        p_f_u = np.concatenate([partial_deri_y, partial_deri_u_lat, partial_deri_psi, partial_deri_omega], axis=1)


        return p_l_u, p_f_u

    def set_state(self, s):
        self._state = s

class StateModel(Dynamics_Config):

    def __init__(self):

        super(StateModel, self).__init__()

    def StateFunction(self, state, control):  # 连续状态方程，state：torch.Size([1024, 2])，control：torch.Size([1024, 1])

        # 状态输入
        y = state[:, 0]
        u_lateral = state[:, 1]
        beta = u_lateral / self.u       # 质心侧偏角：torch.Size([1024])
        psi = state[:, 2]
        omega_r = state[:, 3]           # 横摆角速度：torch.Size([1024])

        # 控制输入
        delta = control[:, 0]  # 前轮转角：torch.Size([1024])
        delta.requires_grad_(True)

        # 前后轮侧偏角
        # alpha_1 = -delta + torch.atan(beta + self.a * omega_r / self.u)
        # alpha_2 = torch.atan(beta - self.b * omega_r / self.u)
        # # 前后轮侧偏角（对前轮速度与x轴夹角xi以及后轮侧偏角alpha_2做小角度假设）
        alpha_1 = -delta + beta + self.a * omega_r / self.u
        alpha_2 = beta - self.b * omega_r / self.u

        # 前后轮侧偏力
        F_y1 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_1)) * self.F_z1
        F_y2 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_2)) * self.F_z2

        # 状态输出：torch.Size([1024])
        deri_y = self.u * torch.sin(psi) + u_lateral * torch.cos(psi)
        deri_u_lat = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_psi = omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        # # 状态输出（对前轮转角delta做小角度假设）
        # deri_beta = (F_y1 + F_y2) / (self.m * self.u) - omega_r
        # deri_omega_r = (self.a * F_y1 - self.b * F_y2) / self.I_zz

        # 按行拼接：torch.Size([2, 1024])
        deri_state = torch.cat((deri_y[np.newaxis, :],
                                deri_u_lat[np.newaxis, :],
                                deri_psi[np.newaxis, :],
                                deri_omega_r[np.newaxis, :]), 0)

        partial_deri_y, = torch.autograd.grad(deri_y, delta, retain_graph=True)
        partial_deri_omega_r, = torch.autograd.grad(deri_omega_r, delta, retain_graph=True)
        partial_deri_u_lat, = torch.autograd.grad(deri_u_lat, delta, retain_graph=True)
        partial_deri_psi, = torch.autograd.grad(deri_psi, delta, retain_graph=True)

        partial_deri = torch.cat((partial_deri_y[np.newaxis, :],
                                partial_deri_u_lat[np.newaxis, :],
                                partial_deri_psi[np.newaxis, :],
                                partial_deri_omega_r[np.newaxis, :]), 0)

        return deri_state, F_y1, F_y2, alpha_1, alpha_2, partial_deri.detach().numpy()

def test():
    statemodel = Dynamic_Model()
    control = 0.01 * np.ones([statemodel.BATCH_SIZE, 1])
    s, r, f_xu, position, _, _ = statemodel.step(control)
    print(statemodel.get_PIM_deri(control))
    statemodel.check_done()
    print(f_xu)

def test_partial_deri():
    statemodel = StateModel()
    state = torch.from_numpy(np.array([[0,0,0,0,0]], dtype='float32'))
    control = torch.tensor([[0.3]])
    deri_state, F_y1, F_y2, alpha_1, alpha_2, partial_deri = statemodel.StateFunction(state, control)
    print(partial_deri)
    statemodel2 = Dynamic_Model()
    statemodel2.set_real_state(np.array([0.0,0.0,0.0,0.0,0.0]))
    control = np.array([[0.3]])
    _, pfu = statemodel2.get_PIM_deri(control)
    pfu[:, 1] = pfu[:, 1] / 20
    print(pfu)

if __name__ == "__main__":
    test_partial_deri()
