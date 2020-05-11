from __future__ import print_function
import torch
import numpy as np
from Config import StateConfig
import math

PI = 3.1415926



class StateModel(StateConfig):

    def __init__(self, linearity = False):
        self._state = np.zeros([self.BATCH_SIZE, 5])
        self._reset_index = np.zeros(self.BATCH_SIZE)
        self._random_init()
        self.linearity = linearity
        # super(StateModel, self).__init__()

    def _random_init(self):
        self._state[:, 0] = self.rho_epect + self.rho_range / 4 * np.random.normal(self.BATCH_SIZE)
        self._state[:, 1] = 1 * PI * np.random.rand(self.BATCH_SIZE)
        self._state[:, 2] = 0.15 * np.random.rand(self.BATCH_SIZE)
        self._state[:, 3] = 0.1 * np.random.rand(self.BATCH_SIZE)
        self._state[:, 4] = 0.05 * np.random.rand(self.BATCH_SIZE)
        init_state = self._state
        self.init_state = init_state

    def _reset_state(self):
        for i in range(self.BATCH_SIZE):
            if self._reset_index[i] == 1:
                self._state[i, :] = self.init_state[i, :]

    def set_zero_state(self):
        self._state = np.array([self.rho_epect, 0., 0., 0., 0.])[np.newaxis, :]

    def check_done(self):
        threshold = np.kron(np.ones([self.BATCH_SIZE, 1]), np.array([self.rho_range, self.psi_range, self.beta_range]))
        check_state = self._state[:, [0, 2, 3]] - np.kron(np.ones([self.BATCH_SIZE, 1]), np.array([self.rho_epect, 0., 0.]))
        sign_error = np.sign(np.abs(check_state) - threshold) # if abs is over threshold, sign_error = 1
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
        beta = state2d[:, 0]
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
            F_y1 = -self.mu * self.F_z1 * np.sin(self.C * np.arctan(self.B * alpha_1))
            F_y2 = -self.mu * self.F_z2 * np.sin(self.C * np.arctan(self.B * alpha_2))
        else:
            # linear tyre model
            F_y1 = self.k1 * alpha_1
            F_y2 = self.k2 * alpha_2

        deri_beta = (np.multiply(F_y1, np.cos(delta)) + F_y2) / (self.m * self.u) - omega_r
        deri_omega_r = (np.multiply(self.a * F_y1, np.cos(delta)) - self.b * F_y2) / self.I_zz

        # when delta is small
        # deri_beta = (F_y1 + F_y2) / (self.m * self.u) - omega_r
        # deri_omega_r = (self.a * F_y1 - self.b * F_y2) / self.I_zz

        deri_state = np.concatenate((deri_beta[np.newaxis, :], deri_omega_r[np.newaxis, :]), 0)

        return deri_state.transpose(), F_y1, F_y2, alpha_1, alpha_2

    def _sf_with_axis_transform(self, control):
        """
        state function with the axis transform, the true model of RL problem
        Parameters
        ----------
        control : np.array
            shape: [batch, 1]

        Returns
        -------
        state_dot ： np.array
            shape: [batch, 4], the gradient of the state
        """
        # state [\rho, \theta, \psi, \beta, \omega]
        assert len(self._state.shape) == 2

        rho = self._state[:, 0]
        theta = self._state[:, 1]
        psi = self._state[:, 2]
        beta = self._state[:, 3]
        omega = self._state[:, 4]

        rho_dot = -self.u * np.sin(psi) - self.u * np.tan(beta) * np.cos(psi)
        theta_dot = (self.u * np.cos(psi) - self.u * np.tan(beta) * np.sin(psi)) / rho
        psi_dot = omega - theta_dot
        state2d = self._state[:, 3:]  # .reshape(2, -1)
        state2d_dot, F_y1, F_y2, _, _ = self._state_function(state2d, control)
        state_dot = np.concatenate([rho_dot[:, np.newaxis],
                                    theta_dot[:, np.newaxis],
                                    psi_dot[:, np.newaxis],
                                    state2d_dot], axis=1)

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
        f_xu = state_dot[:,[0, 2, 3, 4]]

        # done
        done = False
        if np.abs(self._state[0, 0] - self.rho_epect) > self.rho_range:
            done = True

        # \theta is not a state
        s = self._state[:, [0, 2, 3, 4]]

        # \theta is useful when plotting the trajectory
        mask = self._state[:, 1]
        s[:, 0] -= self.rho_epect

        return s, l, f_xu,  mask, F_y1, F_y2

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
        l += 20 * np.power(self._state[:, 0] - self.rho_epect, 2)[:, np.newaxis]
        l += 0.2 * np.power(self._state[:, 2], 2)[:, np.newaxis]
        # if np.abs(control-0.4) > 0.1:
        l += 10 * np.power(control, 2)
        return l

    def get_state(self):
        s = self._state[:, [0, 2, 3, 4]]
        s[:, 0] -= self.rho_epect
        return s

    def get_all_state(self):
        s = self._state
        return s

    def set_state(self, origin_state):
        self._state[:, [0, 2, 3, 4]] = origin_state
        self._state[:, 0] += self.rho_epect

    def set_real_state(self, state):
        self._state = state


    def get_PIM_deri(self, control):
        p_l_u = 2 * 2 * control
        # approximate partial derivative of f(x,u) with respect to u todo:是否能有准确的求导方式
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

        para_beta = - self.mu * self.g / self.u / self.L * self.b
        para_omega = - self.mu * self.a * self.b * self.m * self.g / self.I_zz / self.L
        temp1 = np.cos(self.C * np.arctan(self.B * alpha_1))
        temp2 = np.multiply(-self.C * self.B * np.reciprocal(1 + (self.B * alpha_1) ** 2), np.cos(delta))
        deri = np.multiply(temp1, temp2) - np.multiply(np.sin(self.C * np.arctan(self.B * alpha_1)), np.sin(delta))

        # Fiala
        partial_deri_beta = para_beta * deri
        partial_deri_omega = para_omega * deri

        # # linear
        # partial_deri_beta = - self.k1 / (self.m * self.u) * (np.cos(delta) + np.multiply(alpha_1, np.sin(delta)))
        # partial_deri_omega = - self.a * self.k1 / self.I_zz * (np.cos(delta) + np.multiply(alpha_1, np.sin(delta)))

        partial_deri_rho = np.zeros(shape)
        partial_deri_psi = np.zeros(shape)
        p_f_u = np.concatenate([partial_deri_rho, partial_deri_psi, partial_deri_beta, partial_deri_omega], axis=1)


        return p_l_u, p_f_u

    def set_state(self, s):
        self._state = s

def test():
    statemodel = StateModel()
    control = 0.01 * np.ones([statemodel.BATCH_SIZE, 1])
    s, r, f_xu, mask, _, _ = statemodel.step(control)
    statemodel.get_PIM_deri(control)
    statemodel.check_done()
    print(f_xu)


if __name__ == "__main__":
    test()
