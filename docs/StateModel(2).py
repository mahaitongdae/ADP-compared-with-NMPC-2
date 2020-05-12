from __future__ import print_function
import math
import torch
import numpy as np
from Config import StateConfig
import matplotlib.pyplot as plt
from matplotlib.pylab import MultipleLocator


class StateModel(StateConfig):

    def __init__(self):

        super(StateModel, self).__init__()

    def StateFunction(self, state, control):  # 连续状态方程，state：torch.Size([1024, 2])，control：torch.Size([1024, 1])

        # 状态输入
        beta = state[:, 0]     # 质心侧偏角：torch.Size([1024])
        omega_r = state[:, 1]  # 横摆角速度：torch.Size([1024])

        # 控制输入
        delta = control[:, 0]  # 前轮转角：torch.Size([1024])
        delta.requires_grad_(True)

        # 前后轮侧偏角
        alpha_1 = -delta + torch.atan(beta + self.a * omega_r / self.u)
        alpha_2 = torch.atan(beta - self.b * omega_r / self.u)
        # # 前后轮侧偏角（对前轮速度与x轴夹角xi以及后轮侧偏角alpha_2做小角度假设）
        # alpha_1 = -delta + beta + self.a * omega_r / self.u
        # alpha_2 = beta - self.b * omega_r / self.u

        # 前后轮侧偏力
        F_y1 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_1)) * self.F_z1
        F_y2 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_2)) * self.F_z2

        # 状态输出：torch.Size([1024])
        deri_beta = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m * self.u) - omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        # # 状态输出（对前轮转角delta做小角度假设）
        # deri_beta = (F_y1 + F_y2) / (self.m * self.u) - omega_r
        # deri_omega_r = (self.a * F_y1 - self.b * F_y2) / self.I_zz

        # 按行拼接：torch.Size([2, 1024])
        deri_state = torch.cat((deri_beta[np.newaxis, :], deri_omega_r[np.newaxis, :]), 0)

        partial_beta = torch.autograd.grad(deri_beta, delta, retain_graph=True)
        partial_omega = torch.autograd.grad(deri_omega_r, delta)


        return deri_state, F_y1, F_y2, alpha_1, alpha_2, partial_beta, partial_omega


def test():
    statemodel = StateModel()
    num = 5
    state = torch.rand((num, 2))
    control = torch.rand((num, 1))
    deri_state, F_y1, F_y2, alpha_1, alpha_2, _, _ = statemodel.StateFunction(state, control)
    print('加油')
    print(deri_state)
    alpha = torch.range(0, 20, 0.1)
    mu = StateModel.D * torch.sin(StateConfig.C * torch.atan(StateConfig.B * alpha / 180 * math.pi))
    plt.plot(alpha.numpy(), mu.numpy(), '#79B9E5', linewidth=0.8)
    plt.show()

def test_new():
    statemodel = StateModel()
    state = torch.from_numpy(np.array([[0,0,0,0,0]], dtype='float32'))
    control = torch.tensor([[0.3]])
    deri_state, F_y1, F_y2, alpha_1, alpha_2, grad1, grad2 = statemodel.StateFunction(state, control)



if __name__ == "__main__":
    test_new()
