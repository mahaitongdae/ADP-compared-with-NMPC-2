import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Policy, Critic
from datetime import datetime
import os
S_DIM = 4
A_DIM = 1
POLY_DEGREE = 2
LR_P = 1e-2


def Trajectory(x, k=1 / 5):
    lane_position = 1 * np.sin(k * x)
    lane_angle = 1 * np.arctan(k * np.cos(k * x))

    return lane_position, lane_angle

policy = Policy(S_DIM, A_DIM, POLY_DEGREE, LR_P)
# value = Value(S_DIM, VALUE_POLY_DEGREE, LR_V)
value = Critic(S_DIM, A_DIM)

load_dir = "./Results_dir/2020-05-12-21-49-100"
policy.load_parameters(load_dir)
value.load_parameters(load_dir)

plt.figure(3)
statemodel_plt = Dynamic_Model.Dynamic_Model()
statemodel_plt.set_real_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
state = statemodel_plt.get_state()
x = np.array([0.])
longitudinal_position = x
plot_length = 500
control = []
for i in range(plot_length):
    s = statemodel_plt.get_state()
    lane_position, lane_angle = Trajectory(x)
    s_r = s
    s_r[:, 0] = s_r[:, 0] - lane_position
    s_r[:, 2] = s_r[:, 2] - lane_angle
    u = policy.predict(s_r)
    s, _, _, x, F_y1, F_y2 = statemodel_plt.step(u)
    state = np.append(state, s, axis=0)
    longitudinal_position = np.append(longitudinal_position, x, axis=0)
    control = np.append(control, u)
plt.plot(longitudinal_position, state[:, 0])
plt.figure(4)
plt.plot(range(plot_length+1), state[:, 1], label='psi')
plt.plot(range(plot_length+1), state[:, 2], label='beta')
plt.plot(range(plot_length+1), state[:, 3], label='omega')
plt.legend(loc='upper right')
plt.figure(5)
plt.plot(range(plot_length), control)
plt.show()
# np.savetxt('state.txt',state)