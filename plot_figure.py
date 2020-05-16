import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
from datetime import datetime
import os
S_DIM = 4
A_DIM = 1
POLY_DEGREE = 2
LR_P = 1e-2


def Trajectory(x, k=1 / 10):
    lane_position = 1 * np.sin(k * x)
    lane_angle = 1 * np.arctan(k * np.cos(k * x))

    return lane_position, lane_angle

# policy = Policy(S_DIM, A_DIM, POLY_DEGREE, LR_P)
policy = Actor(S_DIM, A_DIM)
# value = Value(S_DIM, VALUE_POLY_DEGREE, LR_V)
value = Critic(S_DIM, A_DIM)

load_dir = "./Results_dir/2020-05-16-16-10-3000"
policy.load_parameters(load_dir)
value.load_parameters(load_dir)

plt.figure(3)
statemodel_plt = Dynamic_Model.StateModel()
# statemodel_plt.set_state(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
state = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
state_history = state[:,0:4].detach().numpy()
x = np.array([0.])
longitudinal_position = x
plot_length = 200
control = []
for i in range(plot_length):
    lane_position, lane_angle = Trajectory(x)
    s_r = state.detach().numpy()
    # s_r[:, 0] = s_r[:, 0] - lane_position
    # s_r[:, 2] = s_r[:, 2] - lane_angle
    u = policy.forward(torch.from_numpy(s_r[:, 0:4]))
    state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel_plt.step(state, u)
    s = state_next.detach().numpy()
    state_history = np.append(state_history, s[:, 0:4], axis=0)
    longitudinal_position = np.append(longitudinal_position, s[:, -1], axis=0)
    control = np.append(control, u.detach().numpy())
    state = state_next
    x = s[:, -1]
plt.plot(longitudinal_position, state_history[:, 0], label='trajectory')
plt.legend(loc='upper right')
plt.figure(4)
plt.plot(range(plot_length+1), state_history[:, 1], label='u_lat')
plt.plot(range(plot_length+1), state_history[:, 2], label='psi')
plt.plot(range(plot_length+1), state_history[:, 3], label='omega')
plt.legend(loc='upper right')
plt.figure(5)
plt.plot(range(plot_length), control)
plt.show()
# np.savetxt('state.txt',state)