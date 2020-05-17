import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
from datetime import datetime
from Config import Dynamics_Config
import os
from utils import myplot
S_DIM = 4
A_DIM = 1
POLY_DEGREE = 2
LR_P = 1e-2

def plot_trajectory(picture_dir):
    config = Dynamics_Config()
    comparison_dir = "./Results_dir/comparison_method"
    if os.path.exists(os.path.join(comparison_dir, 'MPC_state.txt')) == 0 or \
        os.path.exists(os.path.join(comparison_dir, 'Open_loop_state.txt')) == 0:
        print('No comparison state data!')
    else:
        mpc_state = np.loadtxt(os.path.join(comparison_dir, 'MPC_state.txt'))
        Open_loop_state = np.loadtxt(os.path.join(comparison_dir, 'Open_loop_state.txt'))
        plt.figure()
        mpc_error = (mpc_state[:, 4], mpc_state[:, 0] - np.sin(config.k_curve * mpc_state[:, 4]))
        open_loop_error =  (Open_loop_state[:, 4], Open_loop_state[:, 0] - np.sin(config.k_curve * Open_loop_state[:, 4]))
        data = [mpc_error, open_loop_error]
                # (mpc_state[:, 4], np.sin(config.k_curve * mpc_state[:, 4]))]
        myplot(data, 2, "xy",
               fname=os.path.join(picture_dir,'trajectory.png'),
               xlabel="longitudinal position [m]",
               ylabel="Lateral position error [m]",
               legend=["MPC", "Open-loop"]
               ) # color_list=["blue",], ylim=[-2, 2]

def plot_adp(log_dir):
    policy = Actor(S_DIM, A_DIM)
    value = Critic(S_DIM, A_DIM)
    load_dir = log_dir
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)
    statemodel_plt = Dynamic_Model.StateModel()
    state = torch.tensor([[0.0, 0.0, 1/30, 0.0, 0.0]])
    state.requires_grad_(False)
    x_ref = statemodel_plt.reference_trajectory(state[:, -1])
    state_r = state.detach().clone()
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref
    state_history = state.detach().numpy()
    x = np.array([0.])
    plot_length = 500
    control_history = []
    for i in range(plot_length):
        u = policy.forward(state_r[:, 0:4])
        state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel_plt.step(state, u)
        state_r_old, _, _, _, _, _, _ = statemodel_plt.step(state_r, u)
        state_r = state_r_old.detach().clone()
        state_r[:, [0, 2]] = state_next[:, [0, 2]]
        x_ref = statemodel_plt.reference_trajectory(state_next[:, -1])
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref
        state = state_next.clone().detach()
        s = state_next.detach().numpy()
        state_history = np.append(state_history, s, axis=0)
        control_history = np.append(control_history, u.detach().numpy())
    trajectory = (state_history[:, -1], state_history[:, 0])
    myplot(trajectory, 1, "xy",
           fname=os.path.join(log_dir, 'trajectory.png'),
           xlabel="longitudinal position [m]",
           ylabel="Lateral position [m]",
           legend=["trajectory"],
           legend_loc="upper right"
    )
    u_lat = (state_history[:, -1], state_history[:, 1])
    psi =(state_history[:, -1], state_history[:, 2])
    omega = (state_history[:, -1], state_history[:, 3])
    data = [u_lat, psi, omega]
    legend=["u_lat", "$\psi$", "$\omega$"]
    myplot(data, 3, "xy",
           fname=os.path.join(log_dir, 'other_state.png'),
           xlabel="longitudinal position [m]",
           legend=legend
           )
    control_history = (state_history[1:, -1], control_history)
    myplot(control_history, 1, "xy",
           fname=os.path.join(log_dir, 'control.png'),
           xlabel="longitudinal position [m]",
           ylabel="steering angle"
           )
    comparison_dir = "./Results_dir/comparison_method"
    np.savetxt(os.path.join(comparison_dir, 'ADP_state.txt'), state_history)
    np.savetxt(os.path.join(comparison_dir, 'ADP_control.txt'), control_history)


if __name__ == '__main__':
    Figures_dir = './Figures/'
    # plot_trajectory(Figures_dir)
    plot_adp("Results_dir/2020-05-17-20-57-final")