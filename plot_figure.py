import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
from datetime import datetime
from Config import Dynamics_Config
import os
from utils import myplot, smooth
S_DIM = 4
A_DIM = 1
POLY_DEGREE = 2
LR_P = 1e-2

def plot_comparison(picture_dir):
    '''
    Plot comparison figure among ADP, MPC & open-loop solution.
    Trajectory, tracking error and control signal plot

    Parameters
    ----------
    picture_dir: string
        location of figure saved.

    '''
    config = Dynamics_Config()
    comparison_dir = "Results_dir/Comparison_Data"
    if os.path.exists(os.path.join(comparison_dir, 'MPC_state.txt')) == 0 or \
        os.path.exists(os.path.join(comparison_dir, 'Open_loop_state.txt')) == 0:
        print('No comparison state data!')
    else:
        mpc_state = np.loadtxt(os.path.join(comparison_dir, 'MPC_state.txt'))
        open_loop_state = np.loadtxt(os.path.join(comparison_dir, 'Open_loop_state.txt'))
        adp_state = np.loadtxt(os.path.join(comparison_dir, 'ADP_state.txt'))

        mpc_trajectory = (mpc_state[:, 4], mpc_state[:, 0])
        open_loop_trajectory = (open_loop_state[:, 4], open_loop_state[:, 0])
        adp_trajectory = (adp_state[:, 4], mpc_state[:, 0])
        trajectory_data = [mpc_trajectory, open_loop_trajectory, adp_trajectory]
        myplot(trajectory_data, 3, "xy",
               fname=os.path.join(picture_dir, 'trajectory.png'),
               xlabel="longitudinal position [m]",
               ylabel="Lateral position error [m]",
               legend=["MPC", "Open-loop", "ADP"],
               legend_loc="upper left"
               )

        mpc_error = (mpc_state[:, 4], mpc_state[:, 0] - np.sin(config.k_curve * mpc_state[:, 4]))
        open_loop_error =  (open_loop_state[:, 4], open_loop_state[:, 0] - np.sin(config.k_curve * open_loop_state[:, 4]))
        adp_error = (adp_state[:, 4], mpc_state[:, 0] - np.sin(config.k_curve * adp_state[:, 4]))
        error_data = [mpc_error, open_loop_error, adp_error]
        myplot(error_data, 3, "xy",
               fname=os.path.join(picture_dir,'trajectory_error.png'),
               xlabel="longitudinal position [m]",
               ylabel="Lateral position error [m]",
               legend=["MPC", "Open-loop","ADP"],
               legend_loc="upper left"
               )

        mpc_psi_error = (mpc_state[:, 4], mpc_state[:, 2] -
                         np.arctan(config.k_curve * np.cos(config.k_curve * mpc_state[:, 4])))
        open_loop_psi_error = (open_loop_state[:, 4], open_loop_state[:, 2] -
                               np.arctan(config.k_curve * np.cos(config.k_curve * open_loop_state[:, 4])))
        adp_psi_error = (adp_state[:, 4], adp_state[:, 2] -
                         np.arctan(config.k_curve * np.cos(config.k_curve * adp_state[:, 4])))
        psi_error_data = [mpc_psi_error, open_loop_psi_error, adp_psi_error]
        myplot(psi_error_data, 3, "xy",
               fname=os.path.join(picture_dir, 'head_angle_error.png'),
               xlabel="longitudinal position [m]",
               ylabel="head angle error [degree]",
               legend=["MPC", "Open-loop", "ADP"],
               legend_loc="upper left"
               )

        mpc_control = np.loadtxt(os.path.join(comparison_dir, 'MPC_control.txt'))
        Open_loop_control = np.loadtxt(os.path.join(comparison_dir, 'Open_loop_control.txt'))
        adp_control = np.loadtxt(os.path.join(comparison_dir, 'ADP_control.txt'))
        mpc_control_tuple = (mpc_state[1:, 4] ,mpc_control)
        Open_loop_control_tuple = (open_loop_state[:, 4], Open_loop_control)
        adp_control_tuple = (adp_state[1:, 4], adp_control)
        control_plot_data = [adp_control_tuple, mpc_control_tuple, Open_loop_control_tuple]
        myplot(control_plot_data, 3, "xy",
               fname=os.path.join(picture_dir, 'control.png'),
               xlabel="longitudinal position [m]",
               ylabel="steering angle [degree]",
               legend=["ADP", "MPC", "Open-loop"],
               legend_loc="upper left"
               )


def adp_simulation_plot(log_dir):
    '''
    Simulate and plot trajectory and control after ADP training algorithm.

    Parameters
    ----------
    log_dir: string
        location of data and figures saved.

    '''
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
           legend_loc="upper left"
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
    control_history_plot = (state_history[1:, -1], control_history)
    myplot(control_history_plot, 1, "xy",
           fname=os.path.join(log_dir, 'control.png'),
           xlabel="longitudinal position [m]",
           ylabel="steering angle"
           )
    comparison_dir = "Results_dir/Comparison_Data"
    np.savetxt(os.path.join(comparison_dir, 'ADP_state.txt'), state_history)
    np.savetxt(os.path.join(comparison_dir, 'ADP_control.txt'), control_history)
    np.savetxt(os.path.join(log_dir, 'ADP_state.txt'), state_history)
    np.savetxt(os.path.join(log_dir, 'ADP_control.txt'), control_history)


if __name__ == '__main__':
    Figures_dir = './Figures/'
    plot_comparison(Figures_dir)