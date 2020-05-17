"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a circle road

    [Method]
    Model predictive control(MPC) as comparison

"""
from matplotlib import pyplot as plt
from Solver import Solver
from Config import Dynamics_Config
import numpy as np
import os

def main():
    log_dir = "Results_dir/Comparison_Data"
    config = Dynamics_Config()
    solver=Solver()
    x = [0.0, 0.0, 0.033, 0.0, 0.0]
    state_history = np.array(x)
    control_history = np.empty([0, 1])
    for i in range(config.NP_TOTAL):
        state, control = solver.mpc_solver(x, config.NP)
        x = state[1]
        u = control[0]
        state_history = np.append(state_history, x)
        control_history = np.append(control_history, u)
        x = x.tolist()
    state_history = state_history.reshape(-1,config.DYNAMICS_DIM)
    np.savetxt(os.path.join(log_dir, 'MPC_state.txt'), state_history)
    np.savetxt(os.path.join(log_dir, 'MPC_control.txt'), control_history)

if __name__ == '__main__':
    main()