"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a curve road

    [Method]
    Approximate dynamic programming

    """
import Dynamic_Model
import numpy as np
import torch
import os
from agent import Actor, Critic
from Train import Train
from datetime import datetime
from plot_figure import adp_simulation_plot

if __name__ == '__main__':

    # Parameters
    MAX_ITERATION = 5000
    LR_P = 1e-4
    LR_V = 3e-4
    S_DIM = 4
    A_DIM = 1
    TRAIN_FLAG = 1
    LOAD_PARA_FLAG = 1

    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # ADP solutions with structured policy
    policy = Actor(S_DIM, A_DIM, lr=LR_P)
    value = Critic(S_DIM, A_DIM, lr=LR_V)
    statemodel = Dynamic_Model.StateModel()
    state_batch = statemodel.get_state()
    iteration_index = 0
    if LOAD_PARA_FLAG == 1:
        load_dir = "./Results_dir/2020-05-17-16-43-30000-linear-FS30"
        policy.load_parameters(load_dir)
        value.load_parameters(load_dir)
    if TRAIN_FLAG == 1 :
        train = Train()
        train.agent_batch = statemodel.initialize_agent()
        while True:
            train.update_state(policy, statemodel)
            value_loss = train.update_value(policy, value, statemodel)
            policy_loss = train.update_policy(policy,value)
            iteration_index += 1
            if iteration_index % 1 == 0:
                log_trace = "iteration:{:3d} |"\
                            "policy_loss:{:3.3f} |" \
                            "value_loss:{:3.3f}".format(iteration_index, float(policy_loss), float(value_loss))
                print(log_trace)
                check_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
                check_value = value.predict(check_state)
                check_policy = policy.predict(check_state)
                check_info = "zero state value:{:2.3f} |"\
                             "zero state policy:{:1.3f}".format(float(check_value), float(check_policy))
                print(check_info)

            if iteration_index % 10000 == 0 or iteration_index == MAX_ITERATION:
                # ==================== Set log path ====================
                log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(iteration_index))
                os.makedirs(log_dir, exist_ok=True)
                value.save_parameters(log_dir)
                policy.save_parameters(log_dir)

            if iteration_index >= MAX_ITERATION:
                train.print_loss_figure(MAX_ITERATION, log_dir)
                train.save_data(log_dir)
                adp_simulation_plot(log_dir)
                break








