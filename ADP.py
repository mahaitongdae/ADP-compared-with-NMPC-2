import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
from Train import Train
from datetime import datetime
import os

if __name__ == '__main__':
    """
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    RL example for lane keeping problem in a circle road

    [Method]
    Model predictive control and approximate dynamic programming

    """
    # Parameters
    Q = np.diag([10, 0.2, 0, 0])
    R = 20
    N = 314
    NP = 10
    MAX_ITERATION = 500
    LR_P = 1e-3
    LR_V = 1e-2
    S_DIM = 4 # TODO:change if oneD success
    A_DIM = 1
    POLY_DEGREE = 2
    VALUE_POLY_DEGREE = 2
    BATCH_SIZE = 512
    TRAIN_FLAG = 1
    LOAD_PARA_FLAG = 0
    MODEL_PRINT_FLAG = 1
    PEV_MAX_ITERATION = 100000
    PIM_MAX_ITERATION = 2000



    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # ADP solutions with structured policy
    # policy = Policy(S_DIM, A_DIM, POLY_DEGREE, LR_P)
    policy = Actor(S_DIM, A_DIM)
    # value = Value(S_DIM, VALUE_POLY_DEGREE, LR_V)
    value = Critic(S_DIM, A_DIM)
    statemodel = Dynamic_Model.StateModel()
    state_batch = statemodel.get_state()
    iteration_index = 0
    if LOAD_PARA_FLAG == 1:
        load_dir = "./Results_dir/2020-05-16-17-40-5000"
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
                # check_state = torch.tensor([[0.5]])
                check_value = value.predict(check_state)
                check_policy = policy.predict(check_state)
                check_info = "zero state value:{:2.3f} |"\
                             "zero state policy:{:1.3f}".format(float(check_value), float(check_policy))
                print(check_info)

            if iteration_index % 1000 == 0 or iteration_index == MAX_ITERATION:
                # ==================== Set log path ====================
                log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(MAX_ITERATION))
                os.makedirs(log_dir, exist_ok=True)
                value.save_parameters(log_dir)
                policy.save_parameters(log_dir)

            if iteration_index >= MAX_ITERATION:
                break







