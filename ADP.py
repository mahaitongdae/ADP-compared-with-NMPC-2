import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
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
    MAX_ITERATION = 400
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
        load_dir = "./Results_dir/2020-05-15-13-38-30000"
        policy.load_parameters(load_dir)
        value.load_parameters(load_dir)
    if TRAIN_FLAG == 1 :
        while True:
            state_batch.detach_()
            state_batch.requires_grad_(True)
            called_state_batch = state_batch[:, 0:4]  #TODO: reset after oneD
            # called_state_batch = state_batch.clone()    #TODO: diminish after oneD
            control = policy.forward(called_state_batch)
            state_batch_next, f_xu, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel.step(state_batch, control)
            # state_batch_next, f_xu, utility = statemodel.step_oneD(state_batch, control)

            # Discrete Policy Evaluation
            called_state_batch_next = state_batch_next[:, 0:4]
            value_next = value.forward(called_state_batch_next)
            target_value = utility.detach() + value_next.detach()
            value_now = value.forward(called_state_batch)
            equilibrium_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
            value_equilibrium = value.forward(equilibrium_state)
            value_loss = 1/2 * torch.mean(torch.pow((target_value - value_now), 2))\
                         + 0.1 * torch.pow(value_equilibrium, 2)
            state_batch.requires_grad_(False)
            value.zero_grad()
            value_loss.backward(retain_graph=True) # PIM differentiate need backpropogating through value
            value.opt.step()

            # # Continuous Policy Evaluation
            # value_loss = value.update_continuous(called_state_batch, utility, f_xu)
            # # print('PEV | ' + 'value loss:{:3.3f}'.format(float(value_loss)))


            # Discrete Policy Improvement
            value_next = value.forward(called_state_batch_next)
            policy_loss = torch.mean(utility + value_next) # Hamilton
            policy.zero_grad()
            policy_loss.backward()
            policy.opt.step()


            # # continuous Policy Improvement
            # p_V_x = value.get_derivative(called_state_batch)
            # policy_loss = policy.update_continuous(utility, p_V_x, f_xu)

            # Check done
            after_check_state = statemodel.check_done_oneD(state_batch_next)
            state_batch = after_check_state.clone()
            iteration_index += 1
            if iteration_index % 200 == 0:
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
            if iteration_index >= MAX_ITERATION:
                break

        # ==================== Set log path ====================
        log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(MAX_ITERATION))
        os.makedirs(log_dir, exist_ok=True)
        value.save_parameters(log_dir)
        policy.save_parameters(log_dir)








