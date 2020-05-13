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
    MAX_ITERATION = 1000
    LR_P = 1e-3
    LR_V = 1e-2
    S_DIM = 4
    A_DIM = 1
    POLY_DEGREE = 2
    VALUE_POLY_DEGREE = 2
    BATCH_SIZE = 512
    TRAIN_FLAG = 1
    LOAD_PARA_FLAG = 0
    MODEL_PRINT_FLAG = 1
    PEV_MAX_ITERATION = 100000
    PIM_MAX_ITERATION = 2000

    # ==================== Set log path ====================
    log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-"+str(MAX_ITERATION))
    os.makedirs(log_dir, exist_ok=True)

    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # ADP solutions with structured policy
    # policy = Policy(S_DIM, A_DIM, POLY_DEGREE, LR_P)
    policy = Actor(S_DIM, A_DIM)
    # value = Value(S_DIM, VALUE_POLY_DEGREE, LR_V)
    value = Critic(S_DIM, A_DIM)
    statemodel = Dynamic_Model.StateModel()
    iteration_index = 0
    if LOAD_PARA_FLAG == 1:
        load_dir = "./Results_dir/2020-05-12-23-01-100"
        policy.load_parameters(load_dir)
        value.load_parameters(load_dir)
    if TRAIN_FLAG == 1 :
        while True:
            with torch.autograd.set_detect_anomaly(True):
                state_batch = statemodel.get_called_state()
                state_batch.detach()
                # all_state_batch = statemodel.get_all_state()
                control = policy.forward(state_batch)
                f_xu, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel.step(control)

                # # Discrete Policy Evaluation
                # value_next = value.predict(state_batch_next)
                # target = utility + value_next
                # # value_loss, grad_value = value.update(state_batch, utility, f_xu)
                # value_loss = value.update(state_batch, target)

                # Continuous Policy Evaluation

                value_loss = value.update_continuous(state_batch, utility, f_xu)
                # print('PEV | ' + 'value loss:{:3.3f}'.format(float(value_loss)))


                # # Discrete Policy Improvement
                # statemodel_pim = Dynamic_Model.Dynamic_Model()
                # PIM_iteration = 0
                # policy.reset_parameters()
                # while True:
                #     statemodel_pim.set_state(all_state_batch)
                #     control = policy.predict(state_batch)
                #     state_batch_next_pim, utility, f_xu, x, _, _ = statemodel_pim.step(control)
                #     p_l_u, p_f_u = statemodel_pim.get_PIM_deri(control)
                #     p_V_x_next = value.get_derivative(state_batch_next_pim)
                #     # policy_loss, grad_policy = policy.update(state_batch, hamilton, p_l_u, p_V_x, p_f_u)
                #     V_next = value.predict(state_batch_next_pim)
                #     policy_loss, grad_policy = policy.update_discrete(state_batch, utility, V_next, p_l_u, p_V_x_next, 0.01 * p_f_u)
                #     if MODEL_PRINT_FLAG == 1:
                #         if PIM_iteration % 500 == 0:
                #             print('PIM | iteration:{:3d} | '.format(PIM_iteration)+'policy loss:{:3.3f}'.format(policy_loss))
                #     PIM_iteration += 1
                #     if policy_loss < 1e-2 or PIM_iteration > PIM_MAX_ITERATION:
                #         break

                # continuous Policy Improvement
                p_V_x = value.get_derivative(state_batch)
                policy_loss = policy.update_continuous(utility, p_V_x, f_xu)

                # Check done
                statemodel.check_done()
                iteration_index += 1
                # x_test = np.array([1,0.2,0.2,0.2])
                # if iteration_index % 1000 == 0:
                    # print("iteration:", iteration_index, "value", value.predict(x_test), "policy:", policy.predict(x_test), "\n")
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
                if iteration_index >= MAX_ITERATION:
                    break

        value.save_parameters(log_dir)
        policy.save_parameters(log_dir)








