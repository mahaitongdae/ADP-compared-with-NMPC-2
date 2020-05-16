import Dynamic_Model
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Actor, Critic
from datetime import datetime
import os
from Config import GeneralConfig

class Train(GeneralConfig):
    def __init__(self):
        super(Train, self).__init__()

        self.agent_batch = torch.empty([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self.state_batch = torch.empty([self.BATCH_SIZE, self.STATE_DIM])
        self.init_index = np.ones([self.BATCH_SIZE, 1])

        self.x_forward = []
        self.u_forward = []
        self.L_forward = []

        for i in range(self.FORWARD_STEP):
            self.u_forward.append([])
            self.L_forward.append([])
        for i in range(self.FORWARD_STEP+1):
            self.x_forward.append([])

        # for i in range(self.BATCH_SIZE):
        #     self.buffer = np.zeros([self.BUFFER_SIZE, self.DYNAMICS_DIM])
        #     self.buffer_all.append(self.buffer)

    def update_state(self, policy, dynamics):
        self.agent_batch = dynamics.check_done(self.agent_batch)
        self.agent_batch.detach_()
        self.state_batch = self.agent_batch[:, 0:4]
        # self.state_batch.requires_grad_(True)
        self.control = policy.forward(self.state_batch)
        self.agent_batch_next, self.f_xu, self.utility, self.F_y1, self.F_y2, _, _ = \
            dynamics.step(self.agent_batch, self.control)
        self.agent_batch = self.agent_batch_next.detach()

    def update_value(self, policy, value, dynamics):

        for i in range(self.FORWARD_STEP):
            if i == 0:
                self.x_forward[i] = self.agent_batch.detach()
                self.u_forward[i] = policy.forward(self.x_forward[i][:, 0:4])
                self.x_forward[i + 1], _, self.L_forward[i],_, _, _, _ = dynamics.step(self.x_forward[i], self.u_forward[i])
            else:
                self.u_forward[i] = policy.forward(self.x_forward[i][:, 0:4])
                self.x_forward[i + 1], _, self.L_forward[i],_, _, _, _ = dynamics.step(self.x_forward[i], self.u_forward[i])
        self.state_batch_next = self.x_forward[-1][:, 0:4]
        self.value_next = value.forward(self.state_batch_next)
        self.utility = torch.zeros([self.FORWARD_STEP, self.BATCH_SIZE], dtype=torch.float32)
        for i in range(self.FORWARD_STEP):
            self.utility[i] = self.L_forward[i].clone()
        self.sum_utility = torch.sum(self.utility,0)
        target_value = self.sum_utility.detach() + self.value_next.detach()
        value_now = value.forward(self.state_batch)
        equilibrium_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        value_equilibrium = value.forward(equilibrium_state)
        value_loss = 1 / 2 * torch.mean(torch.pow((target_value - value_now), 2)) \
                     + 1 * torch.pow(value_equilibrium, 2)
        self.state_batch.requires_grad_(False)
        value.zero_grad()
        value_loss.backward()  # PIM differentiate need backpropogating through value?
        value.opt.step()

        return value_loss.detach().numpy()

    def update_policy(self, policy, value):
        self.value_next = value.forward(self.state_batch_next)
        policy_loss = torch.mean(self.sum_utility + self.value_next)  # Hamilton
        policy.zero_grad()
        policy_loss.backward()
        policy.opt.step()

        return policy_loss.detach().numpy()


