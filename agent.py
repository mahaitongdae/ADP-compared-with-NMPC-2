import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from torch.distributions import Normal
from torch.nn import init
import torch.nn.functional as F

LOG2Pi = np.log(2 * np.pi)
PI = np.pi

class Actor(nn.Module):
    def __init__(self, input_size, output_size, order=1, lr=0.001):
        super(Actor, self).__init__()

        # generate polynomial feature using sklearn
        self.out_size = output_size
        po = PolynomialFeatures(degree=order)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names()) - 1
        self._pipeline = Pipeline([('poly', po)])
        self._out_gain = PI / 9
        self._norm_matrix = 0.1 * torch.tensor([1, 1, 1, 1], dtype=torch.float32)

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(self._poly_feature_dim, 256), # TODO: change when publish
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )
        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
        temp = torch.mul(x, self._norm_matrix)
        x = torch.mul(self._out_gain, self.layers(temp))
        return x

    def _initialize_weights(self):
        """
        initial parameter using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def loss_function(self, utility, p_V_x, f_xu):

        hamilton = utility + torch.diag(torch.mm(p_V_x, f_xu.T))
        loss = torch.mean(hamilton)
        return loss

    def update(self, state, target_v):
        """
        update parameters
        Parameters
        ----------
        state: state batch, shape [batch, state dimension]
        target_v: shape [batch, 1]

        Returns
        -------

        """

        target_v = torch.as_tensor(target_v).detach()
        value_base = self.forward(self._zero_state)
        i = 0
        while True:
            v = self._evaluate0(state)
            v_loss = torch.mean((v - target_v) * (v - target_v)) + 10 * torch.pow(value_base, 2)
            self.opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self.opt.step()
            i += 1
            if v_loss.detach().numpy() < 0.1 or i >= 5:
                break

        return v_loss.detach().numpy()

    def update_continuous(self, utility, p_V_x, f_xu):
        """
        update parameters
        Parameters
        ----------
        state: state batch, shape [batch, state dimension]
        target_v: shape [batch, 1]

        Returns
        -------

        """

        # actor_base = self.forward(self._zero_state)
        i = 0
        while True:
            u_loss = self.loss_function(utility, p_V_x, f_xu) # + 0 * torch.pow(actor_base, 2)
            self.opt.zero_grad()  # TODO
            u_loss.backward(retain_graph=True)
            self.opt.step()
            i += 1
            if u_loss.detach().numpy() < 0.1 or i >= 0:
                break

        return u_loss.detach().numpy()

    def preprocess(self, X):
        """
        transform raw data to polynomial features
        Parameters
        ----------
        X :  np.array
            shape : (batch, state_dim)
        Returns
        -------
        out :  np.array
            shape : (batch, poly_dim)
        """

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return torch.Tensor(self._pipeline.fit_transform(X)[:, 1:])

    def predict(self, x):

        return self.forward(x).detach().numpy()

    def save_parameters(self, logdir):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))

    def load_parameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir,'actor.pth')))


class Critic(nn.Module):
    """
    value function with polynomial feature
    """

    def __init__(self, input_size, output_size, order=1, lr=0.001):
        super(Critic, self).__init__()

        # generate polynomial feature using sklearn
        self.out_size = output_size
        po = PolynomialFeatures(degree=order)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names()) - 1
        self._pipeline = Pipeline([('poly', po)])

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(self._poly_feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            # nn.Identity()
            nn.ReLU()
        )
        self._norm_matrix = 0.1 * torch.tensor([2, 1, 10, 10], dtype=torch.float32)

        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()


        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0]) # TODO: oneD change

    def _evaluate0(self, state):
        """
        convert state into polynomial features, and conmpute state
        Parameters
        ----------
        state: current state [batch, feature dimension]

        Returns
        -------
        out: value tensor [batch, 1]
        """

        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))
        elif len(state.shape) == 2:
            state = state
        state_tensor = self.preprocess(state)
        out = self.forward(state_tensor)
        return out

    def predict(self, state):
        """
        Parameters
        ----------
        state: current state [batch, feature dimension]

        Returns
        -------
        out: value np.array [batch, 1]
        """

        return self.forward(state).detach().numpy()

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
        x = torch.mul(x, self._norm_matrix)
        x = self.layers(x)
        return x

    def _evaluate0(self, state):
        """
        convert state into polynomial features, and conmpute state
        Parameters
        ----------
        state: current state [batch, feature dimension]

        Returns
        -------
        out: value tensor [batch, 1]
        """

        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))
        elif len(state.shape) == 2:
            state = state
        state_tensor = self.preprocess(state)
        out = self.forward(state_tensor)
        return out

    def predict(self, state):
        """
        Parameters
        ----------
        state: current state [batch, feature dimension]

        Returns
        -------
        out: value np.array [batch, 1]
        """

        return self.forward(state).detach().numpy()

    def loss_function(self, state, utility, f_xu):

        # state.require_grad_(True)
        V = self.forward(state)
        partial_V_x, = torch.autograd.grad(torch.sum(V), state, create_graph=True)
        partial_V_x.view([len(state), -1])
        hamilton = utility.detach() + torch.diag(torch.mm(partial_V_x, f_xu.T.detach()))
        loss = 1 / 2 * torch.mean(torch.pow(hamilton, 2))
        return loss

    def update(self, state, target_v):
        """
        update parameters
        Parameters
        ----------
        state: state batch, shape [batch, state dimension]
        target_v: shape [batch, 1]

        Returns
        -------

        """

        target_v = torch.as_tensor(target_v).detach()
        value_base = self.forward(self._zero_state)
        i = 0
        while True:
            v = self._evaluate0(state)
            v_loss = torch.mean((v - target_v) * (v - target_v)) + 10 * torch.pow(value_base, 2)
            self.opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self.opt.step()
            i += 1
            if v_loss.detach().numpy() < 0.1 or i >= 20:
                break

        return v_loss.detach().numpy()

    def update_continuous(self, state, utility, f_xu):
        """
        update parameters
        Parameters
        ----------
        state: state batch, shape [batch, state dimension]
        target_v: shape [batch, 1]

        Returns
        -------

        """

        self._zero_state.requires_grad_(True)
        value_base = self.forward(self._zero_state)
        i = 0
        while True:
            v_loss = self.loss_function(state, utility, f_xu) + 0.1 * torch.pow(value_base, 2)
            self.opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True) # TODO: retain_graph=True operation?
            self.opt.step()
            i += 1
            if v_loss < 0.1 or i >= 0:
                break
        self._zero_state.requires_grad_(False)
        return v_loss.detach().numpy()

    def preprocess(self, X):
        """
        transform raw data to polynomial features
        Parameters
        ----------
        X :  np.array
            shape : (batch, state_dim)
        Returns
        -------
        out :  np.array
            shape : (batch, poly_dim)
        """

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return torch.Tensor(self._pipeline.fit_transform(X)[:, 1:])

    def _initialize_weights(self):
        """
        initial paramete using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def get_derivative(self, state):
        # state.requires_grad_(True)
        predict = self.forward(state)
        derivative, = torch.autograd.grad(torch.sum(predict), state)
        return derivative.detach()

    def save_parameters(self, logdir):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        torch.save(self.state_dict(), os.path.join(logdir, "critic.pth"))

    def load_parameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir,'critic.pth')))

def test():
    x = np.array([[1,2,3,4],[4,3,2,1]])
    critic = Critic(4,1)
    out = critic.predict(x)
    deri = critic.get_derivative(x)
    print(out, deri)

if __name__ == '__main__':
    test()
