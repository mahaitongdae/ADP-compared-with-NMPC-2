import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from torch.distributions import Normal
from torch.nn import init

LOG2Pi = np.log(2 * np.pi)

class Policy(object):
    def __init__(self, input_size, out_size, degree, lr=0.001):
        """
        policy function with polynomial feature, linear approximation
        Parameters
        ----------
        input_size : int
            state dimension
        output_size : int
            action dimension
        degree : int
            the max degree of polynomial function
        """
        self._out_dim = out_size
        self.lr = lr
        po = PolynomialFeatures(degree=degree)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names())
        self._pipeline = Pipeline([('poly', po)])

        self.reset_parameters()
        self._logsigma = np.zeros((1, self._out_dim))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.adam_t = 0
        self.adam_m_sig = np.zeros((1, self._out_dim))
        self.adam_v_sig = np.zeros((1, self._out_dim))

        self.adam_m_mu = np.zeros((self._poly_feature_dim, self._out_dim))
        self.adam_v_mu = np.zeros((self._poly_feature_dim, self._out_dim))

    def reset_parameters(self):
        """
        Reset parameters of policy approximate.

        """
        self._w = np.random.random([self._poly_feature_dim, self._out_dim]) * 0.0005
        self._w[0, 0] = 0.0

    def predict(self, X):
        """
        Doing prediction
        Parameters
        ----------
        X : np.array
            shape (1, state_dim)

        Returns
        -------
        action : np.array
            shape : (1, action_dim)
        """
        action = self.preprocess(X).dot(self._w)
        for i in range(action.shape[0]):
            if action[i] >= np.pi / 6:
                action[i] = np.pi / 6
            elif action[i] <= -np.pi / 6:
                action[i] = -np.pi / 6

        return action

    def update(self, state,  hamilton, p_l_u, p_V_x, p_f_u):
        """

        Parameters
        ----------
        state : np.array state of agent
            shape (batch, state_dim)
        p_l_u : np.array partial derivative of utility with respect to control
            shape (batch, action_dim)
        p_V_x : np.array partial derivative of value function with respect to state_batch_next
            shape (batch, state_dim)
        p_f_u : np.array partial derivative of state_dot with respect to control
            shape (batch, action_dim)

        Returns
        -------

        """
        s = state
        Xp = self.preprocess(s)
        loss = hamilton
        p_H_u = p_l_u + np.diag(p_V_x.dot(p_f_u.T))[:,np.newaxis]
        grad_w = 1 / s.shape[0] * np.dot(p_H_u.T, Xp)

        # grad_w = np.mean(grad_w, axis=0)[:, np.newaxis]
        self._w -= self.lr * grad_w.T

        return loss, grad_w

    def gradient_decent_discrete(self, state, p_l_u, p_V_x_next, p_f_u):
        s = state
        Xp = self.preprocess(s)
        temp = p_l_u + np.diag(p_V_x_next.dot(p_f_u.T))[:, np.newaxis]
        grad_w = 1 / s.shape[0] * np.dot(temp.T, Xp)
        self._w -= self.lr * grad_w.T
        return grad_w

    def update_discrete(self, state, utility, V_next, p_l_u, p_V_x_next, p_f_u):
        s = state
        Xp = self.preprocess(s)
        grad_w = self.gradient_decent_discrete(state,p_l_u,p_V_x_next,p_f_u)
        # state_batch_next, utility, f_xu, mask, _, _ = statemodel_pim.step(control)
        loss = utility + V_next # TODO:V也要每一步更新 那么只能放在外面 这一步没有办法包装在Policy中
        loss = loss.mean()
        if loss < 0:
            print("pause")
        return loss, grad_w

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
        return self._pipeline.fit_transform(X)

    def get_w(self):
        return self._w

    def set_w(self, x):
        assert x.shape == self._w.shape
        # assert y.shape == self._logsigma.shape
        self._w = x
        # self._logsigma = y

    def save_parameters(self, logdir):
        policy_w = self.get_w()
        np.save(os.path.join(logdir, "actor"),policy_w)

    def load_parameters(self, load_dir):
        policy_w = np.load(os.path.join(load_dir, 'actor.npy'))
        self.set_w(policy_w)

class Actor(nn.Module):
    """
    stochastic policy with linear approximation and polynomial feature
    """

    def __init__(self, input_size, output_size, order=1, lr=0.01):
        super(Actor, self).__init__()
        # generate polynomial feature using sklearn
        self.out_size = output_size
        po = PolynomialFeatures(degree=order)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names()) - 1
        self._pipeline = Pipeline([('poly', po)])

        # initial parameters of actor
        log_std = np.zeros(output_size)
        self.log_sigma = torch.nn.Parameter(torch.as_tensor(log_std))

        self.layers = nn.Sequential(
            nn.Linear(self._poly_feature_dim, output_size),
            nn.Identity()
        )
        self._initialize_weights()

        # init optimizor
        self._opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        mu: mean of action
        sigma: covarience of action
        dist: Gaussian(mu, dist)
        """
        mu = self.layers(x)
        mu = torch.tanh(mu) * 3.14159 / 9
        sigma = torch.exp(self.log_sigma)
        dist = Normal(mu, sigma)

        return mu, sigma, dist

    def _initialize_weights(self):
        """
        xavier initial of parmeters
        """
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def choose_action(self, state):
        """
        choose_action according to current state
        Parameters
        ----------
        state:  np.array; shape (batch, N_S)

        Returns
        -------
        action: shape (batch, N_a)
        mu: action without stochastic, shape (batch, N_a)
        """
        if len(state.shape) == 1:
            state = state.reshape((-1, state.shape[0]))
        elif len(state.shape) == 2:
            state = state
        state_polynormial = self.preprocess(state)
        mu, sigma, dist = self.forward(state_polynormial)
        action = dist.rsample()
        action = action.detach().numpy()
        mu = mu.detach().numpy()

        # action clipping
        if action[[0]] > 3.14159 / 9:
            action[[0]] = 3.14159 / 9
        if action[[0]] < -3.14159 / 9:
            action[[0]] = -3.14159 / 9
        return action, mu

    def update(self, s_batch, a_batch, bootstrap):
        """[summary]

        Parameters
        ----------
        s_batch : [batch, N_s]
        a_batch : [batch, N_a]
        bootstrap : [batch, 1]
            [description]

        Returns
        -------
        a_loss
        """
        if len(s_batch.shape) == 1:
            s_batch = s_batch.reshape((-1, s_batch.shape[0]))

        # convert into tensor
        s_batch = self.preprocess(s_batch)
        s_batch = torch.as_tensor(s_batch).detach()
        bootstrap = torch.as_tensor(bootstrap).detach()
        a_batch = torch.as_tensor(a_batch).detach()
        mu, sigma, dist = self.forward(s_batch)
        log_pi = dist.log_prob(a_batch)
        log_pi = log_pi.reshape(-1)
        bootstrap = bootstrap.reshape(-1)

        # actor loss
        a_loss = torch.mean(-log_pi * bootstrap)

        # update
        self._opt.zero_grad()
        a_loss.backward()
        self._opt.step()

        return a_loss

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
        return torch.Tensor(self._pipeline.fit_transform(X)[:, 1:])

    def save_parameter(self, logdir):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))


class Critic(nn.Module):
    """
    value function with polynomial feature
    """

    def __init__(self, input_size, output_size, order=1, lr=0.01):
        super(Critic, self).__init__()

        # generate polynomial feature using sklearn
        self.out_size = output_size
        po = PolynomialFeatures(degree=order)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names()) - 1
        self._pipeline = Pipeline([('poly', po)])

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(self._poly_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Identity()
        )
        # initial optimizer
        self._opt = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
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
        return self._evaluate0(state).detach().numpy()

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
        i = 0
        while True:
            v = self._evaluate0(state)
            v_loss = torch.mean((v - target_v) * (v - target_v))
            self._opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self._opt.step()
            i += 1
            if v_loss.detach().numpy() < 0.1 or i >= 100:
                break

        return v_loss

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
        state_tensor = self.preprocess(state)
        state_tensor.requires_grad_(True)
        predict = self.forward(state_tensor)
        derivative, = torch.autograd.grad(torch.sum(predict), state_tensor)
        return derivative.detach().numpy()

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
