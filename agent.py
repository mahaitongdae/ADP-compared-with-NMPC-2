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
        self._w = np.random.normal(0.0,0.005,[self._poly_feature_dim, self._out_dim])
        self._w[0, 0] = 0.0
        # self._w = np.zeros([self._poly_feature_dim, self._out_dim])

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
        policy_w = np.load(os.path.join(load_dir, 'actor.pth'))
        self.set_w(policy_w)

class Actor(nn.Module):
    def __init__(self, input_size, output_size, order=1, lr=0.001):
        super(Actor, self).__init__()

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
        x = self.layers(x)
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
            self._opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self._opt.step()
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
            self._opt.zero_grad()  # TODO
            u_loss.backward(retain_graph=True)
            self._opt.step()
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

        # zeros state value
        self._zero_state = torch.tensor([0.0]) # TODO: oneD change

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
            self._opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self._opt.step()
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
            v_loss = self.loss_function(state, utility, f_xu) + 10 * torch.pow(value_base, 2)
            self._opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True) # TODO: retain_graph=True operation?
            self._opt.step()
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
