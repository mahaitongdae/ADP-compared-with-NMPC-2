import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from Dynamic_Model import Dynamic_Model

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal
from torch.nn import init

LOG2Pi = np.log(2 * np.pi)


class Critic(nn.Module):
    """
    value funtion with polynomial feature
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
        # initial optimizor
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
        update paramters
        Parameters
        ----------
        state: state batch, shape [batch, state dimension]
        target_v: shape [batch, 1]

        Returns
        -------

        """

        v = self._evaluate0(state)
        target_v = torch.as_tensor(target_v).detach()
        v_loss = torch.mean((v - target_v) * (v - target_v))

        for _ in range(30):
            self._opt.zero_grad()  # TODO
            v_loss.backward(retain_graph=True)
            self._opt.step()
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


class Value(object):
    def __init__(self, input_size, degree, lr=1e-3):
        """
        Value function with polynomial feature, linear approximation
        Update with gradient descent
        Parameters
        ----------
        input_size : int
            state dimension
        degree : int
            the max degree of polynomial function
        """

        po = PolynomialFeatures(degree=degree)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names())
        self._pipeline = Pipeline([('poly', po)])

        # self._w = np.zeros((self._poly_feature_dim, 1))
        self._w = 0.05 * np.random.rand(self._poly_feature_dim, 1)
        self.lr = lr

    def reset_grad(self):
        self._w = 0.05 * np.random.rand(self._poly_feature_dim, 1)

    def predict(self, X):
        """
        Doing prediction
        Parameters
        ----------
        X : np.array
            shape (batch, state_dim)

        Returns
        -------
        out : np.array
            shape : (batch, 1)
        """

        return self.preprocess(X).dot(self._w)

    def gradient_decent(self, x, utility, f_xu):

        p_V_x = self.get_derivative(x)
        hamilton = utility + np.diag(p_V_x.dot(f_xu.T))[:, np.newaxis]
        loss = 0.5 * np.power(hamilton, 2).mean()
        deri_H_w = []
        for i in range(x.shape[0]):
            temp = np.concatenate((np.array([1]), x[i, :]))
            pp_V_xw = np.zeros([self._poly_feature_dim, x.shape[1]])
            pp_V_xw[[1, 5, 6, 7, 8], 0] = temp
            pp_V_xw[[2, 6, 9, 10, 11], 1] = temp
            pp_V_xw[[3, 7, 10, 12, 13], 2] = temp
            pp_V_xw[[4, 8, 11, 13, 14], 3] = temp
            p_H_w = pp_V_xw.dot(f_xu[i, :])
            deri_H_w.append(p_H_w)
        deri_H_w = np.array(deri_H_w)
        grad_w = 1 / x.shape[0] * np.dot(hamilton.T, deri_H_w)
        self._w -= self.lr * grad_w.T

        return loss, grad_w


    def update(self, x, utility, f_xu):
        """

        Parameters
        ----------
        X : np.array
            shape: (batch, state_dim)
        y : np.array
            shape: (batch, 1)

        Returns
        -------

        """
        # assert X.shape[0] == y.shape[0]
        while True:
            loss, grad_w = self.gradient_decent(x, utility, f_xu)
            print("loss:",loss,"grad_w",grad_w,"weights",self.get_w())
            if loss < 1e-1:
                break


        return loss, grad_w

    def gradient_decent_discrete(self, state, V_next, utility, type='SGD'): #TODO: 收敛速度较慢，考虑SGD，mini-batch


        if type=='BGD':
            s = state
            V = self.predict(s)
            Xp = self.preprocess(s)
            temp = utility + V_next - V
            grad_w = 1 / s.shape[0] * np.dot(-temp.T, Xp)
            self._w -= self.lr * grad_w.T
        elif type == 'SGD':
            index = np.random.choice(state.shape[0])
            s = state[index][np.newaxis,:]
            V = self.predict(s)
            Xp = self.preprocess(s)
            V_next = V_next[index][np.newaxis,:]
            utility = utility[index][np.newaxis,:]
            temp = utility + V_next - V
            grad_w = 1 / s.shape[0] * np.dot(-temp.T, Xp)
            self._w -= self.lr * grad_w.T


        return grad_w

    def adam(self, state, V_next, utility, max_iteration):
        Xp = self.preprocess(state)
        y = utility + V_next

        # 初始化
        m, dim = Xp.shape
        self.momentum = 0.1  # 冲量
        self.threshold = 1e-2  # 停止迭代的错误阈值
        self.iterations = max_iteration  # 迭代次数
        self.error = 0  # 初始错误为0

        self.beta1 = 0.9  # 算法作者建议的默认值
        self.beta2 = 0.999  # 算法作者建议的默认值
        self.e = 0.00000001  # 算法作者建议的默认值

        mt = np.zeros([1, dim])
        vt = np.zeros([1, dim])

        self.reset_grad()

        for i in range(self.iterations):
            error = 1 / (2 * m) * np.dot((self.predict(state) - y).T,
                                         (self.predict(state) - y))
            if abs(error) <= self.threshold:
                break

            index = np.random.choice(state.shape[0])
            # index = i % dim
            gradient = Xp[index] * (Xp[index].dot(self._w) - y[index])[np.newaxis, :]
            mt = self.beta1 * mt + (1 - self.beta1) * gradient
            vt = self.beta2 * vt + (1 - self.beta2) * (gradient ** 2)
            mtt = mt / (1 - (self.beta1 ** (i + 1)))
            vtt = vt / (1 - (self.beta2 ** (i + 1)))
            vtt_sqrt = np.sqrt(vtt)
            grad_w = np.multiply(mtt, np.reciprocal(vtt_sqrt + self.e))
            self._w = self._w - self.lr * grad_w.T
            if i % 5000 == 0:
                print('PEV | iteration:{:3d} | '.format(i)+'value loss:{:3.3f}'.format(float(error)))

        return float(error), grad_w

    def update_discrete(self, state, state_next, utility, max_iteration, model_print_flag, type = 'SGD',):
        s_next = state_next
        V_next = self.predict(s_next)
        V_batch = self.predict(state)
        temp = utility + V_next - V_batch
        loss = 0.5 * np.power(temp, 2).mean()
        iteration = 0
        if type == 'Adam':
            loss,grad_w = self.adam(state,V_next,utility,max_iteration)
        else:
            while True:
                grad_w = self.gradient_decent_discrete(state, V_next, utility,type=type)
                V_batch = self.predict(state)
                temp = utility + V_next - V_batch
                loss = 0.5 * np.power(temp, 2).mean()
                if model_print_flag == 1:
                    if iteration % 5000 == 0:
                        print('Iteration:{:3d} | '.format(iteration)+'Value loss:{:3.3f}'.format(loss))
                iteration += 1
                if loss < 1:
                    break

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
            # / np.array([1, 1, 8])
        return self._pipeline.fit_transform(X )

    def get_w(self):
        return self._w

    def set_w(self, x):
        assert x.shape == self._w.shape
        # assert y.shape == self._logsigma.shape
        self._w = x
        # self._logsigma = y

    def get_derivative(self, s): #todo:if POLY_DEGREE changes, needed to rewrite
        """
        calculate partial derivative of value with respect to state, i.e. p_V_x

        Parameter
        ---------
        s: np.array state
            shape:[batch, state_dim]
        Return
        ------
        p_V_x: np.array
            shape: [batch, state_dim]
        """
        shape = s.shape
        deri_state = np.concatenate((np.ones([shape[0], 1]), s), axis=1)
        # POLY_DEGREE = 2
        w_a = self._w[[1, 5, 6, 7, 8],:]
        w_b = self._w[[2, 6, 9, 10, 11],:]
        w_c = self._w[[3, 7, 10, 12, 13],:]
        w_d = self._w[[4, 8, 11, 13, 14],:]
        deri_a = deri_state.dot(w_a)
        deri_b = deri_state.dot(w_b)
        deri_c = deri_state.dot(w_c)
        deri_d = deri_state.dot(w_d)
        # POLY_DEGREE = 1

        p_V_x = np.concatenate((deri_a, deri_b, deri_c, deri_d), axis=1)

        return p_V_x


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

        self.reset_grad()
        self._logsigma = np.zeros((1, self._out_dim))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.adam_t = 0
        self.adam_m_sig = np.zeros((1, self._out_dim))
        self.adam_v_sig = np.zeros((1, self._out_dim))

        self.adam_m_mu = np.zeros((self._poly_feature_dim, self._out_dim))
        self.adam_v_mu = np.zeros((self._poly_feature_dim, self._out_dim))

    def reset_grad(self):
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

        return self.preprocess(X).dot(self._w)

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

    @staticmethod
    def log_prob(x, mu, sigma):
        """

        Parameters
        ----------
        x : np.array
            shape : (batch, action_dim), action
        mu : np.array
            shape : (batch, action_dim), mean
        sigma : np.array
            shape : (1, action_dim), std variance
        Returns
        -------
        logprob : np.array
            shape : (batch, 1), log probability of actions
        grad_mu : np.array
            shape : (batch, action_dim), d log pi / d mu
        grad_logsigma : np.array
            shape : (batch, action_dim), d log pi / d log sigma
        """
        sigma = sigma + 1e-8
        logprob = - (x - mu) ** 2 / 2 / sigma / sigma - np.log(sigma) - 0.5 * LOG2Pi
        logprob = np.prod(logprob, axis=1, keepdims=True)
        grad_mu = (x - mu) / sigma / sigma
        grad_logsigma = (x - mu) ** 2 / sigma / sigma - 1.0
        return logprob, grad_mu, grad_logsigma

    def get_w(self):
        return self._w

    def set_w(self, x):
        assert x.shape == self._w.shape
        # assert y.shape == self._logsigma.shape
        self._w = x
        # self._logsigma = y

    def adam(self, grad_mu, grad_sig):
        self.adam_t += 1
        self.adam_m_mu = self.beta1 * self.adam_m_mu + (1 - self.beta1) * grad_mu
        self.adam_v_mu = self.beta2 * self.adam_v_mu + (1 - self.beta2) * grad_mu ** 2
        adam_m_mu_hat = self.adam_m_mu / (1 - pow(self.beta1, self.adam_t))
        adam_v_mu_hat = self.adam_v_mu / (1 - pow(self.beta2, self.adam_t))
        self._w += self.lr * adam_m_mu_hat / (np.sqrt(adam_v_mu_hat) + self.epsilon)

        self.adam_m_sig = self.beta1 * self.adam_m_sig + (1 - self.beta1) * grad_sig
        self.adam_v_sig = self.beta2 * self.adam_v_sig + (1 - self.beta2) * grad_sig ** 2
        adam_m_sig_hat = self.adam_m_sig / (1 - pow(self.beta1, self.adam_t))
        adam_v_sig_hat = self.adam_v_sig / (1 - pow(self.beta2, self.adam_t))
        self._logsigma += self.lr * adam_m_sig_hat / (np.sqrt(adam_v_sig_hat) + self.epsilon)


class CriticGD(object):
    def __init__(self, input_size, degree, lr=1e-3):
        """
        Value function with polynomial feature, linear approximation
        Update with gradient descent
        Parameters
        ----------
        input_size : int
            state dimension
        degree : int
            the max degree of polynomial function
        """

        po = PolynomialFeatures(degree=degree)
        po.n_input_features_ = input_size
        self._poly_feature_dim = len(po.get_feature_names())
        self._pipeline = Pipeline([('poly', po)])

        self._w = np.zeros((self._poly_feature_dim, 1))
        self.lr = lr

    def predict(self, X):
        """
        Doing prediction
        Parameters
        ----------
        X : np.array
            shape (batch, state_dim)

        Returns
        -------
        out : np.array
            shape : (batch, 1)
        """

        return self.preprocess(X).dot(self._w)

    def update(self, X, y):
        """

        Parameters
        ----------
        X : np.array
            shape: (batch, state_dim)
        y : np.array
            shape: (batch, 1)

        Returns
        -------

        """
        assert X.shape[0] == y.shape[0]
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        Xp = self.preprocess(X)

        y_old =Xp.dot(self._w)
        # t = y_old - y
        # print("2", y_old.shape)
        loss = np.mean((y_old - y) ** 2)
        grad_w = 2 * (y_old - y) * Xp

        grad_w = np.mean(grad_w, axis=0)
        grad_w = grad_w.reshape((-1,1))
        # print("Xp",np.max(Xp), np.min(Xp))
        # print(np.max((y_old - y)), np.min((y_old - y)), y.shape, y_old.shape)
        # print(loss)
        self._w -= self.lr * grad_w
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
            # / np.array([1, 1, 8])
        return self._pipeline.fit_transform(X )

    def get_w(self):
        return self._w

    def set_w(self, x):
        assert x.shape == self._w.shape
        # assert y.shape == self._logsigma.shape
        self._w = x
        # self._logsigma = y


class ActorLinear(object):
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

        # self._w = np.zeros((self._poly_feature_dim, self._out_dim))
        self._w = np.random.random([self._poly_feature_dim, self._out_dim]) * 0.2
        self._w[0, 0] = 0.0
        self._logsigma = np.zeros((1, self._out_dim))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.adam_t = 0
        self.adam_m_sig =  np.zeros((1, self._out_dim))
        self.adam_v_sig =  np.zeros((1, self._out_dim))

        self.adam_m_mu = np.zeros((self._poly_feature_dim, self._out_dim))
        self.adam_v_mu = np.zeros((self._poly_feature_dim, self._out_dim))


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
        mu = self.preprocess(X).dot(self._w)
        sigma = np.exp(self._logsigma)
        ep = np.random.randn(*mu.shape)
        action = mu + ep * sigma
        return action, (mu, self._logsigma)

    def update(self, s_his, a_his, adv):
        """

        Parameters
        ----------
        s_his : np.array
            shape (batch, state_dim)
        a_his : np.array
            shape (batch, action_dim)
        adv : np.array
            shape (batch, 1)

        Returns
        -------

        """
        # print(s_his.shape, a_his.shape, adv.shape)
        _, (mu, log_sig) = self.predict(s_his)

        # print(mu.shape, log_sig.shape)
        s_his_p = self.preprocess(s_his)
        # loss
        # print(s_his_p.shape)
        logpi, grad_mu, grad_logsigma = self.log_prob(a_his, mu, np.exp(log_sig))

        loss = np.mean(logpi * adv)
        grad_mu = grad_mu.reshape(-1, 1, self._out_dim)
        s_his_p = s_his_p.reshape(-1, self._poly_feature_dim, 1)
        adv_3d = adv.reshape(-1,1,1)
        # t = grad_mu * s_his_p * adv_3d
        # print(grad_logsigma.shape, adv.shape)
        w_mu_grad = np.mean(grad_mu * s_his_p * adv_3d, axis=0)
        w_logsig_grad = np.mean(grad_logsigma * adv, axis=0, keepdims=True)

        # SGD
        self._w += self.lr * w_mu_grad
        self._logsigma += self.lr * w_logsig_grad

        # ADAM
        # self.adam(w_mu_grad, w_logsig_grad)

        return loss #  , w_mu_grad, w_logsig_grad

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

    @staticmethod
    def log_prob(x, mu, sigma):
        """

        Parameters
        ----------
        x : np.array
            shape : (batch, action_dim), action
        mu : np.array
            shape : (batch, action_dim), mean
        sigma : np.array
            shape : (1, action_dim), std variance
        Returns
        -------
        logprob : np.array
            shape : (batch, 1), log probability of actions
        grad_mu : np.array
            shape : (batch, action_dim), d log pi / d mu
        grad_logsigma : np.array
            shape : (batch, action_dim), d log pi / d log sigma
        """
        sigma = sigma + 1e-8
        logprob = - (x - mu) ** 2 / 2 / sigma / sigma - np.log(sigma) - 0.5 * LOG2Pi
        logprob = np.prod(logprob, axis=1,keepdims=True)
        grad_mu = (x - mu) / sigma / sigma
        grad_logsigma = (x - mu) ** 2 / sigma / sigma - 1.0
        return logprob, grad_mu, grad_logsigma

    def get_w(self):
        return self._w, self._logsigma

    def set_w(self, x, y):
        assert x.shape == self._w.shape
        assert y.shape == self._logsigma.shape
        self._w = x
        self._logsigma = y

    def adam(self, grad_mu, grad_sig):
        self.adam_t += 1
        self.adam_m_mu = self.beta1 * self.adam_m_mu + (1-self.beta1) * grad_mu
        self.adam_v_mu = self.beta2 * self.adam_v_mu + (1-self.beta2) * grad_mu ** 2
        adam_m_mu_hat = self.adam_m_mu / (1 - pow(self.beta1, self.adam_t))
        adam_v_mu_hat = self.adam_v_mu / (1 - pow(self.beta2, self.adam_t))
        self._w += self.lr * adam_m_mu_hat / (np.sqrt(adam_v_mu_hat) + self.epsilon)

        self.adam_m_sig = self.beta1 * self.adam_m_sig + (1 - self.beta1) * grad_sig
        self.adam_v_sig = self.beta2 * self.adam_v_sig + (1 - self.beta2) * grad_sig ** 2
        adam_m_sig_hat = self.adam_m_sig / (1 - pow(self.beta1, self.adam_t))
        adam_v_sig_hat = self.adam_v_sig / (1 - pow(self.beta2, self.adam_t))
        self._logsigma += self.lr * adam_m_sig_hat / (np.sqrt(adam_v_sig_hat) + self.epsilon)


# class ActorNN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ActorNN, self).__init__()
#         self.out_size = output_size
#         po = PolynomialFeatures(degree=5)
#         po.n_input_features_ = input_size
#         self._poly_feature_dim = len(po.get_feature_names())
#         self._pipeline = Pipeline([('poly', po)])
#         log_std = np.zeros(output_size)
#         self.log_sigma = torch.nn.Parameter(torch.as_tensor(log_std))
#
#         self.layers = nn.Sequential(
#             nn.Linear(self._poly_feature_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_size),
#             nn.Identity()
#         )
#         self._initialize_weights()
#
#
#
#     def forward(self, x):
#
#         mu = self.layers(x)
#         sigma = torch.exp(self.log_sigma)
#         dist = Normal(mu, sigma)
#
#         return mu, sigma, dist
#
#     def _initialize_weights(self):
#         # print(self.modules())
#
#         for m in self.modules():
#             # print(m)
#             if isinstance(m, nn.Linear):
#                 # print(m.weight.data.type())
#                 # input()
#                 # m.weight.data.fill_(1.0)
#                 init.xavier_uniform_(m.weight)
#                 init.constant_(m.bias, 0.0)
#                 # print(m.weight)
#
#     def clip_but_pass_gradient(self, x, l=-2., u=2.):
#         clip_up = (x > u).float()
#         clip_low = (x < l).float()
#         clip_value = (u - x)*clip_up + (l - x)*clip_low
#         return x + clip_value.detach()
#
#     def choose_action(self, state):
#         """
#         :param state: np.array; shape (N_S,)
#         :return: action (N_a,)
#         """
#
#         state_T = torch.Tensor(state).reshape((-1, state.shape[0]))
#         state_T = self.preprocess(state_T)
#         mu, sigma, dist = self.forward(state_T)
#         action = dist.rsample()
#         # action = mu
#         # action = self.clip_but_pass_gradient(action)
#         action = action.detach().numpy()[0]
#         return action
#
#     def loss(self,s_his, a_his, adv):
#
#         s_his = self.preprocess(s_his)
#         # print(s_his.shape)
#         s_his = torch.as_tensor(s_his).detach()
#         a_his = torch.as_tensor(a_his).detach()
#         adv = torch.as_tensor(adv).detach()
#
#         mu, sigma, dist = self.forward(s_his)
#         print(sigma)
#         log_pi = dist.log_prob(a_his)
#         # temp = -log_pi * adv
#         log_pi = log_pi.reshape(-1)
#         a_loss = torch.mean(-log_pi * adv)
#         temp = -log_pi * adv
#         # print(temp.shape)
#         return a_loss
#
#     def preprocess(self, X):
#         """
#         transform raw data to polynomial features
#         Parameters
#         ----------
#         X :  np.array
#             shape : (batch, state_dim)
#         Returns
#         -------
#         out :  np.array
#             shape : (batch, poly_dim)
#         """
#         X = X.numpy()
#         if len(X.shape) == 1:
#             X = X.reshape(1, -1)
#         return torch.Tensor(self._pipeline.fit_transform(X))
#
#
# class ActorNN0(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ActorNN0, self).__init__()
#         self.out_size = output_size
#         log_std = np.zeros(output_size)
#         self.log_sigma = torch.nn.Parameter(torch.as_tensor(log_std))
#
#         self.layers = nn.Sequential(
#             nn.Linear(input_size,32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_size),
#             nn.Identity()
#         )
#         # self.layers = nn.Sequential(
#         #     nn.Linear(input_size, 1),
#         #
#         #     nn.Identity()
#         # )
#         self._initialize_weights()
#
#     def forward(self, x):
#
#         mu = self.layers(x)
#         sigma = torch.exp(self.log_sigma)
#         dist = Normal(mu, sigma)
#
#         return mu, sigma, dist
#
#     def _initialize_weights(self):
#         # print(self.modules())
#
#         for m in self.modules():
#             # print(m)
#             if isinstance(m, nn.Linear):
#                 # print(m.weight.data.type())
#                 # input()
#                 # m.weight.data.fill_(1.0)
#                 init.xavier_uniform_(m.weight)
#                 init.constant_(m.bias, 0.0)
#                 # print(m.weight)
#
#
#     def predict(self, state):
#         """
#         :param state: np.array; shape (N_S,)
#         :return: action (N_a,)
#         """
#
#         state_T = torch.Tensor(state).reshape((-1, state.shape[0]))
#         # state_T = self.preprocess(state_T)
#         mu, sigma, dist = self.forward(state_T)
#         action = dist.rsample()
#         # action = mu
#         # action = self.clip_but_pass_gradient(action)
#         action = action.detach().numpy()[0]
#         return action, None
#
#     def loss(self,s_his, a_his, adv):
#
#         # s_his = self.preprocess(s_his)
#         # print(s_his.shape)
#         s_his = torch.as_tensor(s_his).detach()
#         a_his = torch.as_tensor(a_his).detach()
#         adv = torch.as_tensor(adv).detach()
#
#         mu, sigma, dist = self.forward(s_his)
#         # print(sigma)
#         log_pi = dist.log_prob(a_his)
#         # print("log_pi",torch.max(log_pi), torch.min(log_pi))
#         # temp = -log_pi * adv
#         # log_pi = log_pi.reshape(-1)
#         a_loss = torch.mean(-log_pi * adv)
#         # temp = -log_pi * adv
#
#         return a_loss



if __name__ == "__main__":


    s = np.array([[1., 2., 3., 4.],[4., 3., 2., 1.]])
    value = Value(4, 2)
    value.set_w(np.random.randint(1, 5, [15, 1]))
    value.predict(s)
    print(value.preprocess(s))
    print(value.get_w())
    deri = value.get_derivative(s)
    print(deri)
    utility = np.array([[1.],[2.]])
    hamilton = np.array([[3.],[2.]])
    f_xu = np.array([[1,1,2,2],[2,2,1,1]])
    value.update(s,hamilton,f_xu )

    # p_l_u = np.random.randint(1,5, [2, 1])
    # p_f_u = np.random.randint(1,5, [2, 4])
    # policy = Policy(4,1,2)
    # loss = policy.update(s, utility, value_next, p_l_u, deri, p_f_u)
    # print(policy.get_w())
