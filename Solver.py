"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    OCP example for lane keeping problem in a circle road

    [Method]
    Open loop solution

"""
from  casadi import *
from matplotlib import pyplot as plt
from Config import Dynamics_Config
import math
#constants

class Solver(Dynamics_Config):
    def __init__(self):
        self.U_LOWER = - math.pi / 9
        self.U_UPPER = math.pi / 9
        self._sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.X_init = [0.0, 0.0, 0.1, 0.0, 0.0]
        self.zero = [0., 0., 0., 0., 0.]


        super(Solver, self).__init__()

    def mpc_solver(self, x_init, predict_steps):
        # Open loop solution
        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # discrete dynamic model
        self.f = vertcat(
            x[0] + self.Ts * (self.u * sin(x[2]) + x[1] * cos(x[2])),
            x[1] + self.Ts * (-self.D * self.F_z1 * sin(
                self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u))) * cos(u[0])
                              - self.D * self.F_z2 * sin(
                        self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u))) / self.m - self.u * x[3]),
            x[2] + self.Ts * (x[3]),
            x[3] + self.Ts * (self.a * (-self.D * self.F_z1 * sin(
                self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u)))) * cos(u[0])
                              - self.b * (-self.D * self.F_z2 * sin(
                        self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u)))) / self.I_zz),
            x[4] + self.Ts * (self.u * cos(x[2]) - x[1] * sin(x[2]))
        )

        # Create solver instance
        self.F = Function("F", [x, u], [self.f])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init
        ubw += x_init

        for k in range(1, predict_steps + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [self.U_LOWER]
            ubw += [self.U_UPPER]

            Fk = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += self.zero
            ubg += self.zero
            w += [Xk]
            lbw += [-inf, -inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, inf]


            # Cost function
            F_cost = Function('F_cost', [x, u], [0.1 * (x[0] - sin(self.k_curve * x[4])) ** 2
                                                 + 0.1 * (x[2] - arctan(self.k_curve * cos(self.k_curve * x[4]))) ** 2
                                                 + 0.001 * u[0] ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([predict_steps, self.DYNAMICS_DIM])
        control = np.zeros([predict_steps, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(predict_steps):
            state[i] = state_all[nt * i: nt * i + nt - 1].reshape(-1)
            control[i] = state_all[nt * i + nt - 1]
        return state, control


