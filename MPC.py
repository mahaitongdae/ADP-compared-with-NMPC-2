"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    OCP example for lane keeping problem in a circle road

    [Method]
    Model predictive control

"""
from  casadi import *
from matplotlib import pyplot as plt

# Vehicle parameters
a = 1.14       # distance c.g.to front axle(m)
L = 2.54       # wheel base(m)
b = L - a      # distance c.g.to rear axle(m)
m = 1500.      # mass(kg)
I_zz = 2420.0  # yaw moment of inertia(kg * m ^ 2)
C = 1.43       # parameter in Pacejka tire model
B = 14.        # parameter in Pacejka tire model
u_long = 20         # longitudinal velocity(m / s)
g = 9.81
mu = 1.0
k1 = -88000    # front axle cornering stiffness for linear model (N / rad)
k2 = -94000    # rear axle cornering stiffness for linear model (N / rad)
Is = 1.        # steering ratio
Ts = 0.1       # control signal period
v_long = 15.   # longitudinal speed
N = 314        # total simulation steps

F_z1 = m * g * b / L
F_z2 = m * g * a / L

#constants
T  = 0.1
Np = 20
Npt = 314
nx = 5
nu = 1
V_REF = 2.
kf = -88000.
kr = -94000.
U_LOWER = - inf
U_UPPER = inf
rho_epect = 100.0

if __name__ == '__main__':
    # Open loop solution
    sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
    x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    X_init = [100.0, 0.0, 0.1, 0.0, 0.0]
    zero = [0., 0., 0., 0., 0.]

    # Dynamic model

    # linear
    # f = vertcat(
    #     x[0] + T * (-u_long * sin(x[2])  - u_long * tan(x[3]) * cos(x[2])),
    #     x[1] + T * (x[3] - (u_long * cos(x[1]) - u_long * tan(x[2]) * sin(x[1])) / 100),
    #     x[2] + T * ((kf*(-u[0] + x[2] + a * x[3] / u_long) * cos(u[0]) + k2 * (x[2] - b * x[3] / u_long)) / (m * u_long) - x[3]),
    #     x[3] + T * (a * (kf * (-u[0] + x[2] + a * x[3] / u_long) * cos(u[0]) - b * (kr * (x[2] - b * x[3] / u_long))) / I_zz)
    # )

    # discrete
    f = vertcat(
        x[0] + T * (-u_long * sin(x[2]) - u_long * tan(x[3]) * cos(x[2])),
        x[1] + T * ((u_long * cos(x[2]) - u_long * tan(x[3]) * sin(x[2])) / 100 ),
        x[2] + T * (x[4] - (u_long * cos(x[2]) - u_long * tan(x[3]) * sin(x[2])) / 100 ),
        x[3] + T * ((-mu * F_z1 * sin(C * arctan(B * (-u[0] + (x[3] + a * x[4] / u_long)))) * cos(u[0])
                     - mu * F_z2 * sin(C * arctan(B * (x[3] - b * x[4] / u_long)))) / (m * u_long) - x[4]),
        x[4] + T * ((a * (-mu * F_z1 * sin(C * arctan(B * (-u[0] + (x[3] + a * x[4] / u_long))))) * cos(u[0])
                     - b * ( -mu * F_z2 * sin(C * arctan(B * (x[3] - b * x[4] / u_long))))) / I_zz)
    )

    # Create solver instance
    F = Function("F", [x, u], [f])

    # Empty NLP
    w = []; lbw=[]; ubw=[];lbg = [];ubg = []
    G = []; J = 0

    # Initial conditions
    Xk = MX.sym('X0', nx)
    w += [Xk]
    lbw += X_init
    ubw += X_init

    for k in range(1, Npt+1):

        # Local control
        Uname = 'U' + str(k - 1)
        Uk = MX.sym(Uname, nu)
        w += [Uk]; lbw += [U_LOWER]; ubw += [U_UPPER]

        Fk = F(Xk, Uk)
        Xname = 'X' + str(k)
        Xk = MX.sym(Xname, nx)

        # Dynamic Constriants
        G += [Fk - Xk]
        lbg += zero
        ubg += zero
        w += [Xk]
        lbw += [-inf, -inf, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, inf]

        # Cost function
        F_cost = Function('F_cost', [x, u], [10 * (x[0] - 100) ** 2 + 0.2 * x[2] ** 2 + 20 * u[0] ** 2])
        J += F_cost(w[k * 2], w[k * 2 - 1])


    # Create NLP solver
    nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
    S = nlpsol('S', 'ipopt', nlp, sol_dic)

    # Solve NLP
    r = S(lbx=lbw, ubx=ubw, x0=0,lbg=lbg,ubg=ubg)
    print(r['x'])
    state_all = np.array(r['x'])
    state = np.zeros([Npt, nx]); control = np.zeros([Npt, nu])


    for i in range(314):
        state[i] = state_all[6*i : 6*i+5].reshape(-1)
        control[i] = state_all[6 * i + 5]

    # Draw figures
    plt.figure(1)
    plt.plot(range(314), control)
    plt.figure(2)
    plt.subplot(111, projection='polar')
    plt.plot(state[:, 1], state[:, 0])
    plt.show()

    # MPC solution

