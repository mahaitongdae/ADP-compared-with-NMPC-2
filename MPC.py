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
Np = 40
Npt = 314
nx = 5
nu = 1
V_REF = 2.
kf = -88000.
kr = -94000.
U_LOWER = - inf
U_UPPER = inf
rho_epect = 100.0

def reference_trajectory(x, length, k = 1/5):
    """
    Generate reference trajectory of sin curve.
    Parameters
    ----------
    x: np.array([nx, 1])
        start longitudinal position of reference trajectory.
    length: steps of prediction horizon, int
    k: int
        k in the curve shape of sin(kx), int

    Returns
    -------
    reference_trajectory: np.array([nx,3])
        reference trajectory of [y, psi, x]
        y: lateral position
        psi: vehicle yaw angle
        x: vehicle longitudinal position

    """
    reference_trajectory = np.zeros([length,3])
    psi = np.arctan(k * np.cos(k * x))
    for i in range(length):
        x = x + T * (v_long * np.cos(psi))
        y = np.sin(k * x)
        psi = np.arctan(k * np.cos(k * x))
        reference_trajectory[i, :] = np.array([y, psi, x])
    return reference_trajectory

def MPC_slover(x_init, Np):
    # Open loop solution
    sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
    x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    X_init = x_init
    zero = [0., 0., 0., 0., 0.]

    # discrete vehicle dynamics
    # x:
    f = vertcat(
        x[0] + T * (u_long * sin(x[2]) + x[1] * cos(x[2])),
        x[1] + T * (-mu * F_z1 * sin(C * arctan(B * (-u[0] + (x[1] + a * x[3]) / u_long))) * cos(u[0])
                    - mu * F_z2 * sin(C * arctan(B * ((x[1] - b * x[3]) / u_long))) / m - u_long * x[3]),
        x[2] + T * (x[3]),
        x[3] + T * (a * (-mu * F_z1 * sin(C * arctan(B * (-u[0] + (x[1] + a * x[3]) / u_long)))) * cos(u[0])
                    - b * (-mu * F_z2 * sin(C * arctan(B * ((x[1] - b * x[3]) / u_long)))) / I_zz),
        x[4] + T * (u_long * cos(x[2]) - x[1] * sin(x[2]))
    )

    # Create solver instance
    F = Function("F", [x, u], [f])

    # Empty NLP
    w = [];    lbw = [];    ubw = [];    lbg = [];    ubg = []
    G = [];    J = 0

    # Initial conditions
    Xk = MX.sym('X0', nx)
    w += [Xk]
    lbw += X_init
    ubw += X_init

    # sin reference
    reference = reference_trajectory(x_init[4], Np)

    for k in range(1, Np + 1):
        # Local control
        Uname = 'U' + str(k - 1)
        Uk = MX.sym(Uname, nu)
        w += [Uk];        lbw += [U_LOWER];        ubw += [U_UPPER]

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
        F_cost = Function('F_cost', [x, u], [10 * (x[0] - reference[k-1, 0]) ** 2
                                             + 0.2 * (x[2] - reference[k-1, 1]) ** 2 + 20 * u[0] ** 2])
        J += F_cost(w[k * 2], w[k * 2 - 1])

    # Create NLP solver
    nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
    S = nlpsol('S', 'ipopt', nlp, sol_dic)

    # Solve NLP
    r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
    # print(r['x'])
    u = np.array(r['x'])[nx+1]
    x_next = np.array(r['x'])[nx+2:nx+2+nx].reshape(-1)

    state_all = np.array(r['x'])
    state = np.zeros([Np, nx]);
    control = np.zeros([Np, nu])
    nt = nx + nu  # total variable per step

    for i in range(Np):
        state[i] = state_all[nt * i: nt * i + nt - 1].reshape(-1)
        control[i] = state_all[nt * i + nt - 1]

    plt.figure(1)
    plt.plot(state[1:, 4], state[1:, 0], label='estimate trajectory')
    plt.plot(reference[:, -1], reference[:, 0], label='reference trajectory')
    # plt.plot(range(Np), state[:, 0], label='estimate trajectory')
    # plt.plot(range(Np), reference[:, 0], label='reference trajectory')
    plt.legend(loc='upper right')
    plt.show()

    return control, state

def main():
    x_init = [0., 0., 0.2, 0., 0.]
    u_history = np.zeros([Npt, 1])
    x_history = np.zeros([Npt, nx])
    for i in range(20):
        control, state = MPC_slover(x_init, Np)
        u_history[i] = control[0]
        x_history[i] = state[1].reshape(-1)
        x_init = x_history[i].tolist()
        print("steps:",i)

    plt.figure(1)
    plt.plot(x_history[:,-1],u_history)
    plt.figure(2)
    plt.plot(x_history[:,-1], x_history[:,0])
    plt.show()

if __name__ == '__main__':
    main()

