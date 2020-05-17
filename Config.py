from __future__ import print_function

class GeneralConfig(object):
    BATCH_SIZE = 256
    DYNAMICS_DIM = 5
    STATE_DIM = 4
    ACTION_DIM = 1
    BUFFER_SIZE = 5000
    FORWARD_STEP = 30
    GAMMA_D = 1

    NP = 50
    NP_TOTAL = 500


class Dynamics_Config(GeneralConfig):
    a = 1.14       # distance c.g.to front axle(m)
    L = 2.54       # wheel base(m)
    b = L - a      # distance c.g.to rear axle(m)
    m = 1500.      # mass(kg)
    I_zz = 2420.0  # yaw moment of inertia(kg * m ^ 2)
    C = 1.43       # parameter in Pacejka tire model
    B = 14.        # parameter in Pacejka tire model
    u = 20         # longitudinal velocity(m / s)
    g = 9.81
    D = 0.75
    k1 = -88000    # front axle cornering stiffness for linear model (N / rad)
    k2 = -94000    # rear axle cornering stiffness for linear model (N / rad)
    Is = 1.        # steering ratio
    Ts = 0.1       # control signal period
    N = 314        # total simulation steps

    F_z1 = m * g * b / L    # Vertical force on front axle
    F_z2 = m * g * a / L    # Vertical force on rear axle

    k_curve = 1/30          # curve shape of sin(kx)

    # ADP reset state range
    y_range = 5
    psi_range = 1.7
    beta_range = 1.0



def test():

    print('加油')


if __name__ == "__main__":
    test()

