from __future__ import print_function


class StateConfig:
    a = 1.14       # distance c.g.to front axle(m)
    L = 2.54       # wheel base(m)
    b = L - a      # distance c.g.to rear axle(m)
    m = 1500.      # mass(kg)
    I_zz = 2420.0  # yaw moment of inertia(kg * m ^ 2)
    D = 0.75       # parameter in Pacejka tire model
    C = 1.43       # parameter in Pacejka tire model
    B = 14.        # parameter in Pacejka tire model
    u = 10         # longitudinal velocity(m / s)
    g = 9.81

    F_z1 = m * g * b / L
    F_z2 = m * g * a / L


def test():

    print('加油')


if __name__ == "__main__":
    test()

