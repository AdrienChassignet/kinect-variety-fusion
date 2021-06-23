import numpy as np

def build_constraints_matrix(q0, q1, q2, d0, d1, d2, q, d):
    g1 = d1/d0
    g2 = d2/d0
    g3 = d/d0
    a = g1*q1[0] - q0[0]
    b = g2*q2[0] - q0[0]
    c = g3*q[0] - q0[0]
    l = g1*q1[1] - q0[1]
    m = g2*q2[1] - q0[1]
    n = g3*q[1] - q0[1]
    r = g1 - 1
    s = g2 - 1
    t = g3 - 1

    return np.array([
        [a**2 - l**2, 2*a*b - 2*l*m, b**2 - m**2, 2*a*c - 2*l*n, 2*b*c - 2*m*n, c**2 - n**2],
        [l**2 - r**2, 2*l*m - 2*r*s, m**2 - s**2, 2*l*n - 2*r*t, 2*m*n - 2*s*t, n**2 - t**2],
        [a*l, b*l + a*m, m*b, c*l + a*n, c*m + b*n, n*c],
        [a*r, b*r + a*s, s*b, c*r + a*t, c*s + b*t, t*c],
        [l*r, m*r + l*s, s*m, n*r + l*t, n*s + m*t, t*n]
    ])

def F(x, cst):
    """
    Define the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs]
    x = [u, v, g1, g2, g3]
    """
    [u0, v0, u1, v1, u2, v2, coeffs] = cst
    [u, v, g1, g2, g3] = x
    a = g1*u1 - u0
    b = g2*u2 - u0
    c = g3*u - u0
    l = g1*v1 - v0 
    m = g2*v2 - v0
    n = g3*v - v0
    r = g1 - 1
    s = g2 - 1
    t = g3 - 1
    return np.array([
        coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*(a*c-l*n) + 2*coeffs[4]*(b*c-m*n) + c**2 - n**2,
        coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*(l*n-r*t) + 2*coeffs[4]*(m*n-s*t) + n**2 - t**2,
        coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b + coeffs[3]*(l*c+n*a) + coeffs[4]*(m*c+b*n) + c*n,
        coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b + coeffs[3]*(r*c+t*a) + coeffs[4]*(s*c+b*t) + c*t,
        coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s + coeffs[3]*(l*t+n*r) + coeffs[4]*(m*t+s*n) + t*n   
    ])

def sum_of_squares_of_F(x, cst):
    return np.sum(F(x,cst)**2)