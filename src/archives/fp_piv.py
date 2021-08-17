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

def build_constraints_matrix_norm(q0, q1, q2, d0, d1, d2, q, d):
    a = d1*q1[0] - d0*q0[0]
    b = d2*q2[0] - d0*q0[0]
    c = d*q[0] - d0*q0[0]
    l = d1*q1[1] - d0*q0[1]
    m = d2*q2[1] - d0*q0[1]
    n = d*q[1] - d0*q0[1]
    r = d1 - d0
    s = d2 - d0
    t = d - d0

    return np.array([
        [a**2 - l**2, 2*a*b - 2*l*m, b**2 - m**2, 2*a*c - 2*l*n, 2*b*c - 2*m*n, c**2 - n**2],
        [l**2 - r**2, 2*l*m - 2*r*s, m**2 - s**2, 2*l*n - 2*r*t, 2*m*n - 2*s*t, n**2 - t**2],
        [a*l, b*l + a*m, m*b, c*l + a*n, c*m + b*n, n*c],
        [a*r, b*r + a*s, s*b, c*r + a*t, c*s + b*t, t*c],
        [l*r, m*r + l*s, s*m, n*r + l*t, n*s + m*t, t*n]
    ])

def get_structure_coefficients(q0, q1, q2, d0, d1, d2, pts, d_pts):
    coeffs = []
    resids = []
    for pt_idx in range(len(pts[0])):
        Kis = []
        for view in range(len(pts)):
            Kis.append(build_constraints_matrix(q0[view], q1[view], q2[view], d0[view], d1[view], d2[view], pts[view][pt_idx], d_pts[view][pt_idx]))
        Kis = np.concatenate(Kis)
        
        _, D, V = np.linalg.svd(Kis)
        coeffs.append(V[-1]/V[-1,-1])
        resids.append(np.sum(np.matmul(Kis, coeffs[-1])**2))
        # print("Residual error: ", np.matmul(Kis, coeffs[-1]))
        # print("Residual error sum of squares: ", np.sum(np.matmul(Kis, coeffs[-1])**2))
        # print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])
    return coeffs, resids

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
    """
    Return the sum of squares of the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs]
    x = [u, v, g1, g2, g3]
    """
    return np.sum(F(x,cst)**2)

def F_3var(x, cst):
    """
    Define the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2] = cst
    [u, v, d3, _, _] = x
    a = d1*u1/d0 - u0
    b = d2*u2/d0 - u0
    c = d3*u/d0 - u0
    l = d1*v1/d0 - v0 
    m = d2*v2/d0 - v0
    n = d3*v/d0 - v0
    r = d1/d0 - 1
    s = d2/d0 - 1
    t = d3/d0 - 1
    return np.array([
        coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*(a*c-l*n) + 2*coeffs[4]*(b*c-m*n) + c**2 - n**2,
        coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*(l*n-r*t) + 2*coeffs[4]*(m*n-s*t) + n**2 - t**2,
        coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b + coeffs[3]*(l*c+n*a) + coeffs[4]*(m*c+b*n) + c*n,
        coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b + coeffs[3]*(r*c+t*a) + coeffs[4]*(s*c+b*t) + c*t,
        coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s + coeffs[3]*(l*t+n*r) + coeffs[4]*(m*t+s*n) + t*n   
    ])

def F_3var_norm(x, cst):
    """
    Define the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2] = cst
    [u, v, d3, _, _] = x
    a = d1*u1 - d0*u0
    b = d2*u2 - d0*u0
    c = d3*u - d0*u0
    l = d1*v1 - d0*v0 
    m = d2*v2 - d0*v0
    n = d3*v - d0*v0
    r = d1 - d0
    s = d2 - d0
    t = d3 - d0
    return np.array([
        coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*(a*c-l*n) + 2*coeffs[4]*(b*c-m*n) + c**2 - n**2,
        coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*(l*n-r*t) + 2*coeffs[4]*(m*n-s*t) + n**2 - t**2,
        coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b + coeffs[3]*(l*c+n*a) + coeffs[4]*(m*c+b*n) + c*n,
        coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b + coeffs[3]*(r*c+t*a) + coeffs[4]*(s*c+b*t) + c*t,
        coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s + coeffs[3]*(l*t+n*r) + coeffs[4]*(m*t+s*n) + t*n   
    ])

def F_3var_norm_fact(x, cst):
    """
    Define the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2] = cst
    [u, v, d3, _, _] = x
    a = d1*u1 - d0*u0
    b = d2*u2 - d0*u0
    l = d1*v1 - d0*v0 
    m = d2*v2 - d0*v0
    r = d1 - d0
    s = d2 - d0
    c1 = coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*d0*(l*v0 - a*u0) + 2*coeffs[4]*d0*(m*v0 - b*u0) + (d0*u0)**2 - (d0*v0)**2
    c2 = coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*d0*(r - l*v0) + 2*coeffs[4]*d0*(s - m*v0) + (d0*v0)**2 - d0**2
    c3 = coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b - coeffs[3]*d0*(l*u0 + a*v0) - coeffs[4]*d0*(m*u0 + b*v0) + u0*v0*d0**2
    c4 = coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b - coeffs[3]*d0*(r*u0 + a) - coeffs[4]*d0*(s*u0 + b) + u0*d0**2
    c5 = coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s - coeffs[3]*d0*(r*v0 + l) - coeffs[4]*d0*(s*v0 + m) + v0*d0**2
    alpha = a*coeffs[3] + b*coeffs[4] - u0*d0
    beta = l*coeffs[3] + m*coeffs[4] - v0*d0
    gamma = r*coeffs[3] + s*coeffs[4] - d0
    return np.array([
        d3*(u*(d3*u + 2*alpha) - v*(d3*v + 2*beta)) + c1,
        d3*(v*(d3*v + 2*beta) - d3 - 2*gamma) + c2,
        d3*(d3*u*v + u*beta + v*alpha) + c3,
        d3*(d3*u + u*gamma + alpha) + c4,
        d3*(d3*v + v*gamma + beta) + c5 
    ])

def call_F_3var_arg(F_3var, cst, x):
    return F_3var(x, cst)

def sum_of_squares_of_F_3var(x, cst):
    """
    Return the sum of squares of the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    return np.sum(F_3var(x,cst)**2)

def sum_of_squares_of_F_3var_norm(x, cst):
    """
    Return the sum of squares of the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    return np.sum(F_3var_norm(x,cst)**2)

def sum_of_squares_of_F_3var_norm_fact(x, cst):
    """
    Return the sum of squares of the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    return np.sum(F_3var_norm_fact(x,cst)**2)

def minimize_brute_force(x0, cst, bounds, px_width=25, d_width=500):
    u0 = int(x0[0])
    v0 = int(x0[1])
    d0 = int(x0[2])
    x =  x0
    minF = sum_of_squares_of_F_3var(x0, cst)

    for u in range(u0 - px_width, u0 + px_width + 1):
        if u in range(bounds[0][0], bounds[1][0] + 1):
            for v in range(v0 - px_width, v0 + px_width + 1):
                if v in range(bounds[0][1], bounds[1][1] + 1):
                    for d in range(d0 - d_width, d0 + d_width + 1):
                        if d in range(bounds[0][2], bounds[1][2] + 1):
                            xi = [u, v, d, 1, 1]
                            evalF = sum_of_squares_of_F_3var(xi, cst)
                            if evalF < minF:
                                print("Found better solution: ", evalF, xi)
                                minF = evalF
                                x = xi

    return x, minF