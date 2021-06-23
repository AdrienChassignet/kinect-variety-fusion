from pickle import load
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720

FOLDER_NAME = "data/"
TIMESTAMP = "_20210608"

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

def J(x, cst):
    """
    Define the Jacobian of the system F.
    cst = [u0, v0, u1, v1, u2, v2, coeffs]
    x = [u, v, g1, g2, g3]
    """
    [u0, v0, u1, v1, u2, v2, coeffs] = cst
    [u, v, g1, g2, g3] = x
    df1du = 2*u*g3**2 - 2*g3*u0 + 2*g3*coeffs[3]*(g1*u1-u0) + 2*g3*coeffs[4]*(g2*u2-u0)
    df1dv = -2*v*g3**2 + 2*g3*v0 - 2*g3*coeffs[3]*(g1*v1-v0) - 2*g3*coeffs[4]*(g2*v2-v0)
    df1dg1 = 2*g1*coeffs[0]*(u1**2-v1**2) + 2*(v1*v0-u1*u0)*(coeffs[0]+coeffs[1]+coeffs[3]) + 2*g2*coeffs[1]*(u1*u2-v1*v2) + 2*g3*coeffs[3]*(u1*u-v1*v)
    df1dg2 = 2*g2*coeffs[2]*(u2**2-v2**2) + 2*(v2*v0-u2*u0)*(coeffs[1]+coeffs[2]+coeffs[4]) + 2*g1*coeffs[1]*(u1*u2-v1*v2) + 2*g3*coeffs[4]*(u2*u-v2*v)
    df1dg3 = 2*g3*(u**2-v**2) + 2*(v*v0-u*u0)*(coeffs[3]+coeffs[4]+1) + 2*g1*coeffs[3]*(u1*u-v1*v) + 2*g2*coeffs[4]*(u2*u-v2*v)

    df2du = 0
    df2dv = 2*v*g3**2 + 2*g3*(-v0 + coeffs[3]*(g1*v1-v0) + coeffs[4]*(g2*v2-v0))
    df2dg1 = 2*g1*coeffs[0]*(v1**2-1) + 2*(1-v1*v0)*(coeffs[0]+coeffs[1]+coeffs[3]) + 2*g2*coeffs[1]*(v1*v2-1) + 2*g3*coeffs[3]*(v1*v-1)
    df2dg2 = 2*g2*coeffs[2]*(v2**2-1) + 2*(1-v2*v0)*(coeffs[1]+coeffs[2]+coeffs[4]) + 2*g1*coeffs[1]*(v1*v2-1) + 2*g3*coeffs[4]*(v2*v-1)
    df2dg3 = 2*g3*(v**2-1) + 2*(1-v*v0)*(coeffs[3]+coeffs[4]+1) + 2*g1*coeffs[3]*(v1*v-1) + 2*g2*coeffs[4]*(v2*v-1)

    df3du = g3*coeffs[3]*(g1*v1-v0) + g3*coeffs[4]*(g2*v2-v0) + g3*(g3*v-v0)
    df3dv = g3*coeffs[3]*(g1*u1-u0) + g3*coeffs[4]*(g2*u2-u0) + g3*(g3*u-u0)
    df3dg1 = 2*g1*coeffs[0]*u1*v1 - (v1*u0+u1*v0)*(coeffs[0]+coeffs[1]+coeffs[3]) + g2*coeffs[1]*(u1*v2+v1*u2) + g3*coeffs[3]*(v1*u+u1*v)
    df3dg2 = 2*g2*coeffs[2]*u2*v2 - (v2*u0+u2*v0)*(coeffs[1]+coeffs[2]+coeffs[4]) + g1*coeffs[1]*(u1*v2+v1*u2) + g3*coeffs[4]*(v2*u+u2*v)
    df3dg3 = 2*g3*u*v - (u*v0+v*u0)*(coeffs[3]+coeffs[4]+1) + g1*coeffs[3]*(v1*u+u1*v) + g2*coeffs[4]*(v2*u+u2*v)

    df4du = g3*coeffs[3]*(g1-1) + g3*coeffs[4]*(g2-1) + g3*(g3-1)
    df4dv = 0
    df4dg1 = 2*g1*coeffs[0]*u1 - (u0+u1)*(coeffs[0]+coeffs[1]+coeffs[3]) + g2*coeffs[1]*(u1+u2) + g3*coeffs[3]*(u+u1)
    df4dg2 = 2*g2*coeffs[2]*u2 - (u0+u2)*(coeffs[1]+coeffs[2]+coeffs[4]) + g1*coeffs[1]*(u1+u2) + g3*coeffs[4]*(u+u2)
    df4dg3 = 2*g3*u - (u+u0)*(coeffs[3]+coeffs[4]+1) + g1*coeffs[3]*(u+u1) + g2*coeffs[4]*(u+u2)

    df5du = 0
    df5dv = g3*coeffs[3]*(g1-1) + g3*coeffs[4]*(g2-1) + g3*(g3-1)
    df5dg1 = 2*g1*coeffs[0]*v1 - (v1+v0)*(coeffs[0]+coeffs[1]+coeffs[3]) + g2*coeffs[1]*(v2+v1) + g3*coeffs[3]*(v1+v)
    df5dg2 = 2*g2*coeffs[2]*v2 - (v2+v0)*(coeffs[1]+coeffs[2]+coeffs[4]) + g1*coeffs[1]*(v2+v1) + g3*coeffs[4]*(v2+v)
    df5dg3 = 2*g3*v - (v0+v)*(coeffs[3]+coeffs[4]+1) + g1*coeffs[3]*(v1+v) + g2*coeffs[4]*(v2+v)

    return np.array([
        [df1du, df1dv, df1dg1, df1dg2, df1dg3],
        [df2du, df2dv, df2dg1, df2dg2, df2dg3],
        [df3du, df3dv, df3dg1, df3dg2, df3dg3],
        [df4du, df4dv, df4dg1, df4dg2, df4dg3],
        [df5du, df5dv, df5dg1, df5dg2, df5dg3],
    ])

def Newton_system(F, J, cst, x, max_iter=100, eps=1e-4):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    cst are the known coefficients in the system F.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = F(x, cst)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, cst), -F_value)
        x = x + delta
        F_value = F(x, cst)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    return x, iteration_counter

def iterative_newton(F, J, cst, x0, max_iter=100, eps=1e-4):
    x = x0
    nb_iter = 0
    for k in range(max_iter):
        # Solve J(xn)*( xn+1 - xn ) = -F(xn):
        diff = np.linalg.solve(J(x, cst), -F(x, cst))
        x = x + diff
        # Stop condition:
        if np.linalg.norm(diff) < eps:
            print('Convergence!, nb iter:', k )
            nb_iter = k
            break

    else: # only if the for loop end 'naturally'
        print('Not converged')

    return x, nb_iter-1

def load_images(visu=False):
    rgb_cam1 = cv2.imread(FOLDER_NAME+"rgb_left"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
    depth_cam1 = np.load(FOLDER_NAME+"depth_left"+TIMESTAMP+".npy")
    rgb_cam2 = cv2.imread(FOLDER_NAME+"rgb_right"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
    depth_cam2 = np.load(FOLDER_NAME+"depth_right"+TIMESTAMP+".npy")

    rgb_gt = cv2.imread("data/rgb_mid_20210608.jpg", cv2.IMREAD_COLOR)
    rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
    depth_gt = np.load("data/depth_mid_20210608.npy")
    
    if visu:
        fig = plt.figure("Input views")
        ax = fig.add_subplot(2, 2, 1)
        imgplot = plt.imshow(rgb_cam1)
        ax.set_title('RGB Cam1')
        ax = fig.add_subplot(2, 2, 2)
        imgplot = plt.imshow(rgb_cam2)
        ax.set_title('RGB Cam2')
        ax = fig.add_subplot(2, 2, 3)
        imgplot = plt.imshow(depth_cam1)
        ax.set_title('Depth Cam1')
        ax = fig.add_subplot(2, 2, 4)
        imgplot = plt.imshow(depth_cam2)
        ax.set_title('Depth Cam2')

        fig = plt.figure("Ground truth cam")
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(rgb_gt)
        ax.set_title('RGB')
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(depth_gt)
        ax.set_title('Depth')
        plt.show()

    return rgb_cam1, depth_cam1, rgb_cam2, depth_cam2, rgb_gt, depth_gt

def main():
    rgb_cam1, depth_cam1, rgb_cam2, depth_cam2, rgb_gt, depth_gt = load_images(visu=False)

    #Q0 bottom-right corner
    q01 = (u01, v01) = (676, 449)
    d01 = depth_cam1[v01][u01]
    q02 = (u02, v02) = (757, 448)
    d02 = depth_cam2[v02][u02]
    q0v = (u0v, v0v) = (658, 451)
    d0v = depth_gt[v0v][u0v]
    #Q1 top-right corner
    q11 = (u11, v11) = (675, 345)
    d11 = depth_cam1[v11][u11]
    q12 = (u12, v12) = (754, 335)
    d12 = depth_cam2[v12][u12]
    q1v = (u1v, v1v) = (657, 346)
    d1v = depth_gt[v1v][u1v]
    #Q2 4 squares left and 3 up from Q0
    q21 = (u21, v21) = (616, 404)
    d21 = depth_cam1[v21][u21]
    q22 = (u22, v22) = (696, 402)
    d22 = depth_cam2[v22][u22]
    q2v = (u2v, v2v) = (598, 406)
    d2v = depth_gt[v2v][u2v]
    #Q end of ruler contact with ground
    q1 = (u1, v1) = (491, 195)
    d1 = depth_cam1[v1][u1]
    q2 = (u2, v2) = (700, 200)
    d2 = depth_cam2[v2][u2]
    qv = (uv, vv) = (541, 203)
    dv = depth_gt[vv][uv]

    print("Check depths: ", d01, d02, d0v, d11, d12, d1v, d21, d22, d2v, d1, d2, dv)

    K1s = build_constraints_matrix(q01, q11, q21, d01, d11, d21, q1, d1)
    K2s = build_constraints_matrix(q02, q12, q22, d02, d12, d22, q2, d2)
    Ks = np.concatenate((K1s, K2s), axis=0)
    _, D, V = np.linalg.svd(Ks)
    print(D)
    coeffs = V[-1]/V[-1,-1]
    print("Structure coefficients: ", coeffs)

    theta = -0.75
    mu = 1.75
    c4 = 1139/1036
    c5 = -2.25
    z = 61/37
    gt_coeffs = [(1+theta**2)*(z)**2+c4**2, theta*mu*z**2+c4*c5, mu**2*z**2+c5**2, c4, c5]
    print("Ground truth structure coefficients: ", gt_coeffs)

    expected = np.array([uv, vv, d1v/d0v, d2v/d0v, dv/d0v])
    print("Expected u and v: ", expected)
    tol = 1e-4
    x0 = np.array([u1, v1, d11/d01, d21/d01, d1/d01])
    # x0 = np.array([700, 450, .8, .9, 1.2])
    # x0 = expected
    # x0 = np.array([0,0,1,1,1])
    print("Initials u, v, g1, g2 and g3 = ", x0)
    # x, n = Newton_system(F, J, [u0v,v0v,u1v,v1v,u2v,v2v,gt_coeffs], x0, max_iter=1000, eps=1e-5)
    # print("Nb of iterations = ", n, " to get x = ", x)
    # error_norm = np.linalg.norm(expected - x, ord=2)
    # # assert error_norm < tol, 'norm of error =%g' % error_norm
    # print('norm of error =%g' % error_norm)
    # print("F(root)=", F(x, [u0v,v0v,u1v,v1v,u2v,v2v,gt_coeffs]))

    print("---- fsolve method ----")
    x = fsolve(F, x0, args=[u0v,v0v,u1v,v1v,u2v,v2v,coeffs])
    print("Solution with fsolve: ", x)
    print("F(x) = ", F(x, [u0v,v0v,u1v,v1v,u2v,v2v,coeffs]))

    print("F(gt_view1)=", F([u1, v1, d11/d01, d21/d01, d1/d01], [u01,v01,u11,v11,u21,v21,coeffs]))

if __name__ == "__main__":
    main()


"""
Points for images of the 06/07-16h55m45s

#Q0 top-left corner
    q01 = (u01, v01) = (371, 466)
    d01 = depth_cam1[v01][u01]
    q02 = (u02, v02) = (747, 466)
    d02 = depth_cam2[v02][u02]
    q0v = (u0v, v0v) = (559, 474)
    d0v = depth_gt[v0v][u0v]
    #Q1 top-right corner
    q11 = (u11, v11) = (490, 481)
    d11 = depth_cam1[v11][u11]
    q12 = (u12, v12) = (905, 483)
    d12 = depth_cam2[v12][u12]
    q1v = (u1v, v1v) = (693, 489)
    d1v = depth_gt[v1v][u1v]
    #Q2 bottom-left corner
    q21 = (u21, v21) = (335, 587)
    d21 = depth_cam1[v21][u21]
    q22 = (u22, v22) = (734, 583)
    d22 = depth_cam2[v22][u22]
    q2v = (u2v, v2v) = (535, 592)
    d2v = depth_gt[v2v][u2v]
    if True:
        #Q bottom-right corner
        q1 = (u1, v1) = (458, 616)
        d1 = depth_cam1[v1][u1]
        q2 = (u2, v2) = (900, 618)
        d2 = depth_cam2[v2][u2]
        qv = (uv, vv) = (675, 623)
        dv = depth_gt[vv][uv]
    else:
        # A chair right lombar support
        q1 = (u1, v1) = (558, 446)
        d1 = depth_cam1[v1][u1]
        q2 = (u2, v2) = (853, 447)
        d2 = depth_cam2[v2][u2]
        qv = (uv, vv) = (714, 457)
        dv = depth_gt[vv][uv]
"""