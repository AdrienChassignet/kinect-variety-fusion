from numpy.linalg.linalg import LinAlgError
import open3d as o3d
import cv2
import numpy as np 
import copy
from helpers import *
import time
from scipy.spatial.transform import Rotation as R
from math import sqrt

def project_points(input_points, rot, t, K):
    img_points, _ = cv2.projectPoints(input_points, rot, t, K, None)
    img = np.zeros([100,100,3], dtype=np.uint8) 
    for point in img_points:
        img[int(point[0][1]), int(point[0][0]), :] = [255,255,255]
    img[int(img_points[0][0][1]), int(img_points[0][0][0]), :] = [0,0,255]
    img[int(img_points[1][0][1]), int(img_points[1][0][0]), :] = [0,255,0]
    img[int(img_points[2][0][1]), int(img_points[2][0][0]), :] = [255,100,0]

    return [point for pt in img_points for point in pt], img

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

def infer_point_position(q0, q1, q2, d0, d1, d2, coeffs):
    g1 = d1/d0
    g2 = d2/d0
    a = g1*q1[0] - q0[0]
    b = g2*q2[0] - q0[0]
    l = g1*q1[1] - q0[1]
    m = g2*q2[1] - q0[1]
    r = g1 - 1
    s = g2 - 1
    # print(a,b,l,m,r,s)
    # print(a*coeffs[3] + b*coeffs[4])
    # print(l*coeffs[3] + m*coeffs[4])
    # print(r*coeffs[3] + s*coeffs[4])
    # print((a**2-l**2)*coeffs[0] + (2*a*b-2*l*m)*coeffs[1] + (b**2-m**2)*coeffs[2])
    # print((l**2-r**2)*coeffs[0] + (2*l*m-2*r*s)*coeffs[1] + (m**2-s**2)*coeffs[2])
    # print(a*l*coeffs[0] + (b*l+a*m)*coeffs[1] + m*b*coeffs[2])
    # print(a*r*coeffs[0] + (b*r+a*s)*coeffs[1] + s*b*coeffs[2])
    # print(r*l*coeffs[0] + (s*l+r*m)*coeffs[1] + m*s*coeffs[2])
    alpha = a*coeffs[3] + b*coeffs[4] - q0[0]
    beta = l*coeffs[3] + m*coeffs[4] - q0[1]
    gamma = r*coeffs[3] + s*coeffs[4] - 1
    mu1 = a*l*coeffs[0] + (b*l+a*m)*coeffs[1] + m*b*coeffs[2] - (beta+q0[1])*q0[0] - (alpha+q0[0])*q0[1] + q0[0]*q0[1]
    mu2 = a*r*coeffs[0] + (b*r+a*s)*coeffs[1] + s*b*coeffs[2] - (gamma+1)*q0[0] - (alpha+q0[0]) + q0[0]
    mu3 = r*l*coeffs[0] + (s*l+r*m)*coeffs[1] + m*s*coeffs[2] - (gamma+1)*q0[1] - (beta+q0[1]) + q0[1]

    delta = (2*gamma*mu1 - 2*alpha*beta*gamma)**2 - 4*(mu1 - alpha*beta)*(mu1*gamma**2 - mu2*beta*gamma - mu3*alpha*gamma + mu2*mu3)
    g3 = (2*alpha*beta*gamma - 2*gamma*mu1 - sqrt(delta)) / (2*(mu1 - alpha*beta))
    u = (-mu2 - g3*alpha) / (g3**2 + gamma*g3)
    v = (-mu3 - g3*beta) / (g3**2 + gamma*g3)
    print("distance -sqrt: ", g3*d0, " // u: ", u, " / v: ", v)
    g3 = (2*alpha*beta*gamma - 2*gamma*mu1 + sqrt(delta)) / (2*(mu1 - alpha*beta))
    u = (-mu2 - g3*alpha) / (g3**2 + gamma*g3)
    v = (-mu3 - g3*beta) / (g3**2 + gamma*g3)
    print("distance +sqrt: ", g3*d0, " // u: ", u, " / v: ", v)
    if g3 < 0:
        g3 = (2*alpha*beta*gamma - 2*gamma*mu1 - sqrt(delta)) / (2*(mu1 - alpha*beta))

    u = (-mu2 - g3*alpha) / (g3**2 + gamma*g3)
    v = (-mu3 - g3*beta) / (g3**2 + gamma*g3)

    return u, v, g3

def infer_point_position2(q0, q1, q2, d0, d1, d2, coeffs):
    g1 = d1/d0
    g2 = d2/d0
    a = g1*q1[0] - q0[0]
    b = g2*q2[0] - q0[0]
    l = g1*q1[1] - q0[1]
    m = g2*q2[1] - q0[1]
    r = g1 - 1
    s = g2 - 1

    alpha = a*coeffs[3] + b*coeffs[4]
    beta = l*coeffs[3] + m*coeffs[4]
    gamma = r*coeffs[3] + s*coeffs[4]

    x = a*l*coeffs[0] + (l*b+m*a)*coeffs[1] + m*b*coeffs[2] - alpha*beta
    y = 2*gamma*x
    z = (gamma**2)*(a*l*coeffs[0] + (l*b+m*a)*coeffs[1] + m*b*coeffs[2]) \
        + gamma*((a*r*coeffs[0] + (r*b+s*a)*coeffs[1] + s*b*coeffs[2])*beta + \
                (r*l*coeffs[0] + (l*s+m*r)*coeffs[1] + m*s*coeffs[2])*alpha) \
        + (a*r*coeffs[0] + (r*b+s*a)*coeffs[1] + s*b*coeffs[2])*(r*l*coeffs[0] + (l*s+m*r)*coeffs[1] + m*s*coeffs[2])

    delta = y**2 - 4*x*z
    if delta > 0:
        g3 = 1 + (-y + sqrt(delta))/(2*x)
        u = q0[0]/g3 - (a*r*coeffs[0] + (r*b+s*a)*coeffs[1] + s*b*coeffs[2] + (g3 - 1)*alpha) / (g3*(gamma + g3 - 1))
        v = q0[1]/g3 - (r*l*coeffs[0] + (l*s+m*r)*coeffs[1] + m*s*coeffs[2] + (g3 - 1)*beta) / (g3*(gamma + g3 - 1))
        print("distance +sqrt: ", g3*d0, " // u: ", u, " / v: ", v)
        g3 = 1 + (-y - sqrt(delta))/(2*x)
        u = q0[0]/g3 - (a*r*coeffs[0] + (r*b+s*a)*coeffs[1] + s*b*coeffs[2] + (g3 - 1)*alpha) / (g3*(gamma + g3 - 1))
        v = q0[1]/g3 - (r*l*coeffs[0] + (l*s+m*r)*coeffs[1] + m*s*coeffs[2] + (g3 - 1)*beta) / (g3*(gamma + g3 - 1))
        print("distance -sqrt: ", g3*d0, " // u: ", u, " / v: ", v)
        return u, v, g3
    else:
        print("Complex solutions for u, v and g3")
        return 0,0,0

def F(cst, x):
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

def J(cst, x):
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
    F_value = F(cst, x)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        try:
            delta = np.linalg.solve(J(cst, x), -F_value)
        except LinAlgError:
            print("Singular matrix in np.linalg.solve, after ", iteration_counter, " iterations.")
            return x, -1
        else:
            x = x + delta
            F_value = F(cst, x)
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
        diff = np.linalg.solve(J(cst, x), -F(cst, x))
        x = x + diff
        # Stop condition:
        if np.linalg.norm(diff) < eps:
            print('Convergence!, nb iter:', k )
            nb_iter = k
            break

    else: # only if the for loop end 'naturally'
        print('Not converged')

    return x, nb_iter-1

#--------------------------------------------------------------------------------

points = np.array([[0, 0, 0.],
                [-150, -150, 300.],
                [400, 500, 150.],
                [500, -500, 200.],
                [250, 150, 250.],
                [-200, 200, 100.],
                [500, 300, 250.]])

pc1 = xyz2pcd(points)
pc1.paint_uniform_color([0., 0., 0.])
pc1.colors[1] = [1,0,0]
pc1.colors[2] = [0,1,0]
pc1.colors[3] = [0,0,1]

# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pc1)
# opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.8, 0.8, 0.8])
# viewer.run()
# viewer.destroy_window()

points = points[1:]
# proj_depth = np.array([np.linalg.norm(p) for p in points])
proj_depth = points[:,-1]
print(proj_depth)

K1 = np.array([[10., 0., 50.],
                [0., 10., 50.],
                [0., 0., 1.]])

img1_points, img1 = project_points(points, np.eye(3), np.zeros(3), K1)
print("Source 1: ", img1_points)

t12 = np.array([10., 0., 0.])
img2_points, img2 = project_points(points, np.eye(3), t12, K1)
print("Source 2: ", img2_points)

t13 = np.array([0., 10., 0.])
img3_points, img3 = project_points(points, np.eye(3), t13, K1)
print("Source 3: ", img3_points)

t14 = np.array([10., 10., 0.])
img4_points, img4 = project_points(points, np.eye(3), t14, K1)
print("Source 4: ", img4_points)

# cv2.imshow("input 1", img1)
# cv2.imshow("input 2", img2)
# cv2.imshow("input 3", img3)
# cv2.waitKey()
# cv2.destroyAllWindows()

t1v = np.array([5., 0., 0.])
img_new_points, img_new = project_points(points, np.eye(3), t1v, K1)
print("Virtual view: ", img_new_points)

m = points.shape[0]
coeffs = []

for i in range(3, m):
    K1is = build_constraints_matrix(img1_points[0], img1_points[1], img1_points[2], proj_depth[0],
                                    proj_depth[1], proj_depth[2], img1_points[i], proj_depth[i])
    K2is = build_constraints_matrix(img2_points[0], img2_points[1], img2_points[2], proj_depth[0],
                                    proj_depth[1], proj_depth[2], img2_points[i], proj_depth[i])
    K3is = build_constraints_matrix(img3_points[0], img3_points[1], img3_points[2], points[0][2],
                                    points[1][2], points[2][2], img3_points[i], points[i][2])
    K4is = build_constraints_matrix(img4_points[0], img4_points[1], img4_points[2], points[0][2],
                                    points[1][2], points[2][2], img4_points[i], points[i][2])
    Kis = np.concatenate((K1is, K2is), axis=0) #, K3is, K4is
    _, D, V = np.linalg.svd(Kis)
    print("Point ", i, " last element in D is: ", D[-1])
    coeffs.append(V[:,-1]/V[-1,-1])
    print("Structure coefficients: ", coeffs[-1])

    # u,v,g3 = infer_point_position2(img_new_points[0], img_new_points[1], img_new_points[2],
    #                                 proj_depth[0], proj_depth[1], proj_depth[2], V[:,-1]/V[-1,-1])
    # print("u=", u, "v=", v, "d=", g3*points[0][2])

# Point  [250, 150, 250.]

ratio = 50*sqrt(299)
p = 167.71133027/ratio
q = 725.860117171/ratio
x = 488.676117511/ratio
y = 107.518620206/ratio
z = 46.210372027/ratio
theta = -p/q
mu = 1/q

c4 = -(x + theta*y)
c5 = -mu*y
c1 = (1+theta**2)*(z**2) + c4**2
c2 = theta*mu*(z**2) + c4*c5
c3 = (mu**2)*(z**2) + c5**2
test_coeffs = [c1, c2, c3, c4, c5]
print("Ground truth coefficients : ", test_coeffs)
print(img_new_points[3], proj_depth[3])
# u,v,g3 = infer_point_position(img_new_points[0], img_new_points[1], img_new_points[2],
#                                     proj_depth[0], proj_depth[1], proj_depth[2], test_coeffs)

# Point [-200, 200, 100.]
gt_coeffs_2 = [0.171886845825051, 0.108565781739248, 0.178849144634526, -0.354097691879594, -0.340908369568667]
# print(img_new_points[4], proj_depth[4])
# u,v,g3 = infer_point_position(img_new_points[0], img_new_points[1], img_new_points[2],
#                                     proj_depth[0], proj_depth[1], proj_depth[2], gt_coeffs_2)

# Point [500, 300, 250.]
gt_coeffs_3 = [0.699177960453233, 0.219728949122417, 0.093312597200622, -0.827149522328371, -0.270384359031326]
# print(img_new_points[5], proj_depth[5])
# u,v,g3 = infer_point_position(img_new_points[0], img_new_points[1], img_new_points[2],
#                                     proj_depth[0], proj_depth[1], proj_depth[2], gt_coeffs_3)

gt_coeffs = [test_coeffs, gt_coeffs_2, gt_coeffs_3]
u0 = img_new_points[0][0]
v0 = img_new_points[0][1]
u1 = img_new_points[1][0]
v1 = img_new_points[1][1]
u2 = img_new_points[2][0]
v2 = img_new_points[2][1]

M = np.array([[u1*v1*c1,           c2*(v1*u2+v2*u1)/2, v1*c4/2, u1*c4/2],
              [c2*(v1*u2+v2*u1)/2, v2*u2*c3,           v2*c5/2, u2*c5/2],
              [v1*c4/2,            v2*c5/2,            0,       1/2    ],
              [u1*c4/2,            u2*c5/2,            1/2,     0      ]])
w,v = np.linalg.eig(M)
print(w)
print(v)
# Retrieve u, v, g1, g2 and g3 in new view with Newton's method
for i in range(3, m):
    expected = np.array([img_new_points[i][0], img_new_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]])
    print("Expected u and v: ", expected)
    tol = 1e-4
    x0 = np.array([img1_points[i][0], img1_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]])
    # x0 = np.array([1, 1, .1, .1, .1])
    print("Initials u, v, g1, g2 and g3 = ", x0)
    x, n = Newton_system(F, J, [u0,v0,u1,v1,u2,v2,gt_coeffs[i-3]], x0, max_iter=1000, eps=1e-5)
    print("Nb of iterations = ", n, " to get x = ", x)
    error_norm = np.linalg.norm(expected - x, ord=2)
    # assert error_norm < tol, 'norm of error =%g' % error_norm
    print('norm of error =%g' % error_norm)


# x²+x*2*2.8359-y²-y*2*5.9852 -30.6309=0,y²+y*2*5.9852-z²-z*2*0.3176+38.8591=0, 5.9852x + xy + 2.8359y + 16.3308=0, 0.3176x + xz + 2.8359z + 0.9042=0, 0.3176y + yz + 5.9852z + 1.9323=0
import sympy

# u, v, g1, g2, g3 = sympy.symbols("u v g1 g2 g3", real=True)
g1 = proj_depth[1]/proj_depth[0]
g2 = proj_depth[2]/proj_depth[0]
g3 = proj_depth[3]/proj_depth[0]
[u0, v0] = img1_points[0]
[u1, v1] = img1_points[1]
[u2, v2] = img1_points[2]
[u, v] = img1_points[3]
a = g1*u1-u0
b = g2*u2-u0
c = g3*u-u0
l = g1*v1-v0
m = g2*v2-v0
n = g3*v-v0
r = g1 - 1
s = g2 - 1
t = g3 - 1

print(F([u0,v0,u1,v1,u2,v2,gt_coeffs[0]], [u,v,g1,g2,g3]))

# eq1 = sympy.Eq(((a)**2 - (g1*v1-v0)**2)*c1 + (2*(g1*u1-u0)*(g2*u2-u0) - 2*(g1*v1-v0)*(g2*v2-v0))*c2 +
#                 ((g2*u2-u0)**2 - (g2*v2-v0)**2)*c3 + (2*(g1*u1-u0)*(g3*u-u0) - 2*(g1*v1-v0)*(g3*v-v0))*c4 +
#                 (2*(g2*u2-u0)*(g3*u-u0) - 2*(g2*v2-v0)*(g3*v-v0))*c5 + (g3*u-u0)**2 - (g3*v-v0)**2, 0)
# eq2 = sympy.Eq((l**2 - r**2)*c1 + (2*l*m - 2*r*s)*c2 + (m**2 - s**2)*c3 + (2*l*n - 2*r*t)*c4 + (2*m*n - 2*s*t)*c5
#                 + n**2 - t**2, 0)
# eq3 = sympy.Eq(a*l*c1 + (l*b+m*a)*c2 + m*b*c3 + (l*c+n*a)*c4 + (m*c+n*b)*c5 + n*c, 0)
# eq4 = sympy.Eq(a*r*c1 + (r*b+s*a)*c2 + s*b*c3 + (r*c+t*a)*c4 + (s*c+t*b)*c5 + t*c, 0)
# eq5 = sympy.Eq(r*l*c1 + (r*m+s*l)*c2 + m*s*c3 + (r*n+t*l)*c4 + (s*n+t*m)*c5 + n*t, 0)
# sympy.solve_poly_system([eq1, eq2, eq3, eq4, eq5], u, v, g1, g2, g3)

