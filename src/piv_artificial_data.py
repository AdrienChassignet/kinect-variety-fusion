import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from helpers import *
from scipy.optimize import fsolve, minimize
from time import time

VISUALIZE = False


def project_points(input_points, rot, t, K):
    img_points, _ = cv2.projectPoints(input_points, rot, t, K, None)
    img = np.zeros([100,100,3], dtype=np.uint8) 
    for point in img_points:
        img[int(point[0][1]), int(point[0][0]), :] = [255,255,255]
    img[int(img_points[0][0][1]), int(img_points[0][0][0]), :] = [0,0,255]
    img[int(img_points[1][0][1]), int(img_points[1][0][0]), :] = [0,255,0]
    img[int(img_points[2][0][1]), int(img_points[2][0][0]), :] = [255,100,0]

    return [point for pt in img_points for point in pt], img

def visualize_point_cloud(points):
    pc1 = xyz2pcd(points)
    pc1.paint_uniform_color([0., 0., 0.])
    pc1.colors[1] = [1,0,0]
    pc1.colors[2] = [0,1,0]
    pc1.colors[3] = [0,0,1]

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pc1)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.8])
    viewer.run()
    viewer.destroy_window()

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

def main():
    # Define the artificial data
    points = np.array([[0, 0, 0.],
                [-150, -150, 300.],
                [400, 500, 150.],
                [500, -500, 200.],
                [250, 150, 250.],
                [-200, 200, 100.],
                [500, 300, 250.]])
    
    if VISUALIZE:
        visualize_point_cloud(points)

    points = points[1:]
    proj_depth = points[:,-1]

    # Create the reference image views of the artificial data
    K1 = np.array([[10., 0., 50.],
                    [0., 10., 50.],
                    [0., 0., 1.]])

    t1 = np.zeros(3)
    img1_points, img1 = project_points(points, np.eye(3), t1, K1)
    print("Points image position in view 1: ", img1_points)

    t12 = t1 + np.array([10., 0., 0.])
    img2_points, img2 = project_points(points, np.eye(3), t12, K1)
    print("Points image position in view 2: ", img2_points)

    # Create the virtual view we want to recreate from the reference ones
    t1v = np.array([5., 0., 0.])
    img_new_points, img_new = project_points(points, np.eye(3), t1v, K1)
    q0v = (u0v, v0v) = img_new_points[0]
    q1v = (u1v, v1v) = img_new_points[1]
    q2v = (u2v, v2v) = img_new_points[2]


    # Estimate the structure coefficients for each 3D points seen in the input views
    m = points.shape[0]
    coeffs = []
    gt_coeffs = [[0.29082426127477906, 0.07868092804751571, 0.025994223505792193, -0.5364839559458426, -0.14812581331103838],
                [0.171886845825051, 0.108565781739248, 0.178849144634526, -0.354097691879594, -0.340908369568667],
                [0.699177960453233, 0.219728949122417, 0.093312597200622, -0.827149522328371, -0.270384359031326]]
    novel_view = np.zeros([100,100,3], dtype=np.uint8) 
    
    for i in range(3, m):
        time_start = time()
        print('--------------------------------------------------')
        print("Processing the point ", points[i])
        print('--------------------------------------------------')

        K1is = build_constraints_matrix(img1_points[0], img1_points[1], img1_points[2], proj_depth[0],
                                        proj_depth[1], proj_depth[2], img1_points[i], proj_depth[i])
        K2is = build_constraints_matrix(img2_points[0], img2_points[1], img2_points[2], proj_depth[0],
                                        proj_depth[1], proj_depth[2], img2_points[i], proj_depth[i])
        Kis = np.concatenate((K1is, K2is), axis=0) 
        _, D, V = np.linalg.svd(Kis)
        # print("Residuals D: ", D)
        coeffs.append(V[-1]/V[-1,-1])
        print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])
        # x1 = [img1_points[i][0], img1_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]]
        # cst = [img1_points[0][0], img1_points[0][1], img1_points[1][0], img1_points[1][1], img1_points[2][0], img1_points[2][1], coeffs[-1]]
        # print("Equations evaluated with found coefficients: ", F(x1, cst))

        # Estimate the position of the current processed point in the new virtual view
        qv = (uv, vv) = img_new_points[i]
        expected = np.array([uv, vv, proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]])
        # print("\nExpected u and v in the new view: ", expected)
        
        # Start the search from the position in reference view
        x0 = np.array([img1_points[i][0], img1_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]])
        # x0 = np.array([50, 50, .1, .1, .1])
        # print("Initialization [u, v, g1, g2, g3] = ", x0)

        cst = [u0v,v0v,u1v,v1v,u2v,v2v,coeffs[-1]]
        x = fsolve(F, x0, args=cst)
        print("Time: ", time()-time_start, "Solution with fsolve: ", x)
        error_norm = np.linalg.norm(expected - x, ord=2)
        # assert error_norm < tol, 'norm of error =%g' % error_norm
        print('norm of error =%g' % error_norm)
        # print("F(x) = ", F(x, cst))

        novel_view[round(x[1]), round(x[0]), :] = [255,255,255]

        time_start = time()
        # Test LS to solve the system
        res = minimize(sum_of_squares_of_F, x0, cst)
        print("LS => time: ", time()-time_start, "\nx=", res["x"], res["fun"])
        print('--------------------------------------------------')

        # #Comparison with GT coefficients computed by hand
        # print("Using ground truth coefficients...")
        # x1 = [img1_points[i][0], img1_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]]
        # cst = [img1_points[0][0], img1_points[0][1], img1_points[1][0], img1_points[1][1], img1_points[2][0], img1_points[2][1], gt_coeffs[i-3]]
        # print("Equations evaluated with ground truth coefficients: ", F(x1, cst))
        # print("Expected u and v in the new view: ", expected)
        # cst_gt = [u0v,v0v,u1v,v1v,u2v,v2v,gt_coeffs[i-3]]
        # x = fsolve(F, x0, args=cst_gt)
        # print("Solution with fsolve: ", x)
        # error_norm = np.linalg.norm(expected - x, ord=2)
        # # assert error_norm < tol, 'norm of error =%g' % error_norm
        # print('norm of error =%g' % error_norm)
        # print("F(x) = ", F(x, cst_gt))

        # # Test LS to solve the system
        # res = minimize(sum_of_squares_of_F, x0, cst_gt)
        # print("\nLS minimization => x=", res["x"], "with residual f: ", res["fun"])
        # print('\n')

    cv2.imshow("Ground truth", img_new)
    cv2.imshow("Novel view", novel_view)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()