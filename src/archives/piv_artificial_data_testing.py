from functools import partial
import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from helpers import *
# from visualization import plot_F_squared_around_x
from scipy.optimize import fsolve, minimize, Bounds, least_squares
from time import time
from fp_piv import build_constraints_matrix_norm

rng = np.random.default_rng(0)

VISUALIZE = False
IMAGE_SIZE = 200
MAX_DEPTH = 400


def project_points(input_points, rot, t, K):
    img_points, _ = cv2.projectPoints(input_points, rot, t, K, None)
    img = np.ones([IMAGE_SIZE,IMAGE_SIZE,3], dtype=np.uint8)*220 
    for point in img_points:
        img[int(point[0][1]), int(point[0][0]), :] = [0,0,0]
    img[int(img_points[0][0][1]), int(img_points[0][0][0]), :] = [30,30,255]
    img[int(img_points[1][0][1]), int(img_points[1][0][0]), :] = [50,220,50]
    img[int(img_points[2][0][1]), int(img_points[2][0][0]), :] = [255,50,30]

    # WARNING: Sub-pixel projection avoided to be closer to reality!!!
    return [np.around(point)/IMAGE_SIZE for pt in img_points for point in pt], img

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

def F(x, cst):
    """
    Define the system of equations in factorized form.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    # Here only coeffs from the cst list is changing in each call of this method => it could be optimized by avoiding recomputing for the fixed parameters
    [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2] = cst
    # [u, v, d3, _, _] = x
    [u,v,d3] = x
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

def call_F(F, cst, x):
    return F(x, cst)

def sum_of_squares_of_F(x, cst):
    """
    Return the sum of squares of the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
    x = [u, v, d3, 1, 1]
    """
    return 0.5*np.sum(F(x,cst)**2)

def main():
    # Define the artificial data
    points = np.array([[0, 0, 0.],
                [-150, -150, 300.],
                [100, 300, 250.],
                [250, -250, 200.],
                [50, 150, 250.],
                [-100, 100, 120.],
                [-150, -200, 150],
                [100, 50, 150.]])

    tmp = [i for i in range(-300, 301) if abs(i)>10]
    new_points = rng.choice(tmp, (200,3))
    new_points[:,-1] = abs(new_points[:,-1])+100
    points = np.vstack([points, new_points])
    
    if VISUALIZE:
        visualize_point_cloud(points)

    NOISE_MAG = MAX_DEPTH*0.05

    points = points[1:]
    d_noise = (rng.random(len(points)) - 0.5) * NOISE_MAG
    proj_depth = (points[:,-1] + d_noise)/MAX_DEPTH

    d_noise2 = (rng.random(len(points)) - 0.5) * NOISE_MAG
    points2 = points - np.array([100, 100, 30.])
    proj_depth2 = (points2[:,-1] + d_noise2)/MAX_DEPTH

    d_noise3 = (rng.random(len(points)) - 0.5) * NOISE_MAG
    points3 = points - np.array([0, 100, -20.])
    proj_depth3 = (points3[:,-1] + d_noise3)/MAX_DEPTH

    d_noise4 = (rng.random(len(points)) - 0.5) * NOISE_MAG
    points4 = points - np.array([100, 0, 20.])
    proj_depth4 = (points4[:,-1] + d_noise4)/MAX_DEPTH

    # Create the reference image views of the artificial data
    f = 15.
    K1 = np.array([[f, 0., IMAGE_SIZE/2],
                    [0., f, IMAGE_SIZE/2],
                    [0., 0., 1.]])

    t1 = np.array([0., 0., 0.])
    img1_points, img1 = project_points(points, np.eye(3), t1, K1)
    print("Points image position in view 1: ", img1_points)
    img2_points, img2 = project_points(points2, np.eye(3), t1, K1)
    print("Points image position in view 2: ", img2_points)
    img3_points, img3 = project_points(points3, np.eye(3), t1, K1)
    print("Points image position in view 3: ", img3_points)
    img4_points, img4 = project_points(points4, np.eye(3), t1, K1)
    print("Points image position in view 4: ", img4_points)

    # Create the virtual view we want to recreate from the reference ones
    t1v = np.array([50., 50., 10.])
    proj_depth_v = (points[:,-1] + t1v[-1])/MAX_DEPTH
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
    novel_view = np.ones([IMAGE_SIZE,IMAGE_SIZE,3], dtype=np.uint8) * 220
    
    resids = []
    error_img_pos = np.empty(0)
    error_depth = np.empty(0)
    timing_opt = []
    cnt_bad = 0
    cnt_high_res = 0
    for i in range(3, m):
        time_start = time()

        K1is = build_constraints_matrix_norm(img1_points[0], img1_points[1], img1_points[2], proj_depth[0],
                                        proj_depth[1], proj_depth[2], img1_points[i], proj_depth[i])
        K2is = build_constraints_matrix_norm(img2_points[0], img2_points[1], img2_points[2], proj_depth2[0],
                                        proj_depth2[1], proj_depth2[2], img2_points[i], proj_depth2[i])
        K3is = build_constraints_matrix_norm(img3_points[0], img3_points[1], img3_points[2], proj_depth3[0],
                                        proj_depth3[1], proj_depth3[2], img3_points[i], proj_depth3[i])
        K4is = build_constraints_matrix_norm(img4_points[0], img4_points[1], img4_points[2], proj_depth4[0],
                                        proj_depth4[1], proj_depth4[2], img4_points[i], proj_depth4[i])
        Kis = np.concatenate((K1is, K2is, K3is, K4is), axis=0) 
        _, D, V = np.linalg.svd(Kis)
        # print("Residuals D: ", D)
        # print("Residuals D: ", D)
        coeffs.append(V[-1]/V[-1,-1])
        # print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])
        # print("Residual error: ", np.matmul(Kis,coeffs[-1]))
        resids.append(np.sum(np.matmul(Kis, coeffs[-1])**2))
        # x1 = [img1_points[i][0], img1_points[i][1], proj_depth[1]/proj_depth[0], proj_depth[2]/proj_depth[0], proj_depth[i]/proj_depth[0]]
        # cst = [img1_points[0][0], img1_points[0][1], img1_points[1][0], img1_points[1][1], img1_points[2][0], img1_points[2][1], coeffs[-1]]
        # print("Equations evaluated with found coefficients: ", F(x1, cst))

        # Estimate the position of the current processed point in the new virtual view
        qv = (uv, vv) = img_new_points[i]*IMAGE_SIZE
        expected = np.array([uv, vv, proj_depth_v[i]*MAX_DEPTH])
        
        # Start the search from the position in reference view
        x0 = np.array([img1_points[i][0], img1_points[i][1], proj_depth[i]])
        # x0 = np.array([50, 50, .1, .1, .1])
        # x0 = np.array([uv/IMAGE_SIZE, vv/IMAGE_SIZE, proj_depth_v[i], 1, 1])

        cst = [u0v,v0v,u1v,v1v,u2v,v2v,coeffs[-1], proj_depth_v[0], proj_depth_v[1], proj_depth_v[2]]

        # x = fsolve(F_3var, x0, args=cst)
        # print("Time: ", time()-time_start, "Solution with fsolve: ", x)
        # error_norm = np.linalg.norm(expected - x, ord=2)
        # # assert error_norm < tol, 'norm of error =%g' % error_norm
        # print('norm of error =%g' % error_norm)
        # print("F(x) = ", F_3var(x, cst))

        time_start = time()
        # Test LS to solve the system
        # Define bounds of the solution space TODO: use max depth value of inputs with security margin
        b = [[0,0,0], [1,1,2]]
        # Convert bounds into constraints
        cons = []
        for idx in range(len(b[0])):
            lower = b[0][idx]
            upper = b[1][idx]
            l = {'type':'ineq', 'fun': lambda x, lb=lower, i=idx: x[i] - lb}
            u = {'type':'ineq', 'fun': lambda x, ub=upper, i=idx: ub - x[i]}
            cons.append(l)
            cons.append(u)
        bnds = Bounds(b[0], b[1], False)

        t_opt_start = time()
        # partial_F = partial(call_F, F, cst)
        # res = least_squares(partial_F, x0, method='lm')
        res = minimize(sum_of_squares_of_F, x0, cst, method='Nelder-mead', bounds=bnds, constraints=cons)
        timing_opt.append((time()- t_opt_start)*1000)

        u = res["x"][0]*IMAGE_SIZE 
        v = res["x"][1]*IMAGE_SIZE
        d = res["x"][2]*MAX_DEPTH
        if round(v) in range(0,IMAGE_SIZE) and round(u) in range(0,IMAGE_SIZE):
            novel_view[round(v), round(u), :] = [0,0,0]
        error_img_pos = np.append(error_img_pos, np.linalg.norm(np.array([uv,vv]) - np.array([u,v]), ord=2))
        error_depth = np.append(error_depth, abs(proj_depth_v[i]*MAX_DEPTH - d))     
        if error_img_pos[-1] > 3.5:
            print('--------------------------------------------------')
            print("Processing the point ", points[i], " idx ", i)
            print('--------------------------------------------------')
            print("LS => time: ", time()-time_start, "\nx=", res["x"], res["fun"])
            print("Reconstructed: ", u, v, d)
            print("\nExpected u and v in the new view: ", expected)
            print("Expected normed solution in the new view: ", uv/IMAGE_SIZE, vv/IMAGE_SIZE, proj_depth_v[i])
            print("Initialization [u, v, d] = ", x0)
            print("Residual error sum of squares: ", np.sum(np.matmul(Kis, coeffs[-1])**2))
            print("Error img and depth: ", error_img_pos[-1], error_depth[-1])
            print("Bad reconstruction")
            cnt_bad += 1
        # if res["fun"] > 1e-7:
        #     cnt_high_res += 1

        # fig = plot_F_squared_around_x(res["x"], cst)
        # plt.show()
    print("Error in pixels for point placement: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_img_pos),
            np.max(error_img_pos), np.min(error_img_pos), np.std(error_img_pos)))
    print("Error in mm for corresponding depth: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_depth),
            np.max(error_depth), np.min(error_depth), np.std(error_depth)))
    print("Residual of coeffs estimation: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(resids),
            np.max(resids), np.min(resids), np.std(resids)))
    print("Time optimizing in ms: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(timing_opt),
            np.max(timing_opt), np.min(timing_opt), np.std(timing_opt)))
    print("Nb of perfect reconstruction: ", m - cnt_bad - 3)

    fig = plt.figure("Views")
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img1)
    ax.set_title('View 1')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(img2)
    ax.set_title('View 2')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(img3)
    ax.set_title('View 3')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(img4)
    ax.set_title('View 4')

    fig = plt.figure("Results")
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_new)
    ax.set_title("Ground truth")
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(novel_view)
    ax.set_title("Novel view")
    plt.show()

if __name__ == "__main__":
    main()