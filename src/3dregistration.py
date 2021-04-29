import numpy as np
import cv2
from matplotlib import pyplot as plt
import teaserpp_python
import open3d as o3d

cam_intrinsics0 = np.array([[504.21252441, 0.          , 319.29864502],
                            [0.          , 504.11572266, 320.29678345],
                            [0.          , 0.          , 1.          ]])
cam_intrinsics1 = np.array([[612.35211182, 0.          , 639.85754395],
                            [0.          , 612.12872314, 363.73092651],
                            [0.          , 0.          , 1.          ]])

def draw_matches(matches, img1, kp1, img2, kp2, window_name="matches"):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    out_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 1500,450)
    # cv2.imshow(window_name, out_matches)
    plt.figure(window_name)
    plt.imshow(out_matches)


# Load data
rgb_cam1 = cv2.imread("data/rgb_20210421-155829.jpg", cv2.IMREAD_COLOR)
rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
depth_cam1 = np.load("data/depth_20210421-155829.npy")
rgb_cam2 = cv2.imread("data/rgb_20210421-155835.jpg", cv2.IMREAD_COLOR)
rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
depth_cam2 = np.load("data/depth_20210421-155835.npy")

# Features extraction
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=200)
# Find keypoints and descriptors directly
kp1, des1 = surf.detectAndCompute(rgb_cam1, None)
kp2, des2 = surf.detectAndCompute(rgb_cam2, None)

# Feature matching
# # Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = matcher.knnMatch(des1,des2,k=2)

# ratio test as per Lowe's paper
good_matches = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good_matches.append(m)

# draw_matches(matches, rgb_cam1, kp1, rgb_cam2, kp2)
# draw_matches(matches, np.uint8((depth_cam1/5000)*255), kp1, np.uint8((depth_cam2/5000)*255), kp2, window_name="Matches on depth images")
print(len(good_matches), len(matches))

# cv2.imshow("depth1", depth_cam1/5000)
# cv2.imshow("depth2", depth_cam2/5000)
plt.show()
# cv2.waitKey()

# Extract matched points depth values
dmap1 = np.array([[],[],[]])
dmap2 = np.array([[],[],[]])
for m in good_matches:
    (x1, y1) = kp1[m.queryIdx].pt
    (x2, y2) = kp2[m.trainIdx].pt
    # Get correct depth value with mean/mode
    neighbors = depth_cam1[int(y1)-1:int(y1)+2, int(x1)-1:int(x1)+2]
    if (neighbors != 0).any():
        dmap1 = np.append(dmap1, ([x1], [y1], [neighbors[neighbors!=0].mean()]), axis=1)

    neighbors = depth_cam2[int(y2)-1:int(y2)+2, int(x2)-1:int(x2)+2]
    if (neighbors != 0).any():
        dmap2 = np.append(dmap2, ([x2], [y2], [neighbors[neighbors!=0].mean()]), axis=1)

# Visualize the depth maps as points clouds and use TEASER++ for the registration

pc1 = o3d.geometry.PointCloud()
pc2 = o3d.geometry.PointCloud()
# pc1.points = o3d.utility.Vector2iVector(depth_cam1)
# pc1.create_from_depth_image(depth_cam1, o3d.camera.PinholeCameraIntrinsic)
pc1.points = o3d.utility.Vector3dVector(np.transpose(dmap1))
pc2.points = o3d.utility.Vector3dVector(np.transpose(dmap2))
pc1.paint_uniform_color([1, 0.706, 0])
pc2.paint_uniform_color([0, 0.651, 0.929])
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc1, 1.0)

rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam1), o3d.geometry.Image(depth_cam1), convert_rgb_to_intensity = False)
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, o3d.camera.PinholeCameraIntrinsic(1280,720,612.35,612.13,639.86,363.73))
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam2), o3d.geometry.Image(depth_cam2), convert_rgb_to_intensity = False)
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(1280,720,612.35,612.13,639.86,363.73))
# o3d.camera.PinholeCameraIntrinsic(1280,720,612.35,612.13,639.86,363.73)
# pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_cam1), o3d.camera.PinholeCameraIntrinsic(1280,720,504.21,504.11,319.30,320.30))

# pcd.paint_uniform_color([1, 0.706, 0])
# pcd2.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([pc1, pc2])


# Populate the parameters
solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 1
solver_params.noise_bound = 0.01
solver_params.estimate_scaling = True
solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 10000
solver_params.rotation_cost_threshold = 1e-16
print("TEASER++ Parameters are:", solver_params)
teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

solver = teaserpp_python.RobustRegistrationSolver(solver_params)
solver.solve(dmap1, dmap2)

solution = solver.getSolution()

# Print the solution
print("Solution is:", solution)