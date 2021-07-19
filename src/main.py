import cv2
from matplotlib import pyplot as plt
import numpy as np
import tools
from time import time
import visualization

from corresponding_points_selector import CorrespondingPointsSelector
from parameterized_image_variety import ParameterizedImageVariety
from texture_mapping import TextureMapping

FOLDER_NAME = "data/artificial_data/74/"
TIMESTAMP = "_20210629-1539"

VISUALIZE = True

def load_rgbd2(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+"photo/"+idx+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = cv2.imread(FOLDER_NAME+"depth/"+idx+".png", cv2.IMREAD_UNCHANGED)

    return rgb_cam, depth_cam

def load_rgbd(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+"rgb_"+idx+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = np.load(FOLDER_NAME+"depth_"+idx+TIMESTAMP+".npy")

    return rgb_cam, depth_cam

def main():
    start_t = time()
    # Load the data
    rgb_cams = []
    depth_cams = []
    cams_idx = range(600,800,25)
    for idx in cams_idx:
        # if i != 2: # keep view 2 for reconstruction
        rgb_cam, depth_cam = load_rgbd2(str(idx))
        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)
    frame_height, frame_width, _ = np.shape(rgb_cams[0])
    max_depth = np.max(depth_cams)
    load_t = time()

    # Extract corresponding points accross the view and the select 3 reference points
    ptSelector = CorrespondingPointsSelector()
    q0, d0, q1, d1, q2, d2, pts, d_pts = ptSelector.points_selection(rgb_cams, depth_cams)
    select_t = time()

    if VISUALIZE:
        fig = visualization.plot_matched_features(rgb_cams, pts, q0, q1, q2)
    visu_t = time()


    # Normalize the pixel position and depth to be on a [0,1] scale for a robust optimization
    pts_px = pts.copy()
    q0, d0, q1, d1, q2, d2, pts, d_pts = tools.normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts, max_px=frame_width, max_d=max_depth)
    norm_t = time()

    virtual_cam = 675
    virtual_view = cams_idx.index(virtual_cam)

    # Define and compute the PIV for the current scene to place the scene matched points in the novel view
    piv = ParameterizedImageVariety(q0, d0, q1, d1, q2, d2, pts, d_pts, virtual_view, frame_width, frame_height, max_depth, debug=False)
    virtual_pts, virtual_d_pts = piv.get_virtual_pts()
    piv_t = time()

    virtual_pts = tools.concatenate_points(virtual_pts, q0[virtual_view], q1[virtual_view], q2[virtual_view], frame_width)
    virtual_d_pts = tools.concatenate_depth_points(virtual_d_pts, d0[virtual_view], d1[virtual_view], d2[virtual_view], max_depth)
    for view in range(len(rgb_cams)):
        pts[view] = tools.concatenate_points(pts[view], q0[view], q1[view], q2[view], frame_width)
        d_pts[view] = tools.concatenate_depth_points(d_pts[view], d0[view], d1[view], d2[view], max_depth)

    print("Loading time: ", load_t - start_t)
    print("Point selection time: ", select_t - load_t)
    print("Normalization time: ", norm_t - visu_t)
    print("PIV computation and reconstruction time: ", piv_t - norm_t)
    print("Total time: ", (piv_t - visu_t) + (select_t - start_t))
    # Visualize the results
    if VISUALIZE:
        fig2 = visualization.plot_point_placement_results(rgb_cams[virtual_view], virtual_pts, pts_px[virtual_view], frame_height, frame_width)
        plt.show()

    text_map = TextureMapping()
    new_img = text_map.create_novel_view_image(virtual_pts, virtual_d_pts, pts, d_pts, rgb_cams, depth_cams)

if __name__ == "__main__":
    main()