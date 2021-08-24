import cv2
from matplotlib import pyplot as plt
import numpy as np
import tools
from time import time
import visualization

from corresponding_points_selector import CorrespondingPointsSelector
from parameterized_image_variety import ParameterizedImageVariety
from texture_mapping import TextureMapping
import box_pts_by_hand

FOLDER_NAME = "data/scene1_small/" #"data/artificial_data/74/" #"data/plantH/" #"data/pillows_smallB/" 
TIMESTAMP = "" #"_20210623-1903" #"_20210629-1539"

VISUALIZE = True

def load_rgbd3(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+idx+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = np.load(FOLDER_NAME+idx+".npy")

    return rgb_cam, depth_cam

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

def main(nn_match_ratio = 0.63, resid_thresh=1e-6, method='lm', cameras=[1,3,4,5,7]):
    start_t = time()
    # Load the data
    rgb_cams = []
    depth_cams = []
    cams_idx = cameras
    for idx in cams_idx:
        # if i != 2: # keep view 2 for reconstruction
        rgb_cam, depth_cam = load_rgbd(str(idx))
        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)
    frame_height, frame_width, _ = np.shape(rgb_cams[0])
    max_depth = np.max(depth_cams)
    load_t = time()

    # Extract corresponding points accross the view and the select 3 reference points
    ptSelector = CorrespondingPointsSelector()
    ptSelector.select_paramters(nn_match_ratio=nn_match_ratio, depth_neighborhood_radius=2, descriptor_ratio=0.75)
    q0, d0, q1, d1, q2, d2, pts, d_pts = ptSelector.points_selection(rgb_cams, depth_cams)
    # q0, d0, q1, d1, q2, d2, pts, d_pts = box_pts_by_hand.get_pts(depth_cams)
    select_t = time()

    if VISUALIZE:
        fig = visualization.plot_matched_features(rgb_cams, pts, q0, q1, q2)
    visu_t = time()


    # Normalize the pixel position and depth to be on a [0,1] scale for a robust optimization
    # pts_px = pts.copy()
    q0, d0, q1, d1, q2, d2, pts, d_pts = tools.normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts, max_px=frame_width, max_d=1*max_depth)
    norm_t = time()

    virtual_cam = 4
    virtual_view = cams_idx.index(virtual_cam)

    # Define and compute the PIV for the current scene to place the scene matched points in the novel view
    piv = ParameterizedImageVariety(q0, d0, q1, d1, q2, d2, pts, d_pts, virtual_view, frame_width, frame_height, max_depth, resid_thresh=resid_thresh, method=method, debug=True)
    virtual_pts, virtual_d_pts, error_res = piv.get_virtual_pts()

    # return len(virtual_pts), error_res

    pts, d_pts = piv.get_updated_pts()
    piv_t = time()

    virtual_pts, virtual_d_pts = tools.rescale_and_concatenate_points(virtual_pts, q0[virtual_view], q1[virtual_view], q2[virtual_view],
                                                                    virtual_d_pts, d0[virtual_view], d1[virtual_view], d2[virtual_view],
                                                                    depth_cams[virtual_view], max_depth, frame_width, frame_height, rescale=False)
    for view in range(len(rgb_cams)):
        pts[view], d_pts[view] = tools.rescale_and_concatenate_points(pts[view], q0[view], q1[view], q2[view],
                                                                        d_pts[view], d0[view], d1[view], d2[view],
                                                                        depth_cams[view], max_depth, frame_width, frame_height)

    print("Loading time: ", load_t - start_t)
    print("Point selection time: ", select_t - load_t)
    print("Normalization time: ", norm_t - visu_t)
    print("PIV computation and reconstruction time: ", piv_t - norm_t)
    print("Total time: ", (piv_t - visu_t) + (select_t - start_t))
    if VISUALIZE:
        fig2 = visualization.plot_point_placement_results(rgb_cams[virtual_view], virtual_pts, pts[virtual_view], frame_height, frame_width)
        # plt.show()

    # Remove GT to properly test the texture mapping
    gt_pts = pts[virtual_view].copy()
    gt_d_pts = d_pts[virtual_view].copy()
    del pts[virtual_view]
    del d_pts[virtual_view]
    del rgb_cams[virtual_view]
    del depth_cams[virtual_view]

    textmap_t = time()
    text_map = TextureMapping()
    new_img = text_map.create_novel_view_image(virtual_pts, virtual_d_pts, pts, d_pts, rgb_cams, depth_cams)
    # new_img = text_map.create_novel_view_image(gt_pts, gt_d_pts, pts, d_pts, rgb_cams, depth_cams)
    print("Texture mapping time: ", time() - textmap_t)
    fig3 = plt.figure("Texture mapping result")
    plt.imshow(new_img)
    plt.show()



if __name__ == "__main__":
    # data = {
    #     "Lowe's ratio": [],
    #     "Pixel error (mean)": [],
    #     "Depth error": [],
    #     "Nb points": [],
    #     "Residual Threshold": [],
    #     "Method":[]
    # }

    # cams = [1,3,4,5,7]
    # # nn_ratios = [.56, .57, .58, .59, .6, .61, .62, .63, .64, .65, .66, .67, .68, .69, .7, .71, .72, .74]
    # nn_ratios = [.56, .58, .6, .62, .64, .66, .68, .7, .72, .74]
    # thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # for method in ['lm', 'nm']:
    #     for resid_thresh in thresholds:
    #         for nn_ratio in nn_ratios:
    #             print("method: ", method, " / nn ratio = ", nn_ratio, " / resid_thresh = ", resid_thresh)
    #             try:
    #                 n, res = main(nn_ratio, resid_thresh, method, cams)
    #                 # for i in range(n):
    #                 #     data["Lowe's ratio"].append(nn_ratio)
    #                 #     data["Pixel error (mean)"].append(res[0][i])
    #                 #     data["Depth error"].append(res[1][i])
    #                 #     data["Nb points"].append(n)
    #                 #     data["Residual Threshold"].append(resid_thresh)
    #                 #     data["Method"].append(method)
    #                 data["Lowe's ratio"].append(nn_ratio)
    #                 data["Pixel error (mean)"].append(np.mean(res[0]))
    #                 data["Depth error"].append(np.mean(res[1]))
    #                 data["Nb points"].append(n)
    #                 data["Residual Threshold"].append(resid_thresh)
    #                 if method == 'lm':
    #                     data["Method"].append('Levenberg-Marquardt')
    #                 elif method == 'nm':
    #                     data["Method"].append('Nelder-Mead')
    #             except:
    #                 print('Error')

    # data = {
    #     "Scene": [],
    #     "Pixel error (mean)": [],
    #     "Depth error (mean)": [],
    #     "Nb points": [],
    #     "Baseline": [],
    #     "Camera configuration":[]
    # }

    # cam_configs = [[0,1,2,3,4,5,6,7,8], [0,2,4,6,8], [1,3,4,5,7], [3,4,5], [1,4,7]]
    # config_names = ['Full', 'Corners', 'Cross', 'Horizontal', 'Vertical']
    # files = ["data/scene1_small/", "data/scene1_wide/", "data/scene2_small/", "data/scene2_wide/"]
    # for i, cams in enumerate(cam_configs):
    #     for j, filename in enumerate(files):
    #         FOLDER_NAME = filename
    #         print("File: ", filename, " // Config: ", config_names[i])
    #         try:
    #             n, res = main(cameras=cams)
    #             data["Scene"].append(j//2 + 1)
    #             data["Pixel error (mean)"].append(np.mean(res[0]))
    #             data["Depth error (mean)"].append(np.mean(res[1]))
    #             data["Nb points"].append(n)
    #             if j%2 == 0:
    #                 data["Baseline"].append('Small')
    #             else:
    #                 data["Baseline"].append('Wide')
    #             data["Camera configuration"].append(config_names[i])
    #         except:
    #             print('Error')


    # import seaborn as sns
    # import pandas as pd

    # df = pd.DataFrame(data)
    # df.to_pickle("scenes_configs.pkl")

    main()