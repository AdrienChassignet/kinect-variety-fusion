import cv2
import tools
import numpy as np
import sys

class CorrespondingPointsSelector():

    def __init__(self):
        self.nn_match_ratio = 0.7
        self.nb_max_features = 15000
        self.descriptor_ratio = 0.9
        self.depth_neighborhood_radius = 0

    def select_paramters(self, nn_match_ratio=0.7, nb_max_features=15000, descriptor_ratio=0.9, depth_neighborhood_radius=0):
        """
        Change parameters of the corresponding point selector.
        nn_match_ratio: Ratio to discriminate bad matches, the lower the pickier
        nb_max_features: The maximum number of features that the algorithm will try to find
        descriptor_ratio: Ratio of the descriptor
        depth_neighborhood_radius: Radius of the neighborhood taken to retrieve depth of a feature point
        """
        self.nn_match_ratio = nn_match_ratio
        self.nb_max_features = nb_max_features
        self.descriptor_ratio = descriptor_ratio
        self.depth_neighborhood_radius = depth_neighborhood_radius

    def points_selection(self, rgb_cams, depth_cams):
        """
        Retrieve matching points among all the views and pick the 3 reference points.
        The 1st ref point Q0 is the centroid of the matched points, the 2nd point Q1 is the
        first extreme point in the lower hull and the 3rd point Q2 is the same with the upper hull.

        Inputs :    - rgb_cams: (n x rgb) List of the RGB images
                    - depth_cams: (n x depth_map) List of the corresponding depth maps
        Outputs :   - q0: List of image position of Q0 in the input views
                    - q1: List of image position of Q1 in the input views
                    - q2: List of image position of Q2 in the input views
                    - pts: List of image position of matched points in the input views
                    - d_pts: List of depth corresponding to matched points in the input views
        """

        pts = self.matched_points_extraction(rgb_cams)
        pts = self.common_points_extraction(pts)

        m = len(pts[0])
        if m < 4:
            print("Error: Too few points matched across the views for the rest of the pipeline.")
            sys.exit()

        d_pts, pts = self.depth_value_extraction(depth_cams, pts)

        q0, q1, q2, d0, d1, d2, pts, d_pts = self.select_reference_points(pts, d_pts)

        return q0, d0, q1, d1, q2, d2, pts, d_pts

    def features_extraction(self, rgb_cams):
        """
        Extract features in the input rgb frames using ORB descriptor from OpenCV
        """

        detector = cv2.ORB_create(self.nb_max_features)
        descriptor = cv2.xfeatures2d.BEBLID_create(self.descriptor_ratio)

        kp = [0 for i in range(len(rgb_cams))]
        des = [0 for i in range(len(rgb_cams))]
        for i, rgb_cam in enumerate(rgb_cams):
            kp[i] = (detector.detect(rgb_cam, None))
            kp[i], des[i] = descriptor.compute(rgb_cam, kp[i])
            des[i] = np.float32(des[i])

        return kp, des

    def common_points_extraction(self, pts):
        """
        Return only the points that are matched in every views
        """

        common_pts = [[] for i in range(len(pts))]
        for idx, pt in enumerate(pts[0]):
            if [] not in [pts[view][idx] for view in range(len(pts))]:
                for view_idx in range(len(pts)):
                    common_pts[view_idx].append(pts[view_idx][idx])

        return common_pts

    def features_matching_accross_all_pairs(self, kp, des):
        """
        Create a list of the features matched between all pairs of input frames.
        The method returns a 2D list of the features that does have a correct match
        with at least one other view. 
        Each row of the list correspond to one view and each column correspond to a feature.
        An empty value means that the feature does not have match in the view of the
        corresponding row.
        Inputs: - kp: A list for each view of the features detected
                - des: A list for each view of the local descriptor of each detected features
        Output: - matched_pts: A 2D list containing, for each view, the features that has a
                            match in at least one other view.
        """

        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matched_pts = [[[] for j in range(self.nb_max_features)] for i in range(len(kp))]
        max_idx = 0
        # Process all pairs of views
        for i in range(0, len(kp)):
            for j in range(i+1,len(kp)):
                    matches = matcher.knnMatch(des[i],des[j],k=2)
                    good_match_idx = 0
                    for idx, (m, n) in enumerate(matches):
                        if m.distance < self.nn_match_ratio * n.distance:
                            # Good matches
                            pt1 = kp[i][m.queryIdx].pt
                            pt1 = (u1, v1) = (int(round(pt1[0])), int(round(pt1[1])))
                            pt2 = kp[j][m.trainIdx].pt
                            pt2 = (int(round(pt2[0])), int(round(pt2[1])))
                            try: # Check if the first feature point has already been selected
                                pt1_idx = matched_pts[i].index(pt1)
                                # If so, match the point in the 2nd view at the corresponding index
                                matched_pts[j][pt1_idx] = pt2 # WARNING: Maybe check if there is already a point here and if it is the same?
                                max_idx -= 1
                            except ValueError:
                                try: # Else check if the second feature point has already been selected
                                    pt2_idx = matched_pts[j].index(pt2)
                                    # If so, match the point in the 1st view at the corresponding index
                                    matched_pts[i][pt2_idx] = pt1 # WARNING: Maybe check if there is already a point here and if it is the same?
                                    max_idx -= 1
                                except ValueError:
                                    matched_pts[i][max_idx + good_match_idx] = pt1
                                    matched_pts[j][max_idx + good_match_idx] = pt2
                            good_match_idx += 1
                    max_idx = max_idx + good_match_idx

        return matched_pts

    def matched_points_extraction(self, rgb_cams):
        """
        Find the feature points that are common among the views.

        Inputs:     - rgb_cams: List of the RGB views.
        Outputs:    - matched_pts: List for each view of a list of the matched points accross all the views. 
        """
        
        kp, des = self.features_extraction(rgb_cams)

        return self.features_matching_accross_all_pairs(kp, des)

    def depth_value_extraction(self, dmap_list, pts_list):
        """
            Extract valid depth of each matched points accross the views.
            The inputs and outputs are list of corresponding data of each view.
            The method extracts the mean of a 3x3 neighborhood in the depth map for each matched
            points and discard the point if no valid depth is available.

            Inputs :    - dmap_list: (n x dmap)List of the depth maps
                        (- image_list: (n x rgb_image) List of the RGB images for potential edge detection)
                        - pts_list: (n x nb_matched_points) List of matched points in each views
            Outputs :   - pts_depth : (n x nb_valid_points) List of depths of the matched valid points
                        - updated_pts : (n x nb_valid_points) List of matched valid points
        """

        updated_pts = [[] for i in range(len(pts_list))]
        pts_depth = [[] for i in range(len(pts_list))]

        for idx in range(len(pts_list[0])): # Check all matched points
            depth = np.zeros(len(pts_list))
            valid = True
            for i in range(len(pts_list)): # Check depth of current point in each view
                if pts_list[i][idx] != []:
                    (u,v) = pts_list[i][idx]
                    neighborhood = tools.get_neighborhood(u, v, self.depth_neighborhood_radius, dmap_list[i])
                    nonzero = neighborhood[np.nonzero(neighborhood)]
                    count = len(nonzero)
                    if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
                        depth[i] = sorted(nonzero)[count//2] #Take median value
                    else:
                        valid = False
                        break
            if valid: # If there is valid depth information in all views we keep the point
                for i in range(len(pts_list)):
                    pts_depth[i].append(depth[i])
                    updated_pts[i].append(pts_list[i][idx])

        return pts_depth, updated_pts

    def select_reference_points(self, pts, d_pts):
        """
        Select the 3 reference points needed for the parameterized image variety.
        Inputs: - pts: list for each view of the corresponding points across the views
                - d_pts: list for each view of the depth of the corresponding points across the views
        Outputs:    - q0, q1, q2: The 3 selected reference point image position in each view
                    - d0, d1, d2: The corresponding depth in each view of the 3 ref points
                    - pts: list for each view of the corresponding points across the views without the 3 ref points
                    - d_pts: list for each view of the depth of the corresponding points across the views without the ref points
        """

        # Use the first view to define the reference points
        ref_view = 0 # len(pts)//2

        centroid = (sum([pt[0] for pt in pts[ref_view]])/len(pts[ref_view]), sum([pt[1] for pt in pts[ref_view]])/len(pts[ref_view]))
        deltas = np.array(pts[ref_view]) - centroid
        centroid_idx = pts[ref_view].index(pts[ref_view][np.argmin(np.einsum('ij,ij->i', deltas, deltas))])

        up_hull, low_hull = tools.convex_hull(pts[ref_view], split=True)
        # Select first extreme point in a counter-clockwise order in both lower and upper hull
        q1_idx = pts[ref_view].index(low_hull[-1])
        q2_idx = pts[ref_view].index(up_hull[-1])
        # TODO: noncollinearity check!

        q0 = []
        d0 = []
        q1 = []
        d1 = []
        q2 = []
        d2 = []

        for i in range(len(pts)):
            q0.append(pts[i][centroid_idx])
            d0.append(d_pts[i][centroid_idx])
            q1.append(pts[i][q1_idx])
            d1.append(d_pts[i][q1_idx])
            q2.append(pts[i][q2_idx])
            d2.append(d_pts[i][q2_idx])
            idxs = [centroid_idx, q1_idx, q2_idx]
            idxs.sort(reverse=True)
            for idx in idxs:
                del pts[i][idx]
                del d_pts[i][idx]

        return q0, q1, q2, d0, d1, d2, pts, d_pts