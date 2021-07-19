from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.twodim_base import tri
import visualization

from delaunator import Delaunator

class TextureMapping():
    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0

    def create_novel_view_image(self, new_pts, new_d_pts, pts, d_pts, rgb_cams, depth_cams):
        self.frame_height, self.frame_width, _ = np.shape(rgb_cams[0])
        new_img = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)

        triangles = self.triangulation(pts, new_pts)
        triangles = self.triangles_z_buffering(triangles, d_pts, new_d_pts)
        self.resolve_visibility()


        fig3 = visualization.plot_triangulation(new_img, new_pts, triangles)
        plt.show()

        return new_img

    def triangulation(self, pts, new_pts):
        """
        Perform Delaunay triangulation in all the input views and the virtual view using the matched points
        Inputs: - pts: List of list of the matched points for all the input views
                - new_pts: List of the reconstructed matched points in the virtual view
        Outputs:    - triangles: List of list of the vertix indexes of the triangles in each view. The last
                    element/list (triangles[-1]) corresponds to the triangulation of the virtual view
        """
        triangles = []

        for view in range(len(pts)):
            triangles.append(Delaunator(pts[view]).triangles)
        triangles.append(Delaunator(new_pts).triangles)

        return triangles

    def triangles_z_buffering(self, triangles, d_pts, new_d_pts):
        """
        Order the triangles in each views in a descending order of depth.
        The 'depth' of a triangle is computed by taking the mean of the depth of its 3 vertices.
        """
        sorted_triangles = []
        for view in range(len(triangles) - 1):
            depth = []
            for i in range(0, len(triangles[view]), 3):
                depth.append((d_pts[view][triangles[view][i]] + d_pts[view][triangles[view][i+1]] + d_pts[view][triangles[view][i+2]])/3)
                depth.append((d_pts[view][triangles[view][i]] + d_pts[view][triangles[view][i+1]] + d_pts[view][triangles[view][i+2]])/3)
                depth.append((d_pts[view][triangles[view][i]] + d_pts[view][triangles[view][i+1]] + d_pts[view][triangles[view][i+2]])/3)
            sorted_triangles.append([t for _, t in sorted(zip(depth, triangles[view]), key=lambda pair: pair[0], reverse=True)])
        depth = []
        for i in range(0, len(triangles[-1]), 3):
            depth.append((new_d_pts[triangles[-1][i]] + new_d_pts[triangles[-1][i+1]] + new_d_pts[triangles[-1][i+2]])/3)
            depth.append((new_d_pts[triangles[-1][i]] + new_d_pts[triangles[-1][i+1]] + new_d_pts[triangles[-1][i+2]])/3)
            depth.append((new_d_pts[triangles[-1][i]] + new_d_pts[triangles[-1][i+1]] + new_d_pts[triangles[-1][i+2]])/3)
        sorted_triangles.append([t for _, t in sorted(zip(depth, triangles[-1]), key=lambda pair: pair[0], reverse=True)])

        return sorted_triangles

    def resolve_visibility(self):
        pass
