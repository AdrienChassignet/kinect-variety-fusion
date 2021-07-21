from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from tools import ProjectedPixel
import visualization

from delaunator import Delaunator

class TextureMapping():
    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0
        self.max_depth = 0

    def create_novel_view_image(self, new_pts, new_d_pts, pts, d_pts, rgb_cams, depth_cams):
        self.frame_height, self.frame_width, _ = np.shape(rgb_cams[0])
        self.max_depth = np.max(depth_cams)
        new_img = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)

        triangles = self.triangulation(pts, new_pts)
        ref_triangles = self.get_reference_triangles(pts, new_pts)
        sorting_idx = self.triangles_z_buffering(triangles, new_d_pts)
        
        self.find_pixel_correspondences(triangles, ref_triangles, sorting_idx)

        fig3 = visualization.plot_triangulation(new_img, new_pts, triangles[-1])
        plt.show()

        return new_img

    def triangulation(self, pts, new_pts):
        """
        Perform Delaunay triangulation in all the input views and the virtual view using the matched points
        Inputs: - pts: List of list of the matched points for all the input views
                - new_pts: List of the reconstructed matched points in the virtual view
        Outputs:    - triangles: List of Scipy.spatial.Delaunay objects
                    (list of the vertix indexes of the triangles in each view). The last
                    element/list (triangles[-1]) corresponds to the triangulation of the virtual view
        """
        triangles = []
        for view in range(len(pts)):
            # triangles.append(Delaunator(pts[view]).triangles)
            triangles.append(Delaunay(pts[view]))
        #triangles.append(Delaunator(new_pts+corners).triangles)
        triangles.append(Delaunay(new_pts))

        return triangles

    def triangles_z_buffering(self, triangles, new_d_pts):
        """
        Order the triangles in the novel view in a descending order of depth.
        The 'depth' of a triangle is computed by taking the mean of the depth of its 3 vertices.
        Expected triangles input to a Scipy.spatial.Delaunay object.
        Return the z-buffer indexes of triangles in the triangulation of the novel view.
        """
        # sorted_triangles_idx = []
        # for view in range(len(triangles) - 1):
        #     depth = np.zeros(len(triangles[view].simplices))
        #     for i, tri in enumerate(triangles[view].simplices):
        #         depth[i] = (d_pts[view][tri[0]] + d_pts[view][tri[0]] + d_pts[view][tri[0]])/3
        #     # sorted_triangles.append([t for _, t in sorted(zip(depth, triangles[view]), key=lambda pair: pair[0], reverse=True)])
        #     sorted_triangles_idx.append(np.argsort(-depth))

        # NOTE: We assume that the z-buffering is the same for all the frames. In other terms, we make the assumption
        #       that the depth of the corresponding triangles does not vary significantly between the views.
        #       This assumption is not really correct but can be accepted in our scenario with no huge change in 
        #       the z-axis or rotations between the views.
        # NOTE bis: Now the texture mapping algorithm does not need to order input views triangles.

        depth = np.array([(new_d_pts[tri[0]] + new_d_pts[tri[1]] + new_d_pts[tri[2]])/3 for tri in triangles[-1].simplices])
        # NOTE: Might be quicker for the sorting operation if we get an int here for the depth to be compared

        return np.argsort(-depth)

    def get_reference_triangles(self, pts, new_pts):
        """
        Get the triangle formed by the 3 reference points Q0, Q1, Q2 in each views.
        Implementation details: the points passed to this class must have in the indexes
        [-7, -6, -5] the position of q0, q1 and q2 in the view.
        Return a list for each view of Scipy.spatial.Delaunay triangluation. The last element is the virtual view.
        """
        ref_triangles = []
        for view in range(len(pts)):
            # triangles.append(Delaunator(pts[view]).triangles)
            ref_triangles.append(Delaunay(pts[view][-7:-4]))
        #triangles.append(Delaunator(new_pts+corners).triangles)
        ref_triangles.append(Delaunay(new_pts[-7:-4]))

        return ref_triangles

    def find_pixel_correspondences(self, triangles, ref_triangles, sorting_order):
        # Get for each pixels in the novel view, the list of the pixels from the input views that are correspondent

        projected_pixels = np.empty((self.frame_height, self.frame_width), dtype=ProjectedPixel)
        tri_nv = triangles[-1]
        # Go through all the triangles of the novel view in a descending depth order
        for t_idx in sorting_order:
            t_v = tri_nv.simplices[t_idx]
            # A_v = tri_nv.points[t_v[0]]
            # B_v = tri_nv.points[t_v[1]]
            # C_v = tri_nv.points[t_v[2]]
            vertices_v = np.array([tri_nv.points[t_v[0]], tri_nv.points[t_v[1]], tri_nv.points[t_v[2]]])
            for view in range(len(triangles) - 1):
                # Get corresponding triangle in processed input view
                t_i = triangles[view].simplices[np.where((triangles[view].simplices == t_v).all(axis=1))]
                if t_i.any(): 
                    # Pick the correct orientation for epipolar lines 
                    
                    t_i = t_i[0]
                    # A_i = triangles[view].points[t_i[0]]
                    # B_i = triangles[view].points[t_i[1]]
                    # C_i = triangles[view].points[t_i[2]]
                    vertices_i = np.array([triangles[view].points[t_i[0]], triangles[view].points[t_i[1]], triangles[view].points[t_i[2]]])
                    # Go trough all the pixel inside this input triangle
                    min_u = int(vertices_i[:,0].min())
                    max_u = int(vertices_i[:,0].max())
                    min_v = int(vertices_i[:,1].min())
                    max_v = int(vertices_i[:,1].max())
                    for u in range(min_u, max_u+1):
                        for v in range(min_v, max_v+1):
                            # Compute barycentric coordinates
                            b = triangles[view].transform[t_idx,:2].dot(np.transpose((u,v) - triangles[view].transform[t_idx,2]))
                            b_coord = [b[0], b[1], 1 - b.sum(axis=0)]
                            if all(n>=0 for n in b_coord): # The pixel (u,v) is within the currently processed triangle
                                # Compute position of the projection of the pixel (u,v) from the input image to the novel one
                                proj_px = tuple(np.matmul(vertices_v.transpose(), b_coord).round().astype('int'))
                                # Compute the 'depth' value (aka pr_bar)

                                # NOTE: use the ref_triangles[-1] to get barycentric coordinates of r

                                r_b = tri_nv.transform[t_idx,:2].dot(np.transpose(proj_px - tri_nv.transform[t_idx,2]))
                                r_b_coord = [r_b[0], r_b[1], 1 - r_b.sum(axis=0)]
                                # Get the position of point r in ref view (r is the intersection between the triangle and
                                # the ray passing through the mapped point)
                                px_back_proj = tuple(np.matmul(vertices_i.transpose(), r_b_coord).round().astype('int'))

                                #NOTE: This is the same point...
                                # What we want to do is to take r as the intersection between the plane spaned by Q0, Q1 and Q2 with
                                # the ray passing through the mapped point. This way r is the point position of another datapoint
                                # than the one currently projected into the novel view.

                                depth = 0
                                # Map the point if there is none at the new position or if the one previously mapped is
                                # occluded by the current point
                                if (projected_pixels[proj_px[::-1]] == None) or (projected_pixels[proj_px[::-1]].depth > depth):
                                    projected_pixels[proj_px[::-1]] = ProjectedPixel(proj_px, view, depth)

        return projected_pixels

                                
                                



    
