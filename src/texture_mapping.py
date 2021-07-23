from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import tools
import visualization

from delaunator import Delaunator

class ProjectedPixel():
    def __init__(self, pixel, view, depth):
        self.pixel = pixel
        self.view = view
        self.depth = depth

class TextureMapping():
    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0
        self.max_depth = 0

    def create_novel_view_image(self, new_pts, new_d_pts, pts, d_pts, rgb_cams, depth_cams):
        self.frame_height, self.frame_width, _ = np.shape(rgb_cams[0])
        self.max_depth = np.max(depth_cams)

        triangles = self.triangulation(pts, new_pts)
        ref_triangles = self.get_reference_triangles(pts, new_pts)
        sorting_idx = self.triangles_z_buffering(triangles[-1], new_d_pts)
        
        # projected_pixels = self.find_pixel_correspondences(triangles, ref_triangles, sorting_idx)
        projected_pixels = self.find_pixel_correspondences_bis(triangles, ref_triangles, d_pts, new_d_pts)

        # triangles = self.full_triangulation(pts, new_pts)
        # projected_pixels = self.find_pixel_correspondences_full_triangles(triangles, pts, new_pts, d_pts, new_d_pts)

        new_img = self.map_inputs_to_novel_view(rgb_cams, projected_pixels)

        # fig3 = visualization.plot_triangulation(new_img, new_pts, triangles[-1])

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

    def full_triangulation(self, pts, new_pts):
        triangles = []
        for view in range(len(pts) + 1):
            tri_list = []
            if view == len(pts):
                n = len(new_pts)
                pts_list = new_pts
            else:
                n = len(pts[view])
                pts_list = pts[view]
            for i in range(n - 2):
                for j in range(i+1, n-1):
                    for k in range(j+1, n):
                        if not tools.collinear(pts_list[i], pts_list[j], pts_list[k]):
                            tri_list.append((i,j,k))
            triangles.append(tri_list)

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

        if isinstance(triangles, Delaunay):
            depth = np.array([(new_d_pts[tri[0]] + new_d_pts[tri[1]] + new_d_pts[tri[2]])/3 for tri in triangles.simplices])
        elif isinstance(triangles, list):
            depth = np.array([(new_d_pts[tri[0]] + new_d_pts[tri[1]] + new_d_pts[tri[2]])/3 for tri in triangles])

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
            # for view in range(len(triangles) - 1):
            for view in [1]:
                # Get corresponding triangle in processed input view
                cor_idx = np.where((triangles[view].simplices == t_v).all(axis=1))[0]
                if cor_idx: 
                    cor_idx = cor_idx[0]
                    t_i = triangles[view].simplices[cor_idx]
                    # t_i = t_i[0]
                    # A_i = triangles[view].points[t_i[0]]
                    # B_i = triangles[view].points[t_i[1]]
                    # C_i = triangles[view].points[t_i[2]]
                    vertices_i = np.array([triangles[view].points[t_i[0]], triangles[view].points[t_i[1]], triangles[view].points[t_i[2]]])
                    vertices_ref = np.array([ref_triangles[view].points[ref_triangles[view].simplices[0][0]], 
                                             ref_triangles[view].points[ref_triangles[view].simplices[0][1]],
                                             ref_triangles[view].points[ref_triangles[view].simplices[0][2]]])
                    # Pick the correct orientation for epipolar lines 
                    # To do so we must find the epipole of the novel view in the current view
                    # This is done by finding the intersection of two epipolar lines
                    # One can compute two epipolar lines using one of the already matched features (vertices of a triangle here)
                    # and the position of this feature projected in the reference triangle. This gives two pixel location in the 
                    # input view that project on the same location in the novel view.
                    # Go trough all the pixel inside this input triangle

                    # Find a first epipolar line:
                    # Projection of vertices of input triangles are known they are the vertices of the novel triangle
                    # What we need now is to get the position of a vertex of this triangle in the ref_triangle of the novel view
                    # b = ref_triangles[-1].transform[0,:2].dot(np.transpose(vertices_v[0] - ref_triangles[-1].transform[0,2]))
                    # b_coord = [b[0], b[1], 1 - b.sum(axis=0)]
                    # v0_ref_proj = tuple(np.matmul(vertices_ref.transpose(), b_coord).round().astype('int'))
                    # # The epipolar line in input view is supported by the v0 of the triangle and the projection
                    # # of the v0' in novel view in the reference triangle
                    # # Second epipolar line:
                    # b = ref_triangles[-1].transform[0,:2].dot(np.transpose(vertices_v[2] - ref_triangles[-1].transform[0,2]))
                    # b_coord = [b[0], b[1], 1 - b.sum(axis=0)]
                    # v1_ref_proj = tuple(np.matmul(vertices_ref.transpose(), b_coord).round().astype('int'))

                    # (x1,y1) = vertices_i[0]
                    # (x2,y2) = v0_ref_proj
                    # (x3,y3) = vertices_i[1]
                    # (x4,y4) = v1_ref_proj
                    # d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                    # if d != 0:
                    #     inter_u = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/d
                    #     inter_v = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/d

                    #This looks incorrect...

                    min_u = int(vertices_i[:,0].min())
                    max_u = int(vertices_i[:,0].max())
                    min_v = int(vertices_i[:,1].min())
                    max_v = int(vertices_i[:,1].max())
                    for u in range(min_u, max_u+1):
                        for v in range(min_v, max_v+1):
                            # Compute barycentric coordinates
                            b = triangles[view].transform[cor_idx,:2].dot(np.transpose((u,v) - triangles[view].transform[cor_idx,2]))
                            b_coord = [b[0], b[1], 1 - b.sum(axis=0)]
                            if all(n>=0 for n in b_coord): # The pixel (u,v) is within the currently processed triangle
                                # Compute position of the projection of the pixel (u,v) from the input image to the novel one
                                proj_px = tuple(np.matmul(vertices_v.transpose(), b_coord).round().astype('int'))
                                # Compute the 'depth' value (aka pr_bar)

                                # Get the position of point r in ref view (r is the intersection between the triangle and
                                # the ray passing through the mapped point)
                                # NOTE: use the ref_triangles[-1] to get barycentric coordinates of r
                                # r_b = ref_triangles[-1].transform[0,:2].dot(np.transpose(proj_px - ref_triangles[-1].transform[0,2]))
                                # r_b_coord = [r_b[0], r_b[1], 1 - r_b.sum(axis=0)]
                                # back_proj_px = tuple(np.matmul(vertices_ref.transpose(), r_b_coord).round().astype('int'))

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

    def find_pixel_correspondences_bis(self, triangles, ref_triangles, d_pts, new_d_pts):
        # Get for each pixels in the novel view, the list of the pixels from the input views that are correspondent
        # This time find for each pixels in the novel view, which point are projected here in input views

        sorting_idx = self.triangles_z_buffering(triangles[-1], new_d_pts)

        projected_pixels = np.empty((self.frame_height, self.frame_width), dtype=object)
        tri_nv = triangles[-1]
        # Go through all the triangles of the novel view in a descending depth order
        for tri_idx in sorting_idx:
            t_v = tri_nv.simplices[tri_idx]
            vertices_v = np.array([tri_nv.points[t_v[0]], tri_nv.points[t_v[1]], tri_nv.points[t_v[2]]])
            min_u = int(vertices_v[:,0].min())
            max_u = int(vertices_v[:,0].max())
            min_v = int(vertices_v[:,1].min())
            max_v = int(vertices_v[:,1].max())
            for u in range(min_u, max_u+1):
                for v in range(min_v, max_v+1):
                    # Compute barycentric coordinates
                    b = tri_nv.transform[tri_idx,:2].dot(np.transpose((u,v) - tri_nv.transform[tri_idx,2]))
                    b_coord = [b[0], b[1], 1 - b.sum(axis=0)]
                    if all(n>=0 for n in b_coord): # The pixel (u,v) is within the currently processed triangle
                        for view in range(len(triangles) - 1):
                            # Get corresponding triangle in processed input view
                            t_i = t_v
                            # cor_idx = np.where((triangles[view].simplices == t_v).all(axis=1))[0]
                            # if cor_idx: 
                            #     cor_idx = cor_idx[0]
                            #     t_i = triangles[view].simplices[cor_idx]
                            vertices_i = np.array([triangles[view].points[t_i[0]], triangles[view].points[t_i[1]], triangles[view].points[t_i[2]]])
                            #     vertices_ref = np.array([ref_triangles[view].points[ref_triangles[view].simplices[0][0]], 
                            #                             ref_triangles[view].points[ref_triangles[view].simplices[0][1]],
                            #                             ref_triangles[view].points[ref_triangles[view].simplices[0][2]]])
                            # Find from which pixel the current (u,v) might have been projected
                            px_back_proj = tuple(np.matmul(vertices_i.transpose(), b_coord).round().astype('int'))
                            # NOTE: maybe here we could store the sub_pixel position in input view
                            # to do a weigthed RGB mapping from the 4 neighbors pixels or bilinear interpolation
                            projected_pixels[v,u] = ProjectedPixel(px_back_proj, view, 0)

        return projected_pixels


    """--------------------------------------------"""

    def find_pixel_correspondences_full_triangles(self, triangles, pts, new_pts, d_pts, new_d_pts):
        # Get for each pixels in the novel view, the list of the pixels from the input views that are correspondent

        sorting_idx = self.triangles_z_buffering(triangles[-1], new_d_pts)

        projected_pixels = np.empty((self.frame_height, self.frame_width), dtype=object)
        tri_nv = triangles[-1]
        # Go through all the triangles of the novel view in a descending depth order
        for tri_idx in sorting_idx:
            t_v = tri_nv[tri_idx]
            vertices_v = np.array([new_pts[t_v[0]], new_pts[t_v[1]], new_pts[t_v[2]]])
            min_u = int(vertices_v[:,0].min())
            max_u = int(vertices_v[:,0].max())
            min_v = int(vertices_v[:,1].min())
            max_v = int(vertices_v[:,1].max())
            for u in range(min_u, max_u+1):
                for v in range(min_v, max_v+1):
                    # Compute barycentric coordinates
                    b_coord = tools.get_barycentric_coordinates(vertices_v[0], vertices_v[1], vertices_v[2], (u,v))
                    if all(n>=0 for n in b_coord): # The pixel (u,v) is within the currently processed triangle
                        for view in range(len(triangles) - 1):
                            # Get corresponding triangle in processed input view
                            cor_idx = [i for i, t in enumerate(triangles[view]) if t == t_v]
                            if cor_idx: 
                                cor_idx = cor_idx[0]
                                t_i = triangles[view][cor_idx]
                                vertices_i = np.array([pts[view][t_i[0]], pts[view][t_i[1]], pts[view][t_i[2]]])
                                # vertices_ref = np.array([ref_triangles[view].points[ref_triangles[view].simplices[0][0]], 
                                #                         ref_triangles[view].points[ref_triangles[view].simplices[0][1]],
                                #                         ref_triangles[view].points[ref_triangles[view].simplices[0][2]]])
                                # Find from which pixel the current (u,v) might have been projected
                                px_back_proj = tuple(np.matmul(vertices_i.transpose(), b_coord).round().astype('int'))
                                projected_pixels[v,u] = ProjectedPixel(px_back_proj, view, 0)

        return projected_pixels

    def map_inputs_to_novel_view(self, rgb_cams, projected_pixels):
        new_img = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)

        for v in range(self.frame_height):
            for u in range(self.frame_width):
                proj_px = projected_pixels[v][u]
                if proj_px != None:
                    new_img[v][u] = rgb_cams[proj_px.view][proj_px.pixel[::-1]]

        return new_img               
                                



    
