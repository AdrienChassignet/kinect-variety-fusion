import numpy as np

def convex_hull(points, split=False):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    From Wikipedia.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    if not split:
      return lower[:-1] + upper[:-1]
    else:
      return lower[:-1], upper[:-1]

def get_neighborhood(u, v, radius, img):
    """
    Return the square neighborhood of a given point in image.
    u and v is the location of the point where we want the neighborhood.
    Radius is the size of the neighborhood (e.g. radius=1 gives a 3x3 neighborhood).
    """
    (height, width) = np.shape(img)
    neighborhood = np.zeros((2*radius+1, 2*radius+1))

    for i in range(-radius, radius+1):
        if v+i >= 0 and v+i < height:
            for j in range(-radius, radius+1):
                if u+j >= 0 and u+j < width:
                    neighborhood[i+radius, j+radius] = img[v+i, u+j]

    return neighborhood

def normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts, max_px, max_d):
    """
    Normalize the pixel coordinates and the depths values of the given
    3 reference points Q0, Q1 and Q2, and the rest of the points.
    The pixel coordinates are normalized by the max_px value which, in 
    the general scenario, corresponds to width of the frame.
    The depth values are normalized by the max_d value which, in the general
    scenario, corresponds to the maximum depth measured in the scene.
    
    Inputs: - q0, q1, q2: The 3 reference point image position in each view
            - d0, d1, d2: The corresponding depth in each view of the 3 ref points
            - pts: list for each view of the matched points across the views
            - d_pts: list for each view of the depth of the corresponding points across the views
            - max_px: the value used to normalize the pixel coordinates
            - max_d: the value used to normalize the depth values
    Outputs:    - q0_n, q1_n, q2_n: The 3 reference point normalized image position in each view
                - d0_n, d1_n, d2_n: The corresponding normalized depth in each view of the 3 ref points
                - pts_n: list for each view of the corresponding normalized points across the views without the 3 ref points
                - d_pts_n: list for each view of the normalized depth of the corresponding points
                        across the views without the ref points 
    """

    nb_views = len(q0)
    q0_n = []
    d0_n = []
    q1_n = []
    d1_n = []
    q2_n = []
    d2_n = []
    pts_n = [[] for i in range(nb_views)]
    d_pts_n = [[] for i in range(nb_views)]

    for i in range(nb_views):
        q0_n.append((q0[i][0]/max_px, q0[i][1]/max_px))
        d0_n.append(d0[i]/max_d)
        q1_n.append((q1[i][0]/max_px, q1[i][1]/max_px))
        d1_n.append(d1[i]/max_d)
        q2_n.append((q2[i][0]/max_px, q2[i][1]/max_px))
        d2_n.append(d2[i]/max_d)
        for j, pt in enumerate(pts[i]):
            pts_n[i].append((pt[0]/max_px, pt[1]/max_px))
            d_pts_n[i].append(d_pts[i][j]/max_d)

    return q0_n, d0_n, q1_n, d1_n, q2_n, d2_n, pts_n, d_pts_n

def rescale_and_concatenate_points(pts, q0, q1, q2, d_pts, d0, d1, d2, dmap, max_d, frame_width, frame_height, rescale=True):
    if rescale:
        new_pts = []
        for pt in pts:
            new_pts.append((round(pt[0]*frame_width), round(pt[1]*frame_width)))
        new_d_pts = [d_pt*max_d for d_pt in d_pts]
    else:
        new_pts = pts.copy()
        new_d_pts = d_pts.copy()
    new_pts.append((round(q0[0]*frame_width), round(q0[1]*frame_width)))
    new_pts.append((round(q1[0]*frame_width), round(q1[1]*frame_width)))
    new_pts.append((round(q2[0]*frame_width), round(q2[1]*frame_width)))
    new_d_pts.append(d0*max_d)
    new_d_pts.append(d1*max_d)
    new_d_pts.append(d2*max_d)

    # corners = [(0, 0), (0, frame_height-1), (frame_width-1, 0), (frame_width-1, frame_height-1)]
    # new_pts += corners
    # #TODO: check if there is a valid depth value here otherwise use the max depth in the frame
    # new_d_pts += [max_d, max_d, max_d, max_d]

    return new_pts, new_d_pts

def get_barycentric_coordinates(A, B, C, P):
    """
    A, B and C are the vertex of the triangle and P is the point we want to express
    in function of these vertices.
    The four points are given as a tuple (x,y) of the pixel coordinates in the frame.
    """
    v0 = (B[0]-A[0], B[1]-A[1])
    v1 = (C[0]-A[0], C[1]-A[1])
    v2 = (P[0]-A[0], P[1]-A[1])
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1. - v - w
    return u, v, w

def collinear(p1, p2, p3):
    return ((p3[1] - p2[1])*(p2[0] - p1[0]) == (p2[1] - p1[1])*(p3[0] - p2[0]))
