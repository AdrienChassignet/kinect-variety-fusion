import numpy as np
from scipy.optimize import minimize, Bounds

class ParameterizedImageVariety():
    def __init__(self, q0, d0, q1, d1, q2, d2, pts, d_pts, virtual_view, frame_width=1280, frame_height=720, max_depth=6000, debug=False):
        self.nb_pts = len(pts[0])
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.pts = pts
        self.d_pts = d_pts
        self.virtual_view = virtual_view

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_depth = max_depth

        self.struct_coeffs = []

        self.debug = debug
        self.resids = []

    def get_virtual_pts(self):
        """
        Main method of the class.
        Compute the PIV, create the novel view by placing the 3 ref points and compute the new positions
        of all the other points of the scene in the novel view.
        Return the pixel coordinates and the depth of the scene points in the novel view.
        """
        self.compute_structure_coefficients()

        q0v, q1v, q2v, d0v, d1v, d2v = self.create_novel_view()

        virtual_pts, virtual_d_pts = self.compute_image_positions_in_virtual_view(q0v, q1v, q2v, d0v, d1v, d2v)

        return virtual_pts, virtual_d_pts

    def create_novel_view(self):
        """
        Define the novel view by definig the 3 reference points pixel coordinates and depth.
        The current method use the ground truth provided with the initialization of the PIV.
        """
        q0v = self.q0[self.virtual_view]
        q1v = self.q1[self.virtual_view]
        q2v = self.q2[self.virtual_view]
        d0v = self.d0[self.virtual_view]
        d1v = self.d1[self.virtual_view]
        d2v = self.d2[self.virtual_view]

        return q0v, q1v, q2v, d0v, d1v, d2v

    def compute_structure_coefficients(self):
        """
        For each point in the scene, compute the structure coefficients associated with the PIV.
        """
        for pt_idx in range(self.nb_pts):
            Kis = []
            for view in range(len(self.pts)):
                if view != self.virtual_view:
                    Kis.append(self.build_constraints_matrix(view, self.pts[view][pt_idx], self.d_pts[view][pt_idx]))
            Kis = np.concatenate(Kis)
            _, D, V = np.linalg.svd(Kis)
            self.struct_coeffs.append(V[-1]/V[-1,-1])
            if self.debug:
                self.resids.append(np.sum(np.matmul(Kis, self.struct_coeffs[-1])**2))

    def build_constraints_matrix(self, view, q, d):
        """
        Create the constraint matrix K, used to compute the PIV, for a given point position q=(u,v)
        with measured depth d in the specified view.
        """
        a = self.d1[view]*self.q1[view][0] - self.d0[view]*self.q0[view][0]
        b = self.d2[view]*self.q2[view][0] - self.d0[view]*self.q0[view][0]
        c = d*q[0] - self.d0[view]*self.q0[view][0]
        l = self.d1[view]*self.q1[view][1] - self.d0[view]*self.q0[view][1]
        m = self.d2[view]*self.q2[view][1] - self.d0[view]*self.q0[view][1]
        n = d*q[1] - self.d0[view]*self.q0[view][1]
        r = self.d1[view] - self.d0[view]
        s = self.d2[view] - self.d0[view]
        t = d - self.d0[view]

        return np.array([
            [a**2 - l**2, 2*a*b - 2*l*m, b**2 - m**2, 2*a*c - 2*l*n, 2*b*c - 2*m*n, c**2 - n**2],
            [l**2 - r**2, 2*l*m - 2*r*s, m**2 - s**2, 2*l*n - 2*r*t, 2*m*n - 2*s*t, n**2 - t**2],
            [a*l, b*l + a*m, m*b, c*l + a*n, c*m + b*n, n*c],
            [a*r, b*r + a*s, s*b, c*r + a*t, c*s + b*t, t*c],
            [l*r, m*r + l*s, s*m, n*r + l*t, n*s + m*t, t*n]
        ])

    def compute_image_positions_in_virtual_view(self, q0v, q1v, q2v, d0v, d1v, d2v):
        """
        Using the computed PIV, retrieve the image position and depth of all the scene points
        in the novel view defined by the corrdinates and depth of the ref points Q0, Q1 and Q2.
        """
        if self.debug:
            error_img_pos = np.zeros(0)
            error_depth = np.zeros(0)
            error_u = np.zeros(0)
            error_v = np.zeros(0)
            results = []
            # csts = []

        virtual_pts = []
        virtual_d_pts = []

        for pt_idx in range(self.nb_pts):
            # Estimate the position of the current processed point in the new virtual view
            qv = self.pts[self.virtual_view][pt_idx]
            
            (uv, vv) = qv
            expected = np.array([uv, vv, self.d_pts[self.virtual_view][pt_idx], 1, 1])
            
            # Start the search from a close position in a reference view
            ref_view = 0 # self.virtual_view+1
            while ref_view == self.virtual_view or self.pts[ref_view][pt_idx] == []:
                ref_view += 1

            x0 = np.array([self.pts[ref_view][pt_idx][0], self.pts[ref_view][pt_idx][1], self.d_pts[ref_view][pt_idx], 1, 1])
            # x0 = expected
            # print("Initialization [u, v, g1, g2, g3] = ", x0)

            cst = [q0v[0],q0v[1],q1v[0],q1v[1],q2v[0],q2v[1],self.struct_coeffs[pt_idx], d0v, d1v, d2v]
            # if self.debug:
            #     csts.append(cst)

            # Minimize sum of squares to solve the system

            # Define bounds of the solution space
            b = [[0,0,0,-np.inf,-np.inf], [1,1,2,np.inf,np.inf]]

            # Convert bounds into constraints
            cons = []
            for idx in range(len(b[0])):
                lower = b[0][idx]
                upper = b[1][idx]
                l = {'type':'ineq', 'fun': lambda x, lb=lower, i=idx: x[i] - lb}
                u = {'type':'ineq', 'fun': lambda x, ub=upper, i=idx: ub - x[i]}
                cons.append(l)
                cons.append(u)
            b = Bounds(b[0], b[1], False)

            res = minimize(self.sum_of_squares_of_F, x0, cst, method='Nelder-mead', constraints=cons, bounds=b, options={'disp': self.debug, 'xatol': 0.000001, 'fatol': 0.000001})

            if res:
                u = round(res["x"][0] * self.frame_width) 
                v = round(res["x"][1] * self.frame_width) 
                d = res["x"][2] * self.max_depth
                try: # Check if the new point project on the same coordinates as a previous one
                    occlusion = virtual_pts.index((u,v))
                    if virtual_d_pts[occlusion] > d: # If the new point is closer to the image plane replace the occluded one
                        del virtual_pts[occlusion]
                        del virtual_d_pts[occlusion]
                        virtual_pts.append((u,v))
                        virtual_d_pts.append(d)
                except ValueError:
                    virtual_pts.append((u,v))
                    virtual_d_pts.append(d)

                if self.debug:
                    error_img_pos = np.append(error_img_pos, np.linalg.norm(np.array([uv,vv])*self.frame_width - np.array([res['x'][0],res['x'][1]])*self.frame_width, ord=2))
                    error_depth = np.append(error_depth, abs(self.d_pts[self.virtual_view][pt_idx]*self.max_depth - d))
                    error_u = np.append(error_u, (uv-res['x'][0])*self.frame_width)
                    error_v = np.append(error_v, (vv-res['x'][1])*self.frame_width)
                    results.append(res['x'])
                #     print("Expected result: ", expected)
                #     print("Solution found: ", res["x"], " / residual being: ", res["fun"])
                #     print("Error px: ", error_img_pos[-1])
                #     print('--------------------------------------------------')
                    

        if self.debug:
            for idx in range(len(error_img_pos)):
                print("{4}- Point {0} ; Residual error for coeffs: {1} ; Pixel error: {2} ; Depth error: {3}".format(self.pts[self.virtual_view][idx], self.resids[idx], error_img_pos[idx], error_depth[idx], idx))
            print("Error in pixels for point placement: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_img_pos),
                    np.max(error_img_pos), np.min(error_img_pos), np.std(error_img_pos)))
            print("Error in mm for corresponding depth: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_depth),
                    np.max(error_depth), np.min(error_depth), np.std(error_depth)))
 
        return virtual_pts, virtual_d_pts


    def F(self, x, cst):
        """
        Define the system of equations.
        cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
        x = [u, v, d3, 1, 1]
        """
        # Here only coeffs from the cst list is changing in each call of this method => it could be optimized by avoiding recomputing for the fixed parameters
        [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2] = cst
        [u, v, d3, _, _] = x
        a = d1*u1 - d0*u0
        b = d2*u2 - d0*u0
        c = d3*u - d0*u0
        l = d1*v1 - d0*v0 
        m = d2*v2 - d0*v0
        n = d3*v - d0*v0
        r = d1 - d0
        s = d2 - d0
        t = d3 - d0
        return np.array([
            coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*(a*c-l*n) + 2*coeffs[4]*(b*c-m*n) + c**2 - n**2,
            coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*(l*n-r*t) + 2*coeffs[4]*(m*n-s*t) + n**2 - t**2,
            coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b + coeffs[3]*(l*c+n*a) + coeffs[4]*(m*c+b*n) + c*n,
            coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b + coeffs[3]*(r*c+t*a) + coeffs[4]*(s*c+b*t) + c*t,
            coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s + coeffs[3]*(l*t+n*r) + coeffs[4]*(m*t+s*n) + t*n   
        ])

    def sum_of_squares_of_F(self, x, cst):
        """
        Return the sum of squares of the system of equations.
        cst = [u0, v0, u1, v1, u2, v2, coeffs, d0, d1, d2]
        x = [u, v, d3, 1, 1]
        """
        return np.sum(self.F(x,cst)**2)