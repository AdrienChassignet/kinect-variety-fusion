from corresponding_points_selector import CorrespondingPointsSelector
import tools
import numpy as np

def get_pts(d_cams):
    cps = CorrespondingPointsSelector()
    q0 = [(514,387), (514,341), (554,294)]
    q1 = [(313,553), (325,478), (385,393)]
    q2 = [(795,489), (765,406), (783,319)]
    pts=[[(378,331), (747,337), (782,547), (282,491)],
         [(390,308), (722,315), (757,474), (304,404)],
         [(445,285), (743,288), (780,391), (375,316)]]
         
    d_pts = [[] for i in range(len(pts))]

    for idx in range(len(pts[0])): # Check all matched points
        depth = np.zeros(len(pts))
        valid = True
        for i in range(len(pts)): # Check depth of current point in each view
            if pts[i][idx] != []:
                (u,v) = pts[i][idx]
                neighborhood = tools.get_neighborhood(u, v, 3, d_cams[i])
                nonzero = neighborhood[np.nonzero(neighborhood)]
                count = len(nonzero)
                if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
                    depth[i] = sorted(nonzero)[count//2] #Take median value
                else:
                    valid = False
                    break
        if valid: # If there is valid depth information in all views we keep the point
            for i in range(len(pts)):
                d_pts[i].append(depth[i])

    d0 = []
    for i, px in enumerate(q0):
        neighborhood = tools.get_neighborhood(px[0], px[1], 3, d_cams[i])
        nonzero = neighborhood[np.nonzero(neighborhood)]
        count = len(nonzero)
        if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
            d0.append(sorted(nonzero)[count//2]) #Take median value
    d1 = []
    for i, px in enumerate(q1):
        neighborhood = tools.get_neighborhood(px[0], px[1], 3, d_cams[i])
        nonzero = neighborhood[np.nonzero(neighborhood)]
        count = len(nonzero)
        if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
            d1.append(sorted(nonzero)[count//2]) #Take median value
    d2 = []
    for i, px in enumerate(q2):
        neighborhood = tools.get_neighborhood(px[0], px[1], 3, d_cams[i])
        nonzero = neighborhood[np.nonzero(neighborhood)]
        count = len(nonzero)
        if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
            d2.append(sorted(nonzero)[count//2]) #Take median value

    return q0, d0, q1, d1, q2, d2, pts, d_pts