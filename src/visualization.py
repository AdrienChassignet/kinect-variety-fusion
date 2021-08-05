import cv2
from matplotlib import pyplot as plt
import numpy as np
from fp_piv import F, F_3var
import copy

def plot_triangulation(rgb_frame, pts, vertices):
    thickness = 1
    color = (255,0,0)
    frame = copy.deepcopy(rgb_frame)
    for i in range(0, len(vertices), 3):
        frame = cv2.line(frame, pts[vertices[i]], pts[vertices[i+1]], color, thickness)
        frame = cv2.line(frame, pts[vertices[i+1]], pts[vertices[i+2]], color, thickness)
        frame = cv2.line(frame, pts[vertices[i+2]], pts[vertices[i]], color, thickness)

    fig = plt.figure("Delaunay triangulation")
    plt.imshow(frame)

    return fig

def plot_matched_features(rgb_cams, pts, q0, q1, q2, name="Matched features"):
    rad = 2
    rgb_cams_cp = copy.deepcopy(rgb_cams)
    for idx in range(len(pts[0])):
        rgb = np.random.rand(3,)*255
        for i in range(len(rgb_cams)):
            c = (round(pts[i][idx][0]), round(pts[i][idx][1]))
            rgb_cams_cp[i] = cv2.circle(rgb_cams_cp[i], c, rad, rgb, -1)
            if idx == 0:
                c0 = (round(q0[i][0]), round(q0[i][1]))
                c1 = (round(q1[i][0]), round(q1[i][1]))
                c2 = (round(q2[i][0]), round(q2[i][1]))
                rgb_cams_cp[i] = cv2.circle(rgb_cams_cp[i], c0, rad, (255,0,0), -1)
                rgb_cams_cp[i] = cv2.circle(rgb_cams_cp[i], c1, rad, (255,0,0), -1)
                rgb_cams_cp[i] = cv2.circle(rgb_cams_cp[i], c2, rad, (255,0,0), -1)

    fig = plt.figure(name)
    for i in range(len(rgb_cams_cp)):
        ax = fig.add_subplot((len(rgb_cams_cp)+2)//3, 3, i+1)
        imgplot = plt.imshow(rgb_cams_cp[i])
        ax.set_title('Cam{}'.format(i))

    return fig

def plot_point_placement_results(rgb_frame, pts, pts_gt, frame_height, frame_width):
    virtual_img = 255 * np.ones([frame_height, frame_width, 3], dtype=np.uint8)
    # virtual_img = cv2.circle(virtual_img, q0v, 7, (255,0,0), -1)
    # virtual_img = cv2.circle(virtual_img, q1v, 7, (255,0,0), -1)
    # virtual_img = cv2.circle(virtual_img, q2v, 7, (255,0,0), -1)
    frame = rgb_frame.copy()
    for idx in range(len(pts)-1):#+3
        if idx < len(pts):
            rgb = np.random.rand(3,)*255
            c = (round(pts_gt[idx][0]), round(pts_gt[idx][1]))
            frame = cv2.circle(rgb_frame, c, 4, rgb, -1)
            virtual_img = cv2.circle(virtual_img, (int(round(pts[idx][0])),int(round(pts[idx][1]))), 3, rgb, -1)
            frame = cv2.circle(rgb_frame, (int(round(pts[idx][0])),int(round(pts[idx][1]))), 2, rgb-40, -1)
        else: #Plot the ref points
            virtual_img = cv2.circle(virtual_img, (int(round(pts[idx][0])),int(round(pts[idx][1]))), 3, (255,0,0), -1)
            frame = cv2.circle(rgb_frame, (int(round(pts[idx][0])),int(round(pts[idx][1]))), 3, (255,0,0), -1)


    fig = plt.figure("Result")
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(frame)
    ax.set_title('Ground truth view')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(virtual_img)
    ax.set_title('Virtual image point placement')

    return fig

def plot_F_squared_around_x(x0, cst, name="Plots of F²"):
    (u0, v0, d3, _, _) = x0
    width = 100
    nb_pt = 1000
    u = np.linspace(u0-width/2, u0+width/2, nb_pt)
    v = np.linspace(v0-width/2, v0+width/2, nb_pt)
    d_scale = 2
    d = np.linspace(d3-width*d_scale, d3+width*d_scale, nb_pt)

    x = np.array([u, v0*np.ones(nb_pt), d3*np.ones(nb_pt), np.ones(nb_pt), np.ones(nb_pt)])
    F_res_u = F_3var(x, cst)
    x = np.array([u0*np.ones(nb_pt), v, d3*np.ones(nb_pt), np.ones(nb_pt), np.ones(nb_pt)])
    F_res_v = F_3var(x, cst)
    x = np.array([u0*np.ones(nb_pt), v0*np.ones(nb_pt), d, np.ones(nb_pt), np.ones(nb_pt)])
    F_res_d = F_3var(x, cst)

    fig = plt.figure(name)
    for eq_idx in range(5):
        ax = fig.add_subplot(1, 6, eq_idx+1)

        ln1 = ax.plot(u, F_res_u[eq_idx]**2, color='red', label='u')
        ax.tick_params(axis='x', labelcolor='red')
        ax2 = ax.twiny()
        ln2 = ax2.plot(v, F_res_v[eq_idx]**2, color='green', label='v')
        ax2.tick_params(axis='x', labelcolor='green')
        ax2.spines['bottom'].set_position(("axes", 1.08))
        ax3 = ax2.twiny()
        ln3 = ax3.plot(d, F_res_d[eq_idx]**2, color='blue', label='d')
        ax3.tick_params(axis='x', labelcolor='blue')
        lns = ln1+ln2+ln3
        ax.legend(lns, [l.get_label() for l in lns])
        ax.set_xlabel('u')
        ax2.set_xlabel('v')
        ax3.set_xlabel('d')
        ax.set_ylabel('F{}²'.format(eq_idx))
        up_bound = 10*max((F_res_u[eq_idx]**2).min(), (F_res_v[eq_idx]**2).min(), (F_res_d[eq_idx]**2).min())
        plt.ylim([0, up_bound])
    
    ax = fig.add_subplot(1,6,6)
    ln1 = ax.plot(u, np.sum(F_res_u**2, axis=0), color='red', label='u')
    ax.tick_params(axis='x', labelcolor='red')
    ax2 = ax.twiny()
    ln2 = ax2.plot(v, np.sum(F_res_v**2, axis=0), color='green', label='v')
    ax2.tick_params(axis='x', labelcolor='green')
    ax2.spines['bottom'].set_position(("axes", 1.08))
    ax3 = ax2.twiny()
    ln3 = ax3.plot(d, np.sum(F_res_d**2, axis=0), color='blue', label='d')
    ax3.tick_params(axis='x', labelcolor='blue')
    lns = ln1+ln2+ln3
    ax.legend(lns, [l.get_label() for l in lns])
    ax.set_xlabel('u')
    ax2.set_xlabel('v')
    ax3.set_xlabel('d')
    ax.set_ylabel('sum of F²')
    plt.ylim([0, 10000])

    
    return fig


def surface_plot_F_squared_around_x(x0, cst, name="Surface plots of F²"):
    (u0, v0, d3, _, _) = x0
    width = 30
    nb_pt = 75
    u = np.linspace(u0-width/2, u0+width/2, nb_pt)
    v = np.linspace(v0-width/2, v0+width/2, nb_pt)
    d_scale = 2
    d = np.linspace(d3-width*d_scale, d3+width*d_scale, nb_pt)

    Uv, Vu = np.meshgrid(u, v)
    Ud, Du = np.meshgrid(u, d)
    Vd, Dv = np.meshgrid(v, d)
    x = np.array([Uv, Vu, d3*np.ones((nb_pt, nb_pt)), np.ones((nb_pt, nb_pt)), np.ones((nb_pt, nb_pt))])
    F_res_uv = F_3var(x, cst)
    x = np.array([Ud, v0*np.ones((nb_pt, nb_pt)), Du, np.ones((nb_pt, nb_pt)), np.ones((nb_pt, nb_pt))])
    F_res_ud = F_3var(x, cst)
    x = np.array([u0*np.ones((nb_pt, nb_pt)), Vd, Dv, np.ones((nb_pt, nb_pt)), np.ones((nb_pt, nb_pt))])
    F_res_vd = F_3var(x, cst)

    fig = plt.figure(name)
    for eq_idx in range(5):
        ax = fig.add_subplot(3, 5, eq_idx+1, projection='3d')
        ax.plot_surface(Uv, Vu, F_res_uv[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('F{}(u,v)²'.format(eq_idx))
        ax.set_title('F{}(u,v)²'.format(eq_idx))

        ax = fig.add_subplot(3, 5, eq_idx+6, projection='3d')
        ax.plot_surface(Ud, Du, F_res_ud[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('u')
        ax.set_ylabel('d')
        ax.set_zlabel('F{}(u,d)²'.format(eq_idx))
        ax.set_title('F{}(u,d)²'.format(eq_idx))

        ax = fig.add_subplot(3, 5, eq_idx+11, projection='3d')
        ax.plot_surface(Vd, Dv, F_res_vd[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('v')
        ax.set_ylabel('d')
        ax.set_zlabel('F{}(v,d)²'.format(eq_idx))
        ax.set_title('F{}(v,d)²'.format(eq_idx))
    
    return fig

def OLD_plot_F_squared_around_x(x0, cst, name="Plots of F²"):
    (u0, v0, g10, g20, g30) = x0
    width = 30
    nb_pt = 75
    u = np.linspace(u0-width/2, u0+width/2, nb_pt)
    v = np.linspace(v0-width/2, v0+width/2, nb_pt)
    d_scale = 200
    l1 = np.linspace(g10-width/d_scale, g10+width/d_scale, nb_pt)
    l2 = np.linspace(g20-width/d_scale, g20+width/d_scale, nb_pt)
    l3 = np.linspace(g30-width/d_scale, g30+width/d_scale, nb_pt)
    Uv, Vu = np.meshgrid(u, v)
    Ul, Lu = np.meshgrid(u, l3)
    Vl, Lv = np.meshgrid(v, l3)
    x = np.array([Uv, Vu, g10*np.ones((nb_pt, nb_pt)), g20*np.ones((nb_pt, nb_pt)), g30*np.ones((nb_pt, nb_pt))])
    F_res_uv = F(x, cst)
    x = np.array([Ul, v0*np.ones((nb_pt, nb_pt)), g10*np.ones((nb_pt, nb_pt)), g20*np.ones((nb_pt, nb_pt)), Lu])
    F_res_ul = F(x, cst)
    x = np.array([u0*np.ones((nb_pt, nb_pt)), Vl, g10*np.ones((nb_pt, nb_pt)), g20*np.ones((nb_pt, nb_pt)), Lv])
    F_res_vl = F(x, cst)

    x = np.array([u, v0*np.ones(nb_pt), g10*np.ones(nb_pt), g20*np.ones(nb_pt), g30*np.ones(nb_pt)])
    F_res_u = F(x, cst)
    x = np.array([u0*np.ones(nb_pt), v, g10*np.ones(nb_pt), g20*np.ones(nb_pt), g30*np.ones(nb_pt)])
    F_res_v = F(x, cst)
    x = np.array([u0*np.ones(nb_pt), v0*np.ones(nb_pt), l1, g20*np.ones(nb_pt), g30*np.ones(nb_pt)])
    F_res_l1 = F(x, cst)
    x = np.array([u0*np.ones(nb_pt), v0*np.ones(nb_pt), g10*np.ones(nb_pt), l2, g30*np.ones(nb_pt)])
    F_res_l2 = F(x, cst)
    x = np.array([u0*np.ones(nb_pt), v0*np.ones(nb_pt), g10*np.ones(nb_pt), g20*np.ones(nb_pt), l3])
    F_res_l3 = F(x, cst)

    fig = plt.figure(name)
    for eq_idx in range(5):
        ax = fig.add_subplot(2, 5, eq_idx+1, projection='3d')
        if eq_idx in [0, 2]:
            ax.plot_surface(Uv, Vu, F_res_uv[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('u')
            ax.set_ylabel('v')
            ax.set_zlabel('F{}(u,v)²'.format(eq_idx))
            ax.set_title('F{}(u,v)²'.format(eq_idx))
        elif eq_idx == 3:
            ax.plot_surface(Ul, Lu, F_res_ul[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('u')
            ax.set_ylabel('g3')
            ax.set_zlabel('F{}(u,g3)²'.format(eq_idx))
            ax.set_title('F{}(u,g3)²'.format(eq_idx))
        elif eq_idx in [1, 4]:
            ax.plot_surface(Vl, Lv, F_res_vl[eq_idx]**2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_xlabel('v')
            ax.set_ylabel('g3')
            ax.set_zlabel('F{}(v,g3)²'.format(eq_idx))
            ax.set_title('F{}(v,g3)²'.format(eq_idx))
        
        ax = fig.add_subplot(2, 5, eq_idx+6)
        ax.plot(u, F_res_u[eq_idx]**2, color='red', label='u')
        ax.tick_params(axis='x', labelcolor='red')
        ax2 = ax.twiny()
        ax2.plot(v, F_res_v[eq_idx]**2, color='green', label='v')
        ax2.tick_params(axis='x', labelcolor='green')
        ax3 = ax.twiny()
        ax3.plot(l1, F_res_l1[eq_idx]**2, color='blue', label='g1')
        ax3.tick_params(axis='x', labelcolor='blue')
        ax4 = ax.twiny()
        ax4.plot(l2, F_res_l2[eq_idx]**2, color='yellow', label='g2')
        ax4.tick_params(axis='x', labelcolor='yellow')
        ax5 = ax.twiny()
        ax5.plot(l3, F_res_l3[eq_idx]**2, color='brown', label='g3')
        ax5.tick_params(axis='x', labelcolor='brown')
        plt.legend()
        plt.ylim([-10, 1000])
    
    return fig