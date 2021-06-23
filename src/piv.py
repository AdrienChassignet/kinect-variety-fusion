from math import sqrt
import numpy as np
import cv2

img1 = np.zeros([200,200,3], dtype=np.uint8) 
img2 = np.zeros([200,200,3], dtype=np.uint8) 

(u01, v01, d01) = (100, 50, 4) # Q0
(u11, v11, d11) = (70, 110, 10) # Q1
(u21, v21, d21) = (120, 120, 10) # Q2
(u1, v1, d1) = (100, 100, 5) # Q
(ub1, vb1, db1) = (160, 40, 5) # Q bis

(u02, v02, d02) = (125, 50, 4) # Q0
(u12, v12, d12) = (80, 110, 10) # Q1
(u22, v22, d22) = (130, 120, 10) # Q2
(u2, v2, d2) = (120, 100, 5) # Q
(ub2, vb2, db2) = (180, 40, 5) # Q bis

img1[v01,u01,:] = [0,0,255]
img1[v11,u11,:] = [0,255,0]
img1[v21,u21,:] = [255,0,0]
img1[v1, u1, :] = [0,255,255]
img1[vb1, ub1, :] = [0,255,255]

(u01, v01, d01) = (.5, .25, 4) # Q0
(u11, v11, d11) = (.35, .55, 10) # Q1
(u21, v21, d21) = (.6, .6, 10) # Q2
(u1, v1, d1) = (.5, .5, 5) # Q
(ub1, vb1, db1) = (.8, .2, 5) # Q bis

img2[v02,u02,:] = [0,0,255]
img2[v12,u12,:] = [0,255,0]
img2[v22,u22,:] = [255,0,0]
img2[v2, u2, :] = [0,255,255]
img2[vb2, ub2, :] = [0,255,255]

(u02, v02, d02) = (.625, .25, 4) # Q0
(u12, v12, d12) = (.4, .55, 10) # Q1
(u22, v22, d22) = (.65, .6, 10) # Q2
(u2, v2, d2) = (.6, .5, 5) # Q
(ub2, vb2, db2) = (.9, .2, 5) # Q bis

cv2.imshow("input 1", img1)
cv2.imshow("input 2", img2)
cv2.waitKey()

g11 = d11/d01
g21 = d21/d01
g31 = d1/d01
gb31 = db1/d01
a1 = g11*u11 - u01
b1 = g21*u21 - u01
c1 = g31*u1 - u01
cb1 = gb31*ub1 - u01
l1 = g11*v11 - v01
m1 = g21*v21 - v01
n1 = g31*v1 - v01
nb1 = gb31*vb1 - v01
r1 = g11 - 1
s1 = g21 - 1
t1 = g31 - 1
tb1 = gb31 - 1

K1s = np.array([
    [a1*a1 - l1*l1, 2*a1*b1 - 2*l1*m1, b1*b1 - m1*m1, 2*a1*c1 - 2*l1*n1, 2*b1*c1 - 2*m1*n1, c1*c1 - n1*n1],
    [l1*l1 - r1*r1, 2*l1*m1 - 2*r1*s1, m1*m1 - s1*s1, 2*l1*n1 - 2*r1*t1, 2*m1*n1 - 2*s1*t1, n1*n1 - t1*t1],
    [a1*l1, b1*l1 + a1*m1, m1*b1, c1*l1 + a1*n1, c1*m1 + b1*n1, n1*c1],
    [a1*r1, b1*r1 + a1*s1, s1*b1, c1*r1 + a1*t1, c1*s1 + b1*t1, t1*c1],
    [l1*r1, m1*r1 + l1*s1, s1*m1, n1*r1 + l1*t1, n1*s1 + m1*t1, t1*n1]
])
K1bs = np.array([
    [a1*a1 - l1*l1, 2*a1*b1 - 2*l1*m1, b1*b1 - m1*m1, 2*a1*cb1 - 2*l1*nb1, 2*b1*cb1 - 2*m1*nb1, cb1*cb1 - nb1*nb1],
    [l1*l1 - r1*r1, 2*l1*m1 - 2*r1*s1, m1*m1 - s1*s1, 2*l1*nb1 - 2*r1*tb1, 2*m1*nb1 - 2*s1*tb1, nb1*nb1 - tb1*tb1],
    [a1*l1, b1*l1 + a1*m1, m1*b1, cb1*l1 + a1*nb1, cb1*m1 + b1*nb1, nb1*cb1],
    [a1*r1, b1*r1 + a1*s1, s1*b1, cb1*r1 + a1*tb1, cb1*s1 + b1*tb1, tb1*cb1],
    [l1*r1, m1*r1 + l1*s1, s1*m1, nb1*r1 + l1*tb1, nb1*s1 + m1*tb1, tb1*nb1]
])

g12 = d12/d02
g22 = d22/d02
g32 = d2/d02
gb32 = db2/d02
a2 = g12*u12 - u02
b2 = g22*u22 - u02
c2 = g32*u2 - u02
cb2 = gb32*ub2 - u02
l2 = g12*v12 - v02
m2 = g22*v22 - v02
n2 = g32*v2 - v02
nb2 = gb32*vb2 - v02
r2 = g12 - 1
s2 = g22 - 1
t2 = g32 - 1
tb2 = gb32 - 1

K2s = np.array([
    [a2*a2 - l2*l2, 2*a2*b2 - 2*l2*m2, b2*b2 - m2*m2, 2*a2*c2 - 2*l2*n2, 2*b2*c2 - 2*m2*n2, c2*c2 - n2*n2],
    [l2*l2 - r2*r2, 2*l2*m2 - 2*r2*s2, m2*m2 - s2*s2, 2*l2*n2 - 2*r2*t2, 2*m2*n2 - 2*s2*t2, n2*n2 - t2*t2],
    [a2*l2, b2*l2 + a2*m2, m2*b2, c2*l2 + a2*n2, c2*m2 + b2*n2, n2*c2],
    [a2*r2, b2*r2 + a2*s2, s2*b2, c2*r2 + a2*t2, c2*s2 + b2*t2, t2*c2],
    [l2*r2, m2*r2 + l2*s2, s2*m2, n2*r2 + l2*t2, n2*s2 + m2*t2, t2*n2]
])
K2bs = np.array([
    [a2*a2 - l2*l2, 2*a2*b2 - 2*l2*m2, b2*b2 - m2*m2, 2*a2*cb2 - 2*l2*nb2, 2*b2*cb2 - 2*m2*nb2, cb2*cb2 - nb2*nb2],
    [l2*l2 - r2*r2, 2*l2*m2 - 2*r2*s2, m2*m2 - s2*s2, 2*l2*nb2 - 2*r2*tb2, 2*m2*nb2 - 2*s2*tb2, nb2*nb2 - tb2*tb2],
    [a2*l2, b2*l2 + a2*m2, m2*b2, cb2*l2 + a2*nb2, cb2*m2 + b2*nb2, nb2*cb2],
    [a2*r2, b2*r2 + a2*s2, s2*b2, cb2*r2 + a2*tb2, cb2*s2 + b2*tb2, tb2*cb2],
    [l2*r2, m2*r2 + l2*s2, s2*m2, nb2*r2 + l2*tb2, nb2*s2 + m2*tb2, tb2*nb2]
])

Ks = np.concatenate((K1s, K2s), axis=0)
U,D,V = np.linalg.svd(Ks)

print("D = ", D)
print("Solution is: ", V[:,-1])
coeffs = V[:,-1]/V[-1,-1]
print("Structure coefficients are: ", coeffs)

Kbs = np.concatenate((K1bs, K2bs), axis=0)
U,D,V = np.linalg.svd(Kbs)

print("D = ", D)
print("Solution is: ", V[:,-1])
coeffs_b = V[:,-1]/V[-1,-1]
print("Structure coefficients are: ",coeffs_b)

# (u0v, v0v, d0v) = (110, 50, 4) # Q0
# (u1v, v1v, d1v) = (74, 110, 10) # Q1
# (u2v, v2v, d2v) = (124, 120, 10) # Q2
(u0v, v0v, d0v) = (0.55, 0.25, 4) # Q0
(u1v, v1v, d1v) = (0.37, .55, 10) # Q1
(u2v, v2v, d2v) = (.62, .6, 10) # Q2

g1 = d1v/d0v
g2 = d2v/d0v
a = g1*u1v - u0v
b = g2*u2v - u0v
l = g1*v1v - v0v
m = g2*v2v - v0v
r = g1 - 1
s = g2 - 1
alpha = a*coeffs[3] + b*coeffs[4] - u0v
beta = l*coeffs[3] + m*coeffs[4] - v0v
gamma = r*coeffs[3] + s*coeffs[4] - 1
mu1 = a*l*coeffs[0] + (b*l+a*m)*coeffs[1] + m*b*coeffs[2] - (beta+v0v)*u0v - (alpha+u0v)*v0v + u0v*v0v
mu2 = a*r*coeffs[0] + (b*r+a*s)*coeffs[1] + s*b*coeffs[2] - (gamma+1)*u0v - (alpha+u0v) + u0v
mu3 = r*l*coeffs[0] + (s*l+r*m)*coeffs[1] + m*s*coeffs[2] - (gamma+1)*v0v - (beta+v0v) + v0v

delta = (2*gamma*mu1 - 2*alpha*beta*gamma)**2 - 4*(mu1 - alpha*beta)*(mu1*gamma**2 - mu2*beta*gamma - mu3*alpha*gamma + mu2*mu3)
g3 = (2*alpha*beta*gamma - 2*gamma*mu1 + sqrt(delta)) / (2*(mu1 - alpha*beta))
if g3 < 0:
    g3 = (2*alpha*beta*gamma - 2*gamma*mu1 - sqrt(delta)) / (2*(mu1 - alpha*beta))

u = (-mu2 - g3*alpha) / (g3**2 + gamma*g3)
v = (-mu3 - g3*beta) / (g3**2 + gamma*g3)

print("u = ", u, " / v = ", v, " / d = ", g3*d0v)

alpha = a*coeffs_b[3] + b*coeffs_b[4] - u0v
beta = l*coeffs_b[3] + m*coeffs_b[4] - v0v
gamma = r*coeffs_b[3] + s*coeffs_b[4] - 1
mu1 = a*l*coeffs_b[0] + (b*l+a*m)*coeffs_b[1] + m*b*coeffs_b[2] - beta*u0v - alpha*v0v + u0v*v0v
mu2 = a*r*coeffs_b[0] + (b*r+a*s)*coeffs_b[1] + s*b*coeffs_b[2] - gamma*u0v - alpha + u0v
mu3 = r*l*coeffs_b[0] + (s*l+r*m)*coeffs_b[1] + m*s*coeffs_b[2] - gamma*v0v - beta + v0v

delta = (2*gamma*mu1 - 2*alpha*beta*gamma)**2 - 4*(mu1 - alpha*beta)*(mu1*gamma**2 - mu2*beta*gamma - mu3*alpha*gamma + mu2*mu3)
g3 = (2*alpha*beta*gamma - 2*gamma*mu1 + sqrt(delta)) / (2*(mu1 - alpha*beta))
if g3 < 0:
    g3 = (2*alpha*beta*gamma - 2*gamma*mu1 - sqrt(delta)) / (2*(mu1 - alpha*beta))

u = (-mu2 - g3*alpha) / (g3**2 + gamma*g3)
v = (-mu3 - g3*beta) / (g3**2 + gamma*g3)

print("u = ", u, " / v = ", v, " / d = ", g3*d0v)