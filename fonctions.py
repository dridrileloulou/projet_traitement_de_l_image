#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:17:27 2025
@author: dridri & Gosan
"""

import numpy as np
import matplotlib.pyplot as plt

def homography_estimate(x1, y1, x2, y2):
    assert(len(x1) == len(y1) == len(x2) == len(y2))
    
    N = len(x1)
    assert(N >= 4)
    A = np.zeros((2*N, 8))
    B = np.zeros(2*N)
    
    for i in range (N):
        
        A[2*i] = [x1[i], y1[i], 1, 0, 0, 0, -x1[i]*x2[i], -y1[i]*x2[i]]
        A[2*i+1] = [0, 0, 0, x1[i], y1[i], 1, -x1[i]*y2[i], -y1[i]*y2[i]]
        
        B[2*i] = x2[i]
        B[2*i+1] = y2[i]
        
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    X = np.append(X, 1)
    
    H = np.reshape(X, (3,3))
    
    return H

def homography_apply(H, x1, y1):
    assert(len(x1) == len(y1))
    
    x2 = []
    y2 = []
    
    for i in range (len(x1)):
        x2.append((H[0,0]*x1[i] + H[0,1]*y1[i] + H[0,2])/ (H[2,0]*x1[i] + H[2,1]*y1[i] + H[2,2]))
        y2.append((H[1,0]*x1[i] + H[1,1]*y1[i] + H[1,2])/ (H[2,0]*x1[i] + H[2,1]*y1[i] + H[2,2]))
    
    return (x2, y2)

def homography_extraction(I1, x, y, w, h):
    
    I2 = np.zeros((h, w))
    
    H = homography_estimate([0, w-1, w-1, 0], [0, 0, h-1, h-1], x, y)
    
    for (i, j) in np.ndindex((h, w)):
        x_ext, y_ext = homography_apply(H, [j], [i])
        x_ext = (int)(x_ext[0])
        y_ext = (int)(y_ext[0])
        I2[i,j] = I1[y_ext, x_ext]
    
    return I2


def homography_projection(I1, I2, x, y):
    
    h_src, w_src, _ = np.shape(I1)
    h_dst, w_dst, _ = np.shape(I2)
    I_final = I2.copy()
    
    H = homography_estimate(x, y, [0, w_src-1, w_src-1, 0], [0, 0, h_src-1, h_src-1])
    
    for (i,j) in np.ndindex((h_dst, w_dst)):
        x_proj, y_proj = homography_apply(H, [j], [i])
        x_proj = (int)(x_proj[0])
        y_proj = (int)(y_proj[0])
        
        if 0 <= x_proj < w_src and 0 <= y_proj < h_src:
            I_final[i,j,:] = I1[y_proj, x_proj, :]
        
    return I_final


def point_in_quad(px, py, xq, yq):
    def cross(x1,y1,x2,y2,x3,y3):
        return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

    s = []
    for k in range(4):
        x1,y1 = xq[k], yq[k]
        x2,y2 = xq[(k+1)%4], yq[(k+1)%4]
        s.append(cross(x1,y1,x2,y2,px,py))

    return (all(v >= 0 for v in s) or all(v <= 0 for v in s))

def homography_cross_projection(I, x1, y1, x2, y2) : 
    h_img, w_img, _ = np.shape(I)
    
    x_carre = np.array([0,1,1,0])
    y_carre = np.array([0,0,1,1])
    
    H1 = homography_estimate(x1, y1, x_carre, y_carre)
    
    H2 = homography_estimate(x_carre, y_carre, x2, y2)
    
    H = np.dot(H2,H1)
    
    H_inv = np.linalg.inv(H)
    
    I_src = I.copy()
    I_final = I.copy()

    h, w = I_final.shape[:2]
    
    for i in range(h_img):
        for j in range(w_img):
            if not point_in_quad(j, i, x2, y2):
                continue
            # Appliquer l'homographie inverse pour trouver la position dans l'image source
            x_proj, y_proj = homography_apply(H_inv, [j], [i])
            x_proj = int(x_proj[0])
            y_proj = int(y_proj[0])

            # Vérifier si la position est dans les limites de l'image
            if 0 <= x_proj < w_img and 0 <= y_proj < h_img:
                I_final[i, j, :] = I_src[y_proj, x_proj, :]

    for i in range(h_img):
        for j in range(w_img):
            if not point_in_quad(j, i, x1, y1):
                continue
            # Appliquer l'homographie inverse pour trouver la position dans l'image source
            x_proj, y_proj = homography_apply(H, [j], [i])
            x_proj = int(x_proj[0])
            y_proj = int(y_proj[0])

            # Vérifier si la position est dans les limites de l'image
            if 0 <= x_proj < w_img and 0 <= y_proj < h_img:
                I_final[i, j, :] = I_src[y_proj, x_proj, :]

    return I_final


def I_to_MIB(I):
    h, w, _ = I.shape
    M = np.ones((h,w))
    B = [(0,0), (w-1, h-1)]
    
    return M, I, B

def MIB_homography(MIB, H):
    M, I, B = MIB
    (x1, y1), (x2, y2) = B

    h, w = M.shape

    # Calcul de la transformation des 4 coins
    xs = [x1, x2, x2, x1]
    ys = [y1, y1, y2, y2]
    xd, yd = homography_apply(H, xs, ys)

    x_min = int(np.floor(min(xd)))
    x_max = int(np.ceil(max(xd)))
    y_min = int(np.floor(min(yd)))
    y_max = int(np.ceil(max(yd)))

    W_ = x_max - x_min + 1
    H_ = y_max - y_min + 1

    M2 = np.zeros((H_, W_), dtype=M.dtype)
    I2 = np.zeros((H_, W_, I.shape[2]), dtype=I.dtype)
    B2 = [(x_min, y_min), (x_max, y_max)]

    # Remplissage destination
    H_inv = np.linalg.inv(H)

    for i in range(H_):
        yg = y_min + i
        for j in range(W_):
            xg = x_min + j

            x_res, y_res = homography_apply(H_inv, [xg], [yg])

            xl = int(x_res[0]) - x1
            yl = int(y_res[0]) - y1

            if 0 <= xl < w and 0 <= yl < h:
                mv = M[yl, xl]
                if mv != 0:
                    M2[i, j] = mv
                    I2[i, j, :] = I[yl, xl, :]

    return (M2, I2, B2)

def MIB_fusion(*MIBS):
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for (_, _, B)  in MIBS:
        [(x1, y1), (x2, y2)] = B

        x_min = x1 if x_min is None else min(x_min, x1)
        y_min = y1 if y_min is None else min(y_min, y1)
        x_max = x2 if x_max is None else max(x_max, x2)
        y_max = y2 if y_max is None else max(y_max, y2)
    
    W = x_max - x_min + 1
    H = y_max - y_min + 1

    M_f = np.zeros((H, W))
    I0 = MIBS[0][1]
    I_f = np.zeros((H, W, I0.shape[2]), dtype=I0.dtype)
    B_f = [(x_min, y_min), (x_max, y_max)]

    for (M, I, B) in MIBS:
        (x1, y1), (x2, y2) = B

        h, w = M.shape
        x_off = x1 - x_min
        y_off = y1 - y_min

        ys = slice(y_off, y_off + h)
        xs = slice(x_off, x_off + w)

        m = (M != 0)

        region_M = M_f[ys, xs]
        region_M[m] = np.maximum(region_M[m], M[m])
        M_f[ys, xs] = region_M

        region_I = I_f[ys, xs, :]
        region_I[m] = I[m]
        I_f[ys, xs, :] = region_I
         
    return (M_f, I_f, B_f)

plt.close('all')

""" TEST pour extraction """
"""
I1 = plt.imread('qr-code-wall.png')

x = [52, 246, 264, 32]
y = [56, 16, 239, 246]
I2 = homography_extraction(I1, x, y, 200, 200)

plt.imshow(I1, cmap='gray')
plt.figure()
plt.imshow(I2, cmap='gray') 

"""

""" TEST POUR PROJECTION"""
"""
I3 = plt.imread('affiche_exterieur.jpg')
I4 = plt.imread('image_rgb.jpg')

# Affiche image pour clic

plt.imshow(I3, cmap='gray')
plt.title("Cliquez sur les 4 points de l'image")
points = plt.ginput(4)
plt.close()

x_2 = np.array([p[0] for p in points])
y_2 = np.array([p[1] for p in points])

I5 = homography_projection(I4, I3, x_2, y_2)
plt.imshow(I5)
"""


""" Test pour projection croisée"""
"""
I6 = plt.imread('affiche_exterieur.jpg')

plt.imshow(I6)
plt.axis('off')
plt.title("Cliquez sur les 4 points de l'image")
points = plt.ginput(4)
plt.close()


plt.imshow(I6, cmap='gray')
plt.axis('off')
plt.title("Cliquez sur les 4 points de l'image")
points_2 = plt.ginput(4)
plt.close()

x_3 = np.array([p[0] for p in points])
y_3 = np.array([p[1] for p in points])
x_4 = np.array([p[0] for p in points_2])
y_4 = np.array([p[1] for p in points_2])

I7 = homography_cross_projection(I6,x_3,y_3,x_4,y_4)
plt.axis('off')
plt.imshow(I7)
"""

""" Test I_to_MIB """
"""
I8 = plt.imread('affiche_exterieur.jpg')

(M, I, B) = I_to_MIB(I8)
print("M :", M)
print("I :", I)
print("B :", B)
"""

""" Test MIB_Homography """
"""
I9 = plt.imread('affiche_exterieur.jpg')
MIB0 = I_to_MIB(I9)

x1 = np.array([0, 10, 10, 0], dtype=float)
y1 = np.array([0, 0, 10, 10], dtype=float)

# Rotation 45° autour du centre (5,5)
theta = np.deg2rad(45)
c, s = np.cos(theta), np.sin(theta)
cx, cy = 5.0, 5.0

def rotate_points(x, y, cx, cy, c, s):
    xt = x - cx
    yt = y - cy
    xr = c*xt - s*yt + cx
    yr = s*xt + c*yt + cy
    return xr, yr

x2, y2 = rotate_points(x1, y1, cx, cy, c, s)

H = homography_estimate(x1, y1, x2, y2)

(M, I, B) = MIB_homography(MIB0, H)
print("M :", M)
print("I :", I)
print("B :", B)

plt.imshow(I)
plt.show()
"""


""" Test MIB_Fusion """
I9  = plt.imread('TestFusion1.png')
I10 = plt.imread('TestFusion2.png')

plt.figure()
plt.imshow(I9)
plt.axis('off')
plt.title("Image 1 (reference) : cliquez 4 points")
pts1 = plt.ginput(4, timeout=0)
plt.close()

plt.figure()
plt.imshow(I10)
plt.axis('off')
plt.title("Image 2 : cliquez les 4 points correspondants")
pts2 = plt.ginput(4, timeout=0)
plt.close()

x1 = np.array([p[0] for p in pts1], dtype=float)
y1 = np.array([p[1] for p in pts1], dtype=float)
x2 = np.array([p[0] for p in pts2], dtype=float)
y2 = np.array([p[1] for p in pts2], dtype=float)

H = homography_estimate(x2, y2, x1, y1)

MIB1 = I_to_MIB(I9)
MIB2 = I_to_MIB(I10)

MIB2_warp = MIB_homography(MIB2, H)

M_f, I_f, B_f = MIB_fusion(MIB1, MIB2_warp)

print("B_f (bounding box globale) =", B_f)
print("I_f shape =", I_f.shape)

plt.figure()
plt.title("Fusion (MIB_fusion)")
plt.imshow(I_f.astype(I9.dtype) if I_f.dtype != I9.dtype else I_f)

plt.show()