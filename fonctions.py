#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:17:27 2025
@author: dridri & Gosan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def perimetre(pt1, pt2, pt3):
    p = (math.dist(pt1, pt2) + math.dist(pt2, pt3)) * 2 

    return p

def aire(pt1, pt2, pt3, pt4):
    x = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y = [pt1[1], pt2[1], pt3[1], pt4[1]]
    a = 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0] - 
                 (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0]))
    return a

def aire_carre_3pts(pt1, pt2, pt3):
    """
    Calcule l'aire à partir de 3 points d'un carré.
    On suppose que pt2 est le sommet de l'angle droit (entre pt1 et pt3).
    """
    # 1. Estimation du 4ème point (pt4)
    # Formule vectorielle : pt4 = pt1 + pt3 - pt2
    pt4_x = pt1[0] + pt3[0] - pt2[0]
    pt4_y = pt1[1] + pt3[1] - pt2[1]
    pt4 = (pt4_x, pt4_y)

    # 2. Utilisation de ta formule du lacé avec les 4 points
    x = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y = [pt1[1], pt2[1], pt3[1], pt4[1]]
    
    a = 0.5 * abs(x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0] - 
                 (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0]))
    return a

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

    if len(I1.shape) == 3:
        I2 = np.zeros((h, w, I1.shape[2]), dtype=I1.dtype)
    else :
        I2 = np.zeros((h, w), dtype=I1.dtype)

    H = homography_estimate([0, w-1, w-1, 0], [0, 0, h-1, h-1], x, y)
    
    for (i, j) in np.ndindex((h, w)):
        x_ext, y_ext = homography_apply(H, [j], [i])
        x_ext = (int)(x_ext[0])
        y_ext = (int)(y_ext[0])
        if len(I1.shape) == 2:
            I2[i,j] = I1[y_ext, x_ext]
        else :
            I2[i,j,:] = I1[y_ext, x_ext,:]

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

if len(sys.argv) <2:
    print("Usage: python3 fonctions.py <test_name>")
    print("Available tests: extraction, projection, cross_projection, I_to_MIB, MIB_homography, MIB_fusion")
    print("Other usage : python3 fonctions.py project <number of images> <image1> <image2> ... <imageN>")
    sys.exit(1)

commande = sys.argv[1]

# --- TEST POUR EXTRACTION ---
if commande == "extraction":
    I1 = plt.imread('qr-code-wall.png')
    x = [52, 246, 264, 32]
    y = [56, 16, 239, 246]
    I2 = homography_extraction(I1, x, y, 200, 200)
    
    plt.subplot(1, 2, 1)
    plt.imshow(I1, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(I2, cmap='gray')
    plt.title("Extraction")
    plt.show()

# --- TEST POUR PROJECTION ---
elif commande == "projection":
    I3 = plt.imread('affiche_exterieur.jpg')
    I4 = plt.imread('image_rgb.jpg')

    plt.imshow(I3)
    plt.title("Cliquez sur les 4 points de destination")
    points = plt.ginput(4)
    plt.close()

    x_2 = np.array([p[0] for p in points])
    y_2 = np.array([p[1] for p in points])

    I5 = homography_projection(I4, I3, x_2, y_2)
    plt.imshow(I5)
    plt.axis('off')
    plt.show()

# --- TEST POUR PROJECTION CROISÉE ---
elif commande == "cross_projection":
    I6 = plt.imread('affiche_exterieur.jpg')

    plt.imshow(I6)
    plt.title("Cliquez sur les 4 points source")
    points = plt.ginput(4)
    plt.close()

    plt.imshow(I6)
    plt.title("Cliquez sur les 4 points destination")
    points_2 = plt.ginput(4)
    plt.close()

    x_3 = np.array([p[0] for p in points])
    y_3 = np.array([p[1] for p in points])
    x_4 = np.array([p[0] for p in points_2])
    y_4 = np.array([p[1] for p in points_2])

    I7 = homography_cross_projection(I6, x_3, y_3, x_4, y_4)
    plt.imshow(I7)
    plt.axis('off')
    plt.show()

# --- TEST POUR I_to_MIB ---
elif commande == "I_to_MIB":
    I8 = plt.imread('affiche_exterieur.jpg')
    (M, I, B) = I_to_MIB(I8)
    print(f"M : {M}\nI : {I}\nB : {B}")

# --- TEST POUR MIB_HOMOGRAPHY ---
elif commande == "MIB_homography":
    I9 = plt.imread('affiche_exterieur.jpg')
    MIB0 = I_to_MIB(I9)

    x1 = np.array([0, 10, 10, 0], dtype=float)
    y1 = np.array([0, 0, 10, 10], dtype=float)

    # Rotation 45° autour du centre (5,5)
    theta = np.deg2rad(45)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = 5.0, 5.0

    xt = x1 - cx
    yt = y1 - cy
    x2 = c*xt - s*yt + cx
    y2 = s*xt + c*yt + cy

    H = homography_estimate(x1, y1, x2, y2)
    (M, I, B) = MIB_homography(MIB0, H)
    
    print(f"M : {M}\nB : {B}")
    plt.imshow(I)
    plt.show()

# --- TEST POUR MIB_FUSION ---
elif commande == "MIB_fusion":
    I9  = plt.imread('TestFusion1.png')
    I10 = plt.imread('TestFusion2.png')

    plt.imshow(I9)
    plt.title("Image 1 (reference) : cliquez 4 points")
    pts1 = plt.ginput(4, timeout=0)
    plt.close()

    plt.imshow(I10)
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
    plt.title("Fusion (MIB_fusion)")
    plt.imshow(I_f.astype(I9.dtype) if I_f.dtype != I9.dtype else I_f)
    plt.show()

# --- pour le challenge ---
elif commande == "challenge":
    I = plt.imread('challenge1.png')
    plt.imshow(I)
    plt.title("Cliquez sur les 4 points pour extraire")
    points = plt.ginput(4, timeout=0)
    plt.axis('off')
    plt.close()
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    I_extracted = homography_extraction(I, x, y, 500, 500)
    plt.imshow(I_extracted)
    plt.title("Image extraite")
    plt.axis('off')

    resultats = []
    for i in range(9):
        plt.imshow(I_extracted)
        plt.title(f"Carré {i+1} : cliquez sur les 3 points")
        pts = plt.ginput(3, timeout=0)
        plt.axis('off')
        plt.close()

        p = perimetre(pts[0], pts[1], pts[2])
        s = aire_carre_3pts(pts[0], pts[1], pts[2])

        resultats.append({
            'id': i + 1,
            'perimetre': p,
            'surface': s
        })

    resultats.sort(key=lambda x: x['surface'], reverse=True)

    surface_max = resultats[0]['surface']
    print("\nClassement par surface décroissante :")
    for res in resultats:
        part_relative = res['surface'] / surface_max
        print(f"Carré {res['id']} : Surface = {res['surface']:.2f} (Part = {part_relative:.2f}), Périmètre = {res['perimetre']:.2f}")

# --- Pour le projet ---
elif commande == "project":
    num_images = int(sys.argv[2])
    images = []
    for i in range(num_images):
        img_path = sys.argv[3 + i]
        img = plt.imread(img_path)
        images.append(img)
    
    plt.imshow(images[0])
    plt.title(f"Cliquez sur les 4 points de l'image 1")
    points_ref = plt.ginput(4, timeout=0)
    plt.axis('off')
    plt.close()
    x_ref = np.array([p[0] for p in points_ref], dtype=float)
    y_ref = np.array([p[1] for p in points_ref], dtype=float)
    
    MIBs = []
    MIBs.append(I_to_MIB(images[0]))

    for i in range(1, num_images):
        plt.imshow(images[i])
        plt.title(f"Image {i+1} : cliquez les 4 points correspondants")
        points = plt.ginput(4, timeout=0)
        plt.axis('off')
        plt.close()

        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)

        H = homography_estimate(x, y, x_ref, y_ref)
        
        MIB_courant = I_to_MIB(images[i])
        MIBs.append(MIB_homography(MIB_courant, H))

    M_f, I_f, B_f = MIB_fusion(*MIBs)

    print("B_f (bounding box globale) =", B_f)
    plt.title("Fusion (MIB_fusion)")
    plt.imshow(I_f.astype(images[0].dtype) if I_f.dtype != images[0].dtype else I_f)
    plt.axis('off')
    plt.show()
else:
    print(f"Erreur : Le test '{commande}' n'est pas reconnu.")