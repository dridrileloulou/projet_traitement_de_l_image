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
    
    H = homography_estimate([0, 0, h, h], [0, w, w, 0], y, x)
    
    for (i, j) in np.ndindex((h, w)):
        x_ext, y_ext = homography_apply(H, [j], [i])
        x_ext = (int)(x_ext[0])
        y_ext = (int)(y_ext[0])
        I2[i,j] = I1[x_ext, y_ext]
    
    return I2



def homography_projection(I1, I2, x, y):
    
    h_src, w_src, _ = np.shape(I1)
    h_dst, w_dst, _ = np.shape(I2)
    I_final = I2.copy()
    
    H = homography_estimate(x, y, [0, 0, h_src, h_src], [0, w_src, w_src, 0])
    
    for (i,j) in np.ndindex((h_dst, w_dst)):
        x_proj, y_proj = homography_apply(H, [j], [i])
        x_proj = (int)(x_proj[0])
        y_proj = (int)(y_proj[0])
        
        if 0 <= x_proj < h_src and 0 <= y_proj < w_src:
            I_final[i,j,:] = I1[x_proj, y_proj, :]
        
    return I_final




#Elle ne marche pas encore
def homography_cross_projection(I, x1, y1, x2, y2) : 
    h_img, w_img, _ = np.shape(I)
    
    x_carre = np.array([0,1,1,0])
    y_carre = np.array([0,0,1,1])
    
    H1 = homography_estimate(x1, y1, x_carre, y_carre)
    
    H2 = homography_estimate(x_carre, y_carre, x2, y2)
    
    H = np.dot(H2,H1)
    
    H_inv = np.linalg.inv(H)
    
    I_final = I.copy()

    h, w = I_final.shape[:2]
    
    for i in range(h_img):
        for j in range(w_img):
            # Appliquer l'homographie inverse pour trouver la position dans l'image source
            x_proj, y_proj = homography_apply(H, [j], [i])
            x_proj = int(x_proj[0])
            y_proj = int(y_proj[0])

            # Vérifier si la position est dans les limites de l'image
            if 0 <= x_proj < h_img and 0 <= y_proj < w_img:
                I_final[i, j, :] = I[x_proj, y_proj, :]

    return I_final

    
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



""" Test sans image fournit par le chat de Mistral"""
"""
x1 = np.array([10, 90, 90, 10])
y1 = np.array([10, 10, 90, 90])
x2 = np.array([20, 80, 80, 20])
y2 = np.array([20, 20, 80, 80])

I_test = np.zeros((100, 100, 3))
I_test[10:90, 10:90, :] = 1  # Zone blanche pour visualiser

I_result = homography_cross_projection(I_test, x1, y1, x2, y2)
plt.imshow(I_result)
"""


plt.show()


