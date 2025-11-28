#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:17:27 2025
@author: dridri & Gosan
"""

import numpy as np

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




