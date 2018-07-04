import numpy as np 
from math import log, ceil, atan, asin, sqrt, cos, sin

def getRotationAndTranslationMatrices(pts11,pts12,pts21,pts22):
    x1,y1 = pts11
    xp1,yp1 = pts21
    x2,y2 = pts12
    xp2,yp2 = pts22
    if x2-x1 == 0:
        #No translation needed
        if y2-y1 == 0:
            return np.identity(2), np.array([0,0])
        try:
            theta = asin((xp2-xp1)/(y2-y1))
        except:
            theta = 0
    else:
        alpha = atan(y2-y1)/(x2-x1)
        theta = (xp2-xp1)/sqrt((x2-x1)**2+(y2-y1)**2) - alpha
    t1 = xp1 - (x1*cos(theta) + y1*sin(theta))
    t2 = yp1 - (y1*cos(theta) - x1*sin(theta))

    rotate = np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    translate = np.array([t1,t2])
    return rotate, translate

def RANSAC(pts1,pts2,N=1000,e=1.0,p=0.99,d=1):
    if len(pts1) != len(pts2):
        return "pts1 must have the same length as pts2"
    s = 2
    best_A = 0
    best_B = 0
    best_num_in = 0

    num_tot = len(pts1)
    samp_count = 0
    while N > samp_count:
        inds = np.random.choice(num_tot,size=num_tot,replace=False)
        R, T = getRotationAndTranslationMatrices(pts1[inds[0]],pts1[inds[1]],pts2[inds[0]],pts2[inds[1]])
        num_in = 0
        for i in inds:
            x1,y1 = np.dot(pts1[i], R) + T
            x2,y2 = pts2[i]
            if sqrt((x2-x1)**2 + (y2-y1)**2) < d:
                num_in += 1
        if num_in > best_num_in:
            best_num_in = num_in
            best_R = R 
            best_T = T
        e0 = 1-float(num_in)/num_tot
        if e0 == 0: 
            break
        if e0 < e:
            e = e0
            N = ceil(log(1-p)/log(1-(1-e)**s))
        samp_count += 1

    R,T,num_in = best_R,best_T,best_num_in
    inliers1 = np.zeros((num_in,2))
    inliers2 = np.zeros((num_in,2))
    j = 0
    for i,pt in enumerate(pts1):
        x1,y1 = np.dot(pt, R) + T
        x2,y2 = pts2[i]
        if sqrt((x2-x1)**2 + (y2-y1)**2) < d:
            inliers1[j] = pt
            inliers2[j] = pts2[i]
            j += 1
    return inliers1, inliers2

