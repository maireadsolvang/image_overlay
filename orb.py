import numpy as np
import cv2 as cv
import os
from itertools import combinations, permutations
from matplotlib import pyplot as plt
from math import log, ceil, atan, asin, sqrt, cos, sin
from RANSAC import RANSAC

bw_maps=['01-36.pgm','06-48.pgm','18-27.pgm','18-45.pgm','29-26.pgm','48-23.pgm','49-54.pgm']

#contains only the one or two smaller rooms
rgb_only_room = ['01-36.ppm','04-17.ppm','18-27.ppm','18-45.ppm','22-35.ppm','29-26.ppm','49-54.ppm','56-58.ppm']
#contains the room(s) and area to the right
rgb_right = ['06-48.ppm','38-27.ppm','48-23.ppm']
#contains the room(s) and area to the left
rgb_left = ['57-25.ppm','38-59.ppm']
#Maps generated that do not contain the room(s)
rgb_no_room = ['false1.ppm','false2.ppm','false3.ppm','false4.ppm']
#Random rgb pictures 
rgb_rand = ['dog.jpg','star_trek.jpg']

rgb_maps = rgb_only_room + rgb_right + rgb_left
compare = list(combinations(rgb_maps,2))

correct = 0
wrong = 0
overlay_num = 0
path = 'rgb' #rgb/ or bw/


if path != '' and path[-1] != '/':
    path += '/'
if output_path != '' and output_path[-1] != '/':
    output_path += '/'
try: 
    os.mkdir(output_path)
except:
    pass


for pair in compare:
    map1 = cv.imread(path+pair[0] ,0)
    map2 = cv.imread(path+pair[1] ,0)


    rows1,cols1 = map1.shape
    rows2,cols2 = map2.shape
    map2 = np.pad(map2, 200, 'constant', constant_values=255)
    map1 = np.pad(map1, 200, 'constant', constant_values=255)


    #Apply a 9x9 Gaussian filter
    map1 = cv.GaussianBlur(map1,(9,9),0)
    map2 = cv.GaussianBlur(map2,(9,9),0)


    # Initiate STAR detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(map1,None)
    kp2, des2 = orb.detectAndCompute(map2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort matches and only use matches with fixed distance threshold
    matches = sorted(matches, key=lambda x:x.distance)
    # matches = filter(lambda x:x.distance < 30, matches)


    points1 = np.zeros((len(matches),2))
    points2 = np.zeros((len(matches),2))
    for i,match in enumerate(matches):
        points1[i] = kp1[match.queryIdx].pt
        points2[i] = kp2[match.trainIdx].pt

    #filter out outliers
    points1,points2 = RANSAC(points1,points2,d=25)

    so = str(overlay_num)

    transform = None
    
    while len(points1) > 20:
        transform = cv.estimateRigidTransform(points1,points2, False)
        if transform is not None:
            ft = transform.flatten()
            a,b,tx,c,d,ty = ft 
            sx = np.sign(a)*pow(a**2+b**2,0.5)
            sy = np.sign(d)*pow(c**2+d**2,0.5)
            #if the scale is ~1 then the transform is likely to be correct
            if 0.9 < sx < 1.1 and 0.9 < sy < 1.1:
                break
            else:
                transform = None

        #Since points are sorted by distance, keep decreasing max distance to get transform
        points1 = points1[:-1]
        points2 = points2[:-1]

    if transform is not None:
        correct+=1
        rows1, cols1 = map1.shape
        rows2, cols2 = map2.shape

        #Pad the smaller image in order to overlay them
        if rows1 > rows2:
            map2 = np.pad(map2, ((0,rows1-rows2),(0,0)), 'constant', constant_values=255)
        else:
            map1 = np.pad(map1, ((0,rows2-rows1), (0,0)), 'constant', constant_values=255)
        if cols1 > cols2:
            map2 = np.pad(map2, ((0,0),(0,cols1-cols2)), 'constant', constant_values=255)
        else:
            map1 = np.pad(map1, ((0,0),(0,cols2-cols1)), 'constant', constant_values=255)

        rows,cols = map1.shape

        #Apply transform
        map1_transform = cv.warpAffine(map1,transform,(cols,rows),borderValue=255)
        #Overlay the Images with 50% transparency each
        overlay = cv.addWeighted(map1_transform, 0.5,map2, 0.5, 0)

        #Save seperate images
        # img3 = cv.drawMatches(map1,kp1,map2,kp2,matches[:10],None, flags=2)
        # plt.imshow(img3),plt.savefig('overlay_'+sfn+'/10_matches')

        # plt.imshow(map1,cmap='gray')
        # plt.savefig('overlay_'+sfn+'/'+sfn+'map1')
        # plt.imshow(map2, cmap='gray')
        # plt.savefig('overlay_'+sfn+'/'+sfn+'map2')
        # plt.imshow(map1_transform, cmap='gray')
        # plt.savefig('overlay_'+sfn+'/'+sfn+'map1_transform')
        # plt.imshow(overlay, cmap='gray'),
        # plt.savefig('overlay_'+sfn+'/'+sfn+'overlay')

        # Save 4 graphs at once
        plt.subplot(221),plt.imshow(map1, cmap='gray'),plt.title('map1 '+pair[0])
        plt.subplot(222),plt.imshow(map2, cmap='gray'),plt.title('map2 '+pair[1])
        plt.subplot(223),plt.imshow(map1_transform, cmap='gray'),plt.title('map1 transform')
        plt.subplot(224),plt.imshow(overlay, cmap='gray'),plt.title('overlaid image')
        plt.savefig(output_path+so+'overlay')
        plt.close()
    else:
        img3 = cv.drawMatches(map1,kp1,map2,kp2,matches[:10],None, flags=2)
        plt.imshow(img3),plt.title('No Working Transform '+pair[0]+'+'+pair[1]),plt.savefig(output_path+so+'no_transform')

    overlay_num += 1
print(correct)