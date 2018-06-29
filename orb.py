import numpy as np
import cv2 as cv
import os
from itertools import combinations, permutations
from matplotlib import pyplot as plt


maps=['01-36.pgm','06-48_97min.pgm','18-27.pgm','18-45.pgm','29-26.pgm','34-27.pgm','48-23big_boi.pgm','49-54.pgm']
# maps = ['01-36.ppm','06-48_97min.ppm','18-27.ppm','18-45.ppm','29-26.ppm','34-27.ppm','48-23big_boi.ppm','49-54.ppm']
compare = combinations(maps,2)

overlay_num = 0
path = 'zold/'
output_path = 'compare_meh/'

try: 
    os.mkdir(output_path)
except:
    pass


for pair in compare:
    map1 = cv.imread(path+pair[0] ,0)
    map2 = cv.imread(path+pair[1] ,0)

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
    matches = filter(lambda x:x.distance < 30, matches)


    points1 = []
    points2 = []
    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2) 

    so = str(overlay_num)
    while True:
        try:
            transform = cv.estimateRigidTransform(points1,points2, False)
        except:
            #No distance threshold can generate a valid transform
            #Save first 10 feature matches 
            img3 = cv.drawMatches(map1,kp1,map2,kp2,matches[:10],None, flags=2)
            plt.imshow(img3),plt.title('No Working Transform '+pair[0]+'+'+pair[1]),plt.savefig(output_path+so+'no_transform')
            transform = False
            break
        if transform is not None:
            #Valid transform found 
            break

        #Since points are sorted by distance, keep decreasing max distance to get transform
        points1 = points1[:-1]
        points2 = points2[:-1]
    
    if transform is not False:
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

        # Display 4 graphs at once
        plt.subplot(221),plt.imshow(map1, cmap='gray'),plt.title('map1 '+pair[0])
        plt.subplot(222),plt.imshow(map2, cmap='gray'),plt.title('map2 '+pair[1])
        plt.subplot(223),plt.imshow(map1_transform, cmap='gray'),plt.title('map1 transform')
        plt.subplot(224),plt.imshow(overlay, cmap='gray'),plt.title('overlaid image')
        plt.savefig(output_path+so+'overlay')
        plt.close()

    overlay_num += 1
