import numpy as np
import cv2 as cv
import os
from itertools import combinations
from matplotlib import pyplot as plt


#to compare multiple maps against each other
# maps = ['2011-01-24-06-18-27.pgm','2011-01-20-07-18-45.pgm','2011-01-21-09-01-36.pgm', '2011-01-25-06-29-26.pgm', '2011-01-27-07-49-54.pgm']
# compare = combinations(maps,2)

#for a single comparison
compare = [('2011-01-24-06-18-27.pgm','2011-01-20-07-18-45.pgm')]

folder_num = 0

for pair in compare:
	map1 = cv.imread(pair[0] ,0)
	map2 = cv.imread(pair[1] ,0)


	# Initiate STAR detector
	orb = cv.ORB_create()

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(map1,None)
	kp2, des2 = orb.detectAndCompute(map2,None)

	# create BFMatcher object
	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	# Match descriptors.
	matches = bf.match(des1,des2)

	#Creates folders to save images + text
	while True:
		try: 
			sfn = str(folder_num)
			os.mkdir('overlay_'+sfn)
			break
		except:
			folder_num += 1

	fout = open('overlay_'+sfn+'/matches.txt', 'w')
	fout.write(str(len(matches)) + ' total matches\n')


	matches = sorted(matches, key=lambda x:x.distance)
	#uncomment to implement with fixed distance threshold
	#matches = filter(lambda x:x.distance < 40, matches)

	points1 = []
	points2 = []
	for match in matches:
		points1.append(kp1[match.queryIdx].pt)
		points2.append(kp2[match.trainIdx].pt)
	points1 = np.array(points1)
	points2 = np.array(points2) 


	while True:
		try:
			transform = cv.estimateRigidTransform(points1,points2, False)
		except:
			#No distance threshold can generate a valid transform
			#Save first 10 feature matches 
			img3 = cv.drawMatches(map1,kp1,map2,kp2,matches[:10],None, flags=2)
			plt.imshow(img3),plt.title('No Working Transform'),plt.savefig(sfn + '/no_transform')
			transform = False
			break
		if transform is not None:
			#First valid transform
			fout.write(str(len(matches)) + ' matches within distance threshold ' + str(matches[-1].distance))
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
			map1 = np.pad(map1, ((0,rows2-rows1), (0,0)), 'constant', constant_values= 255)
		if cols1 > cols2:
			map2 = np.pad(map2, ((0,0),(0,cols1-cols2)), 'constant', constant_values= 255)
		else:
			map1 = np.pad(map1, ((0,0),(0,cols2-cols1)), 'constant', constant_values= 255)

		rows,cols = map1.shape

		#Apply transform 
		map1_transform = cv.warpAffine(map1,transform,(cols,rows),borderValue=255)
		#Overlay the Images with 50% transparency each
		overlay = cv.addWeighted(map1_transform, 0.5,map2, 0.5, 0)

		#Save images
		plt.imshow(map1,cmap='gray')
		plt.savefig('overlay_'+sfn+'/'+sfn+'map1')
		plt.imshow(map2, cmap='gray')
		plt.savefig('overlay_'+sfn+'/'+sfn+'map2')
		plt.imshow(map1_transform, cmap='gray')
		plt.savefig('overlay_'+sfn+'/'+sfn+'map1_transform')
		plt.imshow(overlay, cmap='gray'),
		plt.savefig('overlay_'+sfn+'/'+sfn+'overlay')
		folder_num += 1


	# Display 4 graphs at once
	# plt.subplot(221),plt.imshow(map1, cmap='gray'),plt.title('map1')
	# plt.subplot(222),plt.imshow(map2, cmap='gray'),plt.title('map2')
	# plt.subplot(223),plt.imshow(map1_transform, cmap='gray'),plt.title('map1 transform')
	# plt.subplot(224),plt.imshow(overlay, cmap='gray'),plt.title('overlay')
	# plt.show()