import numpy as np
import cv2

with np.load('logs/magicpoint_synth_homoAdapt_coco/predictions/val/COCO_val2014_000000000136.npz') as data:
	pts = data['pts']

image = cv2.imread('datasets/COCO/val2014/COCO_val2014_000000000136.jpg')
for kp in pts:
	image = cv2.circle(image.copy(), (int(kp[0]/kp[2]), int(kp[1]/kp[2])), 2, (210, 32, 13), 2)

cv2.imshow("image", image)
cv2.waitKey()
