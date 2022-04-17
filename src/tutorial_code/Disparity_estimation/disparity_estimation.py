import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('img1.jpg', 0)
imgR = cv2.imread('img2.jpg', 0)

stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()