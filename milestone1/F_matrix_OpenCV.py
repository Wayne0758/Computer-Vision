import numpy as np
import cv2
import matplotlib.pyplot as plt
from Homography import Homography
from FundamentalMat import FundamentalMat

HMG = Homography()
FM = FundamentalMat()
pts_A = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/a.txt")
pts_B = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/b.txt")
pts_C = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/c.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
F1 = cv2.findFundamentalMat(pts_A[:8], pts_B[:8], method=cv2.FM_8POINT)[0]
F2 = cv2.findFundamentalMat(pts_B[:8], pts_C[:8], method=cv2.FM_8POINT)[0]
F3 = cv2.findFundamentalMat(pts_A[:8], pts_C[:8], method=cv2.FM_8POINT)[0]
print("Estimated Fundamental Matrix F1:\n", F1)
print("Estimated Fundamental Matrix F2:\n", F2)
print("Estimated Fundamental Matrix F3:\n", F3)

img_A = cv2.imread('G:\Machine Learning\CV\IMG_1223.jpg')
img_B = cv2.imread('G:\Machine Learning\CV\IMG_1224.jpg')
FM.draw_epipolar_lines(img_A, img_B, pts_A[8:], pts_B[8:], F1)
