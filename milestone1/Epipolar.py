import numpy as np
import cv2
import matplotlib.pyplot as plt
from Homography import Homography
from FundamentalMat import FundamentalMat

HMG = Homography()
FM = FundamentalMat()
pts_A = HMG.read_coords_from_txt("milestone1\Data\A.txt")
pts_B = HMG.read_coords_from_txt("milestone1\Data\B.txt")
pts_C = HMG.read_coords_from_txt("milestone1\Data\C.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
F1 = FM.compute_fundamental_matrix_normalized(pts_A[:10], pts_B[:10])
F2 = FM.compute_fundamental_matrix_normalized(pts_B[:10], pts_C[:10])
F3 = FM.compute_fundamental_matrix_normalized(pts_A[:10], pts_C[:10])

img_A = cv2.imread('milestone1\Data\IMG_1223.JPG')
img_B = cv2.imread('milestone1\Data\IMG_1224.JPG')
img_C = cv2.imread('milestone1\Data\IMG_1226.JPG')
FM.draw_epipolar_lines(img_A, img_B, img_C, pts_A[10:], pts_B[10:], pts_C[10:], F1, F3)
