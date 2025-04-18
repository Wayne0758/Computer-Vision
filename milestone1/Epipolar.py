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
F1 = FM.compute_fundamental_matrix_normalized(pts_A[:8], pts_B[:8])
F2 = FM.compute_fundamental_matrix_normalized(pts_B[:8], pts_C[:8])
F3 = FM.compute_fundamental_matrix_normalized(pts_A[:8], pts_C[:8])

img_A = cv2.imread('G:\Machine Learning\CV\IMG_1223.jpg')
img_B = cv2.imread('G:\Machine Learning\CV\IMG_1224.jpg')
img_C = cv2.imread('G:\Machine Learning\CV\IMG_1226.jpg')
FM.draw_epipolar_lines(img_A, img_B, img_C, pts_A[8:], pts_B[8:], pts_C[8:], F1, F3)
