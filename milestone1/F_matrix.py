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
print("Estimated Fundamental Matrix F1:\n", F1)
print("Estimated Fundamental Matrix F2:\n", F2)
print("Estimated Fundamental Matrix F3:\n", F3)