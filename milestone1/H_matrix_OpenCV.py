import numpy as np
import cv2
from Homography import Homography

HMG = Homography()
pts_A = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/a.txt")
pts_B = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/b.txt")
pts_C = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/c.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
H1, mask = cv2.findHomography(pts_A[:6], pts_B[:6], method=cv2.RANSAC, ransacReprojThreshold=3.0)
H2, mask = cv2.findHomography(pts_B[:6], pts_C[:6], method=cv2.RANSAC, ransacReprojThreshold=3.0)
H3, mask = cv2.findHomography(pts_A[:6], pts_C[:6], method=cv2.RANSAC, ransacReprojThreshold=3.0)
print("Estimated Homography Matrix 1:\n", H1)
print("Estimated Homography Matrix 2:\n", H2)
print("Estimated Homography Matrix 3:\n", H3)
HA = np.matmul(H1,np.append(pts_A[8],1))
HB = np.matmul(H2,np.append(pts_B[8],1))
HC = np.matmul(H3,np.append(pts_A[8],1))
CA = [HA[0]/HA[2],HA[1]/HA[2]]
CB = [HB[0]/HB[2],HB[1]/HB[2]]
CC = [HC[0]/HC[2],HC[1]/HC[2]]
print("Validation A-B: Point A:",pts_A[8],"Point B:",pts_B[8],"Reconstructed Point B:",CA)
print("Validation B-C: Point B:",pts_B[8],"Point C:",pts_C[8],"Reconstructed Point C:",CB)
print("Validation A-C: Point A:",pts_A[8],"Point C:",pts_C[8],"Reconstructed Point C:",CC)