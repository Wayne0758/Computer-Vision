import numpy as np
from Homography import Homography

HMG = Homography()
pts_A = HMG.read_coords_from_txt(filename = "milestone1\Data\A.txt")
pts_B = HMG.read_coords_from_txt(filename = "milestone1\Data\B.txt")
pts_C = HMG.read_coords_from_txt(filename = "milestone1\Data\C.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)

H1 = HMG.compute_homography_dlt_normalized(pts_A[:10], pts_B[:10]) #10 points
H2 = HMG.compute_homography_dlt_normalized(pts_B[:10], pts_C[:10])
H3 = HMG.compute_homography_dlt_normalized(pts_A[:10], pts_C[:10])
HA = np.matmul(H1,np.append(pts_A[10],1)) 
HB = np.matmul(H2,np.append(pts_B[10],1))
HC = np.matmul(H3,np.append(pts_A[10],1))
CA = [HA[0]/HA[2],HA[1]/HA[2]]
CB = [HB[0]/HB[2],HB[1]/HB[2]]
CC = [HC[0]/HC[2],HC[1]/HC[2]]
print("Validation A-B: Point in A:",pts_A[10],"Point in B:",pts_B[10],"Reconstructed Point:",CA)
print("Validation B-C: Point in B:",pts_B[10],"Point in C:",pts_C[10],"Reconstructed Point:",CB)
print("Validation A-C: Point in A:",pts_A[10],"Point in C:",pts_C[10],"Reconstructed Point:",CC)