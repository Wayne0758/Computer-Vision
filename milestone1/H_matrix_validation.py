import numpy as np
from Homography import Homography

HMG = Homography()
pts_A = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/a.txt")
pts_B = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/b.txt")
pts_C = HMG.read_coords_from_txt(filename = f"G:/Machine Learning/CV/c.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
H1 = HMG.compute_homography_dlt_normalized(pts_A[:6], pts_B[:6])
H2 = HMG.compute_homography_dlt_normalized(pts_B[:6], pts_C[:6])
H3 = HMG.compute_homography_dlt_normalized(pts_A[:6], pts_C[:6])
HA = np.matmul(H1,np.append(pts_A[8],1))
HB = np.matmul(H2,np.append(pts_B[8],1))
HC = np.matmul(H3,np.append(pts_A[8],1))
CA = [HA[0]/HA[2],HA[1]/HA[2]]
CB = [HB[0]/HB[2],HB[1]/HB[2]]
CC = [HC[0]/HC[2],HC[1]/HC[2]]
print("Validation A-B: Point in A:",pts_A[8],"Point in B:",pts_B[8],"Reconstructed Point:",CA)
print("Validation B-C: Point in B:",pts_B[8],"Point in C:",pts_C[8],"Reconstructed Point:",CB)
print("Validation A-C: Point in A:",pts_A[8],"Point in C:",pts_C[8],"Reconstructed Point:",CC)