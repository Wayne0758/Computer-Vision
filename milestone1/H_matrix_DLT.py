import numpy as np
from Homography import Homography

HMG = Homography()

pts_A = HMG.read_coords_from_txt("milestone1\Data\A.txt")
pts_B = HMG.read_coords_from_txt("milestone1\Data\B.txt")
pts_C = HMG.read_coords_from_txt("milestone1\Data\C.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
H1 = HMG.compute_homography_dlt_normalized(pts_A[:10], pts_B[:10])
H2 = HMG.compute_homography_dlt_normalized(pts_B[:10], pts_C[:10])
H3 = HMG.compute_homography_dlt_normalized(pts_A[:10], pts_C[:10])
print("Estimated Homography Matrix A-B:\n", H1)
print("Estimated Homography Matrix B-C:\n", H2)
print("Estimated Homography Matrix A-C:\n", H3)