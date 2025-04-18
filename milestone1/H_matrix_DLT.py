import numpy as np
from Homography import Homography

HMG = Homography()

pts_A = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/a.txt")
pts_B = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/b.txt")
pts_C = HMG.read_coords_from_txt(f"G:/Machine Learning/CV/c.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
H1 = HMG.compute_homography_dlt_normalized(pts_A[:6], pts_B[:6])
H2 = HMG.compute_homography_dlt_normalized(pts_B[:6], pts_C[:6])
H3 = HMG.compute_homography_dlt_normalized(pts_A[:6], pts_C[:6])
print("Estimated Homography Matrix A-B:\n", H1)
print("Estimated Homography Matrix B-C:\n", H2)
print("Estimated Homography Matrix A-C:\n", H3)