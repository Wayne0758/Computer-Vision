import numpy as np
import cv2
import matplotlib.pyplot as plt

pts_A = [(182, 257),
(580, 213),
(592, 539),
(161, 489),
(212, 345),
(242, 344),
(475, 336),
(520, 336),
(284, 343),
(417, 340),
(411, 517),
(275, 502)]
pts_B = [(58, 232),
(574, 247),
(588, 559),
(16, 547),
(109, 352),
(158, 354),
(468, 362),
(513, 362),
(227, 356),
(402, 361),
(397, 556),
(210, 552)]
pts_C = [(192, 185),
(609, 233),
(624, 470),
(182, 513),
(259, 312),
(307, 315),
(547, 324),
(576, 324),
(365, 315),
(501, 321),
(508, 482),
(366, 494)]
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
H1, mask = cv2.findHomography(pts_A[:4], pts_B[:4], method=cv2.RANSAC, ransacReprojThreshold=3.0)
H2, mask = cv2.findHomography(pts_B[:4], pts_C[:4], method=cv2.RANSAC, ransacReprojThreshold=3.0)
H3, mask = cv2.findHomography(pts_A[:4], pts_C[:4], method=cv2.RANSAC, ransacReprojThreshold=3.0)
print("Estimated Homography Matrix 1:\n", H1)
print("Estimated Homography Matrix 2:\n", H2)
print("Estimated Homography Matrix 3:\n", H3)
HA = np.matmul(H1,np.append(pts_A[5],1))
HB = np.matmul(H2,np.append(pts_B[5],1))
HC = np.matmul(H3,np.append(pts_A[5],1))
CA = [HA[0]/HA[2],HA[1]/HA[2]]
CB = [HB[0]/HB[2],HB[1]/HB[2]]
CC = [HC[0]/HC[2],HC[1]/HC[2]]
print("Validation A-B: Point A:",pts_A[5],"Point B:",pts_B[5],"Reconstructed Point B:",CA)
print("Validation B-C: Point B:",pts_B[5],"Point C:",pts_C[5],"Reconstructed Point C:",CB)
print("Validation A-C: Point A:",pts_A[5],"Point C:",pts_C[5],"Reconstructed Point C:",CC)