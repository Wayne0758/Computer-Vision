import numpy as np

def normalize_points_2d(points):
    """
    Normalize 2D points so that:
    - centroid is at (0, 0)
    - average distance to origin is sqrt(2)
    Returns: normalized points, transformation matrix T
    """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    norm_points = (T @ points_h.T).T

    return norm_points[:, :2], T

def compute_homography_dlt_normalized(src_pts, dst_pts):
    """
    Compute homography using normalized DLT algorithm.
    Inputs: src_pts and dst_pts are (N, 2) arrays
    """
    N = src_pts.shape[0]

    # Normalize
    src_norm, T_src = normalize_points_2d(src_pts)
    dst_norm, T_dst = normalize_points_2d(dst_pts)

    A = []
    for i in range(N):
        x, y = src_norm[i]
        x_p, y_p = dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
        A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])

    A = np.array(A)

    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[-1, -1]

# Assume pts_A and pts_B are numpy arrays of shape (N, 2)
# containing corresponding points in image A and image B
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
H1 = compute_homography_dlt_normalized(pts_A[:4], pts_B[:4])
H2 = compute_homography_dlt_normalized(pts_B[:4], pts_C[:4])
H3 = compute_homography_dlt_normalized(pts_A[:4], pts_C[:4])
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