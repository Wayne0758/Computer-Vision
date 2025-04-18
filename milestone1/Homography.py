import numpy as np
import os

class Homography:
    def normalize_points_2d(self, points):
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

    def compute_homography_dlt_normalized(self, src_pts, dst_pts):
        """
        Compute homography using normalized DLT algorithm.
        Inputs: src_pts and dst_pts are (N, 2) arrays
        """
        N = src_pts.shape[0]

        # Normalize
        src_norm, T_src = self.normalize_points_2d(src_pts)
        dst_norm, T_dst = self.normalize_points_2d(dst_pts)

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

    def read_coords_from_txt(self, filename):
        coords = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        x, y = map(int, line.strip().split(','))
                        coords.append((x, y))
                    except:
                        continue
        return coords