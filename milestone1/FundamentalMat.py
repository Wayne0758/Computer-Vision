import numpy as np
import cv2
import matplotlib.pyplot as plt

class FundamentalMat:
    def normalize_points_2d(self, points):
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
        norm_points = (T @ points_h.T).T[:, :2]
        return norm_points, T

    def construct_matrix_A(self, pts1, pts2):
        A = []
        for (x, y), (x_p, y_p) in zip(pts1, pts2):
            A.append([x * x_p, x_p * y, x_p, y_p * x, y * y_p, y_p, x, y, 1])
        return np.array(A)

    def compute_fundamental_matrix_normalized(self, pts1, pts2):
        pts1_norm, T1 = self.normalize_points_2d(pts1)
        pts2_norm, T2 = self.normalize_points_2d(pts2)

        A = self.construct_matrix_A(pts1_norm, pts2_norm)

        _, _, Vt = np.linalg.svd(A)
        F_hat = Vt[-1].reshape(3, 3)

        U, S, Vt = np.linalg.svd(F_hat)
        S[2] = 0
        F_hat_rank2 = U @ np.diag(S) @ Vt

        F = T2.T @ F_hat_rank2 @ T1

        return F / F[2, 2]
        
    def draw_epipolar_lines(self, img1, img2, img3, pts1, pts2, pts3, F1, F3):
        """
        Draws epipolar lines:
        - in img2 for pts1 (using F)
        - in img1 for pts2 (using F.T)

        pts1: points in image 1 (Image A), shape (N, 2)
        pts2: points in image 2 (Image B), shape (N, 2)
        F: fundamental matrix from image 1 to image 2
        """
        img1 = img1.copy()
        img2 = img2.copy()
        img3 = img3.copy()

        # Convert to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        pts3_h = np.hstack([pts3, np.ones((pts2.shape[0], 1))])

        # Compute epipolar lines using formula
        lines_in_21 = (F1 @ pts1_h.T).T     # l' = F x
        lines_in_12 = (F1.T @ pts2_h.T).T   # l  = F^T x'
        lines_in_31 = (F3 @ pts1_h.T).T     # l' = F x
        lines_in_13 = (F3.T @ pts3_h.T).T   # l  = F^T x'

        # Draw epipolar lines and points
        def draw_lines(img, lines, pts, color_img=True):
            h, w = img.shape[:2]
            img_color = img if color_img else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for r, pt in zip(lines, pts):
                a, b, c = r
                x0, y0 = 0, int(-c / b) if b != 0 else 0
                x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
                color = tuple(np.random.randint(0, 255, 3).tolist())
                img_color = cv2.line(img_color, (x0, y0), (x1, y1), color, 5)
                img_color = cv2.circle(img_color, tuple(pt.astype(int)), 5, color, -1)
            return img_color

        img1_lines = draw_lines(img2, lines_in_21, pts2)
        img2_lines = draw_lines(img3, lines_in_31, pts3)

        # Display side-by-side
        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
        plt.title("Epipolar Lines in Image B (from pts in A)")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
        plt.title("Epipolar Lines in Image C (from pts in A)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()