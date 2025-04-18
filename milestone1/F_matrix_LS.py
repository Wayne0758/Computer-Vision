import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_points_2d(points):
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

def construct_matrix_A(pts1, pts2):
    A = []
    for (x, y), (x_p, y_p) in zip(pts1, pts2):
        A.append([x * x_p, x * y_p, x, y * x_p, y * y_p, y, x_p, y_p, 1])
    return np.array(A)

def compute_fundamental_matrix_normalized(pts1, pts2):
    pts1_norm, T1 = normalize_points_2d(pts1)
    pts2_norm, T2 = normalize_points_2d(pts2)

    A = construct_matrix_A(pts1_norm, pts2_norm)

    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_hat)
    S[2] = 0
    F_hat_rank2 = U @ np.diag(S) @ Vt

    F = T2.T @ F_hat_rank2 @ T1

    return F / F[2, 2]

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
F1 = compute_fundamental_matrix_normalized(pts_A[:8], pts_B[:8])
F2 = compute_fundamental_matrix_normalized(pts_B[:8], pts_C[:8])
F3 = compute_fundamental_matrix_normalized(pts_A[:8], pts_C[:8])
print("Estimated Fundamental Matrix F1:\n", F1)
print("Estimated Fundamental Matrix F2:\n", F2)
print("Estimated Fundamental Matrix F3:\n", F3)

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
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

    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    # Compute epipolar lines using formula
    lines_in_2 = (F @ pts1_h.T).T     # l' = F x
    lines_in_1 = (F.T @ pts2_h.T).T   # l  = F^T x'

    # Draw epipolar lines and points
    def draw_lines(img, lines, pts, color_img=True):
        h, w = img.shape[:2]
        img_color = img if color_img else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r, pt in zip(lines, pts):
            a, b, c = r
            x0, y0 = 0, int(-c / b) if b != 0 else 0
            x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img_color = cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
            img_color = cv2.circle(img_color, tuple(pt.astype(int)), 5, color, -1)
        return img_color

    img1_lines = draw_lines(img1, lines_in_1, pts1)
    img2_lines = draw_lines(img2, lines_in_2, pts2)

    # Display side-by-side
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines in Image A (from pts in B)")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines in Image B (from pts in A)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

img_A = cv2.resize(cv2.imread('G:\Machine Learning\CV\IMG_1223.jpg'), (800, 600))
img_B = cv2.resize(cv2.imread('G:\Machine Learning\CV\IMG_1224.jpg'), (800, 600))
draw_epipolar_lines(img_A, img_B, pts_A, pts_B, F1)
