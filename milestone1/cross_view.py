import numpy as np
import cv2
import matplotlib.pyplot as plt

from Homography import Homography
from FundamentalMat import FundamentalMat

def draw_epipolar_line(line, img_shape, color='r', label=None):
    a, b, c = line
    h, w = img_shape[:2]

    if abs(b) > 1e-6:
        x1, x2 = 0, w
        y1 = int((-c - a * x1) / b)
        y2 = int((-c - a * x2) / b)
    else:
        # 垂直线
        x1 = x2 = int(-c / a)
        y1, y2 = 0, h

    return (x1, y1), (x2, y2)

def visualize_epipolar_intersection(F_AB, F_CB, pt_A, pt_C, gt_B, image_B):
    """
    :param F_AB: Fundamental matrix from A to B
    :param F_CB: Fundamental matrix from C to B
    :param pt_A: point in A (x, y)
    :param pt_C: point in C (x, y)
    :param gt_B: ground-truth point in B (x, y)
    :param image_B: image array
    """
    pt_A_h = np.array([*pt_A, 1.0])
    pt_C_h = np.array([*pt_C, 1.0])

    l1 = F_AB @ pt_A_h  # 极线 from A
    l2 = F_CB @ pt_C_h  # 极线 from C

    # 交点 = 极线叉积
    pt_inter = np.cross(l1, l2)
    pt_inter = pt_inter / pt_inter[2]

    # 可视化
    img_vis = image_B.copy()
    h, w = img_vis.shape[:2]

    # 绘制极线
    p1, p2 = draw_epipolar_line(l1, img_vis.shape)
    img_vis = cv2.line(img_vis, p1, p2, (0, 0, 255), 5)  # 红线

    p3, p4 = draw_epipolar_line(l2, img_vis.shape)
    img_vis = cv2.line(img_vis, p3, p4, (255, 0, 0), 5)  # 蓝线

    # 预测交点
    cv2.circle(img_vis, tuple(np.int32(pt_inter[:2])), 20, (0, 0, 255), -1)  # 红点
    # Ground truth
    cv2.circle(img_vis, tuple(np.int32(gt_B)), 20, (0, 255, 0), -1)          # 绿点

    # 误差
    pixel_error = np.linalg.norm(pt_inter[:2] - gt_B)
    print(f"Pixel Error = {pixel_error:.2f} pixels")

    # 显示图像
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines and Intersection in Image B")
    plt.axis('off')
    plt.show()

HMG = Homography()
FM = FundamentalMat()
pts_A = HMG.read_coords_from_txt("milestone1\Data\A.txt")
pts_B = HMG.read_coords_from_txt("milestone1\Data\B.txt")
pts_C = HMG.read_coords_from_txt("milestone1\Data\C.txt")
pts_A = np.array(pts_A, dtype=np.float32)
pts_B = np.array(pts_B, dtype=np.float32)
pts_C = np.array(pts_C, dtype=np.float32)
F_AB = FM.compute_fundamental_matrix_normalized(pts_A[:10], pts_B[:10])
F_CB = FM.compute_fundamental_matrix_normalized(pts_C[:10], pts_B[:10])
# F3 = FM.compute_fundamental_matrix_normalized(pts_A[:10], pts_C[:10])
img_B = cv2.imread('milestone1\Data\IMG_1224.JPG')

visualize_epipolar_intersection(F_AB, F_CB, pts_A[10], pts_C[10], pts_B[10], img_B)
