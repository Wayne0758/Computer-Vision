import h5py
import numpy as np
import cv2
import sqlite3
from pathlib import Path
from tqdm import tqdm

def image_pair_to_pair_id(image_id1, image_id2):
    """COLMAP规定：pair_id = image_id1 + image_id2 * 2^32"""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 + (image_id2 << 32)

def encode_matches(matches):
    """把numpy array编码成COLMAP需要的BLOB格式"""
    return matches.astype(np.uint32).tobytes()

def run_ransac(kpts1, kpts2):
    """用RANSAC基于Fundamental Matrix过滤匹配"""
    if len(kpts1) < 8:
        return None  # Fundamental Matrix至少需要8对
    F, mask = cv2.findFundamentalMat(kpts1, kpts2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence=0.999)
    if mask is None:
        return None
    mask = mask.ravel().astype(bool)
    if mask.sum() < 8:
        return None
    return mask

def build_database(features_h5_path, matches_h5_path, output_db_path):
    features_h5 = h5py.File(features_h5_path, 'r')
    matches_h5 = h5py.File(matches_h5_path, 'r')
    
    output_db_path = Path(output_db_path)
    if output_db_path.exists():
        output_db_path.unlink()  # 如果已存在，先删掉
    conn = sqlite3.connect(str(output_db_path))
    cursor = conn.cursor()

    # 建立基本表结构
    cursor.execute('''
        CREATE TABLE two_view_geometries (
            pair_id INTEGER PRIMARY KEY,
            rows BLOB,
            cols BLOB,
            data BLOB,
            config INTEGER,
            F BLOB
        )
    ''')

    image_name_to_id = {}
    for idx, name in enumerate(features_h5.keys()):
        image_name_to_id[name] = idx + 1  # image_id从1开始

    # 遍历matches
    for name1 in tqdm(matches_h5.keys(), desc="Processing matches"):
        group = matches_h5[name1]
        for name2 in group.keys():
            if name2 not in features_h5:
                continue
            if name1 not in features_h5:
                continue

            matches = group[name2]["matches0"][:]
            valid = matches > -1

            if valid.sum() < 8:
                continue  # 少于8个匹配点，不处理

            kpts1 = features_h5[name1]['keypoints'][:]
            kpts2 = features_h5[name2]['keypoints'][:]

            src_pts = kpts1[valid]
            dst_pts = kpts2[matches[valid]]

            mask = run_ransac(src_pts, dst_pts)
            if mask is None:
                continue

            src_pts = src_pts[mask]
            dst_pts = dst_pts[mask]

            if len(src_pts) < 8:
                continue

            matches_filtered = np.arange(len(src_pts), dtype=np.uint32)

            # 写入数据库
            pair_id = image_pair_to_pair_id(image_name_to_id[name1], image_name_to_id[name2])
            cursor.execute('''
                INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, F)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pair_id, None, None, encode_matches(matches_filtered), 2, None))

    conn.commit()
    conn.close()
    print(f"Saved pruned matches to {output_db_path}")

