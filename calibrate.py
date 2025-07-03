import datetime
import glob
import multiprocessing
import os
import shutil
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

from CamOperation_class import ImageQueue, ImageSetQueue, CameraInterface

def calibrate_right_left(datafile_right, datafile_left):
    # 🎯 ChArUco board parameters
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 13
    SQUARE_SIZE = 40.0  # mm
    MARKER_SIZE = 30.0  # mm
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((CHARUCO_COLS, CHARUCO_ROWS), SQUARE_SIZE, MARKER_SIZE, ARUCO_DICT)

    # 🔍 Load stereo image pairs
    left_images = sorted(glob.glob(datafile_left))
    right_images = sorted(glob.glob(datafile_right))
    assert len(left_images) == len(right_images), "Mismatch in left/right image count!"

    # 📌 Storage for calibration data
    all_corners_l, all_ids_l = [], []
    all_corners_r, all_ids_r = [], []
    obj_points = []

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

    for left_img_path, right_img_path in zip(left_images, right_images):
        img_l = cv2.imread(left_img_path)
        img_r = cv2.imread(right_img_path)

        if img_l is None or img_r is None:
            print(f"❌ Failed to read images: {left_img_path}, {right_img_path}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # 🔍 Detect ArUco markers
        marker_corners_l, marker_ids_l, _ = detector.detectMarkers(gray_l)
        marker_corners_r, marker_ids_r, _ = detector.detectMarkers(gray_r)

        if marker_ids_l is None or marker_ids_r is None:
            print(f"❌ No markers detected in images: {left_img_path}, {right_img_path}")
            continue  # Skip images without detections

        # 可视化检测到的ArUco标记
        img_l_markers = cv2.aruco.drawDetectedMarkers(img_l.copy(), marker_corners_l, marker_ids_l)
        img_r_markers = cv2.aruco.drawDetectedMarkers(img_r.copy(), marker_corners_r, marker_ids_r)
        cv2.imshow("Left Markers", img_l_markers)
        cv2.imshow("Right Markers", img_r_markers)
        cv2.waitKey(50)  # 显示500毫秒

        # 🔹 Interpolate ChArUco corners
        _, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(marker_corners_l, marker_ids_l,
                                                                                  gray_l,
                                                                                  board)
        _, charuco_corners_r, charuco_ids_r = cv2.aruco.interpolateCornersCharuco(marker_corners_r, marker_ids_r,
                                                                                  gray_r,
                                                                                  board)

        if charuco_corners_l is None or charuco_corners_r is None:
            print(f"❌ No ChArUco corners interpolated in images: {left_img_path}, {right_img_path}")
            continue

        # 🔎 Find common ChArUco IDs in both images
        ids_l = set(charuco_ids_l.flatten())
        ids_r = set(charuco_ids_r.flatten())
        common_ids = list(ids_l.intersection(ids_r))

        if len(common_ids) < 10:
            continue  # Skip if too few common points

        # 📌 Keep only common points
        common_corners_l, common_corners_r, common_obj_points = [], [], []
        for i, charuco_id in enumerate(charuco_ids_l.flatten()):
            if charuco_id in common_ids:
                common_corners_l.append(charuco_corners_l[i])

        for i, charuco_id in enumerate(charuco_ids_r.flatten()):
            if charuco_id in common_ids:
                common_corners_r.append(charuco_corners_r[i])

        for charuco_id in common_ids:
            common_obj_points.append(board.getChessboardCorners()[charuco_id])

        all_corners_l.append(np.array(common_corners_l, dtype=np.float32))
        all_corners_r.append(np.array(common_corners_r, dtype=np.float32))
        obj_points.append(np.array(common_obj_points, dtype=np.float32))

    print(f"✅ Collected {len(all_corners_l)} valid stereo pairs.")

    # print(obj_points[0])

    # 🎯 Calibration function
    def calibrate_camera(obj_points, img_points, img_size):
        return cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    img_size = gray_l.shape[::-1]  # (width, height)
    print(img_size)
    ret_l, camera_matrix_l, dist_coeffs_l, _, _ = calibrate_camera(obj_points, all_corners_l, img_size)
    ret_r, camera_matrix_r, dist_coeffs_r, _, _ = calibrate_camera(obj_points, all_corners_r, img_size)

    # 🔹 Stereo Calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    # 从右相机坐标系到左相机坐标系的变换
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, all_corners_l, all_corners_r,
        camera_matrix_l, dist_coeffs_l,
        camera_matrix_r, dist_coeffs_r,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )

    # 🎯 Stereo Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix_l, dist_coeffs_l,
        camera_matrix_r, dist_coeffs_r,
        img_size, R, T, alpha=0
    )
    return camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F, R1, R2, P1, P2, Q
    # 💾 Save all parameters

def calibrate_color_left(datafile_color, datafile_left, K_left, D_left):
    # 🎯 ChArUco board parameters
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 13
    SQUARE_SIZE = 40.0  # mm
    MARKER_SIZE = 30.0  # mm
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((CHARUCO_COLS, CHARUCO_ROWS), SQUARE_SIZE, MARKER_SIZE, ARUCO_DICT)

    # 🔍 Load stereo image pairs
    left_images = sorted(glob.glob(datafile_left))
    color_images = sorted(glob.glob(datafile_color))
    assert len(left_images) == len(color_images), "Mismatch in left/color image count!"

    # 📌 Storage for calibration data
    all_corners_l, all_ids_l = [], []
    all_corners_c, all_ids_c = [], []
    obj_points = []

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

    for left_img_path, color_img_path in zip(left_images, color_images):
        img_l = cv2.imread(left_img_path)
        img_c = cv2.imread(color_img_path)

        if img_l is None or img_c is None:
            print(f"❌ Failed to read images: {left_img_path}, {color_img_path}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        # 🔍 Detect ArUco markers
        marker_corners_l, marker_ids_l, _ = detector.detectMarkers(gray_l)
        marker_corners_c, marker_ids_c, _ = detector.detectMarkers(gray_c)

        if marker_ids_l is None or marker_ids_c is None:
            print(f"❌ No markers detected in images: {left_img_path}, {color_img_path}")
            continue  # Skip images without detections

        # 可视化检测到的ArUco标记
        img_l_markers = cv2.aruco.drawDetectedMarkers(img_l.copy(), marker_corners_l, marker_ids_l)
        img_c_markers = cv2.aruco.drawDetectedMarkers(img_c.copy(), marker_corners_c, marker_ids_c)
        cv2.imshow("Left Markers", img_l_markers)
        cv2.imshow("Color Markers", img_c_markers)
        cv2.waitKey(50)  # 显示500毫秒

        # 🔹 Interpolate ChArUco corners
        _, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(marker_corners_l, marker_ids_l,
                                                                                  gray_l,
                                                                                  board)
        _, charuco_corners_c, charuco_ids_c = cv2.aruco.interpolateCornersCharuco(marker_corners_c, marker_ids_c,
                                                                                  gray_c,
                                                                                  board)

        if charuco_corners_l is None or charuco_corners_c is None:
            print(f"❌ No ChArUco corners interpolated in images: {left_img_path}, {color_img_path}")
            continue

        # 🔎 Find common ChArUco IDs in both images
        ids_l = set(charuco_ids_l.flatten())
        ids_c = set(charuco_ids_c.flatten())
        common_ids = list(ids_l.intersection(ids_c))

        if len(common_ids) < 10:
            continue  # Skip if too few common points

        # 📌 Keep only common points
        common_corners_l, common_corners_c, common_obj_points = [], [], []
        for i, charuco_id in enumerate(charuco_ids_l.flatten()):
            if charuco_id in common_ids:
                common_corners_l.append(charuco_corners_l[i])

        for i, charuco_id in enumerate(charuco_ids_c.flatten()):
            if charuco_id in common_ids:
                common_corners_c.append(charuco_corners_c[i])

        for charuco_id in common_ids:
            common_obj_points.append(board.getChessboardCorners()[charuco_id])

        all_corners_l.append(np.array(common_corners_l, dtype=np.float32))
        all_corners_c.append(np.array(common_corners_c, dtype=np.float32))
        obj_points.append(np.array(common_obj_points, dtype=np.float32))

    print(f"✅ Collected {len(all_corners_l)} valid stereo pairs.")

    # print(obj_points[0])

    # 🎯 Calibration function
    def calibrate_camera(obj_points, img_points, img_size):
        return cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    img_size = gray_l.shape[::-1]  # (width, height)
    print(img_size)
    # ret_l, camera_matrix_l, dist_coeffs_l, _, _ = calibrate_camera(obj_points, all_corners_l, img_size)
    camera_matrix_l, dist_coeffs_l = K_left, D_left
    ret_c, camera_matrix_c, dist_coeffs_c, _, _ = calibrate_camera(obj_points, all_corners_c, img_size)

    # 🔹 Stereo Calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    # 从右相机坐标系到左相机坐标系的变换
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, all_corners_l, all_corners_c,
        camera_matrix_l, dist_coeffs_l,
        camera_matrix_c, dist_coeffs_c,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )

    # 🎯 Stereo Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix_l, dist_coeffs_l,
        camera_matrix_c, dist_coeffs_c,
        img_size, R, T, alpha=0
    )
    return camera_matrix_c, dist_coeffs_c, R, T, E, F, R1, R2, P1, P2, Q
    # 💾 Save all parameters
def open_cam(queue, i):
    print(f"Starting camera{i} process")
    cam_interface = CameraInterface(i)
    cam_interface.enum_device()
    cam_interface.open_devices()
    cam_interface.start_grabbing(queue)

def collect_images(shm_queue, rgb_queue):
    def ensure_folder_empty(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            # 清空文件夹中的所有内容
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    flag = True
    left_save_folder = "dataset/left"
    right_save_folder = "dataset/right"
    rgb_save_folder = "dataset/rgb"
    ensure_folder_empty(left_save_folder)
    ensure_folder_empty(right_save_folder)
    ensure_folder_empty(rgb_save_folder)
    while flag:
        retval = shm_queue.get_image_set()
        left_time = None
        left_image = None
        right_image = None
        rgb_image = None
        if retval is not None:
            image, left_time = retval
            left_image, right_image = image
        else:
            continue
        if left_image is not None and right_image is not None:
            # tip_point, assist_point = image_process(right_image, left_image, img_process, needle_queue)
            # 将左右图像拼接成一幅图像
            combined_image = np.hstack((left_image, right_image))
            rgb_image = rgb_queue.get_image_calibrate(left_time)
            cv2.imshow('Combined Image', combined_image)
            key = cv2.waitKey(1)  # 等待1毫秒，以便窗口刷新
            # 检测按键 's' 保存图像
            if key == ord('s'):

                if rgb_image is not None:
                    # 指定保存图像的文件夹路径
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    left_image_path = os.path.join(left_save_folder, f'left_image_{timestamp}.png')
                    right_image_path = os.path.join(right_save_folder, f'right_image_{timestamp}.png')
                    rgb_image_path = os.path.join(rgb_save_folder, f'rgb_image_{timestamp}.png')
                    cv2.imwrite(left_image_path, left_image)
                    cv2.imwrite(right_image_path, right_image)
                    cv2.imwrite(rgb_image_path, rgb_image)
                    print(f'Images saved at {timestamp}')
            if key == ord('q'):
                flag = False
    left_datafile = "dataset/left/*.png"
    right_datafile = "dataset/right/*.png"
    rgb_datafile = "dataset/rgb/*.png"
    camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F, R1, R2, P1, P2, Q = calibrate_right_left(
        datafile_right=right_datafile, datafile_left=left_datafile)
    print('K_l', camera_matrix_l)
    print('K_r', camera_matrix_r)
    print('D_l', dist_coeffs_l)
    print('D_r', dist_coeffs_r)
    print('R', R)
    print('T', T)
    np.savez("right_to_left_calib_params.npz",
             camera_matrix_l=camera_matrix_l, dist_coeffs_l=dist_coeffs_l,
             camera_matrix_r=camera_matrix_r, dist_coeffs_r=dist_coeffs_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print("✅ Calibration saved as right_to_left_calib_params.npz")
    camera_matrix_c, dist_coeffs_c, R, T, E, F, R1, R2, P1, P2, Q = calibrate_color_left(
        datafile_color=rgb_datafile, datafile_left=left_datafile, K_left=camera_matrix_l, D_left=dist_coeffs_l)
    print('K_rgb', camera_matrix_c)
    print('D_rgb', dist_coeffs_c)
    print('R', R)
    print('T', T)
    np.savez("rgb_to_left_calib_params.npz",
             camera_matrix_rgb=camera_matrix_c, dist_coeffs_rgb=dist_coeffs_c,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print("✅ Calibration saved as rgb_to_left_calib_params.npz")


if __name__ == "__main__":
    queue_size = 200
    img_shape = (1024, 1280)
    shm_queue = ImageSetQueue(queue_size, img_shape, tags=("l", "r"), timestamp_max_deviation=0.05, timestamp_staple=1)
    rgb_queue = ImageQueue(queue_size, (1024, 1280, 3))
    collect_proc = multiprocessing.Process(target=collect_images,args=(shm_queue, rgb_queue))
    camera_r_proc = multiprocessing.Process(target=open_cam, args=(shm_queue, 0))
    camera_l_proc = multiprocessing.Process(target=open_cam, args=(shm_queue, 1))
    camera_rgb_proc = multiprocessing.Process(target=open_cam, args=(rgb_queue, 2))
    collect_proc.start()
    camera_r_proc.start()
    camera_l_proc.start()
    camera_rgb_proc.start()
    collect_proc.join()
    camera_r_proc.join()
    camera_l_proc.join()
    camera_rgb_proc.join()
