import multiprocessing
import tkinter as tk
import cv2
import numpy as np

from CamOperation_class import CameraInterface, ImageSetQueue, ImageProcess, Show_vtk, PointInputApp, ImageQueue

def open_cam(queue, i):
    print(f"Starting camera{i} process")
    cam_interface = CameraInterface(i)
    cam_interface.enum_device()
    cam_interface.open_devices()
    cam_interface.start_grabbing(queue)


def display(shm_queue, needle_queue, real_world_points, img_shape):
    print("Starting display process")
    img_process = ImageProcess(datafile_r_to_l="right_to_left_calib_params.npz", datafile_rgb_to_l="rgb_to_left_calib_params.npz", real_world_points=real_world_points, img_shape=img_shape)
    while True:
        if True:
            retval = shm_queue.get_image_set()
            left_image = None
            right_image = None
            rgb_image = None
            if retval is not None:
                image, _ = retval
                left_image, right_image = image
            else:
                continue
            if left_image is not None and right_image is not None:

                tip_point, assist_point, mapped_point_uv_left = image_process(right_image, left_image, img_process, needle_queue)
                # print('tip_point', tip_point)
                if mapped_point_uv_left is not None:
                    # cv2.circle(right_image, mapped_point_uv_left, 5, (128, 128, 128), -1)
                    # cv2.imshow('right', right_image)
                    # cv2.circle(left_image, mapped_point_uv_left, 5, (0, 0, 0), -1)
                    # cv2.imshow('left', left_image)
                    key = cv2.waitKey(1)
                    # 检测按键‘c’更新靶点及路径
                    if key == ord('c'):
                        print('tip_point', tip_point)
                '''
                # 将左右图像拼接成一幅图像
                combined_image = np.hstack((left_image, right_image))
                cv2.imshow('Combined Image', combined_image)
                key = cv2.waitKey(1)  # 等待1毫秒，以便窗口刷新
                # 检测按键‘c’更新靶点及路径
                if key == ord('c'):
                    if tip_point is not None:
                        needle_dict = {'target_c': tip_point, 'assist': assist_point}
                        needle_queue.put(needle_dict)
                '''
            # else:
                # print("Do Not Get Img")


def image_process(right_image, left_image, img_process, needle_queue):
    left_frame = left_image
    right_frame = right_image

    # 应用极线校正映射
    # left_frame_rectified = cv2.remap(left_frame, img_process.map1_left, img_process.map2_left, cv2.INTER_LINEAR)
    # right_frame_rectified = cv2.remap(right_frame, img_process.map1_right, img_process.map2_right, cv2.INTER_LINEAR)
    left_frame_rectified = left_frame
    right_frame_rectified = right_frame

    # 应用色彩映射
    colored_image = cv2.applyColorMap(left_frame_rectified, cv2.COLORMAP_JET)

    # 阈值分割
    _, left_thresh = cv2.threshold(left_frame_rectified, 220, 255, cv2.THRESH_BINARY)
    _, right_thresh = cv2.threshold(right_frame_rectified, 220, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours_left, _ = cv2.findContours(left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图中标记轮廓
    left_frame_with_contours = cv2.drawContours(left_frame_rectified.copy(), contours_left, -1, (0, 255, 0), 2)
    right_frame_with_contours = cv2.drawContours(right_frame_rectified.copy(), contours_right, -1, (0, 255, 0), 2)

    # 右图像进行区域生长
    right_seeds = img_process.find_seeds(right_frame_rectified)  # 找到所有高亮度区域的中心点作为种子点
    right_result = img_process.region_growing(right_frame_rectified, right_seeds)  # 进行区域生长，得到二值图
    # cv2.imshow('right_result', right_result)
    left_seeds = img_process.find_seeds(left_frame_rectified)  # 找到所有高亮度区域的中心点作为种子点
    left_result = img_process.region_growing(left_frame_rectified, left_seeds)  # 进行区域生长，得到二值图

    # 计算质心
    right_centroids = img_process.calculate_centroids(right_result, right_frame_rectified)
    left_centroids = img_process.calculate_centroids(left_result, left_frame_rectified)
    # print("left_centroids", left_centroids)
    # print("right_centroids", right_centroids)

    # 特征点匹配
    if len(left_centroids) == 0 or len(right_centroids) == 0:
        matches = []
    else:
        # 匹配特征点
        matches = img_process.match_features(left_centroids, right_centroids)
    # print("matches", matches)

    # 打印匹配结果
    # for match in matches:
        # print(f"Left: {match[0]}, Right: {match[1]}")

    # 计算三维坐标
    if len(matches) > 0:
        calculated_3d_points, points2d_3d = img_process.calculate_3d_points(matches, right_frame_with_contours, left_frame_with_contours)
    # cv2.imshow('left_frame_with_contours', left_frame_with_contours)
    # cv2.imshow('right_frame_with_contours', right_frame_with_contours)

    # 计算仿射变换矩阵(用彩色图像验证绝对距离)
    tip_point = None
    assist_point = None
    mapped_point_uv_left = None
    if len(matches) > 0:
        colored_image, tip_point,  assist_point, mapped_point_uv_left = img_process.calculate_affine_matrix(calculated_3d_points, points2d_3d, colored_image, left_frame_with_contours, needle_queue)
    # cv2.imshow('colored_image', colored_image)
    return tip_point, assist_point, mapped_point_uv_left

def p_vtk(needle_queue, rgb_queue, real_world_points):
    Show_vtk(needle_queue, rgb_queue, real_world_points)

def input_point(needle_queue):
    root = tk.Tk()
    app = PointInputApp(root, needle_queue)
    root.mainloop()

if __name__ == "__main__":
    queue_size = 200
    img_shape = (1024, 1280)
    shm_queue = ImageSetQueue(queue_size, img_shape, tags=("l", "r"), timestamp_max_deviation=0.05, timestamp_staple=1)
    rgb_queue = ImageQueue(queue_size, (1024, 1280, 3))
    needle_queue = multiprocessing.Queue()

    # 已知的现实世界中针的四个点（单位：mm）
    real_world_points = np.array([
        [0, 0, 18],
        [40.448864, 44.315792, 18],
        [88, 0, 18],
        [41.017045, -28.59374, 18]
    ], dtype=np.float32)

    display_proc = multiprocessing.Process(target=display, args=(shm_queue, needle_queue, real_world_points, img_shape))
    camera_r_proc = multiprocessing.Process(target=open_cam, args=(shm_queue, 1))
    camera_l_proc = multiprocessing.Process(target=open_cam, args=(shm_queue, 2))
    camera_rgb_proc = multiprocessing.Process(target=open_cam, args=(rgb_queue, 0))
    vtk_proc = multiprocessing.Process(target=p_vtk, args=(needle_queue, rgb_queue, real_world_points))
    input_point_poc = multiprocessing.Process(target=input_point, args=(needle_queue,))
    display_proc.start()
    # time.sleep(10)
    camera_r_proc.start()

    camera_rgb_proc.start()

    camera_l_proc.start()
    vtk_proc.start()
    # input_point_poc.start()
    camera_r_proc.join()
    camera_l_proc.join()
    camera_rgb_proc.join()
    vtk_proc.join()
    # input_point_poc.join()



