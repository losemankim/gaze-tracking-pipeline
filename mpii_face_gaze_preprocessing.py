from typing import Tuple

import cv2
import numpy as np


def get_matrices(camera_matrix: np.ndarray, distance_norm: int, center_point: np.ndarray, focal_norm: int, head_rotation_matrix: np.ndarray, image_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    회전, 스케일링 및 변환 매트릭스를 계산합니다.

    :param camera_matrix: 고유 카메라 매트릭스
    :param distance_norm: 카메라의 정규화된 거리
    :param center_point: 이미지의 중심 위치
    :param focal_norm: 정규화된 초점 거리
    :param head_rotation_matrix: 머리 회전
    :param image_output_size: 출력 이미지의 출력 크기
    :return: 회전, 스케일링 및 변환 매트릭스
    """
    # normalize image
    distance = np.linalg.norm(center_point)  # actual distance between center point and original camera
    z_scale = distance_norm / distance
    #cam_norm: 정규화된 카메라 매트릭스
    cam_norm = np.array([
        [focal_norm, 0, image_output_size[0] / 2],
        [0, focal_norm, image_output_size[1] / 2],
        [0, 0, 1.0],
    ])
    # scaling matrix:z축만 z-scale이 적용됩니다.
    scaling_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    forward = (center_point / distance).reshape(3) # 눈의 방향
    down = np.cross(forward, head_rotation_matrix[:, 0])# 눈의 아래 방향
    down /= np.linalg.norm(down)# 눈의 아래 방향의 크기를 1로 만듭니다.
    right = np.cross(down, forward)# 눈의 오른쪽 방향
    right /= np.linalg.norm(right)# 눈의 오른쪽 방향의 크기를 1로 만듭니다.

    rotation_matrix = np.asarray([right, down, forward])# 회전 매트릭스
    transformation_matrix = np.dot(np.dot(cam_norm, scaling_matrix), np.dot(rotation_matrix, np.linalg.inv(camera_matrix)))# 변환 매트릭스

    return rotation_matrix, scaling_matrix, transformation_matrix


def equalize_hist_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """
    RGB 이미지의 히스토그램을 균일화합니다.

    :param rgb_img: RGB 이미지
    :return: 균등화된 RGB 이미지
    """
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)  # convert from RGB color-space to YCrCb
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])  # equalize the histogram of the Y channel
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)  # convert back to RGB color-space from YCrCb
    return equalized_img


def normalize_single_image(image: np.ndarray, head_rotation, gaze_target: np.ndarray, center_point: np.ndarray, camera_matrix: np.ndarray, is_eye: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 이미지의 정규화 과정은 'is_eye'에 따라 정규화된 눈 이미지 또는 얼굴 이미지를 생성합니다.
    :param 이미지: 원본 이미지
    :param head_rotation: 머리의 회전
    :param gaze_target: 시선의 3D 대상
    :param center_point: 초점을 맞출 얼굴의 3D 점
    :param camera_matrix: 고유 카메라 매트릭스
    :param is_eye: 참이면 눈에 대한 `distance_norm` 및 `image_output_size` 값이 사용됩니다.
    :return: 정규화된 이미지, 정규화된 시선 및 회전 행렬
    """
    # normalized camera parameters
    focal_norm = 960  # focal length of normalized camera #정규화된 카메라의 초점 거리
    distance_norm = 500 if is_eye else 1600  # normalized distance between eye and camera #카메라와 눈 사이의 정규화된 거리
    image_output_size = (96, 64) if is_eye else (96, 96)  # size of cropped eye image #크롭된 눈 이미지의 크기

    # compute estimated 3D positions of the landmarks
    if gaze_target is not None:
        gaze_target = gaze_target.reshape((3, 1))

    head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
    rotation_matrix, scaling_matrix, transformation_matrix = get_matrices(camera_matrix, distance_norm, center_point, focal_norm, head_rotation_matrix, image_output_size)

    img_warped = cv2.warpPerspective(image, transformation_matrix, image_output_size)  # image normalization
    img_warped = equalize_hist_rgb(img_warped)  # equalizes the histogram (normalization)

    if gaze_target is not None:
        # normalize gaze vector
        gaze_normalized = gaze_target - center_point  # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        gaze_normalized = np.dot(rotation_matrix, gaze_normalized)
        gaze_normalized = gaze_normalized / np.linalg.norm(gaze_normalized)
    else:
        gaze_normalized = np.zeros(3)

    return img_warped, gaze_normalized.reshape(3), rotation_matrix
