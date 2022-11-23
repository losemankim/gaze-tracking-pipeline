from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Gdk에서 모니터 크기를 가져옵니다.
    https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    window환경에서 실행시 에러가 발생할 수 있습니다.
    window환경에서 실행시에는 직접 모니터의 크기를 입력해야 합니다.
    :return: 모니터 너비와 높이의 튜플(mm 및 픽셀) 또는 None
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

#TargetOrientation는 Enum 클래스이다.
class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def get_camera_matrix(calibration_matrix_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    `calibration_matrix_path`에서 camera_matrix 및 dist_coefficients를 로드합니다.
    calibration_matrix_path는 `calibrate_camera.py`를 사용하여 생성된 파일입니다. eg) `calibration_matrix.yaml`
    dist_coefficients는 카메라의 렌즈 왜곡을 보정하는 데 사용됩니다.
    :param base_path: 데이터의 기본 경로
    :return: 카메라 고유 행렬 및 dist_coefficients
    """
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients


def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """
    `solvePnP`를 사용하여 `face_model`을 `face_landmarks`에 맞춥니다.
    solvePnP는 3D-2D 점 쌍을 사용하여 카메라의 위치를 계산합니다.
    face_model은 3D 모델이고 face_landmarks는 2D 이미지입니다.
    solvePnPRansac는 노이즈에 덜 민감하며, 더 많은 점 쌍을 사용하여 더 정확한 결과를 제공합니다.
    sovlePnP는 노이즈에 더 민감하며, 더 적은 점 쌍을 사용하여 더 빠른 결과를 제공합니다.
    Rodrigues는 회전 벡터를 회전 행렬로 변환합니다.
    :param camera_matrix: 카메라 고유 행렬
    :param dist_coefficients: 왜곡 계수
    :param shape: 이미지 모양
    :param results: MediaPipe FaceMesh의 출력
    :return: 카메라 좌표계의 전체 얼굴 모델
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks


def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """
    시선 2D to 3D
    피치 및 시선 3d 벡터
    피치란 시선의 방향을 나타내는 벡터이다.
    
    :param gaze: 피치 및 시선 벡터
    :return: 3d 벡터
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])


def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """
    레이 평면 교차 
    시선 광선과 모니터를 나타내는 평면 사이의 교차점을 계산합니다.
    linalg 는 선형대수를 위한 라이브러리입니다.
    :param support_vector: 시선의 지원 벡터 지원벡터란 광선의 시작점을 나타냅니다.
    :param direction_vector: 시선의 방향 벡터
    :param plane_normal: 평면의 법선
    :param plane_d: 평면의 d
    :return: 사람이 화면에서 보고 있는 3D 지점
    """
    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    return np.linalg.solve(matrix, bias).reshape(3)


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    평면 방정식
    x-y 평면의 방정식을 계산합니다.
    평면의 법선 벡터는 회전 행렬에서 z축입니다. 그리고 tmat는 비행기의 한 지점에 제공됩니다.
    assert 는 조건이 맞지 않으면 에러를 발생시킵니다.
    .shape는 행렬의 크기를 반환합니다.
    .size는 행렬의 원소의 개수를 반환합니다.
    :param rmat: 회전 행렬
    :param tmat: 변환 행렬
    :return: (a, b, c, d), 여기서 평면의 방정식은 ax + by + cz = d입니다.
    """
    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
    assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """
    화면의 포인트를 픽셀 단위로 계산합니다.
    :param monitor_mm: 모니터의 크기(mm)
    :param monitor_pixels: 모니터의 픽셀 크기
    :param 결과: 화면의 예측 지점(mm)
    :return: 화면의 포인트(픽셀 단위)
    result[0]은 화면에 있는 점의 x 좌표입니다.
    result_x는 화면의 중심을 기준으로 왼쪽이 음수, 오른쪽이 양수입니다.
    result[1]은 화면에 있는 점의 y 좌표입니다.
    result_y는 화면의 중심을 기준으로 위쪽이 음수, 아래쪽이 양수입니다.
    round()는 반올림을 합니다.
    astype()은 데이터 타입을 바꿉니다.
    asarray()는 데이터를 배열로 바꿉니다.
    tuple()은 데이터를 튜플로 바꿉니다.
    """
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2   
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])#

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])
    
    return tuple(np.asarray([result_x, result_y]).round().astype(int))
