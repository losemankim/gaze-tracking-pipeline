from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml
import math
import time

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
    # direction_vector는 방향벡터입니다. 방향벡터는 광선의 방향을 나타냅니다. 이것의 정확도를 높이기 위해서는 광선의 방향을 나타내는 벡터를 더 정확하게 계산해야 합니다.
    
    a11 = direction_vector[1]  #a11은 방향벡터의 y좌표입니다.
    a12 = -direction_vector[0] #a12는 방향벡터의 x좌표입니다.
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1] #b1은 방향벡터의 y좌표와 지원벡터의 x좌표의 곱에서 방향벡터의 x좌표와 지원벡터의 y좌표의 곱을 뺀 값입니다. 이것은 
    #support_vector와 direction_vector가 평행하지 않다는 가정하에 광선의 방향을 나타내는 벡터를 더 정확하게 계산할 수 있습니다.
    a22 = direction_vector[2] #a22
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)#matrix는 행렬입니다. 행렬은 행렬의 해를 구하는데 있어서 중요한 역할을 합니다.
    bias = np.insert(line_b, 2, plane_d, axis=0) #bias는 편향입니다. 편향은 행렬의 해를 구하는데 있어서 중요한 역할을 합니다.

    return np.linalg.solve(matrix, bias).reshape(3) #linalg.solve는 행렬의 해를 구하는 함수입니다.#reshape는 행렬의 크기를 바꾸는 함수입니다.


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
    # print(result[0])
    # print(result[1])
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] /2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])#
    
    result_y = result[1]
    result_y = result_y   # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])
    
    return tuple(np.asarray([result_x, result_y]).round().astype(int))

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
def blinkRatio(landmarks, right_indices):
    rh_right = landmarks[right_indices[0]]#눈썹의 오른쪽 끝
    rh_left = landmarks[right_indices[8]]#눈썹의 왼쪽 끝
    rv_top = landmarks[right_indices[12]]#눈썹의 위쪽 끝
    rv_bottom = landmarks[right_indices[4]]#눈썹의 아래쪽 끝
    rhDistance = euclaideanDistance(rh_right, rh_left)#눈썹의 오른쪽 끝과 왼쪽 끝의 거리
    rvDistance = euclaideanDistance(rv_top, rv_bottom)#눈썹의 위쪽 끝과 아래쪽 끝의 거리
    reRatio = rhDistance/rvDistance#눈썹의 오른쪽 끝과 왼쪽 끝의 거리와 눈썹의 위쪽 끝과 아래쪽 끝의 거리의 비율
    ratio = reRatio#ratio는 눈썹의 오른쪽 끝과 왼쪽 끝의 거리와 눈썹의 위쪽 끝과 아래쪽 끝의 거리의 비율입니다.
    return ratio 
def landmarksDetection(img, results): #얼굴의 랜드마크를 찾는 함수
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    return mesh_coord

def blinkRatio2(face_model_all_transformed, LEFT_EYE):
    # 1. 좌표 중에서 가장 왼쪽 포인트와 가장 오른쪽 포인트를 찾는다.
    left_eye_coords = face_model_all_transformed[LEFT_EYE, :]
    leftmost_point = np.argmin(left_eye_coords[:, 0])
    rightmost_point = np.argmax(left_eye_coords[:, 0])
    
    # 2. 눈의 가로 길이를 계산한다.
    eye_width = np.abs(left_eye_coords[rightmost_point, 0] - left_eye_coords[leftmost_point, 0])
    
    # 3. 눈을 감았는지 여부를 판단한다.
    top_point = np.argmin(left_eye_coords[:, 1])
    bottom_point = np.argmax(left_eye_coords[:, 1])
    eye_height = np.abs(left_eye_coords[top_point, 1] - left_eye_coords[bottom_point, 1])
    ratio = eye_height / eye_width
def split_screen(pixel,split_num):
    pixel_x = pixel[0]
    width=[] 
    for i in range(1,split_num+1):
        width.append(pixel_x/split_num*i)

    return width
def compare_pw(pw1,pw2):
    #리스트 형식으로 받아온 비밀번호를 비교하는 함수
    if pw1==pw2:
        return True