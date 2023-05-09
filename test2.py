import cv2
import mediapipe as mp
import utils, math
import numpy as np
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
def img_warped(img, src, dst): #이미지를 왜곡시키는 함수
    h, status = cv2.findHomography(src, dst)
    height, width = img.shape[:2]
    img_warp = cv2.warpPerspective(img, h, (width, height))
    return img_warp
def euclaideanDistance(point, point1):#두 점 사이의 거리를 구하는 함수
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
def blinkRatio(landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2
    return ratio 
def landmarksDetection(img, results, draw=False): #얼굴의 랜드마크를 찾는 함수
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord
def main():
    source = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    while True:
        ret, img = source.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)
        if results.multi_face_landmarks:
            mesh_coord=landmarksDetection(img, results, draw=False)
            left_eye_ratio = blinkRatio(mesh_coord, LEFT_EYE, RIGHT_EYE)
            print(left_eye_ratio)
            img_warped_leye=img_warped(img, np.float32([mesh_coord[LEFT_EYE[0]], mesh_coord[LEFT_EYE[8]], mesh_coord[LEFT_EYE[12]], mesh_coord[LEFT_EYE[4]]]), np.float32([[0,0], [200,0], [0,200], [200,200]]))
            cv2.imshow["warped", img_warped_leye]
            if(left_eye_ratio>4.5):
                print("왼쪽 눈을 감았습니다.")
            for i in range(0, len(LEFT_EYE)):
                cv2.circle(img, mesh_coord[LEFT_EYE[i]], 2, (0,255,0), -1)
                cv2.putText(img, str(LEFT_EYE[i]), mesh_coord[LEFT_EYE[i]], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
            for i in range(0, len(RIGHT_EYE)):
                cv2.circle(img, mesh_coord[RIGHT_EYE[i]], 2, (0,255,0), -1)
                cv2.putText(img, str(RIGHT_EYE[i]), mesh_coord[RIGHT_EYE[i]], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
        cv2.imshow('frame', img)
        key = cv2.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
if __name__ == '__main__':
    main()