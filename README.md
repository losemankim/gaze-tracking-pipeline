# 완전한 시선 추적 파이프라인 을 위한 프레임워크
https://github.com/pperle/gaze-tracking-pipeline
원본을 한글화한 문서입니다.


아래 그림은 카메라-화면 시선 추적 파이프라인의 일반적인 표현을 보여줍니다. [1].
웹캠 이미지는 왼쪽에서 오른쪽으로 눈과 얼굴의 정규화된 이미지를 만들기 위해 전처리됩니다. 이러한 이미지는 3D 시선 벡터를 예측하는 모델에 입력됩니다.
사용자의 머리 자세가 알려지면 예측된 시선 벡터를 화면에 투사할 수 있습니다. \
이 프레임워크를 사용하면 입력 이미지만을 기반으로 화면의 시청 위치를 예측하는 실시간 접근 방식을 구현할 수 있습니다.

![camera-to-screen gaze tracking pipeline](./docs/gaze_tracking_pipeline.png)필
1. `pip install -r requirements.txt`(필요한 라이브러리를 받아오기위해 pip을 이용해 requirements받아오기)(그러나 환경에따라 쓸수없는 모듈도 존재함
한글화한 본인은 python3.6.8 window11 사용중입니다. 
requirements 를 받아올때 오류가 있다면 requirements2.txt를 이용해주세요.
2. 필요하다면, 제공된 대화형 스크립트를 사용하여 카메라를 보정하세요`python calibrate_camera.py`, see [Camera Calibration by OpenCV](https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html).
(실제로 파일 이름은 camera_calibrate.py로 되어있음)
3. 더 높은 정확도를 위해, 또한 Takahashiet al.에 설명된 대로 화면의 위치를 보정하는 것이 좋습니다. [Takahashiet al.](https://doi.org/10.2197/ipsjtcva.8.11),opencv에서 제공하는 [OpenCV and matlab implementation](https://github.com/computer-vision/takahashi2012cvpr).
4. 신뢰할 수 있는 예측을 하려면 제안된 모델을 각 사용자에 대해 특별히 보정해야 합니다. 이에 소프트웨어가 제공됩니다.[collect this calibration data](https://github.com/pperle/gaze-data-collection).
5. [Train a model](https://github.com/pperle/gaze-tracking) or [download a pretrained model](https://drive.google.com/drive/folders/1-_bOyMgAQmnwRGfQ4QIQk7hrin0Mexch?usp=sharing). (훈련된 모델을 다운로드 받을 수 도있습니다.)
6. 모든 이전 단계가 이행된 경우(라이브러리,카메라보정,학습모델을 전부 만족해야함), `python main.py --calibration_matrix_path=./calibration_matrix.yaml --model_path=./p00.ckpt` 실행될 수 있고 "빨간색 레이저 포인터"가 화면에 표시되어야 합니다. `main.py` 또한 다음과 같은 여러 시각화 옵션을 제공합니다.
   1. `--visualize_preprocessing` 전처리된 이미지를 시각화하기 위해사용됨
   2. `--visualize_laser_pointer` 사람이 보고 있는 시선을 빨간색 레이저 포인터 점처럼 화면에 표시하려면 아래 이미지에서 오른쪽 모니터를 참조하세요.
   3. `--visualize_3d` 3D 장면에서 머리, 화면 및 시선 벡터를 시각화하려면 아래 이미지의 왼쪽 모니터를 참조하십시오.(멀티모니터를 사용한다면 멀티모니터상에 3d grid 가 나타납니다.
(윈도우 사용자의 경우 pgi를 사용할수 없기 때문에+ 화면정보를 받아올수 없기 때문에 --monitor_mm (모니터의 실제크기) --monitor_pixels(모니터의 해상도) 를 지정해주어야합니다.)
(eg. python main.py --calibration_matrix_path=./calibration_matrix.yaml --model_path=./p14.ckpt --monitor_mm=244,356 --monitor_pixels=1920,1080)

![live-example](./docs/live_example.png)

[1] Amogh Gudi, Xin Li, and Jan van Gemert, “Efficiency in real-time webcam gaze tracking”, in Computer Vision - ECCV 2020 Workshops - Glasgow, UK, August 23-28, 2020, Proceedings, Part I, Adrien Bartoli and Andrea Fusiello, Eds., ser. Lecture Notes in Computer Science, vol. 12535, Springer, 2020, pp. 529–543. DOI : 10.1007/978-3-030-66415-2_34. [Online]. Available: https://doi.org/10.1007/978-3-030-66415-2_34.
