## 📘 AI 학습 정리 (GitHub 기준)

---

### 1. GitHub, Markdown, Colab
- [Github 사용법](Github-guide.md)
- [MarkDown 사용법](MarkDown-guide.md)
- [Colab 사용법](Colab-guide.md)

[GitHub 사용법](https://github.com/jetsonmom/git_test_markdown_sample?tab=readme-ov-file#github-%EC%82%AC%EC%9A%A9%EB%B2%95)

#### ✅ GitHub 계정 만드는 순서 (2025년 기준)
- 웹 브라우저(크롬, 엣지, 사파리 등) 실행
- 주소창에 `https://github.com` 입력 후 접속
- 오른쪽 위 또는 메인 화면의 **Sign up** 클릭
- 자주 사용하는 이메일 주소 입력
- 비밀번호 생성 (영어 대소문자+숫자+특수문자 조합, 예: `Git1234!hub`)
- 사용자 이름(Username) 입력 (영어, 숫자, 하이픈(-)만 가능, 예: `jetsunmom`, `sungsookjang66`)
- 안내에 따라 인증 및 추가 정보 입력 후 가입 완료

#### ✅ Repository(저장소) 만들기 순서
- GitHub 로그인

![image](https://github.com/user-attachments/assets/5ab6b163-b0e4-496e-95e5-97c0e7e166b3)

- 우측 상단 **+** 버튼 클릭 → **New repository** 선택
- 저장소 이름(Repository name) 입력
- 공개(Public)/비공개(Private) 선택
- **Initialize this repository with a README** 체크 (README.md 파일 생성)
- **Create repository** 클릭

![image](https://github.com/user-attachments/assets/254e5e75-be42-421e-a673-636cec99bf76)

---

**Markdown 문법**

#### 🔰 1. 마크다운(Markdown)이란?
- 간단한 문법으로 글을 꾸미는 방법
- HTML보다 쉽고, GitHub의 README.md 등에서 주로 사용

#### 🛠️ 2. GitHub에서 마크다운 사용하기
- 계정 생성 → 저장소 생성 → README.md 파일 추가 → 마크다운 문법으로 내용 작성

**Markdown 문법**

#### 🔰 1. 마크다운(Markdown)이란?
- 간단한 문법으로 글을 꾸미는 방법
- HTML보다 쉽고, GitHub의 README.md 등에서 주로 사용

#### 🛠️ 2. GitHub에서 마크다운 사용하기
- 계정 생성 → 저장소 생성 → README.md 파일 추가 → 마크다운 문법으로 내용 작성

#### ✍️ 3. 기본 마크다운 문법 정리

| 기능       | 문법 예시              | 결과 예시         |
|------------|------------------------|-------------------|
| 제목       | #, ##, ###             | ## 내 프로젝트    |
| 굵게       | **굵게**               | **중요**          |
| 기울임     | *기울임*               | *강조*            |
| 목록       | -, *                   | - 사과- 배    |
| 숫자 목록  | 1., 2.                 | 1. 첫째2. 둘째|
| 링크       | [이름](주소)           | [구글](https://google.com)|
| 이미지     |     | |
| 코드블록   | ```python ... ```
| 인라인 코드| `코드`                 | `a = 3`           |
| 구분선     | ---                    | ---               |
---


**Colab 기초**

![image](https://github.com/user-attachments/assets/ef728171-2b01-4ee3-b307-919023b6e46f)

- Google Colab은 웹 기반 파이썬 노트북 환경
- 주로 데이터 분석, 머신러닝 실습에 활용
- GitHub 저장소와 연동 가능 (파일 불러오기, 저장 등)

- 이용되는 분야
📊 데이터 분석 실습 (pandas, matplotlib 등)
🧠 머신러닝/딥러닝 모델 학습 (TensorFlow, PyTorch 등)
📝 논문 코드 테스트, Kaggle 노트북 공유
👩‍🏫 교육용 실습 환경 (학생들에게 설치 없이 환경 제공 가능)

시작하는 방법
https://colab.research.google.com 접속
Google 계정으로 로그인
새 노트북 만들기 (+ 새 노트북)
코드 셀에 파이썬 코드 입력 후 Shift + Enter로 실행

✅ 자주 쓰는 코드 스니펫 예시
```
# 드라이브 연동
from google.colab import drive
drive.mount('/content/drive')

# 파일 업로드
from google.colab import files
uploaded = files.upload()

# GPU 확인
!nvidia-smi

# 패키지 설치
!pip install pandas

```

---

### 2. Python3

[**Python 공부 정리**]
변수, 자료형, 조건문, 반복문, 함수 등 기초 문법 학습

- [25.06.23](https://github.com/KwonHo-geun/automobile/blob/main/25.06.23.ipynb)
  - [**Python 자동차 제어 예시 코드**](./Python.md)
- [25.06.24](https://github.com/KwonHo-geun/automobile/blob/main/25.06.24.ipynb)
  - [클래스](https://claude.ai/public/artifacts/82c1fb01-030d-4ae3-abde-118676216f64)
  - [딕셔너리](https://claude.ai/public/artifacts/a11af36d-c9fa-4366-9580-379644d1af5d)
  - [**for문을 활용한 예시 코드**](https://github.com/KwonHo-geun/automobile/blob/main/06.25.%EC%9E%90%EC%9C%A8%EC%A3%BC%ED%96%89_%EC%9E%90%EB%8F%99%EC%B0%A8_for%EB%AC%B8.ipynb)

- [25.06.25_리스트](https://claude.ai/public/artifacts/fd98c798-ab20-40a4-8a3b-537503b9849c)
  - [**06.25-주요 변수를 활용한 자율주행 자동차 예시 코드**](https://github.com/KwonHo-geun/automobile/blob/main/25_06_26.ipynb)

- [25.06.27 리스트 + 딕셔너리를 포함한 자율주행자동차 예시 코드](https://github.com/KwonHo-geun/automobile/blob/main/25_06_27_Dict.ipynb)
- [25.06.30 함수, main 함수](https://github.com/KwonHo-geun/automobile/blob/main/25.06.30.ipynb)

- [25.07.01 자율주행 함수 생성](https://github.com/KwonHo-geun/automobile/blob/main/25.07.01.ipynb)

- [25.07.02 자율주행 자동차 class, 함수 실행](https://github.com/KwonHo-geun/automobile/blob/main/25.07.02.ipynb)

- [25.07.03 자율주행 자동차 - Numpy lib.ver](https://github.com/KwonHo-geun/automobile/blob/main/25.07.03.ipynb)

---

### 3. Data structure / Data Sciencs
- [**데이터 구조 개요** ](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/data_structures.md)
- [**Pandas**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/pandas.md): 데이터프레임 생성, 분석, 전처리
- [**25.07.02 Numpy**](https://github.com/KwonHo-geun/automobile/blob/main/25.07.02%20-%20Numpy.ipynb): 고속 수치 연산, 배열 처리
- [**25.07.03 Matplotlib**](https://github.com/KwonHo-geun/automobile/blob/main/25.07.03-Matplotlib.ipynb): 데이터 시각화(그래프, 차트 등)

---

### 4. Machine Learning

- [**Basic**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/ml_basic.md): 지도/비지도 학습, 모델 평가
- [**모델 훈련 및 평가**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/ml_test.md): 학습 데이터 준비, 모델 학습, 성능 평가

---

### 5. OpenCV

- [**OpenCV 기초**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/OpenCV_basic.md): 이미지 읽기, 변환, 필터 적용
- [**이미지 처리**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/image_test.md): 엣지 검출, 객체 인식 등

---

### 6. CNN(합성곱 신경망)

- [**25.07.14 CNN 기본**](https://github.com/KwonHo-geun/hogeun/blob/main/CNN.md): 구조, 원리, 활용 예시
- [**25.07.15 CNN레이어**](https://github.com/KwonHo-geun/automobile/blob/main/25.07.15_CNN_%EC%98%88%EC%8B%9C_%EB%AA%A8%EB%8D%B8.ipynb): 예시 코드
- [**자율주행 관련 코드**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/cnn_test.md): 이미지 분류, 객체 탐지 등

---

### 7. Ultralytics

- 용어정리

| 용어                      | 설명                                                                 | 주요 특징                                | 지원 모델                          | 활용 예시                                 |
|---------------------------|----------------------------------------------------------------------|------------------------------------------|-------------------------------------|-------------------------------------------|
| backbone                  | 이미지 특징을 추출하는 CNN 구조                                      | 다양한 네트워크 구조, 사전 학습 가중치 활용| CSPDarknet, EfficientNet 등         | 특징 추출, 분류, 검출의 기반              |
| head                      | 추출된 특징을 바탕으로 클래스·위치 예측                              | 구조 변경 가능, 다양한 목적별 설계        | YOLO head, Faster R-CNN head 등     | 객체 분류, 바운딩 박스 예측               |
| image size (imgsz=640)    | 입력 이미지 해상도, 값이 크면 정확도↑ 속도↓                          | 모델 성능·속도에 직접 영향               | 모든 CNN 기반 모델                  | 데이터 전처리, 하이퍼파라미터 튜닝        |
| batch size                | 한 번에 처리하는 이미지 수                                            | GPU 메모리에 영향, 과대/과소 학습 예방    | 대부분의 딥러닝 프레임워크          | 모델 학습 성능 최적화                    |
| labels                    | 클래스·바운딩 박스 등 라벨 정보 텍스트 파일                          | 다양한 포맷, 직접 제작 가능              | YOLO, COCO, Pascal VOC              | 라벨링, 검출 학습관리                    |
| class                     | 탐지하려는 객체의 종류                                               | 다중 클래스 지원                        | 맞춤 클래스 세팅 가능               | 사람, 차량, 동물 탐지 등                 |
| train.py, val.py, predict.py| 학습, 검증, 추론 담당 스크립트                                    | 실험 자동화, CLI 사용 가능               | 대부분의 구현체(예: YOLO)           | 학습, 검증, 실시간 추론                  |
| epochs                    | 전체 데이터 반복 학습 횟수                                           | 과적합 방지, 하이퍼파라미터              | 모든 학습 기반 프레임워크            | 모델 학습 반복횟수 조정                  |
| lr, learning rate         | 가중치 업데이트 변화량                                               | 빠른 수렴, 폭발/소멸 방지                | 모든 최적화 기법                    | 학습률 스케줄링, 초매개변수 조정         |
| augmentation              | 데이터 다양성 강화를 위한 기법(회전, 확대 등)                        | 일반화 향상, 오버피팅 방지               | Albumentations, torchvision 등      | 증강 이미지 생성, 데이터셋 보강           |
| overfitting               | 학습 데이터에 과도 적합되어 일반화 성능 저하 현상                    | 검증 데이터로 확인, 조기 종료 전략        | 모든 딥러닝 모델                    | Early Stopping, Dropout 등               |
| conf (confidence)         | 객체일 확률, 임계값 미만 박스 무시                                   | 탐지 임계치, 정밀도-재현율 트레이드오프   | 모든 객체 탐지 모델                 | confidence threshold 설정                |
| iou (Intersection over Union)| 예측·실제 박스 겹침 비율                                         | 평가 척도, 값이 높을수록 정확             | 모든 객체 검출 평가                 | 성능 평가, NMS 등                       |
| NMS (Non-Maximum Suppression)| 겹치는 박스 중 확률 높은 것만 남기는 후처리                      | 중복 박스 제거, 속도 향상                 | 객체 탐지 모델 내장                 | 최종 검출 결과 정제                      |
| runs/detect/predict       | 추론 결과 저장 기본 경로                                             | 자동 폴더 생성, 결과 이미지/영상 제공     | YOLO 등 여러 프레임워크             | 결과 확인, 시각화                        |
| 그리드(Grid)              | 이미지를 S×S 셀로 분할                                               | 공간 구획화, 각 셀 예측 담당              | YOLO, SSD 등                        | 출력 구조 설계, 예측 영역 관리           |
| 그리드 셀(Cell)           | 그리드 내부 단일 칸, 각 셀이 예측 수행                               | 다중 책임 분배, 병렬 예측 구조            | YOLO 등                             | 객체 위치 담당, 예측 구조 핵심           |
| 바운딩 박스(Bounding Box) | 객체 위치를 x, y(중심), w, h(폭,높이)로 표시                         | 위치 정보 제공, 다양한 포맷 지원          | 모든 객체 탐지 모델                 | 객체 위치 시각화, 평가                   |
| 클래스 확률(Class Probability)| 객체가 해당 클래스일 확률                                    | 다중 클래스 지원, 확률 기반               | 대부분의 모델                       | 분류/검출 결과 해석                     |
| 오브젝트성(Objectness Score)| 예측 박스에 객체 존재 확률(0~1)                                 | 임계값 적용, 탐지 결과 필터링             | YOLO, SSD 등                        | 신뢰도 기반 필터링                      |
| 앵커 박스(Anchor Box)     | 다양한 모양/비율 미리 정의 병렬 박스 템플릿                         | 다양한 객체 대응, 학습 전 고정            | YOLO, Faster R-CNN 등               | 박스 예측 정밀도 향상                    |
| Confidence Threshold      | 신뢰도 기준치, 미만 박스 무시                                      | precision-recall 조정                     | 대부분의 모델                       | threshold 튜닝, 결과 품질 관리           |
| 예측 벡터/출력 텐서       | 그리드 셀이 예측하는 좌표, 오브젝트성, 클래스 확률 벡터             | 구조 다양, 출력을 한 번에 반환            | 모든 CNN 기반 객체 검출             | 후처리, 결과 분석                       |


- [**Ultralytics 기본**]
- [**YOLOv8**](https://github.com/KwonHo-geun/automobile/blob/main/25.07.16_YOLOv8%EC%8B%A4%EC%8A%B5%20%EA%B2%B0%EA%B3%BC.ipynb)
- [**YOLOv12**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/YOLOv12_test.md)



---

### 8. TensorRT vs PyTorch
- [**PyTorch_Basic**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/PyTorch_basic.md)
- [**TensorRT**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/TensorRT_test.md)
- [**YOLOv12**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/YOLOv12_test.md)


| 항목        | PyTorch                | TensorRT           |
|-------------|------------------------|--------------------|
| 주요 특징   | 연구/개발 친화적, 유연 | 추론 속도 최적화   |
| 지원 모델   | 다양한 모델            | 주로 추론(배포)    |
| 활용 예시   | 모델 개발, 실험        | 실시간 추론, 배포  |

---

### 9. TAO Toolkit on RunPod

- [**TAO 사용법**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.TAO_install.md): NVIDIA의 Transfer Learning Toolkit
- [**RunPod 연동**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.TAO_Toolkit.md): 클라우드 환경에서 모델 학습/배포

---

### 10. 칼만필터, CARLA, 경로 알고리즘

- [**칼만필터**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.kalman.md): 센서 데이터 융합, 예측/보정
- [**CARLA 시뮬레이터**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.CARLA.md): 자율주행 시뮬레이션 환경
- [**경로 알고리즘**](): 최단 경로 탐색, 경로 계획

---

### 11. ADAS & (ADAS TensorRT vs PyTorch)

- [**ADAS 기본**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.adas_basic.md): 첨단 운전자 지원 시스템 개념
- [**TensorRT vs PyTorch 비교**](https://github.com/jetsonmom/git_test_markdown_sample/blob/main/.vs.md): 실시간성, 추론 속도, 개발 편의성 등 비교

---
