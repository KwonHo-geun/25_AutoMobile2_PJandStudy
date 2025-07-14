# CNN (Convolutional Neural Network) 정리

-  ### CNN이란 무엇인가?
  **Convolutional Neural Network(CNN, 합성곱 신경망)**는 이미지, 음성, 시계열 등 공간적/시간적 패턴이 중요한 데이터를 효과적으로 처리하는 딥러닝 모델.
  인간 시각 피질의 구조에서 영감을 받아 설계되었으며, 주로 이미지 분류, 객체 인식, 의료 영상 분석 등 다양한 분야에서 활용됨.

- CNN Architecture

| Layer 유형              | 주요 역할 및 설명                                                        |
|------------------------|-------------------------------------------------------------------------|
| Input Layer            | 원본 데이터(이미지 등)를 입력받는 계층                                   |
| Convolutional Layer    | 필터(커널)를 통해 입력 데이터의 특징(엣지, 패턴 등) 추출                 |
| Activation Layer       | 비선형성 부여 (주로 ReLU 함수 사용)                                      |
| Pooling Layer          | 특징 맵의 공간적 크기 축소, 위치 변화에 대한 불변성 제공                 |
| Fully Connected Layer  | 추출된 특징을 바탕으로 최종 예측(분류 등) 수행                           |
| Output Layer           | Softmax 등으로 최종 결과 출력                                            |


- ### DeepLearning Model 흐름

입력 -> (Convolution -> Pooling) × N 
= 계층별로 국소 특징 추출 Flatten 
-> Fully Connected Layer-> Activation -> Loss 계산 -> 
Optimizer로 파라미터 업데이트 -> Epoch 반복 + Regularization 적용

- ### CNN 핵심 특징
- 로컬 연결성(Local Connectivity): 각 뉴런은 전체 입력이 아닌 일부 영역(수용영역)만을 바라.
- 가중치 공유(Shared Weights): 동일 필터가 전체 입력에 반복 적용되어 파라미터 수가 대폭 감소.
- 계층적 특징 추출(Hierarchical Feature Extraction): 저수준(엣지 등)에서 고수준(객체 등)까지 점진적으로 특징을 학습.
- 공간 불변성(Translation Invariance): 위치 변화에도 강인한 특징 추출이 가능


- ### CNN 관련 주요 용어 정리

| 용어                       | 설명                                                                                                      |
|----------------------------|---------------------------------------------------------------------------------------------------------|
| Convolution (합성곱)        | 입력 이미지에 커널(필터)을 슬라이딩하면서 내적 연산을 수행해 특징 맵(feature map)을 생성                   |
| Kernel = Filter (커널=필터) | 3×3, 5×5 같은 작은 크기의 가중치 행렬로, 각 국소 영역과 곱해져 입력에서 특정 패턴(엣지, 텍스처 등)을 추출   |
| Stride (스트라이드)         | 커널을 이동시킬 때 한 번에 이동하는 픽셀 수. (Stride=1: 한 픽셀씩, Stride=2: 두 픽셀씩 → 출력 맵 크기 변화) |
| Padding (패딩)              | 입력 주변에 0(또는 다른 값)을 추가해 출력 크기를 조절하거나 경계 정보 손실을 방지 (‘same’: 입력과 같게, ‘valid’: 순수 합성곱) |
| Activation Function         | 합성곱·완전연결 층의 선형 출력에 비선형성을 부여. (대표: ReLU, Sigmoid, Tanh 등)                          |
| Pooling (풀링)              | 특징 맵 크기 축소 및 위치 변동 강인성 부여<br>- Max Pooling: 영역 내 최대값<br>- Average Pooling: 영역 내 평균값 |
| Flatten                     | 다차원 feature map을 1차원 벡터로 변환                                                                    |
| Fully Connected Layer        | 평탄화된 벡터를 입력으로 받아 최종 클래스 점수나 회귀값을 예측                                            |
| Epoch (에폭)                 | 전체 학습 데이터를 한 번 모두 사용해 파라미터를 업데이트한 횟수                                           |
| Batch (배치)                 | 한 번에 신경망에 입력으로 넣어 학습시키는 샘플 묶음. 배치 크기에 따라 학습 안정성과 속도 변화              |
| Loss Function                | 모델 예측값과 실제값 간 차이를 수치화<br>– 분류: Cross-Entropy<br>– 회귀: MSE                             |
| Optimizer                    | 손실 함수를 최소화하도록 파라미터를 업데이트하는 알고리즘<br>ex) SGD, Momentum, RMSprop, Adam 등           |
| Regularization (정규화)      | 모델 복잡도 제어로 과적합 방지<br>– L1/L2 페널티: 가중치 크기 제어<br>– Dropout: 학습 시 뉴런 일부 무작위 비활성화 |

 
