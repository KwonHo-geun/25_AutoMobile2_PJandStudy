# 🚗 25년 2기 - Project_Lane_and_Traffic_Sign_Dectetion

조명 : 2조 
팀 인원 : 권호근(모델 구현), 정주은(PPT제작), 라벨링(이성원, 양근영, 진효상)

---
## 📌 프로젝트 개요
이 프로젝트는 **차선 인식(Lane Detection)** 을 목표로 YOLOv11 객체 탐지 모델과 HuggingFace의 SegFormer를 활용한 **Semantic Segmentation** 기법을 결합하여 구현한 프로젝트이다.  
SegFormer는 전이학습(Transfer Learning)을 적용하여 차선 인식 정확도를 향상시켰으며, 실시간 처리 환경에서도 동작할 수 있도록 최적화하였다.

---


## 📊 데이터셋
- 데이터 종류: 차선 인식용 도로 주행 이미지를 사용하였다.
- 데이터 갯수 : 약 800장

라벨 형식:

- YOLOv11 → Bounding Box(Label txt 파일) 형식을 사용하였다.
- SegFormer → Pixel-level Mask 이미지를 사용하였다.

🛠 데이터 전처리
이 프로젝트의 데이터 전처리는 Roboflow를 활용하여 수행하였다.
Roboflow를 이용하여 이미지 크기 조정, 라벨 형식 변환, 데이터 증강을 진행하였다.
구체적인 전처리 과정은 다음과 같다.

라벨 형식 변환: YOLOv11 학습용 라벨과 SegFormer 학습용 마스크 형식으로 변환하였다.

## 🔧 사용 기술 스택
- **언어(Language)**: Python3
- **딥러닝 프레임워크**: PyTorch
- **모델(Model)**
  - [YOLOv11](https://github.com/ultralytics/ultralytics)
  - [HuggingFace SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
- **전이학습(Transfer Learning)**: 사전 학습된 모델 파라미터 활용
- **기타 라이브러리**
  - OpenCV
  - NumPy
  - Transformers (HuggingFace)
  - Matplotlib

```
Lane-Detection-Project/
│
├── data/                        # 데이터셋
│   ├── train/                   # 학습 데이터
│   │   ├── images/              # 학습용 이미지
│   │   ├── labels/              # YOLO 라벨(txt)
│   │   └── masks/               # SegFormer 마스크
│   ├── valid/                   # 검증 데이터
│   │   ├── images/
│   │   ├── labels/
│   │   └── masks/
│   └── test/                    # 테스트 데이터
│       ├── images/
│       ├── labels/
│       └── masks/
│
├── models/                      # 학습된 모델 가중치
└── notebooks/                   # Google Colab에서 구현
```

---

## ⚙️ 학습 과정
YOLOv11 학습
- 객체 탐지로 표지판과 차선 후보 영역을 탐색하였다.
- 데이터셋은 커스텀 라벨 데이터를 사용하였다.

SegFormer 전이학습
- 차선 영역을 픽셀 단위로 분할하는 Semantic Segmentation을 수행하였다.
- 데이터셋에는 image, label만 있기에 label기반 mask를 따로 만들어주어 학습하였다.

## 📌 결과

### 1. YOLOv11 결과
- 차선 및 차량 탐지 성능이 우수하게 나타났다.
- 원본
  
https://github.com/user-attachments/assets/bf79dd11-7fd3-4e34-b68c-dc73f2dcb001

- 결괴

https://github.com/user-attachments/assets/34618499-f6a5-4b24-ba5e-883ccc5c62e7

### 2. SegFormer 결과
- SegFormer는 차선 영역을 픽셀 단위로 분할하는 Semantic Segmentation을 수행하였다.
- 결과 영상은 존재하지 않는다.
- 본 프로젝트에서는 SegFormer 모델을 Google Colab 환경에서 학습 및 추론하였으나,
  추론 과정에서 메모리 제한 및 실행 시간 제한 문제로 인해 영상 저장이 완료되지 못하였다.  

## 📌 코드 설명

### 1. YOLOv11 코드
```python
!pip install -q ultralytics opencv-python

from ultralytics import YOLO

# YOLOv11 모델 로드
model = YOLO("yolo11n-seg.yaml")  # YAML로 새로운 모델 생성
model = YOLO("yolo11n-seg.pt")    # 사전 학습 모델 로드
model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # YAML + 전이학습

import os

# 데이터셋 경로
DATASET_ROOT = dataset.location 

splits = ["train", "valid", "test"]

for split in splits:
    images_dir = os.path.join(DATASET_ROOT, split, "images")
    labels_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"❌ {split} 폴더의 images 혹은 labels 디렉터리를 찾을 수 없습니다.")
        continue

    # 이미지 파일 개수
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # 라벨 파일 개수
    label_files = [
        f for f in os.listdir(labels_dir)
        if f.lower().endswith(".txt")
    ]

    print(f"📦 {split.upper()} 분할:")
    print(f"   이미지 수 : {len(image_files)}개")
    print(f"   라벨 수   : {len(label_files)}개\n")

   ```

### 2. HuggingFace SegFormer 코드
```
from pathlib import Path
from PIL import Image, ImageDraw

# 마스크 생성 함수
def create_index_masks(base_dir: str, segmentation_txt: bool = True):
    base = Path(base_dir)
    for split in ("train", "valid", "test"):
        img_dir   = base/split/"images"
        label_dir = base/split/"labels"
        mask_dir  = base/split/"masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem     = img_path.stem
            txt_path = label_dir/f"{stem}.txt"
            out_mask = mask_dir/f"{stem}.png"

            W, H = Image.open(img_path).size
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)

            if txt_path.exists():
                for line in txt_path.read_text().splitlines():
                    parts = line.split()
                    cls   = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    if segmentation_txt:
                        pts = [
                            (coords[i] * W, coords[i+1] * H)
                            for i in range(0, len(coords), 2)
                        ]
                        draw.polygon(pts, fill=cls)
                    else:
                        cx, cy, w, h = coords
                        x1 = (cx - w/2) * W
                        y1 = (cy - h/2) * H
                        x2 = (cx + w/2) * W
                        y2 = (cy + h/2) * H
                        draw.rectangle([x1, y1, x2, y2], fill=cls)

            mask.save(out_mask)
```
```
from pathlib import Path
import glob, os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 데이터셋 클래스
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor):
        self.files      = sorted(glob.glob(f"{images_dir}/*.jpg") +
                                 glob.glob(f"{images_dir}/*.png"))
        self.masks_dir  = masks_dir
        self.processor  = processor
        print(f"Found {len(self.files)} images in {images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path  = self.files[idx]
        stem      = Path(img_path).stem
        mask_path = os.path.join(self.masks_dir, f"{stem}.png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        enc = self.processor(images=image,
                             segmentation_maps=np.array(mask, dtype=np.int64),
                             return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

```

3. SegFormer 결과 미생성 원인
Colab에서 SegFormer 추론 결과를 영상(mp4)으로 변환하는 과정에서 GPU 메모리 부족 및 런타임 시간 제한이 발생하였다.
모든 프레임을 처리하기 전에 런타임이 종료되어 영상이 저장되지 못했다.

## 💭프로젝트 끝난 후 느낀점

이번 프로젝트를 진행하면서 여러 가지 새로운 경험을 할 수 있었다.  
우선, 라벨링 작업은 처음 접하는 과정이었는데, 단순히 이미지를 표시하는 수준이 아니라 차선이라는 세밀한 객체를 구분해야 하다 보니 상당한 집중력과 인내심이 필요했다. 처음에는 이 과정이 꽤 낯설고 어려웠지만, 점차 익숙해지면서 라벨링의 중요성과 데이터 품질이 모델 성능에 미치는 영향에 대해 깊이 이해하게 되었다. 무엇보다 직접 만든 데이터셋이 학습에 사용되는 과정이 신기하게 다가왔다.  

또한, 모델 학습과 테스트를 다양한 환경에서 시도해 보았다. Colab과 RunPod를 번갈아 사용하며 각각의 장단점을 직접 체감할 수 있었다. Colab은 접근성이 좋고 사용이 간편했지만, 실행 시간 제한과 메모리 부족 문제가 자주 발생했다. 반면 RunPod는 GPU 환경을 유연하게 사용할 수 있었지만, 환경 설정과 파일 업로드·다운로드 과정이 다소 번거롭게 느껴졌다. 이러한 경험을 통해 환경 선택이 프로젝트 효율성에 얼마나 큰 영향을 미치는지 깨달았다.  

마지막으로, 코드 적용 과정에서도 적지 않은 어려움이 있었다. 특히 마스크(mask) 생성 부분은 모델이 요구하는 입력 형식과 구조를 정확히 맞춰야 했기 때문에 시행착오가 많았다. 데이터 형식 불일치나 차원 문제로 인해 여러 번 오류가 발생했고, 이를 수정하며 모델 구조와 데이터 전처리에 대해 더 깊게 이해할 수 있었다. 이 과정은 힘들었지만, 문제 해결 능력을 키우는 데 큰 도움이 되었다.  

-  
