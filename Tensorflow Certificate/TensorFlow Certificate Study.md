## TensorFlow Certificate Study

**Callbacks**: To stop training at specific threshold during epoch

**Flatten**: A layer that takes input shape square and turns into a simple linear array.

**Convolutional NN**: 

- image 전체를 보는 것이 아니라 부분을 보는것이 핵심 아이디어. 이 '부분'에 해당하는 것이 filter이다.
- `CNN`의 가장 핵심적인 개념은 이미지의 공간정보를 유지한채 학습을 한다는 것입니다.
- image가 filter를 거쳐가면 차원이 작아지는데 이를 방지하는 방법이 padding
  - **Padding**: Zero padding (0으로 둘러쌈)
    - `Padding`은 `Convolution`을 수행하기 전, 입력 데이터 주변을 특정 픽셀 값으로 채워 늘리는 것입니다. `Padding`을 사용하게 되면 입력 이미지의 크기를 줄이지 않을 수 있습니다.
- **적당히 크기도 줄이고, 특정 feature를 강조**할 수 있어야 하는데 그 역할을 `Pooling` 레이어에서 하게 됩니다.
  - **Pooling**
    1. Max Pooling: CNN에서는 주로 Max Pooling 사용
    2. Average Pooling
    3. Min Pooling
- **특징 추출 단계 (Feature Extraction)**:
  - Convolution Layer: 필터를 통해 이미지의 특징을 추출
  - Pooling Layer: 특징을 강화시키고 이미지의 크기를 줄임.
- **CNN's Parameters**
  - Convolution Filter의 개수
  - Filter의 사이즈
  - Padding 여부
  - Stride (필터를 얼마나 움직일것인가)

**ImageGenerator (TensorFlow)**: automatically adjust the size of input pictures.