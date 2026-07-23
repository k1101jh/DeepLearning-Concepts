# Activation Function (활성화 함수)

---
Reference:
- [Deep Sparse Rectifier Neural Networks (Glorot et al., 2011)](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
- [Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415)
- [Sigmoid-Weighted Linear Units for Neural Network Function Approximation (Elfwing et al., 2018)](https://arxiv.org/abs/1702.03118)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Activation Function (활성화 함수 - ReLU, GELU, SiLU, ReLU6 등)
- **관련 분야/카테고리**: Basics / Neural Network Architecture
- **한 줄 요약**: 인공 신경망의 각 레이어 출력에 비선형성(Non-linearity)을 부여하여 복잡한 데이터 패턴 및 표현(Representation)을 학습할 수 있게 만드는 필수 함수이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 선형 연산의 한계점
- **표현력 제한**: 아무리 깊은 신경망(Multi-layer Perceptron)을 쌓더라도 비선형 활성화 함수가 없다면 $W_2(W_1 x + b_1) + b_2 = W_{net} x + b_{net}$ 와 같이 단일 선형 변환으로 축소되어 복잡한 곡선 경계나 고차원 패턴을 표현하지 못합니다.
- **Sigmoid / Tanh의 기울기 소실(Vanishing Gradient)**: 초기 활성화 함수로 많이 쓰이던 Sigmoid나 Tanh는 입출력 양 끝단에서 미분값(기울기)이 0으로 수렴하는 Saturation 영역이 존재하여, 딥러닝 레이어가 깊어질수록 역전파 기울기가 사라지는 문제가 발생했습니다.

### 비선형 활성화 함수 도입을 통한 핵심 해결 목표
- **비선형성 확보**: 복잡한 데이터의 분포와 경계를 근사(Universal Approximation Theorem)할 수 있는 표현 능력을 제공합니다.
- **기울기 전달 원활화**: ReLU계열 및 Smooth 비선형 함수(GELU, SiLU)를 사용하여 딥러닝 모델 학습 시 기울기 소실을 예방하고 학습 속도를 비약적으로 향상시킵니다.
- **모바일/양산 최적화**: ReLU6와 같이 상한선(Upper Bound)을 두어 정밀도가 제한된 임베디드 및 Quantization(모바일 환경) 연산 효율성을 최적화합니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 대표적인 활성화 함수 및 수식 메커니즘

1. **ReLU (Rectified Linear Unit)**
   - 음수 입력은 0으로 차단하고 양수 입력은 그대로 통과시킵니다.
   $$\text{ReLU}(x) = \max(0, x)$$

2. **ReLU6**
   - ReLU의 양수 출력을 최대 6으로 제한하여 모바일 연산(Fixed-point 8-bit quantization)에서 표현 범위를 안정화합니다.
   $$\text{ReLU6}(x) = \min(\max(0, x), 6)$$

3. **GELU (Gaussian Error Linear Unit)**
   - 입력값 $x$가 정규분포를 따른다고 가정할 때, $x$가 자기 자신보다 작거나 같을 확률(Gaussian CDF $\Phi(x)$)을 가중치로 곱해 부드러운 확률적 선택을 수행합니다. Transformer 및 BERT/GPT 계열 표준 활성화 함수입니다.
   $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$
   - 근사식: $\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)$

4. **SiLU (Sigmoid-Weighted Linear Unit, Swish)**
   - Sigmoid 함수를 가중치로 사용하여 입력값이 클수록 선형에 가까워지고, 음수 영역에서는 약간의 음수 값을 허용하는 매끄러운 곡선 함수입니다. EfficientNet, YOLOv5/v8, LLaMA 계열 표준 활성화 함수입니다.
   $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Dying ReLU 현상과 해결 방안
- ReLU는 $x < 0$ 구간에서 미분값이 0이 되므로, 비정상적으로 큰 음수 기울기가 지나갈 경우 뉴런이 permanently 비활성화되어 더 이상 학습되지 않는 Dying ReLU 문제가 발생할 수 있습니다.
- 이를 보완하기 위해 LeakyReLU, GELU, SiLU와 같이 음수 구간에서도 미세한 기울기를 전달하는 완화된 비선형 함수가 고안되었습니다.

### 2) Smooth Non-monotonicity (매끄러운 비단조성)
- GELU나 SiLU의 주요 특징 중 하나는 음수 영역 부근에서 단조 증가(Monotonic)하지 않고 살짝 감소했다가 올라오는 비단조 곡선을 가진다는 점입니다.
- 이 매끄러운 형태가 손실 곡면(Loss Landscape)을 한층 부드럽게 만들어 딥 트랜스포머나 대형 비전 모델의 최적화 효율성을 크게 높여줍니다.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 활용하여 주요 활성화 함수들을 커스텀 모듈 및 표준 API로 사용하는 파이썬 예시 코드입니다.

```python
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomActivationDemo(nn.Module):
    """주요 비선형 활성화 함수(ReLU, ReLU6, GELU, SiLU)의 동작 방식을 시연하는 파이토치 모듈.

    Attributes:
        relu (nn.ReLU): 표준 ReLU 레이어.
        relu6 (nn.ReLU6): 상한값이 6인 모바일 최적화 ReLU6 레이어.
        gelu (nn.GELU): Gaussian Error Linear Unit 레이어.
        silu (nn.SiLU): Sigmoid-Weighted Linear Unit 레이어.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu: nn.ReLU = nn.ReLU()
        self.relu6: nn.ReLU6 = nn.ReLU6()
        self.gelu: nn.GELU = nn.GELU()
        self.silu: nn.SiLU = nn.SiLU()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """입력 텐서에 대해 각 활성화 함수 적용 결과를 반환합니다.

        Args:
            x (torch.Tensor): 다양한 입력값을 포함하는 N차원 연산 텐서.

        Returns:
            Dict[str, torch.Tensor]: 각 활성화 함수명이 키(Key)이고 텐서 연산 결과가 값(Value)인 딕셔너리.
        """
        results: Dict[str, torch.Tensor] = {
            "ReLU": self.relu(x),
            "ReLU6": self.relu6(x),
            "GELU": self.gelu(x),
            "SiLU": self.silu(x),
        }
        return results


# 실행 예시
if __name__ == "__main__":
    # 테스트 샘플 텐서 생성 (음수, 양수, 큰 값 포함)
    sample_input: torch.Tensor = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0, 10.0])
    
    demo_model: CustomActivationDemo = CustomActivationDemo()
    output_dict: Dict[str, torch.Tensor] = demo_model(sample_input)

    print("=== 활성화 함수별 출력 비교 ===")
    for act_name, act_output in output_dict.items():
        print(f"{act_name:7s}: {act_output.tolist()}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | ReLU | ReLU6 | GELU | SiLU (Swish) |
| :--- | :--- | :--- | :--- | :--- |
| **수식 복잡도** | 매우 단순 ($\max(0,x)$) | 매우 단순 ($\min(\max(0,x),6)$) | 상대적으로 복잡 (Gaussian CDF) | 중간 ($x \cdot \sigma(x)$) |
| **연산 속도** | 최고 (단순 조건문) | 최고 (하드웨어 고정점 연산 유리) | 상대적 차이 있음 (근사식 사용) | 빠름 |
| **음수 구간 기울기**| 0 (Dying ReLU 가능성) | 0 | 매끄러운 음수 곡선 | 매끄러운 음수 곡선 |
| **주요 적용 대상** | 일반적인 CNN, MLP | MobileNet, 모바일 Quantization | ViT, BERT, GPT-3/4, RoBERTa | EfficientNet, YOLOv5/v8, LLaMA |

### 장점
- **GELU / SiLU**: 미분 가능하고 부드러운 곡선을 가져 대규모 데이터셋 학습 및 최신 트랜스포머/LLM 구조에서 최선의 성과를 냅니다.
- **ReLU / ReLU6**: 메모리 점유가 적고 연산 비용이 매우 저렴하여 경량화 모바일 모델에 최적입니다.

### 한계점
- 지수 연산($e^x$) 및 복잡한 특수 함수 연산이 포함된 GELU/SiLU는 경량 하드웨어 장치에서 ReLU 대비 약간의 연산 오버헤드가 있을 수 있습니다.

---

## 7. 활용 사례 및 응용

1. **Transformer & LLM (BERT, GPT 시리즈)**
   - Transformer 아키텍처의 Feed-Forward Network(FFN) 내부 활성화 함수로 GELU 및 SwiGLU(SiLU 기반 변형)가 사실상 표준으로 사용됩니다.
2. **최신 컴퓨터 비전 모델 (EfficientNet, YOLOv5~v8)**
   - ConvNeXt 및 YOLO 최신 버전은 SiLU를 사용하여 기존 ReLU 대비 정확도 향상을 이끌어냈습니다.
3. **모바일 온디바이스 모델 (MobileNetV2 / MobileNetV3)**
   - 초경량 신경망에서 8-bit quantization 시 오버플로우를 방지하기 위해 ReLU6 및 Hard-Swish를 채택합니다.
