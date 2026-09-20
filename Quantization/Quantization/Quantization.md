# Quantization (양자화)

---
Reference:
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
- [Introduction to Weight Quantization (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Quantization (양자화)
- **관련 분야/카테고리**: Quantization / Model Compression / LLM Efficiency
- **한 줄 요약**: 높은 정밀도의 연속 실수 파라미터(FP32, FP16, BF16) 및 활성화 값을 정밀도가 낮고 표현 범위가 작은 정수 형태(INT8, INT4, 1-bit 등)로 변환하여 모델 성능 저하를 최소화하면서 연산 속도와 메모리 사용량을 획기적으로 낮추는 기술

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

- **컴퓨팅 자원 및 VRAM 메모리 병목 현상**:
    - 대규모 언어 모델(LLM)과 Vision Transformer 등 최근 딥러닝 모델의 파라미터 크기 및 Self-Attention 연산량이 폭발함
    - 예: 70B 규모 파라미터 LLM을 FP16 정밀도로 로드하려면 최소 140GB 이상의 VRAM이 필요하여 단일 GPU 디바이스 서빙이 불가능함
- **저정밀도 연산의 연산 및 에너지 효율성**:
    - INT8 또는 INT4 연산은 FP16 / FP32 연산에 비해 처리 속도(Throughput)가 빠르고 칩셋 점유 면적 및 에너지 소비 비용이 기하급수적으로 감소함

![alt text](./images/Comparison%20between%20peak%20throughput%20&%20comparison%20of%20the%20corresponding%20energy%20cost%20and%20relative%20area.png)
> **Figure 1. 정밀도별 처리량 및 에너지/면적 비용 비교 (출처: [Survey Paper](https://arxiv.org/abs/2103.13630))**
> - (왼쪽) Titan RTX 및 A100 GPU에서의 비트 정밀도별 최대 처리량 비교
> - (오른쪽) 45nm 공정 기술 기준 정밀도에 따른 에너지 비용과 상대 면적 비용 (정밀도가 낮을수록 에너지 효율성과 연산 처리량이 향상됨)

---

## 3. IEEE 부동 소수점 표기 구조

부동소수점 데이터 표현 방식은 **Sign Bit (부호)**, **Exponent Bits (지수)**, **Mantissa Bits (가수)**의 3가지 구성 요소로 이루어집니다.

- **예시: $-118.625$의 2진수 변환**:
    - ${118}_{10} = {1110110}_{2}$, ${0.625}_{10} = {0.101}_{2}$
    - 정규화: $1.110110101 \times 2^6$
    - 부호부: 음수이므로 `1`
    - 가수부: $11011010100000000000000$ (23비트 채움)
    - 지수부: $6 + \text{Bias}(127) = 133 \to {10000101}_{2}$

| 자료형 | 부호 비트 (Sign) | 지수 비트 (Exponent) | 가수 비트 (Mantissa) | 특징 및 주 용도 |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 1 | 8 | 23 | 표준 단정밀도 부동소수점 (기본 학습용) |
| **FP16** | 1 | 5 | 10 | 반정밀도 부동소수점 (표현 범위가 작음) |
| **BF16** | 1 | 8 | 7 | Bfloat16 (FP32와 동일한 지수 범위, 정밀도는 낮춤) |

![alt text](./images/FP32%20to%20FP16.png)
> **Figure 2. FP32에서 FP16으로의 정밀도 변환 (출처: [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization))**

![alt text](./images/FP32%20to%20BF16.png)
> **Figure 3. FP32에서 BF16으로의 정밀도 변환 (출처: [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization))**

![alt text](./images/FP32%20to%20INT8.png)
> **Figure 4. FP32에서 INT8 정수 영역으로의 양자화 (출처: [Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization))**

---

## 4. 핵심 원리 및 메커니즘 (How?)

### 1) 선형 양자화 (Linear Quantization)
실수 연속 도메인 $r \in [r_{\min}, r_{\max}]$을 저정밀도 정수 도메인 $q \in [q_{\min}, q_{\max}]$으로 매핑하는 수식입니다.

- **양자화 (Quantization)**:
    $$q = \text{round}\left( \frac{r}{S} \right) + Z$$
- **역양자화 (Dequantization)**:
    $$\tilde{r} = S (q - Z)$$
    - $r$: 입력 실수 (Real-valued input)
    - $q$: 출력 정수 (Quantized integer output)
    - $S$: 스케일링 팩터 (Real-valued Scaling Factor)
    - $Z$: 영점 오프셋 (Integer Zero-point)

![alt text](./images/Quantization%20sample.png)
> **Figure 5. 실수 영역에서 양자화 정수 영역으로의 매핑**

![alt text](./images/Dequantization%20sample.png)
> **Figure 6. 정수 영역에서 실수 영역으로의 역양자화**

---

### 2) Uniform vs Non-Uniform Quantization

![alt text](./images/Uniform%20&%20Non-uniform%20quantization.png)
> **Figure 7. Uniform & Non-uniform Quantization 비교 (출처: [Survey Paper](https://arxiv.org/abs/2103.13630))**

- **Uniform Quantization**:
    - 양자화 단계(Step Size) 간격이 전체 범위에서 일정하게 유지되는 방식 ($S$ 고정)
    - 하드웨어 가속기 구현이 단순하고 효율적임
- **Non-Uniform Quantization**:
    - 값의 밀도가 높은 구간은 간격을 촘촘하게, 밀도가 낮은 구간은 넓게 배치하는 방식
    - 신경망 가중치 분포(정규분포 형태)에 맞춰 정밀도를 극대화할 수 있으나 연산 오버헤드가 발생함

---

### 3) Symmetric vs Asymmetric Quantization

![alt text](./images/Symmetric%20and%20Asymmetric.png)
> **Figure 8. Symmetric 및 Asymmetric Quantization 구조**

- **비대칭 양자화 (Asymmetric Quantization)**:
    - 실수 범위 $[r_{\min}, r_{\max}]$를 정수 범위 $[q_{\min}, q_{\max}]$에 맞춰 영점 오프셋 $Z$를 자유롭게 이동시킴
    $$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$$
    $$Z = -\text{round}\left( \frac{r_{\min}}{S} \right) + q_{\min}$$
    - ReLU 출력과 같이 비대칭으로 치우친 데이터 분포 표현에 적합함
- **대칭 양자화 (Symmetric Quantization)**:
    - 영점을 $Z = 0$으로 고정하고 실수 범위를 부호 대칭으로 맞춰 연산을 단순화함
    $$S = \frac{2 \times \max(|r|)}{q_{\max} - q_{\min}}$$
    - **Full Range**: INT8 범위 $[-128, 127]$ 전체 사용
    - **Restricted Range**: INT8 범위 $[-127, 127]$ 사용

---

### 4) 양자화 기본 기법 (Absmax vs Zero-Point)

![alt text](./images/Absmax%20Quantization%20and%20zero-point%20quantization.png)
> **Figure 9. Absmax 양자화와 Zero-point 양자화 비교**

- **Absmax 양자화**:
    - 입력 $X$의 절대값 최대치 $\max(|X|)$로 나눈 후 INT8 상한치(127)를 곱해 $[-127, 127]$로 대칭 매핑함
    $$X_{\text{quant}} = \text{round}\left( \frac{127}{\max(|X|)} \cdot X \right)$$
    $$X_{\text{dequant}} = \frac{\max(|X|)}{127} \cdot X_{\text{quant}}$$
- **Zero-Point 양자화**:
    - 비대칭 범위를 고려하여 $X$의 $\min(X)$과 $\max(X)$ 차이로 스케일링한 뒤 영점 이동을 적용함
    $$\text{scale} = \frac{255}{\max(X) - \min(X)}$$
    $$\text{zeropoint} = -\text{round}(\text{scale} \cdot \min(X)) - 128$$
    $$X_{\text{quant}} = \text{round}(\text{scale} \cdot X + \text{zeropoint})$$

---

### 5) Calibration (보정 과정)
실수 범위의 Outlier(이상치)에 민감하지 않도록 최적의 Clipping Range $[r_{\min}, r_{\max}]$를 결정하는 절차입니다.

![alt text](./images/Calibration.png)
> **Figure 10. Calibration 과정에서의 Clipping 범위 선택**

- **보정 전략**:
    - **Percentile (백분위수)**: 상위 $99.99\%$ 등의 값을 임계치로 자름
    - **MSE 최적화**: 원본 가중치와 양자화 가중치 간의 Mean Squared Error 최소화
    - **KL-Divergence (엔트로피 최적화)**: 원본 연속 분포와 양자화 불연속 분포 간 정보 손실 정보량 최소화

---

### 6) Static vs Dynamic Quantization
- **Dynamic Quantization (동적 양자화)**:
    - 가중치는 사전에 양자화하지만, Activation은 추론 과정에서 입력 텐서의 $\min, \max$를 즉시 계산하여 동적으로 양자화함
    - 각 입력마다 정밀도가 높고 모델 메모리 로드 속도를 줄이나, 실시간 스케일 연산으로 추론 속도 향상이 미미할 수 있음
- **Static Quantization (정적 양자화)**:
    - 훈련 전/후 보정 데이터셋(Calibration Set)을 활용해 Activation의 범위까지 미리 계산하여 고정함
    - 런타임 연산 오버헤드가 없어 연산 속도가 매우 빠름

---

### 7) 양자화 세분성 (Quantization Granularity)
- **Layer-wise Quantization**: 레이어 전체 매개변수에 대해 1개의 스케일 팩터 $S$ 적용 (단순하지만 정확도 낮음)
- **Channel-wise Quantization**: 레이어 내의 각 출력 채널별로 서로 다른 스케일 팩터 $S_i$ 적용 (가장 보편적인 기법)
- **Group-wise Quantization**: 128개 등의 연속 파라미터 그룹 단위로 세밀한 스케일 팩터 적용 (LLM 양자화 표준)

---

## 5. 핵심 양자화 기술 및 최신 LLM 연구 (PTQ, QAT, BitNet, GPTQ, AWQ, QLoRA)

![alt text](./images/PTQ%20and%20QAT.png)
> **Figure 11. Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT) 비교**

### 1) PTQ (Post-Training Quantization)
사전 학습 완료된 모델을 재학습 없이 소량의 보정 데이터로 양자화하는 기법입니다.
- **GPTQ (Optimal Brain Quantization 기반 PTQ)**:
    - 2차 미분 오차(Hessian Matrix $H$)를 계산하여, 특정 가중치 하나를 양자화할 때 발생하는 손실 오차를 남아있는 나머지 가중치들에 역방향으로 일괄 업데이트 보정함
    $$\Delta w_q = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot H^{-1}_{:, q}$$
- **AWQ (Activation-aware Weight Quantization)**:
    - LLM의 활성화 값(Activation) 크기가 특정 1% 채널에 집중되어 있다는 점을 착안하여, 중요한 1% 가중치 채널에 가중치를 보호하는 채널별 스케일링 전처리를 수행함

### 2) QAT (Quantization-Aware Training)

![alt text](./images/QAT.png)
> **Figure 12. Quantization-Aware Training (QAT) 순전파/역전파 흐름**

- 학습 과정에 가짜 양자화(Fake Quantization - Forward 시 양자화 후 역양자화 적용)를 도입하고, 역전파 시에는 Straight-Through Estimator (STE)를 사용해 FP32 가중치를 갱신함

![alt text](./images/Wide%20minima1.png)
> **Figure 13. 양자화를 고려하지 않은 Narrow Minima에서의 양자화 오류 발생**

![alt text](./images/Wide%20minima2.png)
> **Figure 14. QAT 학습을 통한 Wide Minima 형성 및 양자화 오류 감소**

### 3) BitNet (1-bit / 1.58-bit Transformers)

![alt text](./images/BitNet.png)
> **Figure 15. BitNet의 BitLinear 레이어 구조**

- `nn.Linear`를 `BitLinear`로 교체하여 가중치를 삼진수 $\{-1, 0, 1\}$ 또는 이진수 $\{-1, 1\}$로 이진화(Binarize)함
- 곱셈(Floating-point Multiplication) 연산을 완전히 제거하고 단순 정수 가산(Addition)으로 Matrix Multiplication을 수행함

### 4) QLoRA (Quantized LoRA)
- 정규분포 가중치에 최적화된 4-bit 데이터 타입 **NF4 (NormalFloat 4)**, 양자화 스케일을 한 번 더 양자화하는 **Double Quantization**, 순간 GPU 메모리 피크를 완화하는 **Paged Optimizers**를 도입하여 4비트 Base LLM 위에서 16비트 LoRA 미세조정을 가능하게 만듦

### 5) Activation Function 양자화 & Look-Up Table (LUT)
- 비선형 활성화 함수(ReLU, GELU 등)를 INT8 입력에 적용할 때 연산 지연을 방지하기 위해, 입력 정수 범위($[0, 255]$ 등)에 대한 활성화 결과값을 미리 계산해두는 **LUT (Look-Up Table)** 참조 방식을 활용함

---

## 6. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Tuple
import torch
import torch.nn as nn

class AbsmaxSymmetricQuantizer(nn.Module):
    """
    Absmax 방식을 이용한 INT8 대칭 양자화(Symmetric Quantization) 및 역양자화 모듈
    
    Attributes:
        num_bits (int): 양자화 비트 수 (기본값 8)
        qmin (int): 정수 최소값 (-128)
        qmax (int): 정수 최대값 (127)
    """
    def __init__(self, num_bits: int = 8) -> None:
        super().__init__()
        self.num_bits: int = num_bits
        self.qmin: int = -(2 ** (num_bits - 1))
        self.qmax: int = (2 ** (num_bits - 1)) - 1

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        연속 실수 FP32/FP16 텐서를 INT8 정수 텐서로 양자화함
        
        Args:
            x (torch.Tensor): 입력 실수 텐서
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: INT8 정수 텐서 및 스케일 팩터 S
        """
        max_abs = torch.max(torch.abs(x))
        scale = max_abs / float(self.qmax)
        scale = torch.clamp(scale, min=1e-8)

        x_quant = torch.round(x / scale)
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax).to(torch.int8)
        return x_quant, scale

    def dequantize(self, x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        INT8 정수 텐서를 실수 FP32 텐서로 역양자화함
        
        Args:
            x_quant (torch.Tensor): INT8 정수 텐서
            scale (torch.Tensor): 스케일 팩터 S
            
        Returns:
            torch.Tensor: 복원된 FP32 실수 텐서
        """
        return x_quant.to(torch.float32) * scale


if __name__ == "__main__":
    quantizer = AbsmaxSymmetricQuantizer(num_bits=8)
    original_weights = torch.randn(4, 4) * 5.0

    quantized_tensor, scale = quantizer.quantize(original_weights)
    dequantized_tensor = quantizer.dequantize(quantized_tensor, scale)

    mse_error = torch.mean((original_weights - dequantized_tensor) ** 2).item()

    print(f"원본 FP32 가중치:\n{original_weights[0]}")
    print(f"양자화 INT8 가중치:\n{quantized_tensor[0]}")
    print(f"스케일 팩터 S: {scale.item():.6f}")
    print(f"양자화-역양자화 MSE 손실: {mse_error:.6f}")
```

---

## 7. 장단점 및 기법 비교 표

| 비교 항목 | PTQ (Post-Training Quantization) | QAT (Quantization-Aware Training) | GPTQ / AWQ (LLM PTQ) | BitNet (1-bit LLM) |
| :--- | :--- | :--- | :--- | :--- |
| **추가 훈련 비용** | 거의 없음 (소량 보정데이터) | 매우 큼 (전체 재학습 필요) | 소형 (GPU 수분~수시간) | 처음부터 전체 재학습 필요 |
| **양자화 비트** | INT8 위주 | INT8 ~ INT4 | INT4 / INT3 | 1-bit / 1.58-bit 삼진수 |
| **정확도 보존력** | INT8 우수, INT4 급락 | 매우 뛰어남 (Wide Minima) | **INT4에서도 극소 오차 보존** | **동일 파라미터 FP16 능가** |
| **핵심 연산** | 정수 곱셈/가산 변환 | 정수 곱셈/가산 변환 | 정수 곱셈/가산 변환 | **곱셈 제거, 가산 연산만 수행** |

---

## 8. 활용 사례 및 응용
- **vLLM & TensorRT-LLM 엔진**: GPTQ / AWQ INT4 모델의 GPU 커널 서빙 가속화
- **ONNX Runtime & PyTorch Mobile**: 모바일/에지 디바이스 정적/동적 양자화 모델 배포
- **QLoRA 미세조정**: 단일 RTX 3090/4090 GPU 상에서 65B/70B LLM Fine-tuning 실현