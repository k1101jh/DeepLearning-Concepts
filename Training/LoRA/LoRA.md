# LoRA (Low-Rank Adaptation, 저차원 적응 기법)

---
Reference:
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., NeurIPS 2023)](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/index)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: LoRA (Low-Rank Adaptation / 저차원 적응 기법)
- **관련 분야/카테고리**: Training / Fine-Tuning / PEFT (Parameter-Efficient Fine-Tuning)
- **한 줄 요약**: 사전 학습된 대형 모델(LLM/Diffusion)의 가중치를 동결(Freeze)하고, 가중치 변화량을 저차원 분해 행렬($W_0 + B \cdot A$, $r \ll d$)로 우회 학습하여 메모리와 파라미터 수모를 99% 이상 절감하는 효율적 미세조정 기법이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### Full Fine-Tuning의 획기적 비용 문제
- **막대한 GPU 메모리 점유**: 수십억~수천억 개 파라미터를 가진 LLM(예: LLaMA-70B, GPT-3)을 전체 미세조정(Full Fine-Tuning)하려면 가중치뿐만 아니라 Optimizer State(Adam 8 bytes/param), Gradient(4 bytes/param), Activation 메모리가 필요하여 수백 GB~TB 단위의 VRAM이 요구됨.
- **배포 및 저장 공간 한계**: 서빙 시 타깃 데이터셋마다 70GB 이상의 전체 파라미터 체크포인트를 독립적으로 배포해야 하므로 스토리지가 낭비됨.

### LoRA 도입을 통한 핵심 해결 목표
- **학습 파라미터 수 99% 감소**: 가중치 업데이트 행렬 $\Delta W$의 본질적인 랭크(Intrinsic Rank)가 낮다는 점에 착안하여 $0.1\% \sim 1\%$의 파라미터만 학습함.
- **추론 지연(Latency) 0**: 학습 완료 후 저차원 행렬 $B \cdot A$를 기존 가중치 $W_0$에 사전에 더해주면($W = W_0 + B \cdot A$), 추론 시 구조적 오버헤드가 전혀 발생하지 않습니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 저차원 분해 (Low-Rank Matrix Decomposition) 메커니즘
원래의 사전 학습 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$는 동결(Freeze)하고, 가중치 업데이트 행렬 $\Delta W$를 두 개의 저차원 행렬 $B \in \mathbb{R}^{d \times r}$와 $A \in \mathbb{R}^{r \times k}$의 곱으로 정의합니다:

$$h = W x = W_0 x + \Delta W x = W_0 x + \frac{\gamma}{r} B A x$$

여기서 각 기호의 의미는 다음과 같습니다:
- $W_0$: 사전 학습된 고정 가중치 행렬 ($d \times k$) - 기울기 업데이트 안 함 (`requires_grad = False`)
- $B$: 0으로 초기화된 훈련 가능 행렬 ($d \times r$)
- $A$: 가우시안 정규분포 $\mathcal{N}(0, \sigma^2)$로 초기화된 훈련 가능 행렬 ($r \times k$)
- $r$: 랭크(Rank) 차원 ($r \ll \min(d, k)$, 보통 4, 8, 16 사용)
- $\frac{\gamma}{r}$: 스케일링 계수 ($\gamma$는 고정 튜닝 하이퍼파라미터 인자)

### 2) 왜 $A$와 $B$의 초기화가 다른가?
- 학습 시작 시점($t=0$)에 $\Delta W = B \cdot A = 0$이 되어야 모델의 초기 출력이 원본 사전 학습 모델과 100% 동일하게 유지됨.
- 따라서 $B = 0$으로 초기화하고 $A$는 가우시안 분포로 초기화하여 초기 출력을 0으로 맞춥니다.

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Intrinsic Dimension (내재적 차원)
- 대형 모델은 이미 광범위한 언어/시각 표현을 습득했기 때문에, 특정 다운스트림 과제를 학습할 때 필요한 가중치 변화량 $\Delta W$는 극히 저차원 부분 공간(Subspace)에 위치함. 
- 이 때문에 랭크 $r=4$나 $r=8$ 정도의 극소 파라미터만으로도 Full Fine-Tuning과 유사한 성능을 달성함.

### 2) QLoRA (Quantized LoRA)
- 사전 학습 가중치 $W_0$를 4-bit NormalFloat(NF4) 형식으로 양자화하여 메모리를 극도로 압축하고, LoRA 어댑터 행렬 $A, B$만 16-bit 부동소수점으로 학습하는 기술임. 단일 RTX 3090/4090 GPU에서 65B 파라미터 LLM을 파인튜닝할 수 있게 되었습니다.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 활용하여 선형 레이어(Linear Layer)에 LoRA 메커니즘을 결합한 커스텀 `LoRALinear` 모듈 구현 예시임.

```python
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """사전 학습 선형 레이어에 LoRA(Low-Rank Adaptation)를 적용하는 래퍼 클래스.

    Args:
        in_features (int): 입력 피처 차원 수 (k).
        out_features (int): 출력 피처 차원 수 (d).
        r (int, optional): 저차원 랭크(Rank). 기본값은 8.
        lora_alpha (float, optional): LoRA 스케일링 하이퍼파라미터. 기본값은 16.0.
        dropout (float, optional): 입력에 적용할 드롭아웃 비율. 기본값은 0.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.r: int = r
        self.lora_alpha: float = lora_alpha
        self.scaling: float = lora_alpha / r if r > 0 else 1.0

        # 1. 원본 선형 가중치 (Freeze 대상)
        self.pretrained_weight: nn.Parameter = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.pretrained_weight.requires_grad = False  # 동결

        # 2. LoRA 저차원 분해 행렬 (Trainable)
        if r > 0:
            # A 행렬: (r, in_features) -> Kaiming Uniform 초기화
            self.lora_A: nn.Parameter = nn.Parameter(torch.zeros(r, in_features))
            # B 행렬: (out_features, r) -> 0으로 초기화
            self.lora_B: nn.Parameter = nn.Parameter(torch.zeros(out_features, r))
            
            self.dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """LoRA 행렬 A와 B의 파라미터를 초기화함.
        A는 Gaussian 분포로, B는 0으로 초기화하여 초기에 ΔW = 0이 되도록 설정함.
        """
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 연산을 수행합니다: y = W_0 * x + (scaling) * (B * A * x)

        Args:
            x (torch.Tensor): 입력 텐서. (batch_size, ..., in_features)

        Returns:
            torch.Tensor: 출력 텐서. (batch_size, ..., out_features)
        """
        # 1. 고정된 원본 가중치 연산
        result: torch.Tensor = F.linear(x, self.pretrained_weight)

        # 2. LoRA 저차원 어댑터 연산 가산
        if self.r > 0:
            lora_out = F.linear(self.dropout(x), self.lora_A)  # (x @ A^T) -> (..., r)
            lora_out = F.linear(lora_out, self.lora_B)        # ((x @ A^T) @ B^T) -> (..., out_features)
            result += lora_out * self.scaling

        return result

    def merge_weights(self) -> None:
        """추론 시 추가 Latency를 없애기 위해 LoRA 가중치(B * A)를 원본 가중치 W_0에 사전에 합칩니다.
        """
        if self.r > 0:
            # ΔW = B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.pretrained_weight.data += delta_w
            # LoRA 행렬 비활성화
            self.r = 0


# 실행 예시
if __name__ == "__main__":
    batch_size, in_dim, out_dim = 4, 1024, 4096
    dummy_input: torch.Tensor = torch.randn(batch_size, in_dim)

    # LoRA 래퍼 선형 레이어 생성 (r=8)
    lora_layer: LoRALinear = LoRALinear(in_features=in_dim, out_features=out_dim, r=8)

    # 1. 학습 진행 가정 (오직 lora_A, lora_B만 역전파 기울기 계산)
    output: torch.Tensor = lora_layer(dummy_input)
    print(f"LoRA 적용 출력 텐서 크기: {output.shape}")
    print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)}")

    # 2. 추론 단계: 가중치 병합(Merge)
    lora_layer.merge_weights()
    merged_output: torch.Tensor = lora_layer(dummy_input)
    print(f"가중치 병합 후 출력 텐서 크기: {merged_output.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Full Fine-Tuning | Adapter Tuning | LoRA (Low-Rank Adaptation) |
| :--- | :--- | :--- | :--- |
| **학습 파라미터 비율** | 100% | ~1% | **< 0.1% ~ 1%** |
| **GPU 메모리 점유** | 극도로 높음 | 낮음 | **매우 낮음 (QLoRA 적용 시 단일 GPU 가능)** |
| **추론 Latency** | 없음 | 추가적인 시퀀셜 레이어 연산 발생 | **없음 (Merge 가능 $W = W_0 + BA$)** |
| **체크포인트 용량** | 수십~수백 GB | 수십 MB | **수 MB ~ 수십 MB** |

### 장점
- **메모리 및 저장 공간 비약적 절감**: 모델 체크포인트를 용량 수 MB 단위로 손쉽게 관리 및 스위칭 가능.
- **추론 속도 저하 0**: 배포 시 가중치 병합으로 원본 모델과 동일한 연산 속도 보장.

### 한계점
- 랭크 $r$ 및 $\alpha$ 스케일링 인자 하이퍼파라미터 튜닝이 추가로 필요함.

---

## 7. 활용 사례 및 응용

1. **LLM 커스텀 미세조정 (LLaMA, Mistral, Gemma)**
   - 대화형 모델(Instruction Tuning), 특정 도메인(의료, 법률, 코드) 전용 LLM 학습 시 표준 PEFT 도구로 활용.
2. **Stable Diffusion & Flux 스타일 학습**
   - 이미지 생성 모델에서 특정 화풍, 인물, 개념을 추가 학습할 때 몇 MB 용량의 LoRA 파일로 배포 및 혼합 적용.
