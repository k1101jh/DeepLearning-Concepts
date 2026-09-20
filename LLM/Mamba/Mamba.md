# Mamba (Selective State Space Model)

---
Reference:
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Mamba (Selective State Space Model)
- **관련 분야/카테고리**: LLM / Sequence Modeling / State Space Model
- **한 줄 요약**: 입력에 따라 상태 전이 파라미터를 가변화하는 Selective Scan 메커니즘을 통해 Transformer의 $O(N^2)$ 계산 복잡도를 선형시간 $O(N)$으로 감축한 시퀀스 모델

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **Transformer 모델의 계산 및 메모리 병목**:
    - Self-Attention 어텐션 연산은 시퀀스 길이 $N$에 대해 $O(N^2)$의 시간 및 공간 복잡도를 가짐
    - Autoregressive 추론 시 이전 키-값 토큰을 저장하는 KV 캐시 메모리 사용량이 시퀀스 길이에 비례하여 무한히 증가함
- **기존 State Space Model (SSM)의 한계점**:
    - S4 등 기존 LTI(Linear Time-Invariant) SSM은 시퀀스 길이와 무관하게 선형시간 $O(N)$ 추론이 가능함
    - 그러나 모든 입력 토큰에 대해 고정된 상태 전이 매트릭스($A, B, C$)를 사용하므로, 입력 맥락에 따라 중요 정보를 선택적으로 기억하거나 망각하는 Content-based Reasoning 능력이 부족함

## 3. 핵심 원리 및 메커니즘 (How?)
- **연속 시간 SSM (Continuous-time SSM)**:
    - 연속적 입력 $x(t) \in \mathbb{R}$를 은닉 상태 $h(t) \in \mathbb{R}^N$를 거쳐 출력 $y(t) \in \mathbb{R}$로 매핑함
    $$\dot{h}(t) = Ah(t) + Bx(t)$$
    $$y(t) = Ch(t)$$
- **이산화 (Discretization)**:
    - 컴퓨터 연산을 위해 연속 신호를 샘플링 간격 $\Delta$(Step size)를 통해 이산 신호로 변환함 (Zero-Order Hold 기법)
    $$\bar{A} = \exp(\Delta A)$$
    $$\bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B \approx \Delta B$$
    $$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
    $$y_t = C h_t$$
- **Selective Scan Mechanism (선택적 스캔 메커니즘)**:
    - 기존 LTI SSM과 달리 파라미터 $B, C, \Delta$를 입력 $x_t$의 함수(Linear projection)로 정의하여 입력 가변적(Time-varying)으로 변경함
    $$B_t = \text{Linear}_N(x_t), \quad C_t = \text{Linear}_N(x_t), \quad \Delta_t = \text{Softplus}(\text{Parameter} + \text{Linear}_1(x_t))$$
    - $\Delta_t$가 작으면 현재 입력 $x_t$를 무시하고 이전 상태 $h_{t-1}$를 유지하며, $\Delta_t$가 크면 현재 입력을 적극적으로 반영함

## 4. 핵심 세부 개념 및 부가 설명
- **Hardware-aware Parallel Scan (하드웨어 친화적 병렬 스캔)**:
    - 파라미터가 시간에 따라 변하면 Convolution 표현식을 통한 Fast Fourier Transform (FFT) 병렬화가 불가능해짐
    - Mamba는 GPU 메모리 계층 구조를 활용하여 DRAM의 입출력을 줄이고 High-bandwidth SRAM 상에서 Associative Parallel Scan 알고리즘을 직접 수행하여 병렬 연산 속도를 보장함
- **Mamba Architecture Block**:
    - H3 블록과 Gated MLP 구조를 융합하여 Conv1D, Linear Projection, SiLU 활성화 함수, Selective SSM 레이어를 하나로 결합한 차세대 블록 구성

## 5. 코드 구현 예시 (PyTorch / Python)
```python
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Mamba의 핵심인 Selective State Space Model (SSM) 레이어 구현 클래스
    
    Attributes:
        d_model (int): 입력 및 출력 특성 차원
        d_state (int): SSM 은닉 상태 차원 (N)
        dt_rank (int): Δ projection 차원
    """
    def __init__(self, d_model: int = 64, d_state: int = 16, dt_rank: int = 8) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.dt_rank: int = dt_rank

        # 연속 시간 파라미터 A 초기화 (S4D 초기화 방식)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log: nn.Parameter = nn.Parameter(torch.log(A))  # exp(A_log)로 음수 행렬 유지
        self.D: nn.Parameter = nn.Parameter(torch.ones(d_model))  # Skip connection 파라미터

        # 입력 x로부터 B, C, Δ를 추출하는 선택적 Projection 레이어
        self.x_proj: nn.Linear = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj: nn.Linear = nn.Linear(dt_rank, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM의 순전파 연산 수행
        
        Args:
            x (torch.Tensor): 입력 텐서 (Batch_size, Seq_len, d_model)
            
        Returns:
            torch.Tensor: 출력 텐서 (Batch_size, Seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # 입력 x로부터 B, C, delta 투영
        x_dbl = self.x_proj(x)  # (batch_size, seq_len, dt_rank + 2*d_state)
        delta_rank, B_rank, C_rank = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta_rank))  # (batch_size, seq_len, d_model)
        B = B_rank  # (batch_size, seq_len, d_state)
        C = C_rank  # (batch_size, seq_len, d_state)

        # Sequential State Update (개념 이해용 순차 루프)
        ys = []
        h = torch.zeros(batch_size, d_model, self.d_state, device=x.device)  # 은닉 상태 초기화

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, d_model)
            delta_t = delta[:, t, :]  # (batch_size, d_model)
            B_t = B[:, t, :]  # (batch_size, d_state)
            C_t = C[:, t, :]  # (batch_size, d_state)

            # 이산화 변환: A_bar = exp(delta * A), B_bar = delta * B
            delta_A = torch.exp(delta_t.unsqueeze(-1) * A)  # (batch_size, d_model, d_state)
            delta_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch_size, d_model, d_state)

            # 은닉 상태 갱신: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = delta_A * h + delta_B * x_t.unsqueeze(-1)

            # 출력 계산: y_t = C_t * h_t + D * x_t
            y_t = torch.matmul(h, C_t.unsqueeze(-1)).squeeze(-1) + self.D * x_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch_size, seq_len, d_model)
        return y


if __name__ == "__main__":
    # 간단한 가동 테스트 코드
    batch_size: int = 2
    seq_len: int = 16
    d_model: int = 64

    model: SelectiveSSM = SelectiveSSM(d_model=d_model, d_state=16)
    dummy_input: torch.Tensor = torch.randn(batch_size, seq_len, d_model)
    output: torch.Tensor = model(dummy_input)

    print(f"입력 형상: {dummy_input.shape}")
    print(f"출력 형상: {output.shape}")
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Transformer (Self-Attention) | S4 (Linear Time-Invariant SSM) | Mamba (Selective SSM) |
| :--- | :--- | :--- | :--- |
| **연산 복잡도 (Training)** | $O(N^2)$ | $O(N \log N)$ (Convolution) | $O(N)$ (Parallel Scan) |
| **연산 복잡도 (Inference)** | $O(N)$ (KV Cash 메모리 급증) | $O(1)$ (Recurrent Step) | $O(1)$ (Recurrent Step) |
| **입력 가변성 (Content-aware)** | 매우 우수 (All-to-all matching) | 취약 (고정된 상태 전이) | 매우 우수 (Selective Filtering) |
| **KV 캐시 필요 여부** | 필요함 (메모리 병목 원인) | 불필요함 | 불필요함 |

- **장점**:
    - 초장대 시퀀스(Long Context) 처리 시 메모리 및 시퀀스 길이 증가에 선형 비례하는 탁월한 스케일링 효율성
    - Autoregressive 추론 시 KV 캐시가 필요하지 않아 메모리 소요가 $O(1)$로 제한됨
- **한계점**:
    - 하드웨어 가속기(GPU CUDA C++) 단의 커스텀 Parallel Scan 캘리브레이션 없이는 단순 PyTorch 루프 작성 시 처리 속도가 느려질 수 있음

## 7. 활용 사례 및 응용
- **Mamba-2 & State-Space LM**: 오디오, DNA 시퀀스, 텍스트 생성 등超장문 데이터셋 프레임워크
- **Vision Mamba (Vim / VMamba)**: 2D 이미지 패치 시퀀스에 Mamba 알고리즘을 적용하여 ViT 대비 빠른 속도로 시각 인식 수행
- **Hybrid Transformer-Mamba**: Jamba 등 어텐션 블록과 Mamba 블록을 교차 배치하여 추론 속도와 검색 정밀도를 동시 확보
