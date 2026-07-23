# Transformer (트랜스포머)

---
Reference:
- [Attention Is All You Need (Vaswani et al., NIPS 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Transformer (트랜스포머 아키텍처)
- **관련 분야/카테고리**: Transformer / NLP / Computer Vision / Multi-modal
- **한 줄 요약**: 순차적(Sequential) 연산 중심의 RNN/LSTM 대신 셀프 어텐션(Self-Attention) 메커니즘만을 활용하여 모든 토큰 간의 관련성을 병렬(Parallel)로 계산하는 혁신적인 신경망 아키텍처이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 RNN / LSTM 및 CNN 계열의 한계점
- **순차 연산과 병렬화 한계**: 기존 RNN 기반 모델은 시점 $t$의 Hidden State $h_t$를 계산하기 위해 이전 시점 $h_{t-1}$의 계산이 반드시 완료되어야 합니다. 이는 GPU의 높은 병렬 연산 성능을 활용하지 못해 대용량 데이터 학습 속도를 크게 저하시켰습니다.
- **장기 의존성 문제 (Long-Range Dependency / Vanishing Gradient)**: 텍스트나 시퀀스의 길이가 길어질수록 시퀀스의 앞단 정보가 뒤로 전달되면서 손실되거나 왜곡되는 한계가 존재했습니다.
- **지역적 필터의 한계 (CNN)**: CNN은 수용 영역(Receptive field)을 넓히기 위해 레이어를 깊게 쌓아야만 멀리 떨어진 픽셀/토큰 간 관계를 파악할 수 있었습니다.

### Transformer 도입을 통한 핵심 해결 목표
- **완전 병렬화(Parallelization)**: 시퀀스 전체를 한 번에 입력받아 어텐션 매트릭스로 병렬 처리함으로써 데이터셋 및 모델의 확장성(Scalability)을 극대화했습니다.
- **직접적인 토큰 간 정보 교환**: 임의의 두 토큰 간 거리에 상관없이 단 1번의 연산($O(1)$ path length)으로 상호작용 및 맥락 정보를 유기적으로 수집합니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 아키텍처 전체 구조

![Encoder and Decoder](./images/Encoder%20and%20Decoder.png)
> **Figure 1. Transformer 인코더-디코더 구조 (출처: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/))**

Transformer는 인코더(Encoder) 스택과 디코더(Decoder) 스택으로 구성되며, 핵심 블록은 **Multi-Head Self-Attention**, **Positional Encoding**, **Feed-Forward Network (FFN)**, **Layer Normalization & Residual Connection**입니다.

### 2) Scaled Dot-Product Attention 공식
입력 텐서를 행렬 $Q$ (Query), $K$ (Key), $V$ (Value)로 사영한 뒤, 다음 공식을 적용합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

- $Q, K, V$: 차원이 각각 $(d_{model}, d_k), (d_{model}, d_k), (d_{model}, d_v)$인 가중치 행렬로 사영된 입력 토큰 벡터
- $QK^T$: 토큰 간 유사도(Similarity score) 행렬
- $\sqrt{d_k}$: 내적(Dot-product) 값의 차원이 커질수록 softmax 기울기가 소실되는 것을 방지하는 스케일링 인자

### 3) Multi-Head Attention 메커니즘
서로 다른 representation 공간에서 토큰 간의 관계를 수집하기 위해 $h$개의 헤드로 split하여 독립적으로 Attention을 계산한 뒤 연결(Concatenate)합니다:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

![Encoder and Decoder2](./images/Encoder%20and%20Decoder2.png)
> **Figure 2. Multi-Head Attention 분할 및 결합 과정 (출처: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/))**

### 4) Positional Encoding (위치 인코딩)
순서 개념이 없는 Self-Attention의 한계를 극복하기 위해, 각 토큰 위치 $pos$와 차원 인덱스 $i$에 따른 주기적 삼각함수 벡터를 입력 임베딩에 더해줍니다:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Masked Multi-Head Attention (Decoder)
- 디코더에서는 미래 토큰의 정보를 참조하지 못하도록 Self-Attention 시 $i < j$ 인 위치의 어텐션 스코어 행렬 상단에 $-\infty$ (또는 매우 큰 음수)를 할당하여 Softmax 통과 후 확률값이 0이 되도록 마스킹(Causal Masking)합니다.

### 2) Pre-LN vs Post-LN
- **Post-LN (원 논문 구조)**: Sub-layer 연산 후 Add & LayerNorm 적용 ($x_{l+1} = \text{LN}(x_l + \text{SubLayer}(x_l))$). 깊은 레이어 학습 시 Warmup이 필수적임.
- **Pre-LN (최신 LLM 표준)**: Sub-layer 연산 전 LayerNorm 적용 ($x_{l+1} = x_l + \text{SubLayer}(\text{LN}(x_l))$). 학습 안정성이 뛰어나며 최근 GPT-3, LLaMA 등 대부분의 구조에서 표준으로 사용됨.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 이용한 Multi-Head Attention 기반의 Transformer Encoder Layer 기본 구현 클래스 예시입니다.

```python
from typing import Optional
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """PyTorch 기반의 Transformer Encoder Layer 클래스.
    Pre-LN 구조와 Multi-Head Attention, FFN(Feed-Forward Network)을 구사합니다.

    Args:
        d_model (int): 토큰 임베딩 차원 수.
        nhead (int): Self-Attention 헤드 개수.
        dim_feedforward (int): Feed-Forward 신경망 내부 숨은층 차원.
        dropout (float, optional): 드롭아웃 비율. 기본값은 0.1.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Multi-Head Self-Attention 모듈
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # Feed-Forward Network
        self.linear1: nn.Linear = nn.Linear(d_model, dim_feedforward)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear2: nn.Linear = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization 모듈 (Pre-LN 구조 준비)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.dropout1: nn.Dropout = nn.Dropout(dropout)
        self.dropout2: nn.Dropout = nn.Dropout(dropout)

        self.activation: nn.GELU = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """인코더 레이어 순전파 연산을 수행합니다.

        Args:
            src (torch.Tensor): 입력 시퀀스 텐서. 크기: (batch_size, seq_len, d_model)
            src_mask (Optional[torch.Tensor], optional): 마스킹 텐서. 기본값은 None.

        Returns:
            torch.Tensor: 레이어를 통과한 출력 텐서. 크기: (batch_size, seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention Sub-Layer (Pre-LN 적용)
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(
            query=src_norm, key=src_norm, value=src_norm, attn_mask=src_mask
        )
        # Residual Connection
        src = src + self.dropout1(attn_output)

        # 2. Feed-Forward Network Sub-Layer (Pre-LN 적용)
        src_norm2 = self.norm2(src)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        # Residual Connection
        src = src + self.dropout2(ffn_output)

        return src


# 실행 예시
if __name__ == "__main__":
    batch_size: int = 2
    seq_len: int = 16
    d_model: int = 512
    nhead: int = 8

    # 가상의 토큰 시퀀스 텐서 생성
    dummy_input: torch.Tensor = torch.randn(batch_size, seq_len, d_model)

    encoder_layer: TransformerEncoderLayer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead
    )
    output: torch.Tensor = encoder_layer(dummy_input)

    print(f"입력 텐서 형태: {dummy_input.shape}")
    print(f"출력 텐서 형태: {output.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | RNN / LSTM | CNN (ResNet 등) | Transformer |
| :--- | :--- | :--- | :--- |
| **병렬 처리 능력** | 불가 ($O(N)$ sequential steps) | 우수 (수용 영역 제한) | 최고 ($O(1)$ parallel attention) |
| **최대 경로 거리** | $O(N)$ (시퀀스 길이 $N$) | $O(\log_k N)$ | $O(1)$ (모든 토큰 간 직접 연결) |
| **계산 복잡도** | $O(N \cdot d^2)$ | $O(N \cdot k \cdot d^2)$ | $O(N^2 \cdot d)$ (토큰 수가 매우 크면 오버헤드 발생) |
| **위치 정보 인식** | 순차 입력 자체로 자연스럽게 반영 | 공간 커널로 반영 | Positional Encoding 필수 추가 |

### 장점
- **대용량 데이터 확장성**: 완전히 병렬화된 구조로 수천 억 개 파라미터의 대형 아키텍처 학습 가능.
- **글로벌 컨텍스트 파악**: 시퀀스/이미지 전반의 맥락 정보를 효율적으로 추출.

### 한계점
- **$O(N^2)$ 메모리 연산 복잡도**: 시퀀스 길이 $N$이 길어질수록 어텐션 맵 메모리가 제곱으로 폭증함 (이를 완화하기 위해 FlashAttention, Linear Attention 등장).

---

## 7. 활용 사례 및 응용

1. **대규모 언어 모델 (LLM)**
   - GPT-3/GPT-4, LLaMA, Gemini, Claude 등 최신 프론티어 LLM의 절대적인 핵심 기반 구조.
2. **Vision Transformer (ViT)**
   - 이미지를 패치(Patch) 단위 토큰으로 분할하여 컴퓨터 비전 분류, 획득 및 생성 작업에 트랜스포머를 적용.
3. **Multi-modal Models (CLIP, Flamingo, GPT-4V)**
   - 텍스트, 이미지, 오디오 등 이종 모달리티를 공통 어텐션 공간에서 통합하여 처리.