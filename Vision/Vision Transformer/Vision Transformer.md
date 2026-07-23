# Vision Transformer (ViT, 비전 트랜스포머)

---
Reference:
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT Documentation](https://huggingface.co/docs/transformers/model_doc/vit)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Vision Transformer (ViT, 비전 트랜스포머)
- **관련 분야/카테고리**: Vision / Transformer / Computer Vision Backbone
- **한 줄 요약**: 이미지 패치(Image Patch)를 텍스트의 토큰(Token)처럼 취급하여, 합성곱(Convolution) 연산 없이 표준 Transformer Encoder만을 사용하여 이미지 분류 및 비전 과제를 수행하는 파운데이션 비전 아키텍처

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 CNN (Convolutional Neural Network) 아키텍처의 한계점
- **지역적 수용 영역 (Local Receptive Field)**: CNN 커널은 국소적인 영역(3x3, 5x5)의 특징만을 수집하므로, 이미지 전체의 글로벌 맥락(Global Context)이나 멀리 떨어진 픽셀 간의 상관관계를 파악하려면 레이어를 매우 깊게 쌓아야 했습니다.
- **귀납적 편향 (Inductive Bias)에 대한 과도한 의존**: CNN은 이미지의 공간적 불변성(Translation Invariance)과 국소성(Locality)이라는 강력한 귀납적 편향을 구조 자체에 내장하고 있어, 대규모 데이터셋(JFT-300M, ImageNet-22k)이 주어졌을 때 표현 능력의 한계에 부딪혔습니다.

### ViT 도입을 통한 핵심 해결 목표
- **글로벌 어텐션(Global Attention)**: 첫 번째 레이어부터 이미지 모든 패치 간의 상호작용을 계산하여 전체 맥락을 한 번에 파악함
- **아키텍처 통합 (Unified Architecture)**: NLP의 표준 Transformer 아키텍처를 최소한의 수정만으로 비전 도메인에 그대로 이식하여, 모달리티 간 장벽을 없애고 대용량 데이터에서 뛰어난 스케일링 법칙(Scaling Law)을 입증했습니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 이미지 패치 패치화 (Patch Extraction & Linear Projection)
입력 이미지 $X \in \mathbb{R}^{H \times W \times C}$를 높이/너비가 $P$인 $N$개의 2D 패치 $X_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$로 분할합니다 (여기서 패치 개수 $N = \frac{HW}{P^2}$).
분할된 패치는 선형 투영(Linear Projection) 행렬 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$를 통과하여 $D$ 차원의 1D 패치 임베딩 텐서로 사영됨

### 2) Class Token ([CLS] Token) 및 Positional Embedding
- **[CLS] Token**: 시퀀스의 맨 앞에 학습 가능한 분류 전용 토큰 $x_{\text{class}} \in \mathbb{R}^{1 \times D}$을 추가함 전체 패치의 어텐션 정보가 [CLS] 토큰으로 집약되어 최종 분류 헤드(MLP Head)에 입력됨
- **Positional Embedding**: 패치의 2D 위치 정보를 유지하기 위해 학습 가능한 1D 위치 임베딩 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$을 패치 임베딩에 더해줍니다:

$$z_0 = \left[ x_{\text{class}}; X_p^1 E; X_p^2 E; \dots; X_p^N E \right] + E_{\text{pos}}$$

### 3) Transformer Encoder 처리
결합된 $z_0$ 텐서는 $L$개의 표준 Transformer Encoder 블록을 통과합니다:

$$z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$$
$$y = \text{LN}(z_L^0) \quad (\text{최종 } [CLS] \text{ 토큰의 출력 추출})$$

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Inductive Bias (귀납적 편향) 차이
- **CNN**: "가까운 픽셀끼리 관련이 높다(Locality)" 및 "위치가 바뀌어도 패턴은 동일하다(Translation Invariance)"는 강한 픽셀 편향이 아키텍처에 하드코딩되어 있음 소규모 데이터셋 학습에 유리함
- **ViT**: 픽셀 공간 구조에 대한 귀납적 편향이 거의 없습니다 (패치 간 2D 구조 정보조차 스스로 학습함). 따라서 소규모 데이터에서는 오버피팅되기 쉽지만, **대규모 데이터셋 사전 학습(Pre-training)** 시 CNN의 성능 한계를 월등히 뛰어넘습니다.

### 2) Hybrid Architecture (하이브리드 아키텍처)
- 완전한 패치 분할 대신 ResNet 등의 CNN을 통과시켜 추출된 Feature Map의 픽셀을 패치로 사용하는 하이브리드 방식도 가능하며, 중소형 데이터셋에서 높은 안정성을 보함

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 활용하여 이미지 패치 분할, Positional Embedding, Transformer Encoder, [CLS] 토큰 분류 헤드를 구현한 기본 `VisionTransformer` 예시 코드함

```python
from typing import Tuple
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """2D 이미지를 1D 패치 임베딩 시퀀스로 변환하는 모듈.
    Conv2d 레이어를 커널 크기=stride=patch_size로 설정하여 효율적으로 패치화 및 사영을 동시에 수행함

    Args:
        img_size (int): 입력 이미지 해상도 (H=W 가정). 기본값 224.
        patch_size (int): 분할할 1개 패치의 해상도 (P). 기본값 16.
        in_channels (int): 입력 이미지 채널 수. 기본값 3.
        embed_dim (int): 사영할 임베딩 차원 수 (D). 기본값 768.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.img_size: int = img_size
        self.patch_size: int = patch_size
        self.num_patches: int = (img_size // patch_size) ** 2

        # Conv2d로 패치 분할 및 선형 투영(Linear Projection) 단번에 처리
        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서. (batch_size, in_channels, img_size, img_size)

        Returns:
            torch.Tensor: 패치 임베딩 텐서. (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) 분류 아키텍처 모듈.

    Args:
        img_size (int): 입력 이미지 해상도.
        patch_size (int): 패치 크기.
        in_channels (int): 입력 채널 수.
        num_classes (int): 분류 클래스 개수.
        embed_dim (int): 임베딩 차원 수.
        depth (int): Transformer Encoder 레이어 개수.
        num_heads (int): Multi-Head Attention 헤드 개수.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12
    ) -> None:
        super().__init__()
        self.patch_embed: PatchEmbedding = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches: int = self.patch_embed.num_patches

        # [CLS] 토큰 및 Positional Embedding 파라미터 정의
        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder 스택
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 최종 분류 헤드 (MLP Head)
        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.head: nn.Linear = nn.Linear(embed_dim, num_classes)

        # 파라미터 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서. (batch_size, in_channels, img_size, img_size)

        Returns:
            torch.Tensor: 클래스 로짓(Logits) 텐서. (batch_size, num_classes)
        """
        batch_size: int = x.shape[0]

        # 1. 패치 분할 및 임베딩 (batch_size, num_patches, embed_dim)
        x = self.patch_embed(x)

        # 2. [CLS] 토큰 추가 (batch_size, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Positional Embedding 가산
        x = x + self.pos_embed

        # 4. Transformer Encoder 통과
        x = self.encoder(x)

        # 5. [CLS] 토큰 추출 및 최종 분류
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out)
        return logits


# 실행 예시
if __name__ == "__main__":
    # 가상의 이미지 텐서 (Batch: 2, 3채널 224x224)
    dummy_img: torch.Tensor = torch.randn(2, 3, 224, 224)

    # ViT-Base (Patch 16x16) 모델 생성
    vit_model: VisionTransformer = VisionTransformer(
        img_size=224, patch_size=16, num_classes=10, embed_dim=192, depth=4, num_heads=4
    )
    output_logits: torch.Tensor = vit_model(dummy_img)

    print(f"입력 이미지 크기: {dummy_img.shape}")
    print(f"ViT 출력 로짓 크기: {output_logits.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | CNN (ResNet, ConvNeXt) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **기본 연산 단위** | 2D Convolution 필터 ($3 \times 3$) | Self-Attention (패치 토큰 간 내적) |
| **수용 영역 (Receptive Field)** | 국소적 (레이어 깊이에 따라 점진 증가) | 전역적 (첫 레이어부터 이미지 전체 조망) |
| **Inductive Bias** | 매우 높음 (Locality, Translation Invariance) | 매우 낮음 (데이터 중심 표현 학습) |
| **소규모 데이터셋 성능** | 우수함 | 상대적으로 저조 (Overfitting 위험) |
| **대규모 데이터셋 Scalability**| 성능 정체 발생 | 대용량 사전 학습 시 압도적 성능 우위 |

### 장점
- **글로벌 컨텍스트 파악 능력이 우수함**: 이미지 내 멀리 떨어진 개체 간 상관관계를 손쉽게 포착.
- **Multimodal 통합 용이성**: NLP의 Transformer와 구조가 완전히 동일하여 시각-언어(VLM) 통합 모델 구축에 직관적함

### 한계점
- **막대한 사전 학습 데이터 필요성**: ImageNet-1k 수준의 소규모 데이터셋만으로는 사전 학습이 어려우며 JFT-300M 등 대용량 데이터셋이 필수적함
- **$O(N^2)$ 패치 수 연산 복잡도**: 이미지 해상도가 높아지면 패치 수가 늘어나 어텐션 연산량이 제곱으로 증가함 (Swin Transformer 등으로 개량됨).

---

## 7. 활용 사례 및 응용

1. **비전 파운데이션 백본 (CLIP, DINOv2, SAM)**
   - Segment Anything Model (SAM), DINOv2 등 최신 비전 기초 모델의 표준 이미지 인코더 백본으로 사용.
2. **Multimodal & VLM (LLaVA, Qwen-VL, GPT-4V)**
   - 시각 언어 모델에서 이미지를 토큰화하여 LLM 패스웨이에 전달하는 비주얼 프론트엔드로 활용.
