# Vision-Language Model (VLM, 시각-언어 모델)

---
Reference:
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP) (Radford et al., ICML 2021)](https://arxiv.org/abs/2103.00020)
- [Visual Instruction Tuning (LLaVA) (Liu et al., NeurIPS 2023)](https://arxiv.org/abs/2304.08485)
- [Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., NeurIPS 2022)](https://arxiv.org/abs/2204.14198)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Vision-Language Model (VLM, 시각-언어 파운데이션 모델)
- **관련 분야/카테고리**: LLM / Multi-Modal / Vision & Language Alignment
- **한 줄 요약**: 이미지(시각 정보)와 텍스트(언어 정보)의 모달리티 간 임베딩 공간을 연결/정렬(Alignment)하거나, 비전 인코더와 대형 언어 모델(LLM)을 프로젝션 커넥터로 결합하여 이미지에 대한 자연어 이해, 추론 및 질의응답을 수행하는 다중모달 AI 모델이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 단일 모달리티 (Unimodal) 모델의 한계점
- **시각 모델 (Vision Only)**: 이미지 분류(Classification)나 객체 탐지(Detection) 모델은 고정된 1,000개 클래스 레이블(예: ImageNet) 안에서만 작동하여, 새로운 클래스나 자연어 문맥 지시를 이해하지 못하는 닫힌 세계(Closed-world) 제약이 있었습니다.
- **언어 모델 (Language Only)**: LLM(GPT-3, LLaMA)은 인간 언어 추론 능력이 뛰어나지만 시각적 세상을 볼 수 없어 텍스트 외부의 물리적 이미지 정보에 접근하지 못했습니다.

### VLM 도입을 통한 핵심 해결 목표
- **열린 세계(Open-world) 시각 인식**: 대규모 웹 이미지-텍스트 쌍(Image-Text Pairs)으로 학습하여 사전 정의되지 않은 클래스도 Zero-shot으로 즉시 인식합니다.
- **이미지 기반 고차원 자연어 추론 (Visual Reasoning)**: 이미지 패치 토큰을 LLM의 토큰 시퀀스로 이식하여, 이미지 속 복잡한 차트 분석, 시각적 질의응답(VQA), 추론을 가능하게 합니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) CLIP 타입: 대조 학습 (Contrastive Learning Alignment)

![CLIP Architecture](https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png)
> **Figure 1. CLIP 대조 학습 및 Zero-shot 분류 메커니즘 (출처: [OpenAI CLIP](https://github.com/openai/CLIP))**

- 이미지 인코더 $f_I(x)$와 텍스트 인코더 $f_T(y)$를 준비합니다.
- Batch Size $N$개의 (이미지, 텍스트) 대각선 쌍의 코사인 유사도는 극대화(Maximize)하고, 대각선이 아닌 이종 쌍의 유사도는 극소화(Minimize)하는 InfoNCE Loss로 두 모달리티 임베딩 공간을 유기적으로 정렬합니다:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j)/\tau)}$$

### 2) LLaVA 타입: Vision Encoder + Projection Connector + LLM

```
[Input Image] -> [Vision Encoder (ViT)] -> [Linear Projector / MLP] -> [Visual Tokens] \
                                                                                       -> [LLM Decoder] -> [Text Response]
                                               [Text Prompt] -------> [Text Tokens]  /
```

- **Vision Encoder (ViT)**: 이미지를 입력받아 이미지 패치 토큰 임베딩 $Z_v$ 추출.
- **Projection Connector (MLP/Cross-Attention)**: 비전 임베딩 차원을 LLM의 입력 차원 $H$로 사영(Projection)하여 비전 토큰 $H_v = W \cdot Z_v$로 변환.
- **LLM Decoder**: 텍스트 토큰과 비전 토큰 시퀀스를 결합하여 다음 토큰(Next Token Prediction)을 자승 생성.

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Visual Instruction Tuning (시각 지시 Tuning)
- 텍스트 LLM의 Instruction Tuning처럼, (이미지, 질문, 답변) 형태의 지시 데이터셋(예: LLaVA-Instruct-150K)을 구축하여 비전 커넥터와 LLM을 미세조정함으로써 단순 이미지 묘사를 넘어 복잡한 비주얼 추론 능력을 획득시킵니다.

### 2) Perceiver Resampler & Q-Former
- 이미지 해상도가 높아지면 비전 토큰 수가 수천 개로 폭증하여 LLM 연산에 과부하가 걸립니다.
- Flamingo의 Perceiver Resampler나 BLIP-2의 Q-Former는 학습 가능한 쿼리 토큰을 사용하여 고정된 $K$개(예: 32개, 64개)의 고축적 비전 토큰으로 압축(Resampling)하는 훌륭한 해결책을 제시합니다.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 활용하여 Vision Encoder(ViT)의 출력을 선형 프로젝션(Linear Projection) 레이어로 사영하여 LLM 임베딩 입력에 결합하는 LLaVA 스타일의 간단한 VLM 패스웨이 모듈 예시입니다.

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn


class SimpleVLMConnector(nn.Module):
    """Vision Encoder(ViT)의 비주얼 피처를 LLM 임베딩 공간 차원으로 변환 및 시퀀스 결합하는 VLM 커넥터 모듈.

    Args:
        vision_dim (int): 비전 인코더 출력 임베딩 차원 수. 기본값 768.
        llm_dim (int): 대형 언어 모델(LLM) 히든 차원 수. 기본값 4096.
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 4096) -> None:
        super().__init__()
        # 2-Layer MLP 프로젝션 커넥터 (GELU 활성화)
        self.projector: nn.Sequential = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """비주얼 토큰과 텍스트 토큰 시퀀스를 하나로 결합합니다.

        Args:
            image_features (torch.Tensor): ViT로부터 추출된 비전 피처 텐서. (batch_size, num_patches, vision_dim)
            text_embeds (torch.Tensor): LLM 텍스트 임베딩 텐서. (batch_size, text_seq_len, llm_dim)

        Returns:
            torch.Tensor: LLM 디코더에 입력될 통합 다중모달 토큰 텐서. (batch_size, num_patches + text_seq_len, llm_dim)
        """
        # 1. 비전 피처를 LLM 차원으로 사영: (batch_size, num_patches, llm_dim)
        visual_tokens: torch.Tensor = self.projector(image_features)

        # 2. 비전 토큰 시퀀스와 텍스트 토큰 시퀀스를 순차적으로 Concatenate
        multimodal_embeds: torch.Tensor = torch.cat((visual_tokens, text_embeds), dim=1)

        return multimodal_embeds


# 실행 예시
if __name__ == "__main__":
    batch_size: int = 2
    num_patches: int = 196    # ViT 14x14 패치 토큰 수
    text_seq_len: int = 32    # 텍스트 프롬프트 토큰 수
    vision_dim: int = 1024
    llm_dim: int = 4096       # LLaMA-7B 히든 차원

    # 가상의 비전 피처 및 텍스트 임베딩 생성
    dummy_vision_feats: torch.Tensor = torch.randn(batch_size, num_patches, vision_dim)
    dummy_text_embeds: torch.Tensor = torch.randn(batch_size, text_seq_len, llm_dim)

    vlm_connector: SimpleVLMConnector = SimpleVLMConnector(vision_dim=vision_dim, llm_dim=llm_dim)
    combined_tokens: torch.Tensor = vlm_connector(dummy_vision_feats, dummy_text_embeds)

    print(f"비전 피처 크기: {dummy_vision_feats.shape}")
    print(f"텍스트 임베딩 크기: {dummy_text_embeds.shape}")
    print(f"LLM 입력 통합 다중모달 시퀀스 크기: {combined_tokens.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Unimodal Vision Model (ResNet/ViT) | Unimodal LLM (GPT-4) | Vision-Language Model (VLM) |
| :--- | :--- | :--- | :--- |
| **입력 모달리티** | 이미지 단독 | 텍스트 단독 | **이미지 + 텍스트 동시 입력** |
| **인식 가능 범위** | 고정된 닫힌 세계 (Closed-world) | 텍스트 세계 | **열린 세계 (Open-world Zero-shot)** |
| **추론 능력** | 라벨 분류 위주 | 문맥 추론 뛰어남 | **이미지를 관찰하고 세부 문맥 추론 (VQA)** |
| **학습 데이터** | 라벨링된 카테고리 데이터 | 대규모 웹 텍스트 | **대규모 웹 (이미지, 텍스트) 멀티모달 데이터** |

### 장점
- **자연어로 자유로운 이미지 제어**: "이 이미지에서 가장 오래된 건물은 어디 있어?"와 같은 대화형 시각 질의응답 가능.
- **Zero-shot 전이 학습**: 재학습 없이 텍스트 프롬프트 변경만으로 새로운 도메인 이미지 분류 가능.

### 한계점
- 고해상도 이미지를 처리할 때 비전 토큰 수가 증가하여 메모리와 연산량이 급증함.

---

## 7. 활용 사례 및 응용

1. **차세대 AI 에이전트 & 자율주행**
   - 도로 상황 이미지를 실시간 분석하고 "좌측 도로 보행자 위험" 판단 문장을 생성하는 의사결정 모듈.
2. **로보틱스 (VLA Models - OpenVLA, RT-2)**
   - VLM의 시각-언어 백본을 로봇 조종 액션 출력과 연결하여 인간의 언어 지시문으로 집안일 로봇 조종.
3. **의료 영상 진단 보조 및 OCR 문서 분석**
   - X-ray/CT 이미지를 보고 진단 소견서 초안을 자동 작성하거나 서식 문서 다이어그램 분석.
