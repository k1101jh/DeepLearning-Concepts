# World Model (월드 모델, 세계 모델)

---
Reference:
- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122)
- [Mastering Diverse Domains through World Models (DreamerV3) (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)
- [V-JEPA: Video Joint Embedding Predictive Architecture (Meta AI, 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: World Model (세계 모델 / 공간 및 비디오 내재 시뮬레이터)
- **관련 분야/카테고리**: Basics / Reinforcement Learning / Video Generation / Spatial AI
- **한 줄 요약**: 관찰된 과거 이미지/비디오 시퀀스와 자신의 행동(Action)을 바탕으로, 환경의 미래 상태(Future Latent State) 및 변화 결과를 내부 잠재 공간(Latent Space)에서 시뮬레이션하여 계획 및 의사결정을 수행하는 AI 모델이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 시상 기반 강화학습(Model-Free RL)의 한계점
- **극심한 표본 효율성(Sample Efficiency) 부재**: Model-Free 강화학습(DQN, PPO 등)은 환경(Environment)과 실제 수천만 번 행동을 직접 부딪히며 시행착오(Trial-and-Error)를 거쳐야 하므로 실제 현실 로봇 적용이 불가능했습니다.
- **물리적 세계 규칙 이해 부재**: 모델이 세상의 물리 법칙(공이 떨어지면 튀어오른다, 물체가 막히면 지나가지 못한다)을 알지 못하고 오직 보상(Reward) 신호에만 반응했습니다.

### World Model 도입을 통한 핵심 해결 목표
- **머릿속 상상 학습 (Dreaming / Imagination Training)**: 실제 환경에 부딪히지 않고, 월드 모델이 생성해내는 내부 잠재 공간 시뮬레이션(Imaginated Trajectory) 안에서 거대한 정책(Policy)을 초고속으로 스스로 훈련시킵니다.
- **물리적 물리 규칙 자율 습득**: 비디오의 다음 프레임 피처를 예측(Self-Supervised Learning)하면서 물리적 입체 구조, 연속성, 물체의 영속성(Object Permanence)을 자연스럽게 습득합니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) World Model의 3대 핵심 구조 (Ha & Schmidhuber)

```
[Observation Image x_t] ---> [Vision Model V (VAE / ViT)] ---> [Latent Code z_t]
                                                                      |
[Action a_t] --------------------------------------------------------+---> [Memory Model M (RNN / Transformer)] ---> [Next State Prediction z_{t+1}]
                                                                      |
                                                                      v
                                                           [Controller C (Policy)] ---> [Action a_{t+1}]
```

1. **Vision Model (V)**: 복잡한 2D 이미지 관찰 $x_t$를 저차원 잠재 벡터 $z_t$로 압축 (VAE 또는 ViT Encoder).
2. **Memory Model (M)**: 과거 잠재 상태 $z_{\le t}$와 행동 $a_t$를 기반으로 다음 시점의 잠재 상태 $z_{t+1}$와 보상 $r_{t+1}$을 예측하는 시퀀스 모델 (RNN, Transformer, RSSM).
3. **Controller (C)**: 잠재 상태 $z_t$와 M의 내부 기억만을 보고 최적의 행동 $a_t$를 결정하는 정책 네트워크.

### 2) V-JEPA (Joint Embedding Predictive Architecture) 메커니즘
Yann LeCun 교수 팀의 V-JEPA는 픽셀(Pixel)을 복원하는 대신, 고차원 잠재 표현(Latent Representation) 공간 상에서 가려진(Masked) 비디오 패치의 피처를 직접 예측하여 불필요한 세부 노이즈(나뭇잎의 흔들림 등)를 무시하고 핵심 물리적 맥락을 표현합니다:

$$\mathcal{L}_{\text{JEPA}} = \| s_y - s_{\hat{y}} \|^2 \quad (\text{잠재 표현 공간 상의 예측 오차 최소화})$$

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) RSSM (Recurrent State Space Model - Dreamer 시리즈)
- 결정론적(Deterministic) 고차원 기억 상태 $h_t$와 확률적(Stochastic) 비결정론적 상태 $z_t$를 동시에 결합하여, 환경의 불확실한 미래 변동성까지 정확히 확률 분포로 모델링하는 최신 월드 모델 아키텍처.

### 2) Video World Simulators (Sora, Gen-2)
- 텍스트 프롬프트와 현재 프레임을 입력받아 수 초 후의 물리적 비디오 장면 변화를 일관성 있게 생성해내는 거대 비디오 디퓨전 기반 월드 모델.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 현재 잠재 상태 $z_t$와 로봇 행동 $a_t$가 주어졌을 때, 미래의 잠재 상태 $z_{t+1}$을 예측하는 간단한 Recurrent World Model (M Model)의 PyTorch 연산 예시 코드입니다.

```python
from typing import Tuple
import torch
import torch.nn as nn


class RecurrentWorldModel(nn.Module):
    """현재 관찰 잠재 상태 z_t와 에이전트 행동 a_t를 입력받아
    미래 잠재 상태 z_{t+1} 및 보상 r_{t+1}을 예측하는 World Model (Memory Module).

    Args:
        latent_dim (int): 관찰 잠재 벡터 차원 수. 기본값 64.
        action_dim (int): 에이전트 행동 벡터 차원 수. 기본값 4.
        hidden_dim (int): 순환 신경망(GRU) 숨은층 차원 수. 기본값 256.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim

        # 입력 이음새: [z_t, a_t] 인코딩
        self.input_embed: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ELU()
        )

        # 상태 전이 순환 모듈 (GRU Cell)
        self.gru_cell: nn.GRUCell = nn.GRUCell(hidden_dim, hidden_dim)

        # 미래 잠재 상태 z_{t+1} 예측 헤드
        self.predict_next_z: nn.Linear = nn.Linear(hidden_dim, latent_dim)

        # 미래 보상 r_{t+1} 예측 헤드
        self.predict_reward: nn.Linear = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """미래 1스텝 상태를 상상(Imagine)하여 예측합니다.

        Args:
            z_t (torch.Tensor): 현재 시점 관찰 잠재 벡터. (batch_size, latent_dim)
            a_t (torch.Tensor): 현재 시점 행한 행동 벡터. (batch_size, action_dim)
            h_prev (torch.Tensor): 이전 시점 GRU 기억 상태. (batch_size, hidden_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 예측된 다음 시점 잠재 상태 z_{t+1} (batch_size, latent_dim)
                - 갱신된 GRU 기억 상태 h_next (batch_size, hidden_dim)
                - 예측된 다음 보상 r_{t+1} (batch_size, 1)
        """
        # 1. z_t와 a_t 결합
        x = torch.cat([z_t, a_t], dim=-1)
        x_embed = self.input_embed(x)

        # 2. GRU 상태 전이
        h_next = self.gru_cell(x_embed, h_prev)

        # 3. 미래 상태 및 보상 예측
        pred_z_next = self.predict_next_z(h_next)
        pred_reward = self.predict_reward(h_next)

        return pred_z_next, h_next, pred_reward


# 실행 예시
if __name__ == "__main__":
    batch_size: int = 4
    world_model: RecurrentWorldModel = RecurrentWorldModel(latent_dim=64, action_dim=4, hidden_dim=256)

    # 가상의 현재 상태 z_t, 행동 a_t, 이전 기억 h_prev 생성
    dummy_z_t: torch.Tensor = torch.randn(batch_size, 64)
    dummy_a_t: torch.Tensor = torch.randn(batch_size, 4)
    dummy_h_prev: torch.Tensor = torch.zeros(batch_size, 256)

    pred_z_next, h_next, pred_reward = world_model(dummy_z_t, dummy_a_t, dummy_h_prev)

    print(f"현재 잠재 상태 z_t 크기: {dummy_z_t.shape}")
    print(f"예측된 미래 잠재 상태 z_{{t+1}} 크기: {pred_z_next.shape}")
    print(f"예측된 미래 보상 r_{{t+1}} 샘플:\n{pred_reward.squeeze()}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Model-Free RL (PPO, SAC) | Model-Based RL (World Model / DreamerV3) |
| :--- | :--- | :--- |
| **상상 속 학습 (Imagination)** | 불가 (실제 상호작용만 가능) | **가능 (상상 속에서 무한 상호작용 및 정책 학습)** |
| **샘플 효율성 (Sample Efficiency)** | 극도로 낮음 (수천만 번 시도 필요) | **매우 높음 (100배 적은 데이터로 수렴)** |
| **세계 물리 법칙 이해** | 없음 | **높음 (스스로 예측하며 물리성 탐구)** |
| **환경 변화 적응력** | 처음부터 다시 훈련 필요 | **월드 모델 재사용 후 쾌속 정책 튜닝** |

### 장점
- **초고속 표본 효율성**: 현실 세계의 위험한 시도를 최소화하고 상상 속에서 안전하게 정책 훈련.
- **자율주행 및 파운데이션 에이전트의 기반**: 물리학적 타당성을 스스로 습득하여 미래 예측 수행.

### 한계점
- 월드 모델의 미래 예측이 부정확할 경우(Compounding Error), 상상 속 정책 학습이 잘못된 방향으로 편향될 위험이 있음.

---

## 7. 활용 사례 및 응용

1. **자율주행 비디오 시뮬레이터 (Sora, Wayve World Model)**
   - 차선 변경, 악천후, 돌발 보행자 등장 시 수초 후 도로 미래 상황을 렌더링하고 제어 계획 수립.
2. **로보틱스 상상 제어 (DreamerV3)**
   - 마인크래프트(Minecraft) 게임에서 0부터 지이아몬드 검을 제작하는 장기 계획(Long-horizon) 과제를 월드 모델 상상 학습만으로 세계 최초로 해결.
