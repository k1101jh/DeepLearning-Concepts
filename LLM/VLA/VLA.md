# Vision-Language-Action Model (VLA, 시각-언어-행동 파운데이션 모델)

---
Reference:
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (Brohan et al., CoRL 2023)](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model (Kim et al., CoRL 2024)](https://arxiv.org/abs/2406.09246)
- [OpenVLA GitHub Repository](https://github.com/openvla/openvla)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Vision-Language-Action Model (VLA, 시각-언어-행동 모델)
- **관련 분야/카테고리**: LLM / Multi-Modal / Robotics / Embodied AI
- **한 줄 요약**: 이미지(카메라 시각 정보)와 자연어 명령어 지시문을 입력받아, 사전 학습된 VLM/LLM의 풍부한 웹 지식을 바탕으로 로봇의 물리적 조종 제어 명령(6-DoF End-Effector Pose & Gripper Action)을 토큰 단위로 직접 출력하는 앤드투앤드(End-to-End) 구체화된 AI(Embodied AI) 모델

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 로봇 제어 (Robotics Policy) 방식의 한계점
- **일반화 능력 부족 (Lack of Generalization)**: 전통적인 모듈형 로봇 파이프라인(객체 인식 $\to$ 포즈 추정 $\to$ 경로 계획 $\to$ 관절 제어)은 특정 환경과 사물에 고정되어 있어서, 처음 보는 물체(Unseen objects)나 새로운 조명/배경 환경에서 동작을 실패했습니다.
- **상식적 추론 능력 부재**: "사과를 집어서 접시 위에 놓아줘"라는 지시문 대신 정형화된 좌표를 입력해야 했으며, "배가 고플 때 먹을 수 있는 과일을 치워줘"와 같은 시각적 상식 및 추론이 불가능했습니다.

### VLA 도입을 통한 핵심 해결 목표
- **웹 규모 대용량 지식의 이식**: 텍스트와 이미지를 대규모로 학습한 VLM(예: PaLM-E, Prismatic VLM)의 가중치를 기반으로 로봇 동작(Action) 데이터를 미세조정하여 일반화 능력을 획득했습니다.
- **자연어 지시문 직관적 제어**: 인간의 모호한 복합 자연어 지시문을 곧바로 6자유도 말단 장치(End-effector) 조작 명령으로 번역함

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 액션의 토큰화 (Action Discretization & Tokenization)
VLA의 핵심 아이디어는 연속적인 로봇 제어 신호를 언어 모델의 텍스트 토큰 어휘(Vocabulary)처럼 이산화(Discretization)하는 것함

로봇의 7차원 동작 상태 벡터 $a_t$:
$$a_t = [\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper\_state}]$$

- 각 위치/회전 변화량을 256개의 이산 구간(Bin)으로 양자화하여, 텍스트 Vocabulary 상의 특정 토큰 ID(예: 토큰 ID 32000 ~ 32256)에 1:1 매핑함

### 2) VLA 아키텍처 연산 파이프라인

```
[Camera Image] ---------> [Vision Encoder (ViT)] ---> [Visual Tokens] \
                                                                        ---> [LLM / VLM Backbone] ---> [Action Tokens (Bins)] ---> [Robot Controller]
[Language Instruction] -> [Text Tokenizer] ----------> [Text Tokens]   /
```

- **입력 시퀀스**: $\text{Tokens} = [\text{Visual Tokens}, \text{Instruction Text Tokens}]$
- **출력 토큰**: 모델은 자동거듭(Autoregressive) 방식으로 다음 7개 토큰(Action Bins)을 순차적으로 생성함
- **데코딩**: 출력된 토큰 ID를 다시 연속적인 실수값 $[\Delta x, \Delta y, \Delta z, \dots]$로 역양자화하여 로봇 모터로 전송함

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Action Bins (액션 빈 양자화)
- 연속적인 변위 $[-0.1, 0.1] \text{ meters}$ 범위를 256개의 동일한 간격의 Bin으로 균등 분할함
- 예: Bin 0 = $-0.1\text{m}$, Bin 128 = $0.0\text{m}$, Bin 255 = $+0.1\text{m}$
- 이를 통해 교사 강요(Teacher-forcing) 교차 엔트로피 손실(Cross-Entropy Loss)을 그대로 사용하여 로봇 정책을 손쉽게 학습시킬 수 있음

### 2) Embodied Instruction Following (구체화된 지시 이행)
- 단순히 "컵을 집어라"뿐만 아니라 "테이블 위에 떨어진 쓰레기를 쓰레기통에 버려라"와 같이 논리적 단계가 필요한 태스크를 시각 정보를 관찰하며 단계적으로 이행함

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 연속적인 7-DoF 로봇 제어 신호를 256개의 이산 액션 토큰 ID로 양자화(Quantize) 및 역양자화(Dequantize)하는 VLA Action Tokenizer의 핵심 PyTorch 구현 코드함

```python
from typing import Tuple
import torch
import torch.nn as nn


class VLAActionTokenizer(nn.Module):
    """연속적인 로봇 말단 제어 변위(7-DoF)를 256개의 이산 토큰 ID로 양자화 및 역양자화하는 VLA 토크나이저 모듈.

    Args:
        num_bins (int, optional): 양자화 이산 구간 개수. 기본값 256.
        min_action (float, optional): 최소 변위 제한값. 기본값 -0.1.
        max_action (float, optional): 최대 변위 제한값. 기본값 0.1.
        vocab_start_idx (int, optional): LLM 어휘 집합 내 액션 토큰 시작 인덱스. 기본값 32000.
    """

    def __init__(
        self,
        num_bins: int = 256,
        min_action: float = -0.1,
        max_action: float = 0.1,
        vocab_start_idx: int = 32000
    ) -> None:
        super().__init__()
        self.num_bins: int = num_bins
        self.min_action: float = min_action
        self.max_action: float = max_action
        self.vocab_start_idx: int = vocab_start_idx

        # Bin 경계값 텐서 생성
        self.register_buffer(
            "bins", torch.linspace(min_action, max_action, num_bins)
        )

    def action_to_tokens(self, continuous_action: torch.Tensor) -> torch.Tensor:
        """연속적인 실수 로봇 액션 벡터를 LLM 토큰 ID 텐서로 변환함

        Args:
            continuous_action (torch.Tensor): 7-DoF 연속 액션 텐서. 크기: (batch_size, 7)

        Returns:
            torch.Tensor: 이산 액션 토큰 ID 텐서. 크기: (batch_size, 7)
        """
        # 액션 범위 클리핑
        clipped_action = torch.clamp(continuous_action, self.min_action, self.max_action)

        # 가장 가까운 Bin 인덱스 검색 (0 ~ num_bins-1)
        discretized_indices = torch.bucketize(clipped_action, self.bins)
        discretized_indices = torch.clamp(discretized_indices, 0, self.num_bins - 1)

        # LLM Vocab 인덱스로 오프셋 가산
        action_token_ids = discretized_indices + self.vocab_start_idx
        return action_token_ids

    def tokens_to_action(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """LLM이 생성한 액션 토큰 ID 텐서를 로봇이 실행 가능한 연속 실수 액션으로 역양자화함

        Args:
            action_token_ids (torch.Tensor): 생성된 액션 토큰 ID. 크기: (batch_size, 7)

        Returns:
            torch.Tensor: 복원된 7-DoF 실수 액션 텐서. 크기: (batch_size, 7)
        """
        # Vocab 오프셋 제거
        bin_indices = action_token_ids - self.vocab_start_idx
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # 해당 Bin의 실수 중심값 복원
        reconstructed_action = self.bins[bin_indices]
        return reconstructed_action


# 실행 예시
if __name__ == "__main__":
    tokenizer: VLAActionTokenizer = VLAActionTokenizer()

    # 가상의 로봇 연속 7-DoF 액션 [dx, dy, dz, droll, dpitch, dyaw, gripper]
    sample_action: torch.Tensor = torch.tensor([
        [0.025, -0.080, 0.001, 0.050, -0.010, 0.000, 1.0],
        [-0.090, 0.010, 0.075, -0.030, 0.020, -0.040, 0.0]
    ])

    # 1. 실수 액션 -> 액션 토큰 ID 변환
    token_ids: torch.Tensor = tokenizer.action_to_tokens(sample_action)
    # 2. 액션 토큰 ID -> 실수 액션 복원
    recovered_action: torch.Tensor = tokenizer.tokens_to_action(token_ids)

    print(f"원본 연속 액션 (Batch 0):\n{sample_action[0]}")
    print(f"생성된 액션 토큰 ID (Batch 0):\n{token_ids[0]}")
    print(f"역양자화 복원 액션 (Batch 0):\n{recovered_action[0]}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | 전통적 로봇 파이프라인 | 일반 RL/BC 로봇 정책 | Vision-Language-Action (VLA) |
| :--- | :--- | :--- | :--- |
| **자연어 지시문 처리** | 불가 (정형 좌표 입력) | 제한적 (단순 템플릿) | **뛰어남 (자연스러운 복합 지시문 이해)** |
| **미조작 객체 일반화** | 매우 저조 | 저조함 | **월등함 (VLM 웹 지식 전이)** |
| **학습 방식** | 모듈별 개별 캘리브레이션 | 단순 모션 모방 | **End-to-End Cross-Entropy 토큰 예측** |
| **추론 파라미터 및 속도**| 소형 (고속) | 소형 (고속) | 대형 (~7B 파라미터, 5~10 Hz 추론) |

### 장점
- **시각-언어 데이터 지식의 성공적 전이**: 인터넷의 이미지-텍스트 데이터 지식을 물리적 제어로 연결함
- **강력한 Zero-Shot 일반화**: 훈련 중 본 적 없는 새로운 음식이나 물체 집기 성공률이 대폭 높음.

### 한계점
- 7B 이상 대형 VLM을 로봇 컨트롤러에 탑재할 때 추론 주파수(Hz)가 낮아 실시간 고속 반응 제어에 제약이 있을 수 있음

---

## 7. 활용 사례 및 응용

1. **지능형 가정용 집안일 로봇 (OpenVLA, RT-2)**
   - "서랍을 열어 약통을 꺼내줘"와 같은 명령을 받고 시각 정보를 분석하여 손잡이를 당기고 물건을 집는 복합 액션 수행.
2. **산업용 물류 자동화 피킹 (Robotic Picking)**
   - 다양한 형태의 상품 상자와 패키지를 지시문과 시각 관찰 기반으로 안전하게 피킹 및 분류.
