# Learning Rate Warmup (학습률 웜업)

---
Reference:
- [Deep Residual Learning for Image Recognition (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (Goyal et al., 2017)](https://arxiv.org/abs/1706.02677)
- [PyTorch lr_scheduler Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Learning Rate Warmup (학습률 웜업)
- **관련 분야/카테고리**: Training / Optimization / Learning Rate Schedule
- **한 줄 요약**: 딥러닝 모델 학습 초기 단계에 학습률(Learning Rate)을 0에 가까운 매우 작은 값부터 목표 학습률까지 단계적으로 증가시켜, 초기의 급격한 가중치 발산과 학습 불안정성을 방지하는 기술이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 방식의 한계점
- **학습 초기의 임의 무작위 초기화(Random Initialization)**: 모델 학습 극초반에는 네트워크의 파라미터가 무작위로 설정되어 있어서, 그래디언트(Gradient)의 방향과 크기가 매우 불안정(noisy & steep)합니다.
- **초기 그래디언트 폭발(Gradient Explosion) 및 발산**: 불안정한 상태에서 곧바로 높은 기본 학습률(Base Learning Rate)을 적용하면 그래디언트 업데이트가 지나치게 커져 모델이 발산(Divergence)하거나 초기 학습 궤적을 벗어날 수 있습니다.
- **Adam / AdamW 계열 최적화기의 분산 추정 오류**: Adam과 같은 적응형(Adaptive) 옵티마이저는 초반 몇 스텝 동안 2차 모멘트(Uncentered Variance $v_t$) 추정치가 충분히 수렴하지 않아 분산이 커지고 그래디언트 업데이트가 제어를 잃을 위험이 큽니다.

### Warmup 도입을 통한 핵심 해결 목표
- 학습 초기 $N$ 스텝/에폭 동안 학습률을 조심스럽게 서서히 올려 가중치가 손실 곡면의 안정적인 영역에 도달하도록 유도합니다.
- 대규모 배치(Large Mini-batch) 학습 및 트랜스포머(Transformer) 기반 아키텍처에서 초기 발산을 완벽히 차단하고 최종 성능을 높입니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 선형 웜업 (Linear Warmup) 공식
가장 널리 쓰이는 선형 웜업에서 현재 학습 스텝 $t$에서의 학습률 $\eta_t$는 다음과 같이 정의됩니다:

$$\eta_t = \eta_{\text{base}} \cdot \left( \frac{t}{T_{\text{warmup}}} \right) \quad \text{for } 1 \le t \le T_{\text{warmup}}$$

여기서 각 기호와 변수의 의미는 다음과 같습니다:
- $\eta_t$: $t$번째 스텝에서의 유효 학습률 (Effective Learning Rate)
- $\eta_{\text{base}}$: 웜업 완료 후 달성하고자 하는 기준/목표 학습률 (Base Learning Rate)
- $t$: 현재 학습 진행 스텝 (1-indexed step counter)
- $T_{\text{warmup}}$: 지정된 전체 웜업 스텝 수 (Warmup Steps)

$t > T_{\text{warmup}}$ 이후부터는 원래 설정한 웜업 후의 스케줄러(Step decay, Cosine Annealing 등)를 따라 학습률을 감소시킵니다.

### 2) Transformer (Vaswani et al.) 웜업 공식
`Attention Is All You Need` 논문에서 제시된 대표적인 웜업 스케줄러 공식은 다음과 같습니다:

$$\eta_t = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5}, t \cdot T_{\text{warmup}}^{-1.5}\right)$$

- $d_{\text{model}}$: Transformer 모델의 임베딩 차원 수
- $t \le T_{\text{warmup}}$ 동안에는 $t \cdot T_{\text{warmup}}^{-1.5}$ 항이 지배하여 학습률이 선형적으로 증가합니다.
- $t > T_{\text{warmup}}$ 이후에는 $t^{-0.5}$ 항이 지배하여 역제곱근(Inverse Square Root) 비율로 점차 감소합니다.

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Warmup과 Decay Scheduler의 연동
Warmup은 단독으로 쓰이기보다 보통 후속 decay 기법과 결합하여 사용됩니다:
- **Linear Warmup + Cosine Annealing**: 초기 웜업 후 코사인 곡선을 따라 학습률을 서서히 줄여 최고 성능을 달성하는 조합으로, ViT 및 LLM 학습의 대세 스케줄러입니다.
- **Linear Warmup + Step Decay**: 특정 에폭마다 학습률을 1/10로 줄이는 전통적인 방식과 결합합니다.

### 2) Large Batch Training에서의 Warmup (Goyal et al., 2017)
- 미니배치 크기를 크게 확대할 경우(예: Batch Size 8k, 32k), 학습 속도를 유지하기 위해 학습률도 배치 크기에 비례하여 증가(`Linear Scaling Rule`)시켜야 합니다.
- 큰 학습률을 바로 적용하면 100% 발산하므로, Goyal et al. 논문에서는 5 에폭 이상의 Gradual Warmup 기법을 적용하여 1시간 만에 ImageNet을 성공적으로 학습시켰습니다.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch의 `torch.optim.lr_scheduler.LambdaLR` 및 커스텀 래퍼를 활용한 Linear Warmup + Cosine Decay 스케줄러 구현 예시입니다.

```python
import math
from typing import List, Dict, Any, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineAnnealingLR(LambdaLR):
    """Linear Warmup과 Cosine Annealing Decay를 결합한 PyTorch 학습률 스케줄러.

    Args:
        optimizer (Optimizer): 학습률을 조절할 대상 PyTorch 옵티마이저.
        warmup_steps (int): 웜업을 진행할 전체 스텝 수.
        max_steps (int): 전체 학습 진행 스텝 수.
        min_lr_ratio (float, optional): 최소 학습률 비율 (base_lr 대비 비율). 기본값은 0.0.
        last_epoch (int, optional): 마지작 에폭/스텝 인덱스. 기본값은 -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        self.warmup_steps: int = warmup_steps
        self.max_steps: int = max_steps
        self.min_lr_ratio: float = min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            # 1단계: Warmup 구간 (current_step <= warmup_steps)
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # 2단계: Warmup 종료 후 구간
            if current_step > self.max_steps:
                return self.min_lr_ratio
                
            # Cosine Decay 구간 계산
            progress: float = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            cosine_decay: float = 0.5 * (1.0 + math.cos(math.pi * progress))
            # min_lr_ratio 보정 적용
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


# 사용 예시
if __name__ == "__main__":
    # 1. 가상 모델 및 옵티마이저 생성
    model: nn.Module = nn.Linear(10, 2)
    optimizer: Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 2. 웜업 100 스텝, 전체 1000 스텝 스케줄러 생성
    total_warmup_steps: int = 100
    total_max_steps: int = 1000
    scheduler: LinearWarmupCosineAnnealingLR = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_steps=total_warmup_steps,
        max_steps=total_max_steps,
        min_lr_ratio=0.01
    )

    # 3. 스텝별 학습률 변화 시뮬레이션
    lrs: List[float] = []
    for step in range(total_max_steps):
        # 파라미터 업데이트 진행 가정
        optimizer.step()
        # 현재 학습률 기록
        current_lr: float = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        # 스케줄러 스텝 이동
        scheduler.step()

    print(f"초기 0 스텝 학습률: {lrs[0]:.6f}")
    print(f"Warmup 완료 지점 (100 스텝) 학습률: {lrs[99]:.6f}")
    print(f"중간 지점 (550 스텝) 학습률: {lrs[549]:.6f}")
    print(f"최종 1000 스텝 학습률: {lrs[-1]:.6f}")
```

---

## 6. 장단점 및 기존 개념과의 비교

### Warmup 적용 여부 비교

| 비교 항목 | Warmup 미적용 (Standard Schedule) | Warmup 적용 (Gradual Warmup Schedule) |
| :--- | :--- | :--- |
| **초기 학습 안정성** | 발산(Divergence) 또는 Loss Spikes 발생 가능성 존재 | 초반 가중치 및 옵티마이저 모멘텀이 매우 안정적으로 정착됨 |
| **최고 학습률 설정** | 높은 학습률 사용에 제한이 있음 | 높은 학습률(Base LR)을 안전하게 활용 가능 |
| **Large Batch 학습** | 배치 크기가 증가할수록 학습 실패 위험 폭증 | Large Mini-batch 학습 성공의 필수 조건 |
| **구현 복잡도** | 단순함 | 스케줄러 설정 코드 및 warmup_steps 튜닝 필요 |

### 장점
- **초기 발산 완벽 방지**: 대형 언어 모델(LLM), Transformer 계열 아키텍처 학습 시 손실 함수 폭발을 방지합니다.
- **성능 수렴성 극대화**: 높은 base learning rate를 사용할 수 있어 전체 학습 속도 및 최종 검증 성능이 대폭 향상됩니다.

### 한계점 및 고려사항
- **Warmup steps 하이퍼파라미터 튜닝**: 전체 스텝 대비 웜업 스텝 비율(보통 전체의 2% ~ 10%)을 적절히 설정해야 합니다. 너무 길면 학습이 느려지고, 너무 짧으면 웜업 효과가 미미합니다.

---

## 7. 활용 사례 및 응용

1. **Transformer 기반 LLM 사전 학습 (BERT, GPT-3, LLaMA 등)**
   - AdamW 옵티마이저와 함께 Linear Warmup + Cosine Decay 전략은 모든 대규모 언어 모델 학습 표준 템플릿입니다.
2. **Vision Transformer (ViT, Swin Transformer)**
   - 합성곱(Convolution)의 귀납적 편향(Inductive Bias)이 부족한 Vision Transformer는 초반 웜업 없이 학습할 경우 손실 값이 수렴하지 않는 현상이 잦습니다.
3. **대규모 데이터셋 초고속 파이프라인 (ImageNet 1-Hour Training)**
   - 대규모 노드 및 대용량 미니배치를 동용하는 분산 학습 파이프라인에서 기본 스케줄러로 채택되고 있습니다.
