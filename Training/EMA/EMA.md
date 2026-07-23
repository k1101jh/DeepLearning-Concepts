# Exponential Moving Average (EMA, 지수 이동 평균)

---
Reference:
- [Polyak Averaging Paper (Polyak & Juditsky, 1992)](https://projecteuclid.org/journals/siam-journal-on-control-and-optimization/volume-30/issue-4/Acceleration-of-Stochastic-Approximation-Procedures/10.1137/0330046.full)
- [PyTorch EMAModel Implementation (timm library)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py)
- [Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)](https://arxiv.org/abs/2105.05233)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Exponential Moving Average (EMA, 지수 이동 평균 가중치)
- **관련 분야/카테고리**: Training / Optimization / Model Weight Averaging
- **한 줄 요약**: 딥러닝 모델 학습 과정에서 시점별 모델 가중치의 지수 이동 평균을 유지하여, 최적화 노이즈를 완화하고 일반화 성능을 높이는 가중치 평탄화(Weight Averaging) 기법이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 방식의 한계점
- **학습 파라미터의 미세 진동(Fluctuation)**: SGD, Adam 등의 Stochastic Gradient Descent 기반 최적화 알고리즘은 미니배치(Mini-batch)의 샘플링 노이즈 때문에 매 스텝(Step)마다 모델 파라미터가 최적점(Loss Minimum) 부근에서 끊임없이 진동합니다.
- **최종 시점 오버피팅 또는 불안정성**: 학습의 가장 마지막 스텝(Last Step) 가중치만 저장하여 사용할 경우, 해당 순간 미니배치 특성에 의해 우연히 성능이 떨어진 시점의 파라미터를 취할 수 있습니다.
- **손실 곡면(Loss Landscape)의 좁은 골짜기**: 딥러닝의 손실 곡면에서 특정 순간의 local minima는 좁고 가파를(sharp minima) 위험이 높아서, 테스트 데이터나 Out-of-distribution 데이터에 대한 일반화 능력이 떨어질 수 있습니다.

### EMA 도입을 통한 핵심 해결 목표
- 최근 가중치에 높은 가중치(Weight)를 주고 과거 가중치에는 지수적으로 감쇄하는 가중치를 주어 최근 변화 궤적을 부드럽게(Smooth) 반영합니다.
- 평가 및 추론 시 원본 파라미터 대신 EMA가 적용된 파라미터(Shadow parameter)를 사용하여, 더 넓고 평평한 손실 골짜기(Flat Minima)에 위치한 모델을 확보함으로써 평가 성능과 학습 안정성을 대폭 향상시킵니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 핵심 공식
학습 스텝 $t$에서의 학습 대상 모델 파라미터를 $\theta_t$, 지수 이동 평균이 반영된 가중치를 $\theta_{\text{EMA}}^{(t)}$라고 할 때, EMA의 업데이트 식은 다음과 같습니다:

$$\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \theta_t$$

여기서 각 기호와 변수의 의미는 다음과 같습니다:
- $\theta_t$: $t$번째 학습 스텝에서 갱신된 실제 모델 파라미터 (Online Model Weights)
- $\theta_{\text{EMA}}^{(t)}$: $t$번째 학습 스텝에서의 지수 이동 평균 파라미터 (Shadow Weights)
- $\beta \in [0, 1)$: 쇠퇴율(Decay rate) 또는 평탄화 계수. 통상 $0.999$, $0.9999$ 등 $1$에 가까운 높은 값을 사용합니다.
- $(1 - \beta)$: 현재 파라미터 $\theta_t$를 반영하는 비율 (Momentum 계수)

### 2) 지수적 감쇄 메커니즘
위 점화식을 과거 $k$번째 스텝까지 전개해 보면 다음과 같습니다:

$$\theta_{\text{EMA}}^{(t)} = (1 - \beta) \sum_{i=0}^{t-1} \beta^i \theta_{t-i} + \beta^t \theta_0$$

- $i$가 증가할수록(즉, 더 과거의 스텝일수록) 계수 $\beta^i$는 지수 함수(Exponential Decay) 형태로 급격히 감소합니다.
- 따라서 최근 업데이트된 파라미터 정보는 강하게 유지되면서도, 여러 스텝에 걸친 가중치가 통합되어 노이즈가 제거됩니다.

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Shadow Weights (그림자 가중치)
- EMA를 적용할 때 기존 학습 모델의 파라미터(`online model`)와 별개로, 동일한 구조 및 크기의 가중치 버퍼(`shadow weights`)를 복사해 유지합니다.
- 학습 단계(Backpropagation 및 Optimizer step)에서는 `online model`만 기울기(Gradient) 계산 및 갱신을 수행하고, 스텝 직후 `shadow weights`를 EMA 공식으로 업데이트합니다.
- 검증/추론 단계에서는 `shadow weights`를 모델에 적용하여 예측을 수행합니다.

### 2) EMA 편향 보정 (Debiasing / Dynamic Decay Rate)
- 학습 초반 $t=1, 2, \dots$ 시점에는 초기값 $\theta_0$의 영향력이 지나치게 우세하여 EMA 파라미터가 0에 가깝거나 초기값에 묶여 느리게 반영될 수 있습니다.
- 이를 보정하기 위해 학습 초반에는 decay rate $\beta$를 유동적으로 낮추어 시작하는 동적 decay 전략을 사용합니다:
  $$\beta_t = \min\left(\beta, \frac{1 + t}{10 + t}\right)$$
- 또는 Adam Optimizer의 bias correction과 유사하게 나누어 주는 편향 보정을 사용하기도 합니다.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch 환경에서 모델 가중치의 EMA를 안전하고 효과적으로 관리할 수 있는 클래스 구현입니다.

```python
import copy
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ModelEMA:
    """학습 중인 PyTorch 모델 가중치의 지수 이동 평균(EMA)을 관리하는 클래스.

    Args:
        model (nn.Module): EMA를 적용할 대상 PyTorch 모델.
        decay (float, optional): EMA 감쇄율(decay rate). 기본값은 0.9999.
        device (Optional[torch.device], optional): EMA 가중치를 연산/저장할 디바이스. None일 경우 모델 디바이스와 동일.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ) -> None:
        # EMA 가중치를 보관할 모델 복제 (매개변수 경사하도 보관 안 함)
        self.module: nn.Module = copy.deepcopy(model).eval()
        self.decay: float = decay
        self.device: Optional[torch.device] = device
        
        if self.device is not None:
            self.module.to(device=self.device)

        # EMA 파라미터 경사하도(gradient) 계산 비활성화
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """현재 학습 모델의 파라미터 값을 바탕으로 EMA 파라미터를 갱신합니다.

        Args:
            model (nn.Module): 현재 optimizer.step()이 완료된 실제 학습 모델.
        """
        # 학습 모델 파라미터 Dict 추출
        msd: Dict[str, torch.Tensor] = model.state_dict()
        # EMA 모델 파라미터 Dict 추출
        esd: Dict[str, torch.Tensor] = self.module.state_dict()

        for k, v in msd.items():
            if k in esd:
                # 부동소수점 타입 매개변수만 EMA 연산 적용
                if v.dtype.is_floating_point:
                    v_ema = esd[k]
                    if self.device is not None:
                        v = v.to(self.device)
                    # EMA 갱신 공식: v_ema = decay * v_ema + (1 - decay) * v
                    v_ema.copy_(v_ema * self.decay + v * (1.0 - self.decay))
                else:
                    # 정수형 버퍼 등(예: num_batches_tracked)은 단순 복사
                    esd[k].copy_(v)

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """평가 및 추론 시 EMA 가중치를 실제 모델에 덮어씌웁니다.
        이후 원복을 위해 교체 전 원본 모델의 state_dict를 반환합니다.

        Args:
            model (nn.Module): EMA 가중치를 덮어씌울 대상 모델.

        Returns:
            Dict[str, torch.Tensor]: 원복에 필요한 교체 직전의 모델 state_dict.
        """
        # 교체 직전 원본 가중치 백업
        original_state_dict: Dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
        # EMA 가중치를 학습 모델로 로드
        model.load_state_dict(self.module.state_dict())
        return original_state_dict

    def restore(self, model: nn.Module, backup_state_dict: Dict[str, torch.Tensor]) -> None:
        """`apply_shadow` 적용 이전의 원본 모델 가중치 상태로 복원합니다.

        Args:
            model (nn.Module): 복원 대상 모델.
            backup_state_dict (Dict[str, torch.Tensor]): `apply_shadow` 실행 시 반환된 백업 state_dict.
        """
        model.load_state_dict(backup_state_dict)


# 사용 예시
if __name__ == "__main__":
    # 1. 샘플 선형 모델 생성
    sample_model: nn.Module = nn.Linear(10, 2)
    ema_helper: ModelEMA = ModelEMA(sample_model, decay=0.99)

    # 2. 임의의 1 스텝 학습 진행 가정
    optimizer = torch.optim.SGD(sample_model.parameters(), lr=0.1)
    inputs = torch.randn(4, 10)
    loss = sample_model(inputs).sum()
    loss.backward()
    optimizer.step()

    # 3. EMA 갱신
    ema_helper.update(sample_model)

    # 4. 검증 시 EMA 가중치 임시 적용 및 평가
    backup = ema_helper.apply_shadow(sample_model)
    # val_loss = evaluate(sample_model) ...
    print("EMA 가중치 적용 완료")

    # 5. 검증 후 다시 원본 가중치 복원
    ema_helper.restore(sample_model, backup)
    print("원본 가중치 복원 완료")
```

---

## 6. 장단점 및 기존 개념과의 비교

### EMA vs 기존 가중치 관리 방식 비교

| 비교 항목 | 일반 가중치 (Last Iteration) | EMA (Exponential Moving Average) | SWA (Stochastic Weight Averaging) |
| :--- | :--- | :--- | :--- |
| **가중치 반영 방식** | 마지막 스텝의 파라미터만 사용 | 매 스텝 가중치의 지수 감쇄 누적 평균 | 일정 학습 주기(예: 매 epoch) 가중치의 등가 평균 |
| **추가 메모리** | 추가 메모리 없음 | 모델 크기만큼의 추가 메모리 필요 (`shadow weights`) | 모델 크기만큼의 추가 메모리 필요 |
| **학습 안정화 성능** | 미니배치 노이즈에 취약함 | 매 스텝 노이즈를 매우 부드럽게 완화 | 넓은 local minima 탐색에 유용함 |
| **적용 시점** | 제한 없음 | 생성 모델(Diffusion/GAN) 및 대형 Vision 모델 | 분류(Classification) 및 일반적인 오버피팅 방지 |

### 장점
- **생성 모델의 품질 획기적 향상**: GAN 및 Diffusion Model 등 생성 결과물의 모드 붕괴(Mode Collapse) 및 노이즈를 대폭 감소시킵니다.
- **Out-of-Distribution 일반화 향상**: Flat Minima 주변 가중치를 얻게 되어 테스트 데이터셋 성능이 안정적입니다.
- **간편한 적용**: Optimizer 종류나 Loss 구조의 변경 없이 독립적으로 추적/적용이 가능합니다.

### 한계점 및 고려사항
- **메모리 사용량 증가**: 모델 파라미터 크기만큼의 메모리를 추가로 점유합니다.
- **Decay 하이퍼파라미터 민감성**: $\beta$가 너무 크면($0.99999$) 최신 학습 상태 반영이 매우 늦어지고, 너무 작으면($0.9$) 평탄화 효과가 떨어집니다.

---

## 7. 활용 사례 및 응용

1. **Diffusion Models (확산 모델)**
   - DDPM, Latent Diffusion (Stable Diffusion), EDM 등 모든 최신 Diffusion 기법에서 EMA 가중치는 생성 고품질 이미지/비디오 추론에 표준으로 사용됩니다.
2. **Generative Adversarial Networks (GAN)**
   - Progressive GAN, StyleGAN 시리즈 등에서 생성기(Generator) 가중치의 진동을 막고 선명한 고화질 이미지를 생성하기 위해 필수적으로 적용됩니다.
3. **Self-Supervised Learning (자기지도학습)**
   - BYOL (Bootstrap Your Own Latent), DINO 등 Target Network의 Teacher 파라미터를 Student 파라미터의 EMA로 갱신하여 붕괴(Collapse) 현상을 방지합니다.
