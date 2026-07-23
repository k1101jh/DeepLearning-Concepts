# Diffusion Model (디퓨전 모델)

---
Reference:
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics (Sohl-Dickstein et al., 2015)](https://proceedings.mlr.press/v37/sohl-dickstein15.html)
- [Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [What are Diffusion Models? (Lilian Weng Blog)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Diffusion Model (확산 모델 / DDPM)
- **관련 분야/카테고리**: Diffusion Model / Generative Model / Image & Video Generation
- **한 줄 요약**: 원본 데이터에 고분산 노이즈를 단계적으로 추가하는 Forward Process와, 역방향으로 노이즈를 제거하며 복원하는 Reverse Process를 Markov Chain으로 모델링하여 고품질 샘플을 생성하는 확률적 생성 모델이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 생성 모델(GAN, VAE, Flow)의 한계점
- **GAN (Generative Adversarial Networks)**: 생성 품질은 우수하지만 모드 붕괴(Mode Collapse), 학습 불안정성(Adversarial Training Instability) 문제가 심각하여 고도의 하이퍼파라미터 튜닝이 요구되었습니다.
- **VAE (Variational Autoencoder)**: 수학적 우도(Likelihood) 최적화 기반으로 학습이 안정적이지만, 복원된 이미지 결과물이 다소 흐릿하게(Blurry) 생성되는 고유한 한계가 존재했습니다.
- **Flow-based Models**: 가역 함수(Invertible Function) 형태의 엄격한 아키텍처 제약이 필요하여 네트워크 표현력이 제한적이었습니다.

### Diffusion Model 도입을 통한 핵심 해결 목표
- **높은 생성이미지 품질과 모드 커버리지(Mode Coverage)**: 픽셀 단위로 점진적인 Denoising 과정을 거쳐 모드 붕괴 없는 안정적 학습 및 극도로 선명하고 정교한 샘플 생성을 달성했습니다.
- **안정적인 손실 함수 최적화**: 적대적 신경망 구조 없이 단순화된 L2 노이즈 예측 손실(MSE Loss)만으로 안정적인 convergence를 보장합니다.

---

## 3. 핵심 원리 및 메커니즘 (How?)

![Diffusion Model](./images/Diffusion%20Model.png)
> **Figure 1. Diffusion Process (Forward)와 Denoising Process (Reverse)의 전체 개념도**

### 1) Forward Process (확산 과정)
원본 데이터 $x_0 \sim q(x)$에 $T$ 단계에 걸쳐 가우시안 노이즈(Gaussian Noise)를 조금씩 주입하는 Markov Chain 과정입니다.

$$q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

여기서 $\beta_t \in (0, 1)$는 스케줄링된 노이즈 계수입니다.
Diffusion Kernel 공식을 사용하면 중간 단계 $t-1$들을 거치지 않고, 원본 $x_0$로부터 arbitrary 시점 $t$의 노이즈 상태 $x_t$를 1단계로 직행 sampling할 수 있습니다:

$$\alpha_t := 1 - \beta_t, \quad \bar{\alpha}_t := \prod_{s=1}^t \alpha_s$$
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$T \to \infty$ 일 경우 $x_T$는 완전히 완전한 무작위 가우시안 노이즈 $\mathcal{N}(0, I)$ 형태가 됩니다.

![Forward process](./images/Forward%20process.png)
> **Figure 2. DDPM의 Forward 및 Reverse Process 단계별 변화 (출처: [Ho et al., 2020](https://arxiv.org/abs/2006.11239))**

### 2) Reverse Process (역방향 복원 과정)
완전한 노이즈 $x_T \sim \mathcal{N}(0, I)$로부터 노이즈를 순차적으로 제거하여 $x_0$를 복원하는 과정입니다. 노이즈 크기 $\beta_t$가 매우 작을 때 역방향 조건부 분포 $p_\theta(x_{t-1}|x_t)$ 역시 가우시안 분포로 근사됩니다:

$$p_\theta(x_{t-1} | x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

신경망(주로 U-Net 아키텍처) $\epsilon_\theta(x_t, t)$는 주입된 진짜 노이즈 $\epsilon$을 예측하도록 학습됩니다.

### 3) 학습 목표 (Loss Function)
Variational Lower Bound (VLB)를 단순화하여, 실제 주입된 노이즈 $\epsilon$과 신경망이 예측한 노이즈 $\epsilon_\theta$ 간의 Mean Squared Error(MSE)를 최소화합니다:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Markov Chain과 Feller 증명
- 맘대로 역방향 분포를 가우시안으로 처리할 수 있는 이론적 근거는 Feller(1949)의 정리에 기반합니다. 시공간 스텝 $\beta_t$가 충분히 작다면 가우시안 확산의 역과정(Reverse process) 역시 동일한 형태의 가우시안 프로세스가 됨이 수학적으로 입증되었습니다.

### 2) Classifier-Free Guidance (CFG)
- 조건부 이미지 생성(Text-to-Image 등) 시 프롬프트 텍스트 $c$의 반영 강도를 제어하는 핵심 기법입니다.
- 조건부 노이즈 예측값과 비조건부 노이즈 예측값의 차이를 강조하여 스케일링합니다:
  $$\tilde{\epsilon}_\theta(x_t, c) = (1 + w) \epsilon_\theta(x_t, c) - w \epsilon_\theta(x_t, \emptyset)$$
  ($w$가 클수록 프롬프트 지시를 매우 강력하게 따름)

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 DDPM의 핵심인 Forward Process 1스텝 주입 및 간단한 L2 Loss 연산을 수행하는 PyTorch 예시 모듈입니다.

```python
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    """DDPM (Denoising Diffusion Probabilistic Models)의 노이즈 주입 및 손실을 연산하는 모듈.

    Args:
        timesteps (int, optional): 확산 단계 타임스텝 총 개수. 기본값은 1000.
        beta_start (float, optional): 초기 스케줄링 노이즈 계수. 기본값은 0.0001.
        beta_end (float, optional): 최종 스케줄링 노이즈 계수. 기본값은 0.02.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ) -> None:
        super().__init__()
        self.timesteps: int = timesteps

        # 선형 노이즈 스케줄링 (Linear Beta Schedule)
        betas: torch.Tensor = torch.linspace(beta_start, beta_end, timesteps)
        alphas: torch.Tensor = 1.0 - betas
        alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)

        # 버퍼 등록 (학습 대상 파라미터가 아닌 스케줄링 상수)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """원본 이미지 x_0에 타임스텝 t에 따른 노이즈를 1스텝으로 즉시 주입합니다 (Diffusion Kernel).

        Args:
            x_0 (torch.Tensor): 원본 샘플 입력 텐서. (batch_size, channels, height, width)
            t (torch.Tensor): 타임스텝 인덱스 텐서. (batch_size,)
            noise (torch.Tensor): 주입할 정규 가우시안 노이즈. (batch_size, channels, height, width)

        Returns:
            torch.Tensor: 노이즈가 주입된 x_t 텐서.
        """
        # 타임스텝 브로드캐스팅 차원 맞춤
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def p_losses(
        self,
        denoise_model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """확산 모델의 L2 예측 손실(MSE Loss)을 계산합니다.

        Args:
            denoise_model (nn.Module): 노이즈 예측을 수행하는 U-Net 신경망.
            x_0 (torch.Tensor): 원본 이미지 텐서.
            t (torch.Tensor): 랜덤 샘플링된 타임스텝.

        Returns:
            torch.Tensor: 계산된 L2 Loss 스칼라값.
        """
        # 1. 무작위 노이즈 생성
        noise: torch.Tensor = torch.randn_like(x_0)

        # 2. Forward Process로 x_t 생성
        x_t: torch.Tensor = self.q_sample(x_0=x_0, t=t, noise=noise)

        # 3. U-Net으로 노이즈 예측
        predicted_noise: torch.Tensor = denoise_model(x_t, t)

        # 4. MSE Loss 계산
        loss: torch.Tensor = F.mse_loss(predicted_noise, noise)
        return loss


# 실행 예시
if __name__ == "__main__":
    diffusion_helper: GaussianDiffusion = GaussianDiffusion(timesteps=1000)
    
    # 1개의 가상 3채널 32x32 이미지
    dummy_x0: torch.Tensor = torch.randn(2, 3, 32, 32)
    dummy_t: torch.Tensor = torch.tensor([100, 500]) # 각각 다른 타임스텝
    
    # 노이즈 주입 연산 확인
    dummy_noise: torch.Tensor = torch.randn_like(dummy_x0)
    x_t: torch.Tensor = diffusion_helper.q_sample(dummy_x0, dummy_t, dummy_noise)
    
    print(f"원본 이미지 텐서 형태: {dummy_x0.shape}")
    print(f"노이즈 주입 텐서 형태: {x_t.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | GAN | VAE | Diffusion Model (DDPM) |
| :--- | :--- | :--- | :--- |
| **샘플 품질** | 매우 뛰어남 | 다소 흐릿함(Blurry) | 극도로 정교하고 뛰어남 |
| **모드 커버리지(다양성)**| 모드 붕괴 위험 높음 | 우수함 | 매우 우수함 (Mode Collapse 없음) |
| **학습 안정성** | 매우 까다로움 (적대적 학습) | 우수함 (ELBO 최적화) | 매우 우수함 (단순 MSE Loss) |
| **추론/샘플링 속도** | 매우 빠름 (1-step forward) | 매우 빠름 (1-step decoder) | 상대적으로 늦음 ($T=1000$ 백워드 스텝 필요 $\to$ DDIM, Consistency Model로 개량 중) |

### 장점
- **최상급 생성 능력**: 미세 텍스처 및 세부 표현력이 우수하며 대규모 텍스트 조건부 생성(Latent Diffusion)의 핵심 아키텍처.
- **안정적 트레이닝**: 손실 함수 발산 위험이 적고 모드 커버리지가 뛰어남.

### 한계점
- **느린 추론 속도**: 이미지 1장을 생성할 때 $T$번의 U-Net 연산을 연속 적용해야 하므로 속도가 느림.

---

## 7. 활용 사례 및 응용

1. **Text-to-Image / Latent Diffusion Models**
   - Stable Diffusion, Midjourney, DALL-E 2/3 등 현대 최고 수준의 이미지 생성 엔진의 절대적 표준.
2. **Video & Audio Generation**
   - Sora, Runway Gen-2 등 고화질 비디오 생성 및 테일러드 음성 오디오 생성.
3. **Medical Imaging & 3D Reconstruction**
   - MRI/CT 초고해상도 복원(Super Resolution) 및 3D Gaussian Splatting / NeRF 표현의 Prior로 적용.
