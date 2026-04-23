# Diffusion Model

---
Reference:
- https://cvpr2022-tutorial-diffusion-models.github.io/
- Diffusion 시초격 논문 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html)
- DDPM 논문 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models

---

## Diffusion Mode

![alt text](./images/Diffusion%20Model.png)

- Diffusion process(forward process)와 Denoising process(reverse process)로 구성된다.
- Markov chain을 정의해서 데이터에 random noise를 천천히 추가한 다음, diffusion process를 역방향으로 진행해서 noise에서 원하는 데이터 샘플을 구성

**Markov chain**
- 각 이벤트의 확률이 이전 이벤트의 상태에만 의존하는 확률적 프로세스
- 다음에 발생할 사건은 지금의 상태에만 의존하고, 그 이전의 상태에는 의존하지 않음
- $P(x^{t+1} | x^{(0)}, ..., x^{(t - 1)}, x^{t}) = P(x^{(t+1)}|x^{(t)})$를 만족

### Diffusion process(forward process)

![alt text](./images/Forward%20process.png)
> DDPM의 forward 및 reverse process

- 실제 데이터 분포에 Gaussian noise를 추가하는 과정
- T 단계 동안 Noise를 추가

$q(x_0):$ 데이터 분포
$q(x_t|x_{t-1}):$ $x_{t-1}$일 때 $x_t$가 되는 데이터 분포

$\displaystyle q(x_t|x_{t-1}) := N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I) \\
=> q(x_{1:T}|x_0) := \prod_{t=1}^{T} q(x_t|x_{t-1})$

- $\beta_t$는 매우 작은 값
- $T -> \infty$면, $x_T$는 isotropic Gaussian distribuion과 동일하다.

**Diffusion process를 더 빠르게 하려면?**

$x_0$에서 $x_t$까지 한 번에 이동하기

여러 단계의 분포를 하나로 합친 $\bar{\alpha}$를 계산

$\displaystyle \bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s) \\
=> q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t) I)
$ (Diffusion Kernel)

다음과 같이 sampling하기
$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{(1 - \bar{\alpha}_t)} \epsilon$
$(\epsilon \sim N(0, I))$

$\beta_t$는 $\bar{\alpha}_T -> 0$이고 $q(x_T|x_0) \approx N(x_T; 0, I)$가 되도록 설계됨

### Denoising process(reverse process)

$$


$q(x^{(t)}|x^{(t-1)})$이 Gaussian(또는 이항) distribution이고, $\beta_t$가 작을 때, $q(x^{(t-1)}|x^{(t)})$ 또한 Gaussian(또는 이항) distribution이 된다. trajectory가 길수록 더 작은 diffusion rate $\beta$를 만들 수 있다.

`* diffusion model이 가능한 이유. time step을 많이 쪼개서 $\beta$를 작게 만들면 역방향 분포를 gaussian 분포로 근사할 수 있다.
[Feller. W. 1949. "On the theory of stochastic processes, with particular reference to applications"]에서 증명
`




