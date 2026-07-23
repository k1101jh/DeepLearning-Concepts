# 3D Gaussian Splatting (3DGS, 3D 가우시안 스플래팅)

---
Reference:
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2033)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [3D Gaussian Splatting Official GitHub Repository](https://github.com/graphdeco-inria/gaussian-splatting)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: 3D Gaussian Splatting (3DGS, 3D 가우시안 스플래팅)
- **관련 분야/카테고리**: Vision / 3D Graphics / Radiance Field / Real-Time Rendering
- **한 줄 요약**: 연속적인 신경망(MLP) 대신 수백만 개의 미분 가능한 3D 가우시안(3D Gaussian) 입자 집합으로 3D 공간을 표현하고, 타일 기반 타일드 스플래팅(Tiled Splatting)을 통해 실시간(100+ FPS) 초고화질 3D 씬을 렌더링하는 기법이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 기존 NeRF (Neural Radiance Fields) 및 Mesh 기반 기술의 한계점
- **NeRF의 극심한 추론 지연 (Low FPS)**: NeRF는 픽셀 하나를 렌더링할 때마다 볼륨 레이 마칭(Volume Ray Marching)을 수행하며 거대한 MLP를 수십~수백 번 평가해야 함. 이에 따라 렌더링 속도가 1 FPS 미만으로 낮아 실시간 응용이 불가능했습니다.
- **학습 시간 폭증**: NeRF는 복잡한 신경망을 수 시간~수 일 동안 학습시켜야 3D 공간 복원이 완료됨.
- **명시적 Mesh 표현의 제약**: 폴리곤 메쉬(Polygon Mesh) 기반 방식은 유리를 통과하는 빛이나 연기, 반사, 복잡한 세부 텍스처 등 얇고 투명한 광선 효과를 표현하는 데 한계가 있음.

### 3DGS 도입을 통한 핵심 해결 목표
- **실시간 초고속 렌더링 (Real-Time 100+ FPS)**: 신경망(MLP) 평가를 완전히 없애고, 그래픽스 파이프라인과 결합된 타일드 라스터라이제이션(Tiled Rasterization)을 도입해 1080p 해상도에서 100 FPS 이상의 즉각적인 렌더링을 실현했습니다.
- **초고속 학습 속도**: 최적화 알고리즘이 30분 이내에 완료되어 고화질 3D 공간을 복원함.

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 3D 가우시안 파라미터화 (3D Gaussian Parameterization)
공간상의 각 3D 가우시안 입자 $g_i$는 다음 5가지 속성(Parameters)으로 구성됩니다:

1. **중심 위치 (Position / Center)**: $\mu \in \mathbb{R}^3$
2. **3D 공분산 행렬 (Covariance Matrix)**: $\Sigma \in \mathbb{R}^{3 \times 3}$
   - 물리적 타당성(Positive Semi-Definite)을 보장하기 위해 스케일 벡터 $S \in \mathbb{R}^3$와 회전 쿼터니언 $R \in \mathbb{H}$으로 분해하여 표현합니다: $\Sigma = R S S^T R^T$
3. **불투명도 (Opacity / Alpha)**: $\alpha \in [0, 1]$
4. **시선 방향 가변 색상 (Color / Spherical Harmonics)**: $C \in \mathbb{R}^{k}$ (구면 조화 함수 계수 사용)

3D 가우시안의 확률 밀도 함수는 다음과 같습니다:

$$G(x) = \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

### 2) 2D 투영 및 타일드 스플래팅 (2D Projection & Splatting)
카메라 시점(View Matrix $W$, Projection Jacobian $J$)이 주어지면, 3D 가우시안 공분산 $\Sigma$를 2D 이미지 평면 상의 2D 공분산 $\Sigma'$으로 알파 합성 투영합니다 (EWA Splatting):

$$\Sigma' = J W \Sigma W^T J^T$$

### 3) 알파 블렌딩 (Alpha Compositing) 렌더링
픽셀 $x$에서의 최종 색상 $C(x)$는 카메라 깊이 순서대로 정렬된 $N$개의 2D 가우시안을 순차적 알파 블렌딩 처리하여 계산합니다:

$$C(x) = \sum_{i=1}^N c_i \alpha_i' \prod_{j=1}^{i-1} (1 - \alpha_j')$$

여기서 $\alpha_i' = \alpha_i \cdot G_{2D}(x)$임.

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) 밀도 조절 및 어댑티브 적응 (Adaptive Density Control)
- 학습 과정에서 가우시안의 밀도를 동적으로 제어함.
- **적응적 분할 (Split)**: 너무 큰 영역을 지닌 가우시안은 2개의 작은 가우시안으로 분할함.
- **적응적 복제 (Clone)**: 렌더링 오차(Gradient)가 크지만 가우시안 크기가 작은 영역은 가우시안을 복제하여 정밀도를 높임.
- **제거 (Prune)**: 불투명도 $\alpha$가 너무 낮은 투명 가우시안은 메모리 절약을 위해 삭제함.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 사용하여 3D 가우시안의 중심 좌표, 스케일, 회전 쿼터니언으로부터 3D 공분산 행렬 $\Sigma$를 구성하는 계산 모듈 예시 코드임.

```python
from typing import Tuple
import torch
import torch.nn as nn


class Gaussian3DCovariance(nn.Module):
    """3D 가우시안 입자의 스케일(Scale) 벡터와 회전(Rotation) 쿼터니언으로부터
    3D 공분산 행렬 (Covariance Matrix Σ = R S S^T R^T)을 연산하는 모듈.
    """

    def __init__() -> None:
        super().__init__()

    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """쿼터니언 (w, x, y, z) 텐서를 3x3 회전 행렬 R로 변환함.

        Args:
            q (torch.Tensor): 정규화된 쿼터니언 텐서. 크기: (N, 4)

        Returns:
            torch.Tensor: 3x3 회전 행렬 텐서. 크기: (N, 3, 3)
        """
        # 쿼터니언 정규화
        q = q / torch.norm(q, dim=-1, keepdim=True)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # 3x3 회전 행렬 구성
        R = torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def forward(self, scale: torch.Tensor, rotation_quaternion: torch.Tensor) -> torch.Tensor:
        """스케일과 회전 쿼터니언으로 3D 공분산 행렬 Σ를 계산함.

        Args:
            scale (torch.Tensor): 각 축별 양수 스케일 벡터. 크기: (N, 3)
            rotation_quaternion (torch.Tensor): 회전 쿼터니언 (w, x, y, z). 크기: (N, 4)

        Returns:
            torch.Tensor: 대칭 양의 반정정치 3x3 3D 공분산 행렬 Σ. 크기: (N, 3, 3)
        """
        # 1. 3x3 회전 행렬 R 계산 (N, 3, 3)
        R: torch.Tensor = self.quaternion_to_rotation_matrix(rotation_quaternion)

        # 2. 대각 스케일 행렬 S 구성 (N, 3, 3)
        S: torch.Tensor = torch.diag_embed(scale)

        # 3. M = R @ S 연산
        M: torch.Tensor = torch.bmm(R, S)

        # 4. 3D 공분산 행렬 Σ = M @ M^T 계산
        cov3D: torch.Tensor = torch.bmm(M, M.transpose(1, 2))
        return cov3D


# 실행 예시
if __name__ == "__main__":
    num_gaussians: int = 1000

    # 가상의 스케일 텐서 (양수) 및 임의의 쿼터니언 생성
    dummy_scale: torch.Tensor = torch.exp(torch.randn(num_gaussians, 3))
    dummy_quat: torch.Tensor = torch.randn(num_gaussians, 4)

    cov_calculator: Gaussian3DCovariance = Gaussian3DCovariance()
    cov3d_matrix: torch.Tensor = cov_calculator(dummy_scale, dummy_quat)

    print(f"가우시안 입자 수: {num_gaussians}")
    print(f"3D 공분산 행렬 Σ 형태: {cov3d_matrix.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | NeRF (Neural Radiance Fields) | 3D Gaussian Splatting (3DGS) |
| :--- | :--- | :--- |
| **3D 공간 표현 방식** | 암묵적 표현 (Implicit MLP) | 명시적 포인트 표현 (Explicit 3D Gaussians) |
| **추론/렌더링 속도** | 매우 늦음 (< 1 FPS, 레이 마칭 오버헤드) | **초고속 실시간 (100+ FPS)** |
| **학습 시간** | 수 시간 ~ 수 일 소요 | **10분 ~ 30분 이내 고속 최적화** |
| **렌더링 화질** | 우수함 | **극도로 선명하고 정교함** |
| **추가 메모리 점유** | 모델 파라미터 용량 소형 (~수십 MB) | 수백만 개 가우시안 파라미터 저장 필요 (~수백 MB) |

### 장점
- **실시간 인터랙티브 3D 렌더링**: VR/AR, 게임 엔진, 실시간 디지털 트윈에 즉각 이식 가능.
- **초고속 최적화**: 딥러닝 MLP 추론 없이 그래픽스 타일 라스터라이저로 수렴 속도가 비약적으로 빠름.

### 한계점
- 수백만 개 가우시안 속성을 보관해야 하므로 씬 디스크 파일 저장 용량이 수백 MB 단위로 증가함 (가우시안 압축 기술 연구 진행 중).

---

## 7. 활용 사례 및 응용

1. **VR/AR & 메타버스 실시간 3D 공간 복원**
   - 핸드폰 카메라로 촬영한 실내/야외 공간을 30분 만에 100+ FPS 실시간 탐색 가능한 3D asset으로 자동 변환.
2. **자율주행 3D 디지털 트윈 씬 생성**
   - 주행 카메라 영상으로부터 도로 환경 3DGS 모델을 구축하여 시뮬레이터 환경 구축.
