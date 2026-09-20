# 3D Gaussian Splatting (3DGS, 3D 가우시안 스플래팅)

---
Reference:
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surajsharma.github.io/2DGS/)
- [Mip-Splatting: Alias-free 3D Gaussian Splatting](https://nerfstudio-project.github.io/mip-splatting/)
- [FastGS: Training 3D Gaussian Splatting in 100 Seconds](https://arxiv.org/abs/2601.00000)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: 3D Gaussian Splatting (3DGS, 3D 가우시안 스플래팅)
- **관련 분야/카테고리**: Vision / 3D Graphics / Radiance Field / Real-Time Rendering
- **한 줄 요약**: 연속적인 신경망(MLP) 대신 수백만 개의 미분 가능한 3D 가우시안(3D Gaussian) 입자 집합으로 3D 공간을 표현하고, 타일 기반 타일드 라스터라이제이션(Tiled Rasterization)을 통해 실시간(100+ FPS) 초고화질 3D 씬을 렌더링하는 기법

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 1) 기존 NeRF 및 Mesh 기반 기술의 한계점
- **NeRF의 극심한 추론 지연 (Low FPS)**:
    - NeRF는 픽셀 하나를 렌더링할 때마다 광선 상의 샘플링 포인트를 따라 볼륨 레이 마칭(Volume Ray Marching)을 수행하고 거대한 MLP를 수십~수백 번 평가해야 함
    - 이에 따라 렌더링 속도가 1 FPS 미만으로 제한되어 실시간 AR/VR 및 게임 응용이 불가능함
- **학습 시간 폭증**:
    - NeRF는 복잡한 암묵적 신경망을 수 시간~수 일 동안 최적화해야 3D 공간 복원이 완료됨
- **명시적 Mesh 표현의 물리적 제약**:
    - 폴리곤 메쉬(Polygon Mesh) 기반 방식은 연기, 투명체, 빛의 반사, 복잡한 초고주파 세부 텍스처 등 얇고 투명한 광선 효과를 자연스럽게 표현하기 어려움

### 2) 3DGS 도입을 통한 핵심 해결 목표
- **실시간 초고속 렌더링 (Real-Time 100+ FPS)**:
    - 신경망(MLP) 평가 연산을 완전히 제거하고, GPU 하드웨어 그래픽스 파이프라인과 결합된 타일드 라스터라이제이션(Tiled Rasterization)을 도입해 1080p 해상도에서 100 FPS 이상의 즉각적인 렌더링을 실현함
- **초고속 학습 속도**:
    - 최적화 알고리즘이 10분~30분 이내에 완료되어 고화질 3D 공간 복원 완료

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) 3D 가우시안 파라미터화 (3D Gaussian Parameterization)
공간상의 각 3D 가우시안 입자 $g_i$는 5가지 핵심 속성(Parameters)으로 구성됩니다:

1. **중심 위치 (Position / Center)**: $\mu \in \mathbb{R}^3$
2. **3D 공분산 행렬 (Covariance Matrix)**: $\Sigma \in \mathbb{R}^{3 \times 3}$
   - 물리적 타당성(Positive Semi-Definite)을 보장하기 위해 스케일 벡터 $S \in \mathbb{R}^3$와 회전 쿼터니언 $R \in \mathbb{H}$으로 분해하여 표현함:
   $$\Sigma = R S S^T R^T$$
3. **불투명도 (Opacity / Alpha)**: $\alpha \in [0, 1]$
4. **시선 방향 가변 색상 (Color / Spherical Harmonics)**: 구면 조화 함수 계수 $C \in \mathbb{R}^{k}$ 사용

3D 가우시안의 확률 밀도 함수는 다음과 같습니다:
$$G(x) = \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

### 2) 2D 투영 및 타일드 스플래팅 (2D Projection & EWA Splatting)
카메라 시점 변환 행렬 $W$와 투영 야코비안(Jacobian) 행렬 $J$가 주어지면, 3D 가우시안 공분산 $\Sigma$를 2D 이미지 평면 상의 2D 공분산 $\Sigma'$으로 알파 합성 투영합니다:

$$\Sigma' = J W \Sigma W^T J^T$$

### 3) GPU 타일 기반 알파 블렌딩 (Alpha Compositing) 렌더링
화면을 $16 \times 16$ 타일 단위로 나누고, 픽셀 $x$에서의 최종 색상 $C(x)$는 카메라 깊이 순서대로 정렬된 $N$개의 2D 가우시안을 순차적 알파 블렌딩 처리하여 연산합니다:

$$C(x) = \sum_{i=1}^N c_i \alpha_i' \prod_{j=1}^{i-1} (1 - \alpha_j')$$
$$\text{단}, \alpha_i' = \alpha_i \cdot G_{2D}(x)$$

---

## 4. 핵심 세부 개념 및 발전 파생 연구

### 1) 밀도 조절 및 어댑티브 적응 (Adaptive Density Control)
학습 과정에서 오차 그래디언트를 기반으로 가우시안의 밀도를 동적으로 제어합니다.
- **적응적 분할 (Split)**: 공간 범위가 너무 큰 가우시안은 2개의 작은 가우시안으로 분할함
- **적응적 복제 (Clone)**: 렌더링 오차(Gradient)가 크지만 가우시안 크기가 작은 부위는 가우시안을 복제하여 표현력을 높임
- **제거 (Prune)**: 불투명도 $\alpha$가 임계값보다 낮거나 크기가 너무 비대해진 투명 가우시안은 제거함

### 2) 2DGS (2D Gaussian Splatting)
- 3D 타원체 대신 2D 평면 디스크 형태의 가우시안을 도입하여 법선 벡터(Normal)를 정교하게 정의하고, 복원 표면의 얇은 껍질(Surface) 아티팩트 및 부유물 노이즈를 근본적으로 해소함

### 3) Mip-Splatting (Alias-Free 3DGS)
- 카메라 시점 이동이나 줌인/줌아웃 시 발생하는 앨리어싱(Aliasing) 및 고주파 아티팩트 문제를 해결하기 위해, 2D low-pass 필터 연산을 2D 공분산 투영 계산에 융합함

### 4) FastGS (Fast 3DGS Training)
- 가우시안 포인트 레이아웃 할당 및 메모리 정렬을 최적화하여 100초 이내에 3DGS 학습 수렴을 완료하는 초고속화 패러다임

---

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Tuple
import torch
import torch.nn as nn

class Gaussian3DCovariance(nn.Module):
    """
    3D 가우시안 입자의 스케일(Scale) 벡터와 회전(Rotation) 쿼터니언으로부터
    3D 공분산 행렬 (Covariance Matrix Σ = R S S^T R^T)을 연산하는 모듈
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        쿼터니언 (w, x, y, z) 텐서를 3x3 회전 행렬 R로 변환함
        
        Args:
            q (torch.Tensor): 정규화된 쿼터니언 텐서 (N, 4)
            
        Returns:
            torch.Tensor: 3x3 회전 행렬 텐서 (N, 3, 3)
        """
        q = q / torch.norm(q, dim=-1, keepdim=True)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def forward(self, scale: torch.Tensor, rotation_quaternion: torch.Tensor) -> torch.Tensor:
        """
        스케일과 회전 쿼터니언으로 3D 공분산 행렬 Σ를 연산함
        
        Args:
            scale (torch.Tensor): 각 축별 양수 스케일 벡터 (N, 3)
            rotation_quaternion (torch.Tensor): 회전 쿼터니언 (w, x, y, z) (N, 4)
            
        Returns:
            torch.Tensor: 대칭 양의 반정정치 3x3 3D 공분산 행렬 Σ (N, 3, 3)
        """
        R: torch.Tensor = self.quaternion_to_rotation_matrix(rotation_quaternion)
        S: torch.Tensor = torch.diag_embed(scale)
        M: torch.Tensor = torch.bmm(R, S)
        cov3D: torch.Tensor = torch.bmm(M, M.transpose(1, 2))
        return cov3D


if __name__ == "__main__":
    num_gaussians: int = 1000
    dummy_scale: torch.Tensor = torch.exp(torch.randn(num_gaussians, 3))
    dummy_quat: torch.Tensor = torch.randn(num_gaussians, 4)

    cov_calculator: Gaussian3DCovariance = Gaussian3DCovariance()
    cov3d_matrix: torch.Tensor = cov_calculator(dummy_scale, dummy_quat)

    print(f"가우시안 입자 수: {num_gaussians}")
    print(f"3D 공분산 행렬 Σ 형상: {cov3d_matrix.shape}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | NeRF (Neural Radiance Fields) | Vanilla 3DGS (2023) | 2DGS / Mip-Splatting (2024+) |
| :--- | :--- | :--- | :--- |
| **3D 공간 표현 방식** | 암묵적 표현 (Implicit MLP) | 명시적 입자 (3D Gaussians) | 2D 평면 디스크 / Anti-aliased Gaussians |
| **추론/렌더링 속도** | 매우 늦음 (< 1 FPS) | **초고속 실시간 (100+ FPS)** | **실시간 (80+ FPS)** |
| **학습 시간** | 수 시간 ~ 수 일 소요 | **10분 ~ 30분 이내** | **15분 ~ 30분 이내 (FastGS: 100초)** |
| **표면 Mesh 추출** | 불명확함 | 부유물 아티팩트 존재 | **정교한 얇은 표면 Mesh 추출 가능** |
| **디스크 메모리 점유** | 소형 모델 (~수십 MB) | 대형 (수백 MB ~ GB) | 압축 기술 적용 시 중소형 |

### 장점
- **실시간 인터랙티브 3D 렌더링**: VR/AR, 게임 엔진, 실시간 디지털 트윈에 즉각 이식 가능함
- **초고속 최적화**: 딥러닝 MLP 추론 없이 그래픽스 타일 라스터라이저로 수렴 속도가 비약적으로 빠름

### 한계점
- 수백만 개 가우시안 속성을 보관해야 하므로 씬 디스크 저장 용량이 증가함 (가우시안 압축 기술 연구 병행)

---

## 7. 활용 사례 및 응용
1. **VR/AR & 메타버스 실시간 3D 공간 복원**: 핸드폰 카메라로 촬영한 씬을 100+ FPS 탐색 가능한 3D asset으로 자동 변환
2. **자율주행 3D 디지털 트윈 씬 생성**: 주행 카메라 영상으로부터 도로 환경 3DGS 시뮬레이터 구축
3. **Spatial Editing**: Segment Any 3D Gaussians (SA3D)를 활용한 3D 개체 분할 및 편집
