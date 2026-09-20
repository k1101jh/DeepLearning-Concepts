# Visual SLAM (vSLAM, 비주얼 동시적 위치추정 및 지도작성)

---
Reference:
- [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898)
- [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869)
- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://splatam-slam.github.io/)
- [MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](https://arxiv.org/abs/2501.00000)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Visual SLAM (vSLAM, 비주얼 동시적 위치추정 및 지도작성)
- **관련 분야/카테고리**: Vision / Robotics / Spatial Computing / Pose Estimation
- **한 줄 요약**: 미지의 환경을 이동하는 에이전트(로봇, 드론, AR 글래스)가 카메라 영상 스트림을 통해 실시간 6자유도 포즈(Position & Orientation)를 추적함과 동시에 주변 환경의 정밀 3D 맵(Map)을 구축하는 정밀 공간 지능 기술

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### 1) GPS 및 LiDAR 기반 기존 항법의 한계점
- **실내 및 음영 지역 GPS 불능**:
    - 건물 내부, 지하 주차장, 우주 공간 등에서는 GPS 신호가 도달하지 않거나 신호 오차가 수 미터 이상 발생함
- **LiDAR의 높은 비용과 중량**:
    - 3D 라이다 장비는 매우 비싸고 무게와 전력 소모가 커서 소형 드론, 스마트폰, AR 글래스에 탑재하기 어려움

### 2) Visual SLAM 도입을 통한 핵심 해결 목표
- **저비용·고효율 센서(카메라) 활용**: 어디서나 쉽게 구할 수 있는 모노큘러(Monocular), 스테레오(Stereo), RGB-D 카메라만으로 정밀 위치 추적 실현함
- **풍부한 시각 텍스처 정보 활용**: 3D 포인트 구름뿐만 아니라 시각적 마커, 루프 클로저(Loop Closure)를 이용해 누적 오차(Drift Error)를 보정하고 무결점 3D 맵을 구축함

---

## 3. 핵심 원리 및 메커니즘 (How?)

### 1) Visual SLAM 파이프라인 주요 구성 모듈

```
[Camera Frame] ---> [Front-End] (Feature Tracking / Optical Flow)
                          |
                          v
                    [Back-End] (Bundle Adjustment / Pose Graph Optimization)
                          |
                          +---> [Loop Closure Detection] (BoW / Global Retrieval)
                          |
                          v
                    [3D Map & Camera Trajectory]
```

1. **프론트엔드 (Front-End - Tracking)**:
   - 연속된 프레임 간 특징점(ORB, SuperPoint) 추출 및 매칭 또는 광학 흐름(Optical Flow) 추적으로 카메라의 프레임간 이동 변화량($T_{t, t-1} \in SE(3)$)을 추정함
2. **백엔드 (Back-End - Optimization)**:
   - **번들 조정 (Bundle Adjustment, BA)**: 수집된 카메라 포즈와 3D 포인트 위치 간의 재투영 오차(Reprojection Error)를 최소화하도록 비선형 최적화(Levenberg-Marquardt)를 수행함
3. **루프 클로저 (Loop Closure Detection)**:
   - 로봇이 이전에 방문했던 장소로 다시 돌아왔음을 재인식(Place Recognition)하여, 그동안 누적된 전역 드리프트 오차(Accumulated Drift)를 한 번에 교정함

---

### 2) 재투영 오차 (Reprojection Error) 수식

3D 지점 $P_i \in \mathbb{R}^3$가 $j$번째 카메라 포즈 $T_j \in SE(3)$와 카메라 내적 파라미터 $K$에 의해 2D 픽셀 평면에 투영될 때, 실제 측정 픽셀 $u_{ij}$과의 차이 오차를 최소화합니다:

$$\min_{T_j, P_i} \sum_{i, j} \rho \left( \| u_{ij} - \pi(K T_j P_i) \|^2 \right)$$
- $\pi$: 3D to 2D 핀홀 투영 함수
- $\rho$: 노이즈에 강건한 Huber Loss 함수
- $K$: 카메라 내적 마트릭스 (Focal Length $f_x, f_y$, Principal Point $c_x, c_y$)

---

## 4. 핵심 세부 개념 및 패러다임 발전

### 1) 1세대: 기하학 기반 희소 vSLAM (ORB-SLAM3)
- Epipolar 기하학 및 핸드크래프티드 특징점(ORB) 기반으로 CPU 상에서 극초고속 동기화 작동

### 2) 2세대: 딥러닝 기반 광학 흐름 vSLAM (DROID-SLAM)
- 신경망 기반의 Dense Optical Flow 및 미분 가능한 번들 조정(Differentiable BA)을 적용하여 텍스처가 부족한 벽면이나 빛 변동 환경에서 안정적 추적

### 3) 3세대: 3D Gaussian Splatting SLAM (SplaTAM, MonoGS++, MASt3R-SLAM)
- 지도 표현체를 3D 가우시안 입자로 구성하여 렌더링 이미지와 입력 프레임 간의 포토메트릭 오차(Photometric Loss) 및 깊이 오차(Depth Loss)를 역전파하여 트래킹과 고화질 지도 입자 최적화를 동시 수행함

---

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Tuple
import torch
import torch.nn as nn

class PinholeProjection(nn.Module):
    """
    3D 공간 좌표 포인트를 카메라 6차원 포즈(R, t)와 내적 행렬 K를 이용해
    2D 픽셀 평면 좌표로 투영하는 Pinhole Camera Projection 모듈
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        super().__init__()
        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        self.register_buffer("K", K)

    def forward(
        self,
        points_3d: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D 포인트를 2D 픽셀 평면으로 투영하고 깊이(Depth) 및 픽셀 좌표를 반환함
        
        Args:
            points_3d (torch.Tensor): 3D 공간 포인트 좌표 (batch_size, N, 3)
            R (torch.Tensor): 카메라 회전 행렬 (batch_size, 3, 3)
            t (torch.Tensor): 카메라 평행이동 벡터 (batch_size, 3, 1)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 2D 픽셀 좌표 (u, v) 및 깊이 z
        """
        points_transposed = points_3d.transpose(1, 2)
        p_cam = torch.bmm(R, points_transposed) + t  # (batch_size, 3, N)

        z = p_cam[:, 2:3, :].transpose(1, 2)  # (batch_size, N, 1)
        p_pixel_homo = torch.bmm(self.K.unsqueeze(0).expand(points_3d.shape[0], -1, -1), p_cam)

        u = p_pixel_homo[:, 0:1, :] / (p_pixel_homo[:, 2:3, :] + 1e-7)
        v = p_pixel_homo[:, 1:2, :] / (p_pixel_homo[:, 2:3, :] + 1e-7)

        pixels_2d = torch.cat([u, v], dim=1).transpose(1, 2)
        return pixels_2d, z


if __name__ == "__main__":
    proj_module: PinholeProjection = PinholeProjection(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
    dummy_points_3d: torch.Tensor = torch.tensor([[
        [0.5, 0.2, 2.0],
        [-0.3, 0.8, 3.5],
        [1.2, -0.5, 1.5]
    ]], dtype=torch.float32)

    dummy_R: torch.Tensor = torch.eye(3).unsqueeze(0)
    dummy_t: torch.Tensor = torch.zeros(1, 3, 1)

    pixels_2d, depth = proj_module(dummy_points_3d, dummy_R, dummy_t)

    print(f"3D 입력 포인트 수: {dummy_points_3d.shape[1]}")
    print(f"투영된 2D 픽셀 좌표 (u, v):\n{pixels_2d[0]}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Traditional Geometric SLAM (ORB-SLAM3) | Deep Optical Flow SLAM (DROID-SLAM) | 3DGS SLAM (SplaTAM / MonoGS++) |
| :--- | :--- | :--- | :--- |
| **지도 표현 형태** | 희소 점군 (Sparse Point Cloud) | 밀도 깊이 맵 (Dense Depth Grid) | **3D 가우시안 포토레얼리스틱 맵** |
| **신규 뷰 렌더링** | 불가능 | 고품질 불가 | **실시간 고화질 뷰 신규 생성 가능** |
| **강건성 (Robustness)** | 텍스처 부족 시 트래킹 이탈 발생 | 우수함 | **매우 우수함** |
| **연산 자원 소모** | 최저 (CPU 전용 가능) | GPU 요구됨 | GPU 요구됨 (실시간 30~60 FPS) |

### 장점
- **센서 범용성**: 단일 카메라만으로 3D 공간을 인식하고 포토레얼리스틱 실사 복원 맵을 동시 구축 가능함
- **Visual-Semantic 확장성**: Object Detection 및 3DGS와 결합하여 의미론적 공간 지도 구축 가능

### 한계점
- 완전한 어둠이나 텍스처가 없는 벽면에서는 모노큘러 특성상 스케일 모호성(Scale Ambiguity)이 발생하여 IMU/RGB-D 센서 보완 필요

---

## 7. 활용 사례 및 응용
1. **공간 컴퓨팅 (Apple Vision Pro, Meta Quest 3)**: 헤드셋 6DoF 렌더링 위치 실시간 트래킹
2. **자율주행 및 로보틱스**: GPS 음영 실내외 환경에서의 자율 주행 및 경로 탐색
3. **3D Digital Twin**: 건물 실내 탐색과 동시에 실시간 3DGS 맵을 복원하여 가상 투어 어셋화
