# Visual SLAM (비주얼 슬램, 동시적 위치추정 및 지도작성)

---
Reference:
- [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM (Campos et al., IEEE T-RO 2021)](https://arxiv.org/abs/2007.11898)
- [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras (Teed & Deng, NeurIPS 2021)](https://arxiv.org/abs/2108.10869)
- [DROID-SLAM Official GitHub Repository](https://github.com/princeton-vl/DROID-SLAM)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Visual SLAM (vSLAM, 비주얼 동시적 위치추정 및 지도작성)
- **관련 분야/카테고리**: Vision / Robotics / Spatial Computing / Pose Estimation
- **한 줄 요약**: 미지의 환경을 이동하는 에이전트(로봇, 드론, AR 글래스)가 온보드 카메라 영상 스트림만을 이용하여 자신의 실시간 6차원 이동 포즈(Position & Orientation)를 추적함과 동시에 3D 맵(Map)을 구축하는 정밀 공간 지능 기술이다.

---

## 2. 등장 배경 및 해결하려는 문제 (Why?)

### GPS 및 LiDAR 기반 기존 항법의 한계점
- **실내 및 음영 지역 GPS 불능**: 음영 건물 내부, 지하, 우주 공간 등에서는 GPS 신호가 도달하지 않거나 신호 오차가 커서 위치를 추적할 수 없음.
- **LiDAR의 높은 비용과 중량**: 3D 라이다 장비는 매우 비싸고 무게와 전력 소모가 커서 소형 드론, 스마트폰, AR 글래스에 탑재하기 어렵습니다.

### Visual SLAM 도입을 통한 핵심 해결 목표
- **저비용·고효율 센서(카메라) 활용**: 어디서나 쉽게 구할 수 있는 모노큘러(Monocular), 스테레오(Stereo), RGB-D 카메라만으로 정밀 센서 융합 위치 추적을 실현함.
- **풍부한 시각적 텍스처 정보 활용**: 3D 포인트 구름뿐만 아니라 시각적 마커, 루프 클로저(Loop Closure)를 이용해 누적 오차(Drift Error)를 보정하고 무결점 3D 맵을 구축함.

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
   - 연속된 프레임 간 특징점(ORB, SuperPoint) 추출 및 매칭 또는 딥러닝 기반 아티팩트 광학 흐름(Optical Flow) 추적으로 카메라의 1차 프레임간 이동 변화량($T_{t, t-1} \in SE(3)$)을 추정함.
2. **백엔드 (Back-End - Optimization)**:
   - **번들 조정 (Bundle Adjustment, BA)**: 수집된 카메라 포즈와 3D 포인트 위치 간의 재투영 오차(Reprojection Error)를 최소화하도록 비선형 최적화(Levenberg-Marquardt)를 수행함.
3. **루프 클로저 (Loop Closure Detection)**:
   - 로봇이 이전에 방문했던 장소로 다시 돌아왔음을 재인식(Place Recognition)하여, 그동안 누적된 전역 드리프트 오차(Accumulated Drift)를 한 번에 교정함.

### 2) 재투영 오차 (Reprojection Error) 수식

3D 지점 $P_i \in \mathbb{R}^3$가 $j$번째 카메라 포즈 $T_j \in SE(3)$와 카메라 내적 파라미터 $K$에 의해 2D 픽셀 평면에 투영될 때, 실제 측정 픽셀 $u_{ij}$과의 차이 오차를 최소화합니다:

$$\min_{T_j, P_i} \sum_{i, j} \rho \left( \| u_{ij} - \pi(K T_j P_i) \|^2 \right)$$

(단, $\pi$는 3D to 2D 핀홀 투영 함수, $\rho$는 노이즈에 강건한 Huber Loss)

---

## 4. 핵심 세부 개념 및 부가 설명

### 1) Traditional (ORB-SLAM) vs Deep Learning (DROID-SLAM)
- **전통적 기법 (ORB-SLAM3)**: 수식 기반 기하학(Epipolar Geometry)과 핸드크래프티드 특징점(ORB Feature)을 사용하여 CPU 상에서 극도로 빠른 초고속 처리가 가능함.
- **딥러닝 기반 (DROID-SLAM)**: 딥러닝 광학 흐름(Optical Flow)과 미분 가능한 번들 조정(Differentiable BA)을 결합하여, 텍스처가 부족한 벽면이나 빛 변동이 심한 극한 환경에서도 튕기지 않고 극도로 안정적으로 추적함.

---

## 5. 코드 구현 예시 (PyTorch / Python)

다음은 PyTorch를 이용하여 3D 공간상의 포인트 집합 $P$가 6자유도 카메라 포즈 $T = [R | t]$와 내적 카메라 행렬 $K$를 통과해 2D 픽셀 좌표로 투영되는 핀홀 카메라 핀홀 재투영 연산 모듈 예시 코드임.

```python
from typing import Tuple
import torch
import torch.nn as nn


class PinholeProjection(nn.Module):
    """3D 공간 좌표 포인트를 카메라 6차원 포즈(R, t)와 내적 행렬 K를 이용해
    2D 픽셀 평면 좌표로 투영하는 Pinhole Camera Projection 모듈.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        super().__init__()
        # 카메라 내적 파라미터 행렬 K (3x3)
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
        """3D 포인트를 2D 픽셀 평면으로 투영하고 깊이(Depth) 및 픽셀 좌표를 반환함.

        Args:
            points_3d (torch.Tensor): 3D 공간 포인트 좌표. 크기: (batch_size, N, 3)
            R (torch.Tensor): 카메라 회전 행렬. 크기: (batch_size, 3, 3)
            t (torch.Tensor): 카메라 평행이동 벡터. 크기: (batch_size, 3, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 2D 픽셀 좌표 (u, v) 텐서 (batch_size, N, 2)
                - 유효 깊이 z 텐서 (batch_size, N, 1)
        """
        # 1. 3D 포인트를 카메라 좌표계로 변환: P_cam = R @ P_world + t
        # (batch_size, N, 3) -> transpose -> (batch_size, 3, N)
        points_transposed = points_3d.transpose(1, 2)
        p_cam = torch.bmm(R, points_transposed) + t  # (batch_size, 3, N)

        # 2. 깊이 z 추출
        z = p_cam[:, 2:3, :].transpose(1, 2)  # (batch_size, N, 1)

        # 3. 카메라 내적 행렬 K 적용: p_pixel_homo = K @ P_cam
        p_pixel_homo = torch.bmm(self.K.unsqueeze(0).expand(points_3d.shape[0], -1, -1), p_cam)

        # 4. 정규화 좌표계 변환 (u/z, v/z)
        u = p_pixel_homo[:, 0:1, :] / (p_pixel_homo[:, 2:3, :] + 1e-7)
        v = p_pixel_homo[:, 1:2, :] / (p_pixel_homo[:, 2:3, :] + 1e-7)

        pixels_2d = torch.cat([u, v], dim=1).transpose(1, 2)  # (batch_size, N, 2)
        return pixels_2d, z


# 실행 예시
if __name__ == "__main__":
    # 가상의 카메라 내적 파라미터 (fx, fy, cx, cy)
    proj_module: PinholeProjection = PinholeProjection(fx=525.0, fy=525.0, cx=319.5, cy=239.5)

    # 3D 공간 상의 포인트 5개 생성
    dummy_points_3d: torch.Tensor = torch.tensor([[
        [0.5, 0.2, 2.0],
        [-0.3, 0.8, 3.5],
        [1.2, -0.5, 1.5],
        [0.0, 0.0, 5.0],
        [-1.0, -1.0, 2.5]
    ]], dtype=torch.float32)

    # 단위 회전 행렬 및 이동 벡터
    dummy_R: torch.Tensor = torch.eye(3).unsqueeze(0)
    dummy_t: torch.Tensor = torch.zeros(1, 3, 1)

    pixels_2d, depth = proj_module(dummy_points_3d, dummy_R, dummy_t)

    print(f"3D 입력 포인트 수: {dummy_points_3d.shape[1]}")
    print(f"투영된 2D 픽셀 좌표 (u, v) 샘플:\n{pixels_2d[0, :3]}")
```

---

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | GPS 기반 위치추정 | LiDAR SLAM | Visual SLAM |
| :--- | :--- | :--- | :--- |
| **센서 가격 및 중량** | 매우 저렴 / 경량 | 매우 고가 / 중량 | **저렴함 / 극도로 경량 (단일 카메라 가능)** |
| **사용 가능 공간** | 실외 한정 | 실내 / 실외 | **실내 / 실외 구분 없이 공간 전역 가능** |
| **시각 텍스처 정보** | 없음 | 없음 (3D 지형 위주) | **풍부함 (시각적 마커 및 객체 인식 연동)** |
| **환경 조명 민감도** | 없음 | 영향 없음 | 조명이 전무한 어둠에서는 성능 저하 가능 |

### 장점
- **센서 접근성 및 범용성**: 스마트폰, AR 헤드셋, 소형 드론에 내장된 카메라만으로 즉시 동작.
- **Visual-Semantic 확장성**: Object Detection 및 3D Gaussian Splatting과 결합하여 의미론적 공간 지도 구축 가능.

### 한계점
- 완전한 어둠이나 텍스처가 전혀 없는 민무늬 흰 벽면에서는 특징점이 없어 트래킹이 일시적으로 꺼질(Tracking Lost) 수 있음.

---

## 7. 활용 사례 및 응용

1. **AR / VR 헤드셋 (Apple Vision Pro, Meta Quest 3)**
   - 공간 컴퓨팅(Spatial Computing)에서 사용자의 두부 6자유도 위치 및 시선을 실시간 추적하여 3D 가상 객체를 현실 공간에 고정.
2. **자율주행차 및 로봇 청소기**
   - GPS가 수신되지 않는 지하 주차장 또는 거실 내부에서 실시간 지도 생성 및 본체 위치 추적.
