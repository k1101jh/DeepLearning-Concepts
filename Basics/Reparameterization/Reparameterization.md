# Reparameterization (Structural Reparameterization)

---
Reference:
- [RepVGG: Making VGG-Style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [GitHub - DingKe/RepVGG](https://github.com/DingKe/RepVGG)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Structural Reparameterization (구조적 재파라미터화)
- **관련 분야/카테고리**: Basics / Model Optimization / Architecture Design
- **한 줄 요약**: 학습 시에는 다중 분기(Multi-branch) 구조로 기울기 흐름과 표현력을 높이고, 추론 시에는 선형 결합 정리를 이용해 단일 합성곱-편향 블록으로 수학적으로 병합하는 네트워크 최적화 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **Multi-branch 아키텍처의 한계점**:
    - ResNet, Inception 등의 Multi-branch (Shortcut/Identity 등) 구조는 학습 시 표현력이 뛰어나고 Gradient Vanishing 문제를 완화함
    - 하지만 추론(Inference) 시 개별 브랜치의 텐서를 메모리 상에 유지하고 합산(Add/Concat)해야 하므로, DRAM 접근 오버헤드(Memory Access Cost, MAC)가 커지고 GPU 파이프라인 효율이 저하됨
- **Plain (Single-path) 모델의 단점**:
    - VGG 스타일의 Single-path (3x3 Conv + ReLU 중첩) 모델은 메모리 접근이 단순하고 추론 및 하드웨어 가속에 매우 효율적임
    - 그러나 레이어가 깊어질수록 학습이 어렵고 성능 한계에 직면함
- **해결 목표**:
    - 학습 단계와 추론 단계의 네트워크 구조를 이원화하여, 학습 시 Multi-branch의 이점과 추론 시 Plain VGG 스타일의 고속 추론 성능을 동시에 달성함

## 3. 핵심 원리 및 메커니즘 (How?)
- **선형 합성곱과 배치 정규화의 등가 변환**:
    - $3 \times 3$ Conv와 Batch Normalization(BN)은 다음과 같은 수학적 등가 1개의 $3 \times 3$ Conv + Bias로 융합 가능함
    $$W'_{i,:,:,:} = \frac{\gamma_i}{\sigma_i} W_{i,:,:,:}$$
    $$b'_i = -\frac{\mu_i \gamma_i}{\sigma_i} + \beta_i$$
- **1x1 Conv 및 Identity 분기의 3x3 Conv 변환**:
    - $1 \times 1$ Conv는 중앙에 1x1 가중치를 배치하고 테두리를 0으로 패딩(Zero-padding)하여 $3 \times 3$ Conv 패러미터로 확장함
    - Identity(Skip Connection) 분기는 입력 차원 채널 단위의 단위 행렬(Identity matrix) 형태를 지닌 $3 \times 3$ Conv 패치로 변환함
- **분기 합산 (Branch Addition)**:
    - 변환된 각 브랜치의 $3 \times 3$ 가중치 $W^{(3 \times 3)}, W^{(1 \times 1)}, W^{(\text{identity})}$와 편향 $b^{(3 \times 3)}, b^{(1 \times 1)}, b^{(\text{identity})}$을 단순 합산함
    $$W_{\text{fused}} = W^{(3 \times 3)} + \text{Pad}(W^{(1 \times 1)}) + W^{(\text{identity})}$$
    $$b_{\text{fused}} = b^{(3 \times 3)} + b^{(1 \times 1)} + b^{(\text{identity})}$$

## 4. 핵심 세부 개념 및 부가 설명
- **Memory Access Cost (MAC) 축소**:
    - 추론 시 브랜치 합산용 임시 메모리 버퍼 할당이 사라져 GPU On-chip 캐시 활용률이 올라가고 속도가 급격히 향상됨
- **Decoupled Training and Inference (학습-추론 분리 디커플링)**:
    - 훈련 중에는 `deploy=False` 상태로 다중 분기 모드로 오차역전파를 수행하고, 훈련 완료 후 `switch_to_deploy()` 함수를 호출하여 평탄화된 단일 레벨 레이어로 고정시킴

## 5. 코드 구현 예시 (PyTorch / Python)
```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepVGGBlock(nn.Module):
    """
    Structural Reparameterization을 적용한 RepVGG 기본 블록 클래스
    
    Attributes:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        stride (int): 합성곱 보폭
        deploy (bool): 추론 전용 병합 모드 여부
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        deploy: bool = False
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.stride: int = stride
        self.deploy: bool = deploy

        self.nonlinearity: nn.ReLU = nn.ReLU()

        if deploy:
            # 추론 모드: 단일 3x3 Conv 레이어만 구성
            self.rbr_reparam: nn.Conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True
            )
        else:
            # 학습 모드: 3x3 Conv+BN, 1x1 Conv+BN, Identity+BN 다중 브랜치 구성
            self.rbr_dense: nn.Sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1: nn.Sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_identity: Optional[nn.BatchNorm2d] = (
                nn.BatchNorm2d(out_channels) if out_channels == in_channels and stride == 1 else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 연산 수행
        
        Args:
            x (torch.Tensor): 입력 텐서 (Batch, Channels, Height, Width)
            
        Returns:
            torch.Tensor: 출력 텐서 (Batch, Out_channels, Out_Height, Out_Width)
        """
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))

        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def _fuse_bn_tensor(self, branch: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conv와 BatchNorm 가중치를 단일 Conv 가중치 및 편향으로 수학적 융합
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # Identity 브랜치용 단위 가중치 생성
            input_dim = self.in_channels
            kernel = torch.zeros((self.in_channels, 1, 3, 3), device=branch.weight.device)
            for i in range(self.in_channels):
                kernel[i, 0, 1, 1] = 1.0
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self) -> None:
        """
        학습 완료 후 다중 분기 가중치를 단일 3x3 Conv 가중치로 융합(Reparameterization)함
        """
        if self.deploy:
            return

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        # 1x1 커널 3x3 패딩 변환
        kernel1x1_padded = F.pad(kernel1x1, (1, 1, 1, 1))

        if self.rbr_identity is not None:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        else:
            kernelid, biasid = 0, 0

        # 가중치 및 편향 합산
        fused_kernel = kernel3x3 + kernel1x1_padded + kernelid
        fused_bias = bias3x3 + bias1x1 + biasid

        # 추론용 레퍼런스 구성
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True
        )
        self.rbr_reparam.weight.data = fused_kernel
        self.rbr_reparam.bias.data = fused_bias

        # 기존 분기 삭제
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

        self.deploy = True


if __name__ == "__main__":
    # 등가 변환 정확도 검증 테스트
    x = torch.randn(1, 16, 32, 32)
    block = RepVGGBlock(in_channels=16, out_channels=16, stride=1, deploy=False)
    block.eval()

    out_train_mode = block(x)
    block.switch_to_deploy()
    out_deploy_mode = block(x)

    diff = (out_train_mode - out_deploy_mode).abs().max().item()
    print(f"학습 모드 vs 추론 융합 모드 출력 최대 오차: {diff:.8f}")
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | ResNet (Multi-branch) | Plain VGG | RepVGG (Structural Reparam) |
| :--- | :--- | :--- | :--- |
| **학습 성능 (Capacity)** | 우수함 | 낮음 (최신 구조 대비) | 우수함 (Multi-branch 학습) |
| **추론 메모리 (Memory Cost)** | 높음 (Shortcut 버퍼 할당) | 최저 (Plain Flow) | 최저 (Plain VGG 구조 병합) |
| **추론 속도 (FPS)** | 보통 | 매우 빠름 | 매우 빠름 |
| **구조 유연성** | 높음 | 낮음 | 높음 |

- **장점**:
    - 별도의 추론 전용 커스텀 쉘 작성 없이 표준 $3 \times 3$ Conv 연산으로 변환되어 TensorRT, OpenVINO 등 엔진과의 호환성이 탁월함
- **한계점**:
    - 비선형 활성화 함수(ReLU 등)가 브랜치 내부에 들어간 경우 선형 결합 법칙이 성립하지 않아 융합 불가능함 (Conv+BN 쌍 구조로 제한)

## 7. 활용 사례 및 응용
- **RepVGG**: VGG 형태의 초고속 백본 네트워크
- **YOLOv6 / YOLOv7 / YOLOv8**: 객체 검출 모델의 RepPAN, RepConv 백본 레이어에 적용하여 에지 디바이스 추론 속도 극대화
- **MobileOne**: 모바일 기기 초저지연 (Sub-1ms) 추론 아키텍처
