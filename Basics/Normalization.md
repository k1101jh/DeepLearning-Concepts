값의 범위를 0~1 사이로 바꾼다.

### InstanceNorm
 - 하나의 instance 값을 정규화 (이미지 한 장을 정규화)
 - 주로 이미지 스타일 변환에 사용

### BatchNorm
 - 각 channel을 Batch 단위로 정규화

### LayerNorm
 - 같은 channel끼리 정규화

### GroupNorm

 - BatchNorm의 문제를 해결
   - Batch 사이즈가 작으면 배치의 평균과 분산이 데이터를 대표하지 못함

 - Diffusion에서 GroupNorm을 사용
   - 이유?([BiT(Big Transfer)](https://doi.org/10.1007/978-3-030-58558-7_29) 4.3절)
      - BN은 이미지의 수가 너무 적으면 성능이 저하된다.
      - 대규모 batch에 걸친 BN 통계 계산은 일반화를 해친다.
      - Global BN을 사용하려면 많은 집계가 필요하므로 상당한 latency가 발생한다.

    - 하지만 GN만 사용하면 BN 사용에 비해 성능이 저하된다.
    -> Weight Standardization(WS)를 추가하면 BN보다 성능이 뛰어나다.
    (WS: Conv Filter의 mean을 0, var을 1로 조정)

