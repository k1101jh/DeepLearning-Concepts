
---
리스트

---
Reference:
- https://gaussian37.github.io/dl-concept-relu6/
---



## ReLU6
기존의 ReLU에서 상한 값을 6으로 둔 함수

$\min(\max(0, x), 6)$

- Test 경과 학습 관점에서 성능이 더 좋았고 최적화 관점(fixed-point)에서 더 좋다.

Fixed-point 관점
- upper bound가 없으면 point를 표현하는 데 수많은 bit를 사용해야 하지만 6으로 상한선을 두면 최대 3개의 bit만 있으면 된다.

- 딥러닝 모델이 학습할 때, sparse한 feature를 더 일찍 학습할 수 있다.


ReLU
ReLU6
GELU
SiLU
