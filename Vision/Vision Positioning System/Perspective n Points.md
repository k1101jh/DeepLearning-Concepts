# Perspective n Points(PnP)

---


---


## Introduction
- 3D point cloud 정보와 2D feature 정보를 이용하여 카메라의 포즈를 추정하는 방법


PnP 알고리즘의 종류
- P3P
- DLT
- EPnP
- Bundle Adjustment

## Direct Linear Transformation(DLT)

3D point P 의 homogeneous coordinates를 $P = (X, Y, Z, 1)^T$라고 하고, image $I_1$에 투영된 normalized homogeneous coordinate를 $x_1 = (u_1, v_1, 1)^T$라 할 때, camera의 pose $\text{R}, \text{T}$가 주어져 있음



## P3P
- 평면으로 projection 된 두 점과 intrinsic parameter을 사용해서 3D 공간상의 두 점의 각도를 계산할 수 있음
![alt text](image.png)
출처: https://velog.io/@koyeongmin/%EA%B8%B0%EC%B4%88-%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-5-P3P

- 점이 3개 있는 경우
![alt text](image-1.png)
출처: https://velog.io/@koyeongmin/%EA%B8%B0%EC%B4%88-%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-5-P3P

- 제 2코사인 법칙을 사용하여 다음과 같은 식을 만듦
$$
\displaystyle
\begin{aligned}

\end{aligned}
$$
- 4차 방정식을 풀었을 때, 3개 점에 대한 깊이 정보 $s_1, s_2, s_3$가 4종류 나옴
- 4번째 점을 사용해서 유일한 해를 찾음

## AP3P

## Efficient Perspective n Points(EPnP)




## SQPnP


## End-to-End Probabilistic PnP

pose를 고정적인 하나의 값으로 표현하지 않고, 확률 분포의 형태로 표현
-> 연속적이기 때문에 미분 가능
-> argmin function을 softargmin으로 대체