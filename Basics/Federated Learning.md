# Federated Learning

---
Reference:
- https://research.google/blog/federated-learning-collaborative-machine-learning-without-centralized-training-data/
- https://medium.com/curg/%EC%97%B0%ED%95%A9-%ED%95%99%EC%8A%B5-federated-learning-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%EC%B1%8C%EB%A6%B0%EC%A7%80-b5c481bd94b7
- https://en.wikipedia.org/wiki/Federated_learning
---
목차
1. [개요](#1-개요)
2. [방법](#2-방법)

## 1. 개요
**Federated Learning(연합 학습)**
여러 클라이언트가 데이터가 탈중앙화된 상황에서 모델을 협력하여 학습하는 기술

### **중앙집중식 연합 학습**
작동 방식
1. 개인 기기에 모델을 다운로드
2. 개인 데이터로 모델을 학습
3. 모델 업데이트 내용을 암호화하여 클라우드로 전송
4. 다른 사용자의 업데이트와 평균을 내어 공유 모델을 개선

### **분산형 연합 학습**
- 클라이언트는 global model을 얻기 위해 스스로를 조정 가능
- 중앙 서버의 조정 없이 상호 연결된 클라이언트 간에 모델 업데이트를 교환
-> single point failure를 방지
- 특정 네트워크 토폴로지는 학습 프로세스의 성능에 영향을 미칠 수 있다.

### **이기종 연합 학습**
- 다른 계산 및 통신 기능을 갖는 이기종 클라이언트에서 연합 학습
- HeteroFL:

### **필요성**
- 데이터 프라이버시
    - 개인 의료 데이터 등 보호해야 하는 데이터가 존재할 때 데이터 유출 없이 모델 훈련이 가능하다.
- 효율성
    - 데이터를 전송하는 것보다 모델의 업데이트 정보만을 주고받는 것이 비용이 적다.

### **어려움**
- SGD와 같은 반복성이 높은 알고리즘은 학습 데이터에 대한 저지연, 고처리량 연결을 필요로 한다.
하지만 연합 학습의 경우, 데이터가 수백만 대의 기기에 매우 고르지 않게 분산되고, 각 기기는 지연 시간이 길고 처리량이 낮으며, 간헐적으로만 학습에 사용할 수 있다.

연합 평균화 알고리즘
- 간단한 SGD보다 더 높은 품질의 업데이트를 계산
    - 업데이트 반복 횟수가 적기 때문에 학습에 더 적은 통신을 사용할 수 있다.


### **세 가지 유형**
- 수평적 연합 학습
    - 유사한 데이터로 학습
- 수직적 연합 학습
    - 상호 보완적인 데이터
    예: 영화와 서핑을 결합하여 음악 선호도를 예측
- 연합 전이 학습
    - 한 가지 작업을 수행하도록 설계된 모델을 다른 작업을 수행하도록 다른 데이터로 학습


### **Federated Stochastic Gradient Descent(FedSGD)**


### **문제점**
- 공격
    - 클라이언트가 모델을 갖기에 공격자가 모델의 파라미터를 알고 있다.
    - targeted attack: 특정 입력값에 대한 성능을 저하시키는 공격
    - untargeted attack: 모델의 성능을 저하시키는 공격

    공격 방법
    1. Model update poisoning
        - 모델 파라미터를 수정하는 공격
        - 누가 공격자인지 발견하기 어렵다.
        - targeted attack, untargeted attack이 가능하다.
    
    2. Data poisoniong
        - 학습 데이터를 오염시켜서 모델이 의도되지 않은 방향으로 학습하도록 하는 공격
        - 특정 뉴런을 학습 단계에서 제외시키는 방법을 주로 사용한다.
        예: 이미지의 특정 영역을 흰색 또는 검은색으로 칠해서 뉴런이 활성화되는 정도를 조절
        - targeted attack, untargeted attack이 가능하다.

    3. Evasion attack
        - 데이터 샘플을 조정해서 의도되지 않은 결과를 만들어내는 공격 방법
        - adversarial examples 사용
        (사람 눈으로 구별할 수 없지만 noise를 추가해서 모델의 손실함수 값을 최대화하는 샘플)