# Generative Adversarial Network(GAN)

---
Reference:
- a
- b
- 미술관에 GAN
---

목차
1. [개요](#1-개요)
2. [방법](#2-방법)

## 1. 개요
**Generative Adversarial Network(GAN)**:

## 2. 사전지식

### 2.1. Auto Encoder
![alt text](./images/Auto%20Encoder%20architecture.png)

[[이미지 출처]](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798)

입력과 출력이 동일한 신경망

- Encoder: 고차원의 이미지를 latent vector(잠재 벡터)로 압축
- Decoder: latent vector를 원본 차원으로 압축 해제

**Mnist test data를 2차원으로 압축한 분포도**
![alt text](./images/Auto%20Encoder%20Mnist%20scatter%20plot.png){: width="50%" height="50%"}

