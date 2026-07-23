InstructGPT

url: https://openai.com/research/instruction-following

![Image](https://cdn.openai.com/instruction-following/draft-20220126f/methods.svg)

Supervised Fine-Tuning(SFM)

1. 고품질의 LLM 출력 데이터셋을 선별
2. labeler가 적절한 출력을 규정
3. 해당 데이터로 Fine-Tuning

일반적인 Fine-Tuning과 다른 점:
 - 일반적인 Fine-Tuning은 특정 작업을 해결하는 방법을 가르치기 위해 수행된다. 이로 인해 모델이 전문화되지만 덜 일반적이게 된다. 반면, SFM은 올바른 스타일이나 행동을 모방하기 위해 Fine-Tuning하여 일반적인 문제 해결 능력을 잃지 않는다.


Reinforcement Learning from Human Feedback(RLHF)