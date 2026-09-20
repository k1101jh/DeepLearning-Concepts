# LLM Utilization Methods (LLM 활용 방법론)

---
Reference:
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023](https://arxiv.org/abs/2210.03629)
- [Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023](https://arxiv.org/abs/2305.10601)
- [Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023](https://arxiv.org/abs/2303.11366)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Utilization%20Methods/LLM%20Utilization%20Methods.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: LLM Utilization Methods (LLM 활용 방법론 및 에이전틱 패턴)
- **관련 분야/카테고리**: Prompt Engineering / Agentic Workflow / Reasoning Patterns
- **한 줄 요약**: 단일 추론 프롬프팅부터 고도화된 추론-액션 연동, 자가 반성, 다중 에이전트 협업 체계까지 LLM의 성능을 극대화하는 전략 및 알고리즘 모음

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단일 Zero-shot 프롬프팅의 복잡 문제 해결 한계**:
  - LLM은 다단계 복잡 추론(수학, 코드 작성, 에이전트 도구 활용 등) 수행 시 중간 사고 과정 없이 한 번에 답을 생성하려 할 경우 환각(Hallucination) 및 논리적 오류 발생
- **방법론 도입을 통한 문제 해결**:
  - 사고 단계 체인화(CoT, ToT), 환경과의 도구 연동(ReAct), 자기 오류 수정(Reflexion)을 통해 복잡 과제 해결 성공률 비약적 상승

## 3. 핵심 원리 및 메커니즘 (How?)
- **주요 추론 및 프롬프팅 패턴 발전 계층 구조**:
  - **Chain-of-Thought (CoT)**: 선형적 사고 단계 생성 ($Input \rightarrow Thought_1 \rightarrow Thought_2 \rightarrow Answer$)
  - **Tree-of-Thoughts (ToT)**: 트리 형태의 다중 사고 경로 탐색 및 DFS/BFS 기반 평가 수식 적용:
    $$Value(s) = \mathbb{E}_{v \in V} [ Score(s, v) ]$$
  - **ReAct (Reasoning + Acting)**: 추론(Thought), 행동(Action), 관찰(Observation)의 반복 루프 ($T_i \rightarrow A_i \rightarrow O_i \rightarrow T_{i+1}$)
  - **Reflexion**: 실패한 경로에 대해 언어적 자기 반성(Verbal Reflection)을 수행하여 메모리에 저장 후 재시도

```text
[User Task]
    │
    ▼
┌──────────────┐    Reasoning    ┌──────────────┐
│ Thought Step │ ──────────────> │ Action Step  │
└──────────────┘                 └──────────────┘
       ▲                                │
       │          Observation           │ Execute Tool
       └────────────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **[Prompt Caching (프롬프트 캐싱 및 접두사 공유)](./Prompt%20Caching/Prompt%20Caching.md)**: 시스템 프롬프트 및 툴 스키마 정적 접두사의 KV 어텐션 캐싱을 통해 **입력 토큰 비용 50~90% 절감 및 TTFT 85% 단축**
- **Plan-and-Execute (계획 및 실행 분리)**:
  - 거대한 목표를 세부 서브 타스크 목록으로 먼저 분할(Decomposition)한 후, 세부 타스크별로 실행 에이전트가 개별 수행하는 분리 방식
- **Self-Reflection / Reflexion (자가 반성)**:
  - 환경의 피드백(오류 메시지, 테스트 실패 등)을 기반으로 자신의 과거 동작 트래젝터리를 스스로 비판 분석하는 인-컨텍스트 학습 기법
- **Multi-Agent Collaboration (다중 에이전트 협업)**:
  - 페르소나(개발자, 검수자, 기획자 등)가 부여된 복수의 에이전트가 대화를 통해 상호 검증하며 최종 결과물을 도출하는 구조

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import List, Dict, Any, Tuple


class ReActAgent:
    """ReAct (Thought -> Action -> Observation) 패턴 및 간단한 Self-Reflection 제어 클래스"""

    def __init__(self, max_steps: int = 5) -> None:
        """Agent 초기화
        
        Args:
            max_steps (int): 최대 반복 추론 단계
        """
        self.max_steps: int = max_steps
        self.memory: List[str] = []

    def mock_llm_reasoning(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """LLM의 추론 및 액션 결정 시뮬레이션 메서드
        
        Args:
            prompt (str): 현재 상황 프롬프트
            
        Returns:
            Tuple[str, str, Dict[str, Any]]: (Thought, Action_Name, Action_Args)
        """
        if "계산" in prompt and "단계 1 완료" not in self.memory:
            return ("두 수의 합을 계산해야 한다.", "calculator", {"expression": "128 + 256"})
        elif "단계 1 완료" in self.memory:
            return ("계산 결과를 바탕으로 최종 답변을 작성한다.", "finish", {"answer": "384"})
        else:
            return ("작업을 이해할 수 없어 자가 반성이 필요하다.", "reflect", {})

    def mock_environment_step(self, action_name: str, args: Dict[str, Any]) -> str:
        """외부 도구 및 환경 실행 관찰(Observation) 반환 메서드
        
        Args:
            action_name (str): 도구 명칭
            args (Dict[str, Any]): 도구 인자
            
        Returns:
            str: 관찰된 결과 문자열
        """
        if action_name == "calculator":
            expr = args.get("expression", "0")
            result = eval(expr)
            return f"계산 결과: {result}"
        return "알 수 없는 액션"

    def run(self, user_query: str) -> str:
        """ReAct 루프 실행 메서드
        
        Args:
            user_query (str): 사용자 요구사항 질문
            
        Returns:
            str: 최종 답변
        """
        print(f"[시작] 사용자 요청: {user_query}")
        
        for step in range(1, self.max_steps + 1):
            current_context = f"Query: {user_query}\nMemory: " + "\n".join(self.memory)
            thought, action_name, action_args = self.mock_llm_reasoning(current_context)
            
            print(f"\n--- Step {step} ---")
            print(f"Thought: {thought}")
            
            if action_name == "finish":
                final_ans = action_args.get("answer", "")
                print(f"Final Answer: {final_ans}")
                return final_ans
                
            elif action_name == "reflect":
                reflection = "이전 액션 실패에 대한 자가 반성: 기본 질문 재해석 필요"
                self.memory.append(f"Reflection: {reflection}")
                print(f"Reflexion 수행: {reflection}")
                
            else:
                obs = self.mock_environment_step(action_name, action_args)
                print(f"Action: {action_name}({action_args})")
                print(f"Observation: {obs}")
                self.memory.append(f"Thought: {thought} | Obs: {obs} | 단계 1 완료")
                
        return "최대 단계 초과로 종료"


# ReAct 실행 테스트
if __name__ == "__main__":
    agent = ReActAgent()
    result = agent.run("128과 256을 더하는 계산 작업을 수행해 줘")
    print("\n최종 실행 결과:", result)
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 기법 | Chain-of-Thought (CoT) | ReAct Pattern | Tree-of-Thoughts (ToT) |
| :--- | :--- | :--- | :--- |
| **추론 구조** | 단일 단방향 사고 체인 | 추론 + 도구 액션 피드백 루프 | 다중 가지 분기 탐색 (Tree Search) |
| **도구 연동** | 불가능 (Internal Reasoning) | 가능 (External Observation) | 선택적 연동 |
| **적용 과제** | 단순 산술 및 논리 추론 | 에이전트 타스크, 웹 검색, 코드 작성 | 게임, 경로 탐색, 복잡 기획 |
| **토큰 소모** | 적음 | 중간 | 매우 큼 |

- **장점**:
  - LLM의 도구 활용 능력 및 복잡 문제 해결력 증대
  - 자가 반성을 통한 실시간 오류 복구 기법 확보
- **한계점**:
  - 다단계 루프 생성에 따른 API 호출 비용 증가 및 레이턴시 상승

## 7. 활용 사례 및 응용
- **AutoGPT / BabyAGI**:
  - Task Decomposition 및 ReAct 루프 기반 자율 에이전트
- **LangGraph / CrewAI**:
  - Multi-Agent Collaboration 및 그래프 기반 Workflow 제어 프레임워크
