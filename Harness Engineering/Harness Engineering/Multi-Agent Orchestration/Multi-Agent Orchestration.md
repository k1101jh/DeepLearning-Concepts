# Multi-Agent Orchestration (다중 에이전트 오케스트레이션)

---
Reference:
- [LangGraph: Multi-Agent Workflows & State Graph Orchestration](https://www.langchain.com/)
- [CrewAI Framework Architecture Guide](https://www.crewai.com/)
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"](https://arxiv.org/abs/2308.08155)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Multi-Agent%20Orchestration/Multi-Agent%20Orchestration.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Multi-Agent Orchestration (다중 에이전트 오케스트레이션 하네스)
- **관련 분야/카테고리**: Harness Engineering / Agent Collaboration / Swarm Architecture / LangGraph
- **한 줄 요약**: 단일 에이전트의 추론 과부하를 방지하기 위해 특화된 페르소나(Supervisor, Coder, Tester 등)를 가진 복수의 에이전트 간 과제 위임, 상태 공유 및 제어 흐름 그래프를 조율하는 시스템 구조

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단일 에이전트의 역할 과부하 (Single Agent Bottleneck)**:
  - 단일 LLM 에이전트에게 기획, 설계, 코딩, 테스트, 문서화 전체를 부여하면 컨텍스트 오버플로우와 툴 선택 혼란으로 작업 완수 실패율 증가
- **복잡 프로젝트의 협업 제어 부재**:
  - 개발 조직처럼 역할을 명확히 나누어 상호 검증(Peer Review)하는 파이프라인 수단 부재
- **Multi-Agent Orchestration 도입을 통한 해결**:
  - 상위 총괄 에이전트(Supervisor)와 세부 전문가 에이전트(Workers)로 분율화하고 하네스가 상태 그래프(State Graph)를 관리하며 흐름을 조율

## 3. 핵심 원리 및 메커니즘 (How?)
- **대표적인 3대 오케스트레이션 토폴로지 (Topologies)**:
  1. **Supervisor-Worker Pattern**: 관리자 에이전트(Supervisor)가 쿼리를 수신하고 하위 작업자 에이전트에게 라우팅 지시
  2. **Hierarchical Plan-and-Execute**: 총괄 플래너가 전체 계획을 세우면 개별 실행 에이전트가 순차 수행
  3. **Swarm / Peer-to-Peer Communication**: 에이전트 간 직접 메시지를 주고받으며 상호 검토 수행

```text
               [User Goal]
                    │
                    ▼
          ┌───────────────────┐
          │ Supervisor Agent  │ ─── (Orchestrator)
          └───────────────────┘
            /       │       \
           /        │        \ (Delegation)
          ▼         ▼         ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Coder    │ │ Reviewer │ │ Tester   │
    │ Agent    │ │ Agent    │ │ Agent    │
    └──────────┘ └──────────┘ └──────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Shared State Management (공유 상태 관리)**:
  - 개별 에이전트가 격리된 프롬프트를 유지하되, 하네스가 중심 상태 딕셔너리(`State`)를 보관하며 파이프라인 통신
- **Conditional Edge & Routing (조건부 분기)**:
  - Tester 에이전트의 결과가 "FAIL"이면 Coder 에이전트로 루프 백(Loop-back)하고, "PASS"이면 완료 노드로 전환하는 하네스 그래프 제어

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any, List


class MultiAgentOrchestrator:
    """Supervisor-Worker 패턴 기반 다중 에이전트 제어 하네스 시뮬레이션 클래스"""

    def __init__(self) -> None:
        """에이전트 역할 및 상태 관리 초기화"""
        self.state: Dict[str, Any] = {"status": "INIT", "code": "", "review": ""}

    def coder_agent(self, task: str) -> str:
        """[Worker Agent] 파이썬 코드를 작성하는 전용 에이전트"""
        return f"# {task} 구현 코드\ndef solution():\n    return 42"

    def reviewer_agent(self, code: str) -> Dict[str, Any]:
        """[Worker Agent] 작정된 코드를 검수하는 전용 에이전트"""
        if "def solution" in code:
            return {"passed": True, "feedback": "코드 리뷰 통과"}
        return {"passed": False, "feedback": "함수 정의 누락"}

    def supervisor_orchestrate(self, user_goal: str) -> Dict[str, Any]:
        """[Supervisor Agent] 작업 위임 및 오케스트레이션 제어 루프
        
        Args:
            user_goal (str): 사용자 요구사항 목표
            
        Returns:
            Dict[str, Any]: 최종 파이프라인 수행 결과
        """
        print(f"[Supervisor] 과제 분해 및 위임 개시: {user_goal}")
        
        # 1. Coder 에이전트에 작업 위임
        self.state["code"] = self.coder_agent(user_goal)
        print("[Supervisor -> Coder] 코드 생성 완료")

        # 2. Reviewer 에이전트에 검수 위임
        review_res = self.reviewer_agent(self.state["code"])
        self.state["review"] = review_res["feedback"]
        print(f"[Supervisor -> Reviewer] 검수 결과: {review_res['feedback']}")

        # 3. 결과 판단 및 상태 업데이트
        if review_res["passed"]:
            self.state["status"] = "COMPLETED"
        else:
            self.state["status"] = "RETRY_REQUIRED"

        return self.state


# 오케스트레이션 하네스 테스트
if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.supervisor_orchestrate("알고리즘 솔루션 작성")
    print("Multi-Agent 최종 상태:", result)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 단일 에이전트 (Single Agent) | Multi-Agent Orchestration |
| :--- | :--- | :--- |
| **역할 분담** | 단일 LLM이 전 영역 수행 | Coder, Tester, Planner 각 페르소나 독립 전문화 |
| **복잡 문제 성공률** | 높지 않음 (컨텍스트 오버플로우) | 비약적으로 높음 (상호 검증 루프 구현) |
| **시스템 구조** | 단순 | 하네스 그래프 상태 관리 아키텍처 복잡 |

- **장점**:
  - 역할 전문화를 통해 복잡 대형 프로젝트의 완수율 극대화
- **한계점**:
  - 여러 에이전트 호출에 따른 토큰 소모량 및 API 비용 증가

## 7. 활용 사례 및 응용
- **LangGraph / CrewAI / AutoGen**:
  - 소프트웨어 개발 생태계의 대표적인 Multi-Agent 오케스트레이션 프레임워크
