# Human-in-the-Loop (HITL 승인 하네스)

---
Reference:
- [LangChain: Human-in-the-loop Agent Architecture](https://www.langchain.com/)
- [Interactive Gateways & Permission Control in Autonomous Agents](https://arxiv.org/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Human-in-the-Loop/Human-in-the-Loop.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Human-in-the-Loop (HITL / 인간 개입 승인 하네스)
- **관련 분야/카테고리**: Harness Engineering / Agent Safety / Permission Control / Interactive Gateways
- **한 줄 요약**: 에이전트가 고위험 액션(`git push`, DB 삭제, 커맨드 실행 등)을 수행하려 할 때 실행을 일시 중단(Pause)하고 사용자의 실시간 검토 및 승인을 거치도록 제어하는 보안 하네스 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **자율 에이전트의 파괴적 실행 리스크**:
  - 완벽 자율 구동 에이전트에 전체 권한을 부여할 경우 환각이나 오판으로 인해 운영 데이터베이스 삭제, 무단 커밋 push, 비용 발생 API 남발 등 돌이킬 수 없는 피해 발생
- **전면 자동화와 안전성 간의 트레이드오프**:
  - 안전을 위해 에이전트 기능을 통제하면 자동화 생산성이 감소하는 문제 존재
- **HITL 기법 도입을 통한 해결**:
  - 저위험 액션(읽기, 분석)은 자동 수행하되, 고위험 액션만 선택적으로 사용자 승인 게이트웨이(Permission Gateway)를 거치도록 하네스가 인터셉트

## 3. 핵심 원리 및 메커니즘 (How?)
- **HITL 하네스 3단계 게이트웨이 파이프라인**:
  1. **Action Assessment**: 에이전트가 호출하려는 툴의 위험도(Risk Level: Low, High, Critical) 판별
  2. **Execution Pause & Prompt**: Critical 액션 시 하네스가 실행을 정지하고 대화 인터페이스에 사용자 승인 모달 출력
  3. **User Feedback Routing**: 사용자의 승인(Approve), 거부(Reject), 또는 변경 지시(Edit) 피드백에 따라 경로 분기

```text
[LLM Agent Tool Action]
           │
           ▼
┌───────────────────────────┐
│ HITL Safety Gateway       │ ─── (Is Action High-Risk?)
└───────────────────────────┘
     │ (NO: Auto Approve)        │ (YES: Pause Execution)
     ▼                           ▼
[Execute Tool]             [User Permission Approval]
                                 │
                                 ├── (Approve) ───> [Execute Tool]
                                 └── (Reject)  ───> [Return Error Hint to LLM]
```

## 4. 핵심 세부 개념 및 부가 설명
- **Breakpoint & Resume (중단점 및 복원)**:
  - 하네스가 세션 상태를 저장(Checkpoint)하고 사용자가 승인 버튼을 누를 때까지 비동기 수면(Sleep) 대기 후 복원
- **Scope-based Permission Allowlist (권한 허용 목록)**:
  - 동일한 `git` 명령이라도 `git status`는 자동 통과, `git push`는 HITL 승인을 요구하는 정밀 정책 통제

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any, Tuple, Callable


class HumanInTheLoopHarness:
    """고위험 액션 호출을 인터셉트하여 사용자의 승인을 대기하는 HITL 하네스 클래스"""

    def __init__(self, high_risk_tools: list[str] = None) -> None:
        """하네스 초기화
        
        Args:
            high_risk_tools (list[str]): 승인이 필요한 고위험 도구 목록
        """
        self.high_risk_tools: list[str] = high_risk_tools or ["delete_database", "git_push", "run_terminal_command"]

    def intercept_and_request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_approval_fn: Callable[[str, Dict[str, Any]], bool]
    ) -> Tuple[bool, str]:
        """[HITL Interceptor] 도구 호출 위험도를 판단하고 사용자 승인을 요청하는 메서드
        
        Args:
            tool_name (str): 실행할 도구 명칭
            arguments (Dict[str, Any]): 도구 인자
            user_approval_fn (Callable): 사용자 승인 콜백 함수
            
        Returns:
            Tuple[bool, str]: (실행 가능 여부, 피드백 메시지)
        """
        if tool_name in self.high_risk_tools:
            print(f"\n[HITL 보초] 고위험 액션 감지: '{tool_name}' ({arguments})")
            # 사용자 승인 콜백 호출 (Interactive Modal / Prompt)
            is_approved = user_approval_fn(tool_name, arguments)
            
            if is_approved:
                return True, "사용자 승인 완료"
            else:
                return False, "사용자에 의해 액션 승인이 거부됨. 대안 경로를 탐색할 것"
                
        return True, "저위험 액션 자동 승인"


# HITL 하네스 검증 테스트
if __name__ == "__main__":
    harness = HumanInTheLoopHarness()

    # 가상 사용자 승인 콜백 (시뮬레이션)
    def mock_user_ui(tool: str, args: Dict[str, Any]) -> bool:
        # 사용자 시뮬레이션: 'git_push'는 승인, 'delete_database'는 거부
        return tool != "delete_database"

    # 1. 고위험 DB 삭제 테스트
    ok1, msg1 = harness.intercept_and_request_approval("delete_database", {"db_name": "prod_db"}, mock_user_ui)
    print("DB 삭제 결과:", ok1, "| 메시지:", msg1)

    # 2. 고위험 Git Push 테스트
    ok2, msg2 = harness.intercept_and_request_approval("git_push", {"branch": "main"}, mock_user_ui)
    print("Git Push 결과:", ok2, "| 메시지:", msg2)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 완전 자율 실행 (Full Auto) | Human-in-the-Loop (HITL) |
| :--- | :--- | :--- |
| **안전성** | 파괴적 명령 및 오판 시 치명적 위험 | 고위험 명령 100% 사전에 인간 차단 및 검증 |
| **생산성** | 최고 (인간 개입 없음) | 높음 (저위험 자동화 + 고위험 선택 승인) |
| **사용자 신뢰** | 에이전트 동작 불안감 존재 | 사용자가 명확한 통제권 보유로 신뢰도 향상 |

- **장점**:
  - 자율 에이전트를 프로덕션 환경에 안심하고 도입할 수 있는 유일한 안전장치
- **한계점**:
  - 사용자 피드백 대기로 인한 비동기 멈춤(Pause) 대기 발생

## 7. 활용 사례 및 응용
- **Antigravity IDE & Terminal Execution**:
  - 쉘 커맨드 실행 및 파일 변경 시 사용자 승인 팝업을 띄우는 하네스 구현
