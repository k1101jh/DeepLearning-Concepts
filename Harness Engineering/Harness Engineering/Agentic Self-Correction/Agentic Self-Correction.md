# Agentic Self-Correction (자가 수정 루프 하네스)

---
Reference:
- [Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback", NeurIPS 2023](https://arxiv.org/abs/2303.17651)
- [Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023](https://arxiv.org/abs/2303.11366)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Agentic%20Self-Correction/Agentic%20Self-Correction.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Agentic Self-Correction (자가 수정 루프 하네스 / CRITIC-Refiner)
- **관련 분야/카테고리**: Harness Engineering / Self-Refinement / Automated Testing / Feedback Loops
- **한 줄 요약**: 에이전트가 코드를 작성한 후 샌드박스 내부의 단위 테스트(Unit Test) 구동 결과나 인터셉터 피드백을 기반으로 스스로 오류 원인을 분석하여 완성본을 도출할 때까지 반복 자가 수정(Self-Refine)을 수행하는 하네스 제어 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단일 생성 코드의 첫 시도 오류율 (First-shot Failure)**:
  - LLM이 한 번에 생성한 코드는 구문 오류(Syntax Error), 엣지 케이스 처리 누락 등 첫 시도 실패율이 높음
- **외부 환경 피드백의 부재**:
  - 에러가 발생했음에도 이를 에이전트 생성 프롬프트로 다시 주입하는 하네스 제어 루프가 없으면 사용자에게 에러 상태로 전달됨
- **Agentic Self-Correction 도입을 통한 해결**:
  - 파이썬 `pytest`나 컴파일러 실행 에러 메시지를 하네스가 인터셉트하여 LLM에게 에러 가이드 프롬프트로 재전송함으로써 90% 이상의 자가 수정 성취

## 3. 핵심 원리 및 메커니즘 (How?)
- **Self-Correction 4단계 반성 및 수정 루프**:
  1. **Initial Code Generation**: 초기 생성 코드 작성
  2. **Sandbox Automated Execution**: 샌드박스 환경에서 단위 테스트(pytest 등) 실행
  3. **Feedback Extraction & CRITIC**: 실패한 Traceback 에러 메시지 포착 및 원인 분석 프롬프트 구성
  4. **Iterative Refinement**: 최대 $N$회 재시도 루프(Iterative Refine)를 돌며 코드 수정

```text
[Initial Generated Code]
           │
           ▼
┌───────────────────────────┐
│ Automated Unit Testing    │ ─── (Execute pytest in Sandbox)
└───────────────────────────┘
     │ (FAIL: Capture Error Traceback)
     ▼
┌───────────────────────────┐
│ CRITIC / Refine Prompt    │ ─── (Self-Feedback Loop)
└───────────────────────────┘
     │ (Iterative Revision)
     ▼
[Success: Return Perfect Code]
```

## 4. 핵심 세부 개념 및 부가 설명
- **Verbal Reinforcement Learning (언어적 강화학습)**:
  - 수치적 보상 대신 에러 메시지 텍스트(Traceback)를 언어적 피드백으로 활용하여 차기 시도에서 올바른 생성을 유도
- **Max Retry Bounding (최대 재시도 상한 통제)**:
  - 무한 루프에 빠지는 것을 막기 위해 하네스가 최대 재시도 횟수(예: Max 3회)를 엄격히 제한

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import sys
from typing import Dict, Any, Tuple


class AgenticSelfCorrectionHarness:
    """단위 테스트 실행 피드백을 기반으로 자가 수정 루프를 제어하는 하네스 파이썬 클래스"""

    def __init__(self, max_refine_attempts: int = 3) -> None:
        """하네스 초기화
        
        Args:
            max_refine_attempts (int): 최대 자가 수정 재시도 횟수
        """
        self.max_refine_attempts: int = max_refine_attempts

    def execute_and_verify(self, code_snippet: str, test_code: str) -> Tuple[bool, str]:
        """[샌드박스 실행] 코드와 단위 테스트를 함께 실행하여 검증하는 시뮬레이션 메서드
        
        Args:
            code_snippet (str): 에이전트 생성 파이썬 코드
            test_code (str): 검증용 단위 테스트 구문
            
        Returns:
            Tuple[bool, str]: (성공 여부, 에러 트레이스백 피드백)
        """
        try:
            local_vars: Dict[str, Any] = {}
            exec(code_snippet + "\n" + test_code, {}, local_vars)
            return True, "모든 단위 테스트 통과"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}"

    def self_correction_loop(self, initial_code: str, test_code: str) -> Dict[str, Any]:
        """[자가 수정 루프] 실패 시 트레이스백을 피드백으로 하여 코드를 재수정하는 루프 메서드
        
        Args:
            initial_code (str): 초기 생성 코드
            test_code (str): 검증 테스트 구문
            
        Returns:
            Dict[str, Any]: 자가 수정 완료 결과
        """
        current_code = initial_code
        
        for attempt in range(1, self.max_refine_attempts + 1):
            passed, feedback = self.execute_and_verify(current_code, test_code)
            print(f"[Self-Correction Attempt {attempt}] 통과 여부: {passed} | 피드백: {feedback}")
            
            if passed:
                return {"success": True, "attempts": attempt, "final_code": current_code}

            # 자가 수정 시뮬레이션 (피드백을 반영하여 오타 수정)
            current_code = current_code.replace("retun", "return")

        return {"success": False, "attempts": self.max_refine_attempts, "error": "최대 자가 수정 횟수 초과"}


# Self-Correction 하네스 테스트
if __name__ == "__main__":
    harness = AgenticSelfCorrectionHarness(max_refine_attempts=3)
    
    # 의도적 오타가 포함된 초기 코드
    buggy_code = "def add(a, b):\n    retun a + b"
    unit_test = "assert add(10, 20) == 30"
    
    result = harness.self_correction_loop(buggy_code, unit_test)
    print("\nSelf-Correction 루프 최종 결과:", result)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 단일 시도 생성 (No Loop) | Agentic Self-Correction |
| :--- | :--- | :--- |
| **코드 완성도** | 1차 시도 버그 포함 시 바로 에러 종료 | 테스트 통과 시까지 자가 수정을 거쳐 완성 |
| **품질 검증** | 수동 사용자 디버깅 필요 | 단위 테스트 기반 자동 자율 검증 및 수정 |
| **성공률** | 보통 | 비약적으로 높음 |

- **장점**:
  - 실행 가능한 완전한 형태의 고품질 코드 자동 도출
- **한계점**:
  - 자가 수정 재시도 루프 수행으로 인한 추론 토큰 소모 증가

## 7. 활용 사례 및 응용
- **SWE-bench Coding Agent**:
  - 소프트웨어 에이전트가 코드를 수정한 후 Pytest 실행 결과를 바탕으로 버그를 자동 자가 수정하는 파이프라인
