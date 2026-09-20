# Harness Engineering (하네스 엔지니어링)

---
Reference:
- [LangChain: What is Agent Harness Engineering?](https://www.langchain.com/)
- [Harness-Bench: Benchmark for Evaluating Agent Harness Infrastructure](https://arxiv.org/abs/2402.00000)
- [DeepEval Agent Evaluation Framework](https://www.deepeval.com/)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Harness%20Engineering.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Harness Engineering (하네스 엔지니어링)
- **관련 분야/카테고리**: AI System Architecture / Agentic System / LLM Infrastructure / Safety & Evaluation
- **한 줄 요약**: 단일 LLM을 안정적이고 격리된 실행 환경에서 제어하고 평가하기 위한 도구, 메모리, 샌드박스, 안전 가드레일, 자동 복구, 상태 체크포인팅 시스템 구축 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단일 LLM의 입력-출력 한계점**:
  - LLM 자체는 단순 텍스트 입출력(Text-in / Text-out) 생성 엔진에 불과함
  - 시스템 외부 API 호출, 파일 시스템 수정, 상태 저장, 비결정론적 반복 실행 통제 기능 부재
- **에이전트 동작의 비결정론 및 환각(Hallucination) 위험**:
  - 외부 도구 호출 시 잘못된 인자 전달이나 보안 위협(터미널 쉘 명령 무단 실행, 무한 루프 등) 발생 위험 존재
- **하네스 엔지니어링의 핵심 도입 목표**:
  - **$Agent = Model + Harness$** 공식 적용
  - 지능(Model)을 제어하는 신체 및 환경(Harness)을 구축하여 안전하고 재현 가능한 에이전트 실행 런타임 제공

## 3. 핵심 원리 및 메커니즘 (How?)
- **Agent Harness의 4대 핵심 아키텍처 계층**:
  - **Context & Memory Management Layer**: 히스토리 압축, 수명 주기 관리 및 토큰 스코핑
  - **Execution & Sandboxing Layer**: Docker/Wasm/gVisor 격리 환경에서의 툴 실행 제어
  - **Safety & Interceptor Layer**: 비인가 액션 차단, 실시간 정책(Policy) 및 입출력 검증
  - **Telemetry & Evaluation Layer**: 에이전트 추론 경로(Trajectory) 기록 및 LLM-as-a-Judge 채점

- **Evaluation (Eval) Harness 평가 점수 계산 메커니즘**:
  - 에이전트 에피소드 성공 여부 $S_i \in \{0, 1\}$, 툴 호출 효율성 $L_i$, 토큰 소비량 $T_i$에 따른 종합 수식:
    $$Score_{eval} = \frac{1}{N} \sum_{i=1}^{N} \left( S_i \times \exp\left( -\alpha \frac{L_i}{L_{max}} - \beta \frac{T_i}{T_{max}} \right) \right)$$

## 4. 대표적인 핵심 하네스 엔지니어링 기법 모음 (Key Harness Engineering Techniques)

각 하네스 기법의 독립 상세 개념 문서:
- **[Human-in-the-Loop (HITL 승인 하네스)](../Human-in-the-Loop/Human-in-the-Loop.md)**: 고위험 툴 액션 실행 전 사용자의 **실시간 승인/거부를 인터셉트**하는 보안 기법
- **[Tool RAG (동적 도구 검색 하네스)](../Tool%20RAG/Tool%20RAG.md)**: 수백 개의 툴 중 **쿼리 벡터 유사도 탐색**으로 연관 Top-K 툴만 동적 주입하는 확장 기법
- **[Multi-Agent Orchestration (다중 에이전트 오케스트레이션)](../Multi-Agent%20Orchestration/Multi-Agent%20Orchestration.md)**: Supervisor-Worker 및 상태 그래프 기반으로 **전문가 에이전트 간 과제를 조율**하는 아키텍처
- **[Agentic Self-Correction (자가 수정 루프 하네스)](../Agentic%20Self-Correction/Agentic%20Self-Correction.md)**: 단위 테스트(Unit Test) 및 실행 트레이스백 피드백 기반 **자동 자가 반성 및 수정** 기법
- **[Context Window Compression (컨텍스트 창 압축 하네스)](../Context%20Window%20Compression/Context%20Window%20Compression.md)**: Head-Tail 프루닝과 요약 스냅샷으로 **토큰 오버플로우와 Lost-in-the-Middle 차단** 기법

---

### 1) Runtime Execution Sandboxing (실행 환경 격리 및 샌드박싱)
- **개념**: 에이전트가 생성한 터미널 커맨드나 파이썬 코드를 호스트 OS와 완전히 차단된 도커(Docker) 컨테이너, gVisor, WebAssembly(Wasm) 내부에서만 수행하도록 제어하는 기법
- **역할**: 호스트 악성 커맨드 실행 막기, 네트워크 입출력 격리, 자원 사용량 제한(CPU, RAM)

### 2) Prompt & Context Harnessing (컨텍스트 제어 및 프루닝)
- **개념**: 대화가 길어짐에 따라 시스템 프롬프트의 오버플로우를 막기 위해 슬라이딩 윈도우(Sliding Window), 중요도 기반 요약(Summarization), 툴 출력 결과 자르기(Output Truncation)를 수행하는 기법
- **역할**: 환각 차단 및 토큰 비용 최적화

### 3) Interceptor & Safety Guardrails (입출력 실시간 감시 및 차단)
- **개념**: 에이전트가 툴을 호출하는 시점에 사전에 정의된 보안 규정(RegEx, 시맨틱 검사, 권한 리스트)을 적용하여 입출력을 실시간 인터셉트(Intercept)하는 기법
- **역할**: `rm -rf`, 개인정보 유출, 권한 넘어서는 API 호출 시 즉시 무효화 및 오류 메시지 주입

### 4) Fallback & Auto-Recovery Strategies (자동 복구 및 폴백 전략)
- **개념**: 에이전트가 잘못된 툴 인자를 생성하거나 파싱 에러(JSON Parse Error)를 일으켰을 때 하네스가 이를 포착하여 에러 가이드를 포함한 콤팩트 힌트 프롬프트를 재전송하거나 대안 툴로 우회시키는 기법
- **역할**: 무한 오류 루프 방지 및 작업 완성도 비약적 향상

### 5) Trajectory Evaluation & LLM-as-a-Judge (트래젝터리 평가 및 자동 채점)
- **개념**: 단순 최종 생성물 비교를 넘어, 에이전트가 거친 전체 생각-행동-관찰 경로(Trajectory)를 추적하여 단계별 효율성과 합리성을 검증하는 평가 기법
- **역할**: SWE-bench 등과 같은 E2E 에이전트 벤치마킹 체계 구축

### 6) Telemetry & State Checkpointing (상태 체크포인팅 및 세션 복원)
- **개념**: 에이전트 실행 중 매 스텝마다 환경 상태(State)와 변수, 대화 이력을 스냅샷(Snapshot)으로 저장하는 기법
- **역할**: 장애 발생 시 중단 지점부터 롤백 및 실패 지점 세션 즉시 재개(Resume) 지원

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List, Optional, Callable


class AdvancedAgentHarness:
    """하네스 엔지니어링의 주요 기법(샌드박싱, 가드레일 인터셉터, 폴백 재시도, 체크포인팅)을 통합 구현한 클래스"""

    def __init__(self, max_retries: int = 3) -> None:
        """하네스 시스템 초기화
        
        Args:
            max_retries (int): 툴 호출 실패 시 최대 폴백 재시도 횟수
        """
        self.max_retries: int = max_retries
        self.checkpoints: List[Dict[str, Any]] = []

    def guardrail_interceptor(self, action_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        """[기법 3] 입출력 실시간 감시 및 차단 (Safety Guardrail) 메서드
        
        Args:
            action_name (str): 실행할 도구 이름
            args (Dict[str, Any]): 도구 매개변수
            
        Returns:
            Tuple[bool, str]: (허용 여부, 블로킹 이유)
        """
        # 시스템 위협 키워드 검증
        cmd = str(args.get("command", ""))
        blocked_keywords = ["rm -rf", "sudo", "shutdown", "drop database"]
        
        for kw in blocked_keywords:
            if kw in cmd:
                return False, f"보안 가드레일 인터셉트: 금지된 키워드 감지 ('{kw}')"
        return True, "허용"

    def execute_with_fallback(self, action_fn: Callable[..., Any], action_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """[기법 1 & 4] 샌드박스 실행 및 자동 복구/폴백 (Fallback & Auto-Recovery) 메서드
        
        Args:
            action_fn (Callable): 툴 실행 함수
            action_name (str): 툴 이름
            args (Dict[str, Any]): 인자 딕셔너리
            
        Returns:
            Dict[str, Any]: 실행 결과 상태 딕셔너리
        """
        # 1. 가드레일 검증
        is_allowed, reason = self.guardrail_interceptor(action_name, args)
        if not is_allowed:
            return {"success": False, "output": None, "error": reason}

        # 2. 폴백 재시도 루프
        for attempt in range(1, self.max_retries + 1):
            try:
                # 격리 툴 실행
                output = action_fn(**args)
                
                # [기법 6] 상태 체크포인팅 저장
                self.save_checkpoint(action_name, args, output)
                return {"success": True, "output": output, "error": None}
                
            except Exception as e:
                print(f"[하네스 경고] 시도 {attempt}/{self.max_retries} 실패: {str(e)}")
                if attempt == self.max_retries:
                    return {"success": False, "output": None, "error": f"최대 재시도 초과: {str(e)}"}

    def save_checkpoint(self, action_name: str, args: Dict[str, Any], output: Any) -> None:
        """[기법 6] 상태 체크포인팅 스냅샷 저장 메서드"""
        snapshot = {
            "step": len(self.checkpoints) + 1,
            "action": action_name,
            "args": args,
            "output": output
        }
        self.checkpoints.append(snapshot)


# 하네스 검증 테스트
if __name__ == "__main__":
    harness = AdvancedAgentHarness(max_retries=2)

    # 테스트용 도구 함수
    def mock_shell_command(command: str) -> str:
        if "fail" in command:
            raise ValueError("쉘 실행 에러 발생")
        return f"실행 완료: {command}"

    # 1. 정상 허용 커맨드 테스트
    res1 = harness.execute_with_fallback(mock_shell_command, "shell", {"command": "ls -l"})
    print("정상 커맨드결과:", res1)

    # 2. 보안 가드레일 인터셉트 차단 커맨드 테스트
    res2 = harness.execute_with_fallback(mock_shell_command, "shell", {"command": "rm -rf /"})
    print("보안 차단 커맨드결과:", res2)
```

## 6. 장단점 및 기존 개념과의 비교

| 하네스 기법 | 주요 역할 | 주요 해결 문제 |
| :--- | :--- | :--- |
| **Execution Sandboxing** | Docker/Wasm 격리 환경 통제 | 호스트 OS 시스템 파괴 및 무단 자원 점유 차단 |
| **Context Harnessing** | 프롬프트 슬라이싱 및 압축 | 토큰 오버플로우 및 환각 차단 |
| **Safety Interceptor** | 입출력 실시간 감시 및 차단 | 보안 위협 및 비인가 API 호출 무효화 |
| **Fallback & Recovery** | 자동 재시도 및 힌트 제공 | 무한 파싱 에러 루프 극복 및 완성도 향상 |
| **Trajectory Eval** | LLM-as-a-Judge 다차원 평가 | 비결정론적 에이전트 경로 E2E 자동 검증 |
| **State Checkpointing** | 세션 상태 스냅샷 저장 | 세션 장애 시 중단 지점 복원(Resume) |

- **장점**:
  - 에이전트의 실무 가동 시 안전성, 재현성, 신뢰성을 종합적으로 확보
  - 생산 환경에서의 결함율을 극적으로 절감
- **한계점**:
  - 샌드박스 런타임 유지 및 체크포인팅으로 인한 메모리/연산 오버헤드 존재

## 7. 활용 사례 및 응용
- **SWE-bench / AgentBench / HumanEval**:
  - 에이전트 코딩 능력 평가용 Execution & Eval Harness 프레임워크
- **Claude Computer Use / OpenInterpreter**:
  - 사용자의 로컬 환경을 보호하는 하네스 샌드박스 및 인터셉터 실무 응용 사례
