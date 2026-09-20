# Context Window Compression (컨텍스트 창 압축 하네스)

---
Reference:
- [LangChain: Conversation Memory & Summary Buffer Management](https://www.langchain.com/)
- [Head-Tail Context Pruning Patterns for LLM Agents](https://arxiv.org/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Context%20Window%20Compression/Context%20Window%20Compression.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Context Window Compression (컨텍스트 창 압축 및 프루닝 하네스)
- **관련 분야/카테고리**: Harness Engineering / Memory Management / Context Optimization / Token Pruning
- **한 줄 요약**: 장기 에이전트 대화 세션에서 필수적인 시스템 프롬프트(Head)와 최근 대화(Tail)는 보존하고, 중반부 대화 이력을 선택적으로 자르거나 요약 스냅샷으로 축약하여 토큰 오버플로우를 막는 하네스 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **컨텍스트 윈도우 한도 초과 (Context Window Overflow)**:
  - 에이전트가 10~20턴 이상 복잡 코딩 및 디버깅을 수행하면 대화 이력이 컨텍스트 한계(예: 128k/200k 토큰)를 초과하여 실행 중단 에러 발생
- **Lost-in-the-Middle 현상**:
  - 프롬프트 중반부에 지나치게 길어진 히스토리가 쌓이면 LLM이 핵심 지침을 잊어버리고 환각 동작을 수행하는 문제 발생
- **Context Window Compression 도입을 통한 해결**:
  - 하네스가 슬라이딩 윈도우(Sliding Window), Head-Tail 프루닝, 요약 체크포인팅을 통해 맥락 보존과 토큰 압축을 동시에 달성

## 3. 핵심 원리 및 메커니즘 (How?)
- **Head-Tail Pruning & Summary Checkpoint 메커니즘**:
  - **Head Section (보존)**: 시스템 프롬프트, 툴 스키마, 주요 미션 정의
  - **Middle Section (압축/삭제)**: 오래된 중반부 대화 이력을 단일 요약 텍스트 $Summary_{ckpt}$로 압축
  - **Tail Section (보존)**: 최근 $M$개의 직전 대화턴 및 관찰(Observation) 결과

```text
[Original Full History]
┌──────────────────┐ ┌──────────────────────────────────────────┐ ┌──────────────────┐
│ Head: Sys Prompt │ │ Middle: Old Turns (10,000+ tokens)      │ │ Tail: Last 3 Turns│
└──────────────────┘ └──────────────────────────────────────────┘ └──────────────────┘
         │                                │                                │
         ▼                                ▼ (Compress to Summary)          ▼
┌──────────────────┐ ┌──────────────────────────────────────────┐ ┌──────────────────┐
│ Head: Sys Prompt │ │ Summary Checkpoint (Compressed 500 tok)  │ │ Tail: Last 3 Turns│
└──────────────────┘ └──────────────────────────────────────────┘ └──────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Observation Truncation (관찰 출력 절단)**:
  - 툴 실행 결과로 수만 줄의 터미널 로그가 생성될 경우, 하네스가 상위 50줄과 하위 50줄만 잘라내고 중반부를 생략 처리하는 기법
- **Token Trigger Threshold (압축 발동 임계값)**:
  - 현재 전체 대화 토큰 수가 한도의 80%에 도달할 때 하네스가 백그라운드 압축 루프를 자동 트리거

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any, List


class ContextCompressionHarness:
    """Head-Tail 프루닝 및 중반부 요약 압축을 제어하는 하네스 파이썬 클래스"""

    def __init__(self, max_token_limit: int = 4000, tail_turns_to_keep: int = 2) -> None:
        """하네스 초기화
        
        Args:
            max_token_limit (int): 압축을 트리거할 토큰 한도
            tail_turns_to_keep (int): 보존할 최근 직전 대화 턴 수
        """
        self.max_token_limit: int = max_token_limit
        self.tail_turns_to_keep: int = tail_turns_to_keep

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """더미 토큰 수 계산 메서드"""
        total_text = "".join([m["content"] for m in messages])
        return len(total_text.split()) * 2

    def compress_context(self, system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """[Head-Tail Pruning] 컨텍스트 창이 한도를 초과할 때 중반부를 압축하는 메서드
        
        Args:
            system_prompt (str): 최전방 Head 시스템 프롬프트
            history (List[Dict[str, str]]): 전체 대화 이력 리스트
            
        Returns:
            List[Dict[str, str]]: 압축이 완료된 최적화 메시지 리스트
        """
        messages = [{"role": "system", "content": system_prompt}] + history
        current_tokens = self.estimate_tokens(messages)

        if current_tokens <= self.max_token_limit or len(history) <= self.tail_turns_to_keep:
            return messages

        print(f"[하네스 경고] 토큰 한도 초과 ({current_tokens}/{self.max_token_limit}) -> Head-Tail 압축 개시")
        
        # Head & Tail 분리
        head = [{"role": "system", "content": system_prompt}]
        tail = history[-self.tail_turns_to_keep:]
        middle = history[:-self.tail_turns_to_keep]

        # Middle 압축 요약
        middle_summary = f"[중반부 {len(middle)}개 대화턴 요약]: 에이전트가 코드를 작성하고 테스트를 진행함"
        compressed_middle = [{"role": "system", "content": middle_summary}]

        return head + compressed_middle + tail


# 컨텍스트 압축 하네스 테스트
if __name__ == "__main__":
    harness = ContextCompressionHarness(max_token_limit=100, tail_turns_to_keep=2)
    
    sys_prompt = "당신은 프론트엔드 개발 에이전트입니다."
    dummy_history = [
        {"role": "user", "content": "1턴: 로그인 페이지 작성해 줘"},
        {"role": "assistant", "content": "1턴 응답: 로그인 페이지 코드 작성 완료... " * 10},
        {"role": "user", "content": "2턴: 대시보드 페이지 작성해 줘"},
        {"role": "assistant", "content": "2턴 응답: 대시보드 페이지 코드 작성 완료... " * 10},
        {"role": "user", "content": "3턴: 최신 테스트 실행해 줘"},
        {"role": "assistant", "content": "3턴 응답: 테스트 실행 완료"}
    ]
    
    compressed = harness.compress_context(sys_prompt, dummy_history)
    print("압축 완료된 메시지 개수:", len(compressed))
    print("압축 메시지 요약 내역:\n", compressed[1])
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 무통제 대화 이력 유지 | Context Window Compression |
| :--- | :--- | :--- |
| **토큰 한도 관리** | 장기 세션 시 100% 토큰 초과 에러 발생 | 슬라이딩 및 요약으로 한도 초과 완전 방지 |
| **Lost-in-the-Middle** | 중반부 문맥 비대화로 에이전트 환각 야기 | 중요 지침(Head) 및 최신 맥락(Tail) 보존 |
| **세션 수명** | 단기 세션으로 제한 | 무한에 가까운 Multi-turn 장기 세션 가능 |

- **장점**:
  - 무한에 가까운 에이전트 multi-turn 세션 지속성 및 안정성 달성
- **한계점**:
  - 중반부 상세 이력 삭제에 따른 일부분맥 세부 정보 손실 가능성 존재

## 7. 활용 사례 및 응용
- **Long-running Autonomous Coding Agent**:
  - 수시간 동안 지속되는 버그 수정 및 코드 리팩토링 에이전트의 메모리 제어 하네스
