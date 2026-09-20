# Prompt Caching (프롬프트 캐싱 및 접두사 공유)

---
Reference:
- [Anthropic Claude Prompt Caching Technical Guide](https://docs.anthropic.com/claude/docs/prompt-caching)
- [OpenAI Prompt Caching & Performance Benchmarks](https://platform.openai.com/docs/guides/prompt-caching)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Utilization%20Methods/Prompt%20Caching/Prompt%20Caching.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Prompt Caching (프롬프트 캐싱 및 접두사 공유 기술)
- **관련 분야/카테고리**: LLM Utilization Methods / Inference Optimization / Cost Reduction / Latency Optimization
- **한 줄 요약**: 반복 전송되는 대용량 정적 접두사(System Prompt, RAG Context, Tool Schema)의 Key-Value 어텐션 텐서를 인퍼런스 엔진에 캐싱하여 입력 토큰 비용을 50~90% 절감하고 응답 대기 시간(TTFT)을 85% 단축시키는 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **에이전트 multi-turn 대화의 과도한 토큰 중복 지불**:
  - LLM 에이전트는 대화가 1회 진행될 때마다 기존의 거대한 시스템 프롬프트, 도구 스키마(Tools), RAG 문서 덩어리를 매턴 반복 전송하여 막대한 비용 발생
- **첫 토큰 생성 레이턴시(Time-To-First-Token / TTFT) 증가**:
  - 입력 토큰 수가 수만 개로 커지면 Prefill 연산 단계 시간이 길어져 사용자의 반응 속도 체감 저하
- **Prompt Caching 도입을 통한 해결**:
  - 동일한 프롬프트 접두사(Prefix)의 어텐션 KV 캐시 상태를 서버 메모리에 재사용함으로써 반복 연산 완전 생략

## 3. 핵심 원리 및 메커니즘 (How?)
- **Prefix Caching 원리와 KV Tensor 저장**:
  - Transformer Decoder의 self-attention 연산 시 $K(Key)$, $V(Value)$ 텐서를 사전 계산하여 캐시 저장
  - 쿼리 요청의 정적 접두사 $P_{static}$와 동적 질문 $Q_{user}$ 분리:
    $$Prompt = [ P_{static} \ (\text{Cached KV}) \, \Vert \, Q_{user} \ (\text{New Prefill}) ]$$

- **절감률 및 성능 이점 수식**:
  - 캐시적용 입력 토큰 읽기 비용 $C_{read}$는 기본 입력 비용 $C_{input}$의 10% 수준:
    $$Cost_{cached} = \left( T_{cached} \times 0.1 \times C_{input} \right) + \left( T_{new} \times C_{input} \right)$$

```text
[Request 1] [System Prompt (Static)] + [User Turn 1 (Dynamic)] ───> [Write KV Cache]
                                                                          │
[Request 2] [System Prompt (Cached 90% Off)] + [User Turn 2] ─────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Static Prefix Structure (정적 접두사 최전방 배치)**:
  - 캐시 적중률(Cache Hit Rate)을 높이기 위해 변하지 않는 시스템 지침, 페르소나, 툴 스키마를 프롬프트 최상단에 배치하고, 가변적인 사용자 질문을 맨 뒤에 배치
- **Anthropic vs OpenAI 방식 차이**:
  - Anthropic: `cache_control: {"type": "ephemeral"}` 중단점을 개발자가 수동 명시하여 90% 파격 할인 제공
  - OpenAI: 1,024 토큰 이상의 정적 접두사에 대해 시스템이 자동(Automatic) 캐싱하여 50% 할인 제공

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class PromptCachingHarness:
    """프롬프트 구조를 정적 접두사와 동적 입력으로 분리하여 캐싱 효율을 최적화하는 파이썬 클래스"""

    def __init__(self, system_instruction: str, tools_schema: List[Dict[str, Any]]) -> None:
        """하네스 초기화
        
        Args:
            system_instruction (str): 고정 시스템 프롬프트
            tools_schema (List[Dict[str, Any]]): 고정 도구 스키마 리스트
        """
        self.static_prefix: str = system_instruction + "\n" + json.dumps(tools_schema, ensure_ascii=False)
        self.cached_tokens_count: int = len(self.static_prefix.split()) * 2  # 더미 토큰 계산

    def build_cached_payload(self, user_query: str) -> Dict[str, Any]:
        """Anthropic/OpenAI 캐시 제어 헤더가 포함된 API 전송 payload를 구성하는 메서드
        
        Args:
            user_query (str): 동적 사용자 입력
            
        Returns:
            Dict[str, Any]: 프롬프트 캐싱 규격 API 페이로드
        """
        # 정적 접두사 섹션에 cache_control 중단점 추가
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.static_prefix,
                        "cache_control": {"type": "ephemeral"}  # Anthropic Prompt Caching 헤더
                    }
                ]
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        return {"messages": messages, "estimated_cached_tokens": self.cached_tokens_count}


# Prompt Caching 하네스 테스트
if __name__ == "__main__":
    harness = PromptCachingHarness(
        system_instruction="당신은 하네스 엔지니어링 전문 AI 에이전트입니다.",
        tools_schema=[{"name": "execute_code", "description": "파이썬 코드를 실행함"}]
    )
    
    payload = harness.build_cached_payload("프롬프트 캐싱의 이점을 설명해 줘")
    print("생성된 Prompt Caching Payload:\n", json.dumps(payload, indent=2, ensure_ascii=False))
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 일반 LLM 요청 (No Caching) | Prompt Caching 적용 |
| :--- | :--- | :--- |
| **입력 토큰 비용** | 매 턴 전체 프롬프트 비용 100% 지불 | 캐시 영역 토큰 비용 **50~90% 절감** |
| **TTFT (첫 토큰 지연)** | 수만 토큰 Prefill 연산으로 지연시간 긺 | KV 캐시 재사용으로 TTFT **최대 85% 단축** |
| **적합한 과제** | 단발성 짧은 질의응답 | Agentic multi-turn 대화, 장문 RAG, 수백 개 툴 바인딩 |

- **장점**:
  - 에이전트 시스템의 가동 비용과 응답 지연을 획기적으로 낮추는 필수 인프라 기법
- **한계점**:
  - 프롬프트 최상단 정적 접두사 구조를 엄격히 유지해야 캐시 적중(Hit) 성공

## 7. 활용 사례 및 응용
- **Claude Code & Enterprise RAG Agent**:
  - 대용량 코드베이스 문맥 및 툴 스키마를 최전방 캐싱하여 초고속 에이전틱 대화 구현
