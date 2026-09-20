# Tool RAG (동적 도구 검색 하네스)

---
Reference:
- [Dynamic Tool Retrieval & Tool RAG Frameworks](https://arxiv.org/)
- [Scalable Agentic Systems with Tool Search Architecture](https://www.langchain.com/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Tool%20RAG/Tool%20RAG.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Tool RAG (Dynamic Tool Retrieval / 동적 도구 벡터 검색 하네스)
- **관련 분야/카테고리**: Harness Engineering / Scalable Agent Architecture / Tool Selection / Context Optimization
- **한 줄 요약**: 에이전트가 보유한 수백 개의 도구(Tools) 중 사용자 쿼리와 연관성이 높은 도구만 임베딩 유사도로 탐색하여 컨텍스트에 동적 주입하는 대규모 툴 확장 하네스 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **모든 툴 스키마 상시 주입의 한계 (Context Bloat & Token Cost)**:
  - 에이전트에 바인딩된 툴 개수가 50~100개 이상으로 증가할 경우 시스템 프롬프트의 토큰 사용량이 급증하여 API 비용 및 지연시간 폭등
- **LLM의 툴 선택 혼란 (Tool Confusion)**:
  - 수십 개의 유사 툴 스키마가 무더기로 주입되면 LLM이 잘못된 툴을 선택하거나 환각 인자를 생성할 확률이 높아짐
- **Tool RAG 도입을 통한 해결**:
  - 도구의 설명(Description)을 벡터 DB에 인덱싱하고, 사용자 쿼리 $Q$가 입력될 때 Top-K 연관 툴만 실시간 검색하여 주입함으로써 해결

## 3. 핵심 원리 및 메커니즘 (How?)
- **Tool RAG 3단계 바인딩 파이프라인**:
  1. **Indexing Stage**: 등록된 모든 툴의 이름과 설명을 텍스트 임베딩 모델로 벡터 데이터베이스(Vector DB)에 인덱싱
  2. **Retrieval Stage**: 사용자 쿼리 $Q$ 입력 시 코사인 유사도 $Sim(Vec(Q), Vec(T_i))$ 계산 후 Top-K(예: Top-3) 툴 추출:
     $$T_{selected} = \operatorname*{Top-K}_{T_i \in T_{all}} \left( \frac{Vec(Q) \cdot Vec(T_i)}{\|Vec(Q)\| \|Vec(T_i)\|} \right)$$
  3. **Dynamic Prompt Injection**: 추출된 Top-K 툴 스키마만 에이전트 생성 컨텍스트 윈도우에 동적 결합

```text
[All Registered Tools (N=500)]
              │ (Vector Indexing)
              ▼
┌───────────────────────────┐
│ Tool Vector Database      │
└───────────────────────────┘
              ▲
              │ (Semantic Vector Search by Query Q)
┌───────────────────────────┐
│ User Input Query Q        │
└───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│ Active Top-K Tools (K=3)  │ ─── (Dynamic Prompt Injection -> LLM)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Hierarchical Tool Routing (계층적 툴 루팅)**:
  - 1차로 툴의 범주(예: DB 툴, GitHub 툴, 파일 툴)를 먼저 카테고리화하고 2차로 세부 툴을 검색하는 다단계 RAG 기법
- **Token Saving & Precision Synergy**:
  - 프롬프트 토큰 사용량을 최대 90% 이상 절감하며, LLM의 올바른 툴 선택 정확도(Tool Selection Accuracy) 증대

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import math
from typing import Dict, Any, List


class ToolRAGHarness:
    """사용자 쿼리와의 유사도를 탐색하여 연관 툴 스키마만 동적 주입하는 Tool RAG 하네스 클래스"""

    def __init__(self) -> None:
        """툴 레지스트리 및 더미 임베딩 DB 초기화"""
        self.tools_registry: Dict[str, Dict[str, Any]] = {
            "sql_query": {"description": "Postgres 데이터베이스에 SQL 쿼리를 실행하여 결과 레코드를 조회함", "keywords": ["sql", "db", "데이터", "조회"]},
            "github_pr_create": {"description": "GitHub 레포지토리에 커밋 코드를 바탕으로 새 Pull Request를 생성함", "keywords": ["github", "pr", "풀리퀘스트", "커밋"]},
            "file_write": {"description": "로컬 호스트 파일 시스템에 텍스트 또는 모듈 코드를 저장함", "keywords": ["파일", "저장", "write", "생성"]}
        }

    def compute_keyword_similarity(self, query: str, keywords: List[str]) -> float:
        """[더미 임베딩 유사도] 쿼리와 키워드 간 단순 매칭 점수를 계산하는 메서드
        
        Args:
            query (str): 사용자 요구사항 쿼리
            keywords (List[str]): 툴 키워드 목록
            
        Returns:
            float: 측정된 유사도 점수
        """
        score = 0.0
        for kw in keywords:
            if kw in query:
                score += 1.0
        return score

    def retrieve_top_k_tools(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """[Tool RAG Retrieval] 쿼리에 맞는 Top-K 연관 툴만 동적 추출하는 메서드
        
        Args:
            query (str): 사용자 쿼리
            k (int): 추출할 최대 툴 개수
            
        Returns:
            List[Dict[str, Any]]: 추출된 Top-K 툴 스키마 목록
        """
        scored_tools: List[Tuple[float, str, Dict[str, Any]]] = []
        
        for tool_name, info in self.tools_registry.items():
            sim = self.compute_keyword_similarity(query, info["keywords"])
            scored_tools.append((sim, tool_name, info))

        # 유사도 내림차순 정렬
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        
        selected_tools = [
            {"name": t[1], "description": t[2]["description"]}
            for t in scored_tools[:k] if t[0] > 0
        ]
        return selected_tools


# Tool RAG 하네스 테스트
if __name__ == "__main__":
    harness = ToolRAGHarness()
    query = "작성한 파이썬 코드를 로컬 파일에 저장해 줘"
    
    active_tools = harness.retrieve_top_k_tools(query, k=1)
    print("Tool RAG 동적 추출 결과:", active_tools)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | All Tools Static Injection | Tool RAG Dynamic Harness |
| :--- | :--- | :--- |
| **확장성 (Scalability)** | 툴 30개 이상 시 토큰 초과로 불가능 | 1,000개 이상의 툴 보유 가능 (O(K) 확장) |
| **프롬프트 토큰 비용** | 매우 큼 (상시 전체 주입) | 극적으로 작음 (Top-K 툴만 주입) |
| **툴 선택 정확도** | 툴이 많아 혼란(Tool Confusion) 발생 | 좁혀진 연관 툴 선택으로 정확도 비약 상승 |

- **장점**:
  - 수백~수천 개의 툴 생태계를 단일 에이전트에서 유연하게 운용 가능
- **한계점**:
  - 벡터 RAG 탐색 레이어 구축 및 임베딩 연산 레이턴시 1회 발생

## 7. 활용 사례 및 응용
- **Large-scale Enterprise Agent System**:
  - 사내 500개 마이크로서비스 API를 단일 에이전트의 툴 RAG 기반으로 동적 연결
