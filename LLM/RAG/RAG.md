# RAG (Retrieval-Augmented Generation)

---
Reference:
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: RAG (Retrieval-Augmented Generation, 검색 증강 생성)
- **관련 분야/카테고리**: LLM / NLP / Knowledge Management
- **한 줄 요약**: 질의와 관련된 외부 지식 DB(Vector Store, Knowledge Graph)의 정보 데이터를 실시간으로 검색하여 LLM의 입력 컨텍스트로 제공함으로써 생성의 정확성을 높이는 지식 증강 기법

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **LLM 파라미터 지식의 한계점**:
    - **환각 (Hallucination)**: LLM이 학습하지 않았거나 불확실한 지식을 사실처럼 그럴듯하게 창작하는 현상
    - **학습 시점 절단 (Knowledge Cut-off)**: 사전 학습 데이터 수집 이후의 실시간 최신 정보 반영 불가
    - **사내/비공개 데이터 보안 접근 불가**: 기업 내부 전문 지식 문서에 대한 파라미터 학습 접근 제약
- **Fine-tuning의 단점**:
    - 외부 지식이 추가되거나 변경될 때마다 고비용의 재학습(Fine-tuning)을 수행해야 하며, 지식의 정확한 인출 컨트롤이 어려움

## 3. 핵심 원리 및 메커니즘 (How?)
- **1단계: 외부 문서 인덱싱 (Indexing)**:
    - 원본 문서(PDF, Markdown 등)를 적절한 크기의 텍스트 텍스트 분할(Chunking) 조각으로 나눔
    - 임베딩 모델(Embedding Model)을 사용하여 텍스트 텍스트 분할을 고차원 세맨틱 벡터 공간에 투영하여 Vector DB에 저장함
- **2단계: 세맨틱 정보 검색 (Retrieval)**:
    - 사용자 질의(Query)가 입력되면 임베딩 모델을 통해 벡터 변환을 수행함
    - Vector DB에서 Cosine Similarity 또는 Euclidean Distance 기준 Top-K 관련 텍스트 분할을 검색함
- **3단계: 컨텍스트 증강 생성 (Generation)**:
    - 검색된 Top-K 텍스트 텍스트 분할을 사용자 질의와 함께 프롬프트 컨텍스트에 삽입함
    - LLM은 증강된 프롬프트를 바탕으로 근거에 기반한 출력을 생성함

## 4. 핵심 세부 개념 및 부가 설명
- **Advanced RAG (고급 RAG)**:
    - **Pre-Retrieval (Query Rewriting / Expansion)**: 사용자 질문을 검색에 최적화되도록 재작성
    - **Post-Retrieval (Reranking)**: Cross-Encoder 등을 활용하여 검색된 텍스트 텍스트 분할의 관련도 순위를 재정렬하여 노이즈 제거
- **GraphRAG (지식 그래프 결합 RAG)**:
    - 기존 벡터 검색은 문서 간 종합적 맥락 분석 및 전체 텍스트 요약(Global Querying) 능력이 부족함
    - GraphRAG는 LLM을 이용해 문단 내 Entity-Relation 네트워크 지식 그래프(Knowledge Graph)를 생성하고 Community Detection 및 요약본을 만들어 전체적인 글로벌 맥락 질의에 완벽 부합하도록 확장함

## 5. 코드 구현 예시 (PyTorch / Python)
```python
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

class SimpleRAGRetriever:
    """
    Cosine Similarity 기반의 간단한 RAG Retriever & Generator 파이프라인 예시
    """
    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim: int = embedding_dim
        self.documents: List[str] = []
        self.doc_embeddings: Optional[torch.Tensor] = None

    def add_documents(self, docs: List[str]) -> None:
        """
        문서 보관소 등록 및 가상 임베딩 생성 (실제 사용 시 HuggingFace/OpenAI 임베딩 연동)
        
        Args:
            docs (List[str]): 등록할 문서 리스트
        """
        self.documents.extend(docs)
        # 임의의 결정론적 임베딩 생성 (예시용)
        embeddings = []
        for doc in docs:
            torch.manual_seed(len(doc))
            vec = torch.randn(self.embedding_dim)
            embeddings.append(F.normalize(vec, p=2, dim=0))
        
        new_embeds = torch.stack(embeddings)
        if self.doc_embeddings is None:
            self.doc_embeddings = new_embeds
        else:
            self.doc_embeddings = torch.cat([self.doc_embeddings, new_embeds], dim=0)

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        질의에 대한 유사 문서 Top-K 검색
        
        Args:
            query (str): 사용자 검색 질의
            top_k (int): 반환할 상위 문서 개수
            
        Returns:
            List[Tuple[str, float]]: (문서 내용, 유사도 점수) 리스트
        """
        if self.doc_embeddings is None:
            return []

        torch.manual_seed(len(query))
        query_vec = F.normalize(torch.randn(self.embedding_dim), p=2, dim=0)

        # 코사인 유사도 연산: (Query, Doc Embeddings)
        scores = torch.matmul(self.doc_embeddings, query_vec)  # (Num_docs,)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(self.documents)))

        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            results.append((self.documents[idx], score))
        return results

    def generate_prompt(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        """
        검색된 컨텍스트 문서를 결합한 RAG 증강 프롬프트 빌드
        
        Args:
            query (str): 사용자 질문
            retrieved_docs (List[Tuple[str, float]]): 검색된 문서 정보
            
        Returns:
            str: 증강된 프롬프트 문자열
        """
        context_str = "\n".join([f"- {doc}" for doc, _ in retrieved_docs])
        prompt = f"""다음 제공된 참고 문맥만을 바탕으로 질문에 정확하게 답변하세요.

[참고 문맥]
{context_str}

[질문]
{query}

[답변]"""
        return prompt


if __name__ == "__main__":
    # RAG 파이프라인 데모
    rag = SimpleRAGRetriever(embedding_dim=32)
    sample_docs = [
        "Mamba는 Selective State Space Model을 활용하여 O(N) 선형복잡도를 제공한다.",
        "RepVGG는 학습 시 Multi-branch 구조를 사용하고 추론 시 3x3 Conv 하나로 가중치를 융합한다.",
        "PyTorch 2.0 torch.compile은 TorchDynamo와 Inductor를 사용해 모델을 컴파일 가속한다."
    ]
    rag.add_documents(sample_docs)

    user_query = "RepVGG의 가중치 융합 방식은 무엇인가요?"
    retrieved = rag.retrieve(user_query, top_k=1)
    augmented_prompt = rag.generate_prompt(user_query, retrieved)

    print("=== 증강된 프롬프트 ===")
    print(augmented_prompt)
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | RAG (Retrieval-Augmented) | Fine-tuning (재학습) | Prompt Engineering |
| :--- | :--- | :--- | :--- |
| **외부 지식 업데이트** | 즉시 가능 (Vector DB만 업데이트) | 고비용 재학습 필요 | 불가능 (컨텍스트 제한) |
| **환각 방지 (Hallucination)** | 매우 우수 (근거 출처 명시) | 보통 (잘못된 학습 시 지속) | 낮음 |
| **도메인 특화 서식 습득** | 보통 | 매우 우수 | 보통 |
| **비용 및 리소스** | 저비용 (검색 인프라 구축) | 고비용 (GPU 계산) | 최소 비용 |

- **장점**:
    - 근거 데이터 출처(Citation)를 투명하게 제시하여 모델 답변의 신뢰도 극대화
    - 실시간 최신 정보 및 권한별 접근 제어가 수월함
- **한계점**:
    - 검색 품질이 나쁘면 잘못된 문맥이 주입되어 답변 품질이 떨어질 수 있음 (Garbage In, Garbage Out)

## 7. 활용 사례 및 응용
- **사내 Knowledge Management Q&A**: 사내 규정, 엔지니어링 Wiki 기반 답변 생성봇
- **GraphRAG**: 의료/법률 등 거대한 연결 지식 데이터셋의 요약 및 전역 질의응답
- **Multimodal RAG**: 이미지, 도면 데이터 기반 멀티모달 프롬프트 조율
