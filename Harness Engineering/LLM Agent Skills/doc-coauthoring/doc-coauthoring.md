# Doc Co-authoring Skill (문서 공동 작성 스킬)

---
Reference:
- [Anthropic Specification for Document Co-authoring Workflows](https://github.com/anthropics/skills)
- [Structured Collaborative Writing Patterns for LLMs](https://arxiv.org/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/doc-coauthoring/doc-coauthoring.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Doc Co-authoring Skill (문서 공동 작성 스킬)
- **관련 분야/카테고리**: LLM Agent Skill / Collaborative Writing / Technical Documentation
- **한 줄 요약**: 단번에 완벽한 장문 문서를 생성하려 하지 않고, 개요 구성 $\rightarrow$ 대화형 질문/맥락 수집 $\rightarrow$ 섹션별 반복 수정 4단계를 거쳐 사용자와 함께 고품질 기술 문서를 완성하는 협업 스킬

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단일 일괄 생성(One-shot Generation)의 한계점**:
  - LLM이 한 번에 수십 페이지의 기획서나 사양서를 생성하면 사용자 의도와 어긋난 부분이 다수 포함되어 전체 문서를 수정하기 극도로 어려워짐
- **맥락 및 세부 요구사항 누락**:
  - 초반 프롬프트에 담기지 않은 비즈니스 로직이나 맥락이 무시된 채 환각 내용으로 문서가 작성되는 문제 발생
- **Doc Co-authoring 스킬 도입을 통한 해결**:
  - 구조화된 인터뷰 및 단계적 작성(Iterative Co-authoring) 절차를 강제하여 사용자가 명확한 통제권을 유지한 채 명확한 지식을 전달하도록 유도

## 3. 핵심 원리 및 메커니즘 (How?)
- **Doc Co-authoring 4단계 라이프사이클**:
  1. **Phase 1: Scope & Outline Definition**: 문서 목적을 정의하고 목차(Outline) 및 섹션 구조를 먼저 동의받음
  2. **Phase 2: Targeted Interview**: 불분명한 요구사항에 대해 명확한 2~3개 질문을 던져 정보 수집
  3. **Phase 3: Sectional Draft**: 확정된 정보를 바탕으로 섹션 단위(Section-by-section) 초안 작성
  4. **Phase 4: Feedback & Verification**: 사용자 피드백을 반영하여 각 섹션을 정제 및 교정

```text
[Initial User Request]
          │
          ▼
┌───────────────────────────┐
│ Phase 1: Define Outline   │
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Phase 2: Q&A Interview    │ ─── (Collect Context)
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Phase 3: Sectional Draft  │ ─── (Write Chunk by Chunk)
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Phase 4: Refine & Verify  │ ─── (Final Approval)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Section-by-Section Drafting (섹션별 단편 작성)**:
  - 한번에 한 섹션씩만 작성하여 검토를 받음으로써 사용자의 피드백 피로도를 줄이고 정확도를 극대화
- **Targeted Questioning (표적 질의)**:
  - 막연한 질문 대신 "A 옵션과 B 옵션 중 어떤 아키텍처를 선호하시나요?" 형태의 선택지 제시 인터뷰 수행

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any, List, Optional


class DocCoAuthoringHarness:
    """에이전트와 사용자 간의 단계별 문서 공동 작성 파이프라인 제어 클래스"""

    def __init__(self, doc_title: str) -> None:
        """하네스 초기화
        
        Args:
            doc_title (str): 작성할 문서 제목
        """
        self.doc_title: str = doc_title
        self.current_phase: int = 1
        self.outline: List[str] = []
        self.sections_content: Dict[str, str] = {}

    def advance_phase(self, user_input: str) -> Dict[str, Any]:
        """사용자 피드백에 따라 작성 단계를 전환하고 다음 액션을 반환하는 메서드
        
        Args:
            user_input (str): 사용자의 입력 피드백
            
        Returns:
            Dict[str, Any]: 현재 상태 및 에이전트 액션 가이드
        """
        if self.current_phase == 1:
            # 개요 확정 단계
            self.outline = ["1. 개요 및 목적", "2. 시스템 아키텍처", "3. 구현 상세"]
            self.current_phase = 2
            return {
                "phase": 2,
                "action": "INTERVIEW",
                "message": "개요가 설정되었습니다. 2단계 아키텍처의 주요 데이터베이스 종류를 지정해 주세요.",
                "outline": self.outline
            }
            
        elif self.current_phase == 2:
            # 섹션 작성 단계 진입
            self.current_phase = 3
            self.sections_content["1. 개요 및 목적"] = f"본 문서는 {self.doc_title}에 대한 상세 기술 사양서이다."
            return {
                "phase": 3,
                "action": "DRAFT_SECTION",
                "section": "1. 개요 및 목적",
                "content": self.sections_content["1. 개요 및 목적"]
            }

        return {"phase": self.current_phase, "action": "COMPLETE", "message": "문서 작성 완료"}


# 문서 공동 작성 스킬 하네스 테스트
if __name__ == "__main__":
    harness = DocCoAuthoringHarness("에이전트 시스템 기획서")
    
    res1 = harness.advance_phase("개요 동의함")
    print("Phase 1 -> 2 결과:", res1)
    
    res2 = harness.advance_phase("PostgreSQL 사용 예정")
    print("Phase 2 -> 3 결과:", res2)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | One-shot 일괄 생성 | Doc Co-authoring Skill 적용 |
| :--- | :--- | :--- |
| **품질 및 정확도** | 환각 포함 및 요구사항 누락 가능성 높음 | 단계별 피드백 수렴으로 요구사항 100% 반영 |
| **수정 용이성** | 전체 장문 문서 재수정 어려움 | 섹션 단위 원자적 검토 및 손쉬운 개정 |
| **사용자 통제권** | 에이전트에 완전히 종속됨 | 사용자가 주도적으로 문서의 방향성 통제 |

- **장점**:
  - 높은 완성도와 검증된 정확성을 갖춘 기술 문서 생산
- **한계점**:
  - 단계별 대화 및 인터뷰 진행으로 인한 대화 세션 단계 추가

## 7. 활용 사례 및 응용
- **Technical Spec & RFP Writing**:
  - 엔지니어링 설계 문서, API 명세서, 요구사항 정의서 공동 작성 파이프라인
