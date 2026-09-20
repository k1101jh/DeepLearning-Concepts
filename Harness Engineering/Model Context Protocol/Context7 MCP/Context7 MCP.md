# Context7 MCP Server (라이브러리 문서 주입 MCP)

---
Reference:
- [Context7 Documentation & Integration Guide](https://modelcontextprotocol.io/)
- [Anthropic MCP Servers Directory & Registry](https://glama.ai/mcp)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/Context7%20MCP/Context7%20MCP.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Context7 MCP Server (컨텍스트7 라이브러리 문서 주입 MCP 서버)
- **관련 분야/카테고리**: Model Context Protocol / Library Documentation / Hallucination Prevention
- **한 줄 요약**: 최신 버전 프레임워크 및 라이브러리의 공식 API 문서 스펙을 실시간 검색하여 에이전트 컨텍스트에 주입함으로써 LLM의 구버전 API 환각(Hallucination)을 전면 차단하는 MCP 서버

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **LLM 학습 데이터 컷오프(Knowledge Cutoff) 및 API 변경 문제**:
  - LLM은 학습 시점 이후 업데이트된 최신 프레임워크(Next.js 14/15, PyTorch 2.x, Pydantic v2 등)의 파괴적 변경(Breaking Changes)을 알지 못해 이미 폐기된(Deprecated) 구버전 메소드를 생성하는 환각 발생
- **정확한 최신 라이브러리 스펙 주입 필요성**:
  - 개발자가 일일이 공식 웹사이트 문서를 복사하여 프롬프트에 붙여넣는 오버헤드 존재
- **Context7 MCP 도입을 통한 해결**:
  - LLM 에이전트가 코드를 작성하는 즉시 해당 패키지/버전의 최신 공식 스펙을 검색하여 프롬프트에 동적 바인딩

## 3. 핵심 원리 및 메커니즘 (How?)
- **Context7 MCP 3단계 문서 주입 파이프라인**:
  1. **Package Detection**: 코드 내 `import` 구문 또는 요구사항에서 대상 패키지 및 버전 식별
  2. **Live Docs Search**: Context7 실시간 실적 인덱싱 DB에서 해당 라이브러리의 표준 API 용례 검색
  3. **Context Injection**: 검색된 핵심 마크다운 API 스펙 문서를 `Resources` 또는 `Tools` 형태로 에이전트 프롬프트에 주입

```text
[LLM Code Generation]
          │ (Need latest library API docs)
          ▼
┌───────────────────────────┐
│ Context7 MCP Server       │ ─── (Search Up-to-date Version Docs)
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Dynamic Docs Context      │ ─── (Inject to LLM Window -> No Hallucination)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Version-Specific Indexing (버전 특정 인덱싱)**:
  - 동일한 라이브러리라도 버전 $v1.x$와 $v2.x$ 간의 차이점을 구별하여 정확한 버전 문서만 스코핑
- **Token-Efficient Chunking (토큰 효율적 튜닝)**:
  - 수백 페이지의 웹 문서 전체를 전송하지 않고 필요한 API 메소드 시그니처 래핑 영역만 콤팩트 추출

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class Context7MCPServer:
    """Context7 방식의 라이브러리 최신 API 문서 검색 및 프롬프트 주입 MCP 서버 파이썬 구현체"""

    def __init__(self) -> None:
        """문서 인덱스 DB 초기화"""
        self.doc_index: Dict[str, Dict[str, str]] = {
            "pydantic_v2": {
                "version": "2.5.0",
                "docs": "BaseModel 사용 시 class Config 대신 model_config = ConfigDict(...)를 사용해야 함"
            },
            "nextjs_v14": {
                "version": "14.1.0",
                "docs": "App Router 사용 시 pages/ 디렉토리 대신 app/page.tsx 구조를 준수함"
            }
        }

    def search_library_docs(self, library_name: str) -> Dict[str, Any]:
        """[MCP Tool] 지정된 라이브러리의 최신 API 문서를 검색하는 메서드
        
        Args:
            library_name (str): 검사할 패키지 명칭
            
        Returns:
            Dict[str, Any]: 검색 결과 문서 및 버전 스펙
        """
        key = library_name.lower()
        if key in self.doc_index:
            info = self.doc_index[key]
            return {
                "found": True,
                "library": library_name,
                "version": info["version"],
                "api_docs_snippet": info["docs"]
            }
        return {"found": False, "message": "해당 라이브러리 문서를 찾을 수 없음"}


# Context7 MCP 서버 테스트
if __name__ == "__main__":
    server = Context7MCPServer()
    res = server.search_library_docs("pydantic_v2")
    print("Context7 문서 조회 결과:", json.dumps(res, indent=2, ensure_ascii=False))
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 단일 LLM 자체 지식 (Out-of-box) | Context7 MCP 연동 |
| :--- | :--- | :--- |
| **API 정확도** | 학습 컷오프 이후 최신 버전에 환각 발생 | 최신 라이브러리 버전 API 스펙 100% 정확 반영 |
| **개발 효율성** | 개발자가 구버전 에러 수동 디버깅 | 에이전트가 알아서 최신 문서를 참조하여 코드 작성 |
| **토큰 소모** | 불필요한 시험 착오로 토큰 낭비 | 필요한 API 시그니처 콤팩트 주입으로 절감 |

- **장점**:
  - LLM의 대표 한계점인 구버전 API 환각 문제를 근본적으로 해소
- **한계점**:
  - Context7 외부 인덱싱 서버에 대한 네트워크 조회 1회 필요

## 7. 활용 사례 및 응용
- **IDE Agent (Cursor / Claude Code)**:
  - 개발자가 최신 신규 프레임워크 도입 시 Context7 MCP를 연결하여 환각 없는 정확한 보일러플레이트 작성
