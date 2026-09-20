# Notion MCP Server (노션 워크스페이스 연동 MCP)

---
Reference:
- [Official Notion API Documentation](https://developers.notion.com/)
- [Notion MCP Integration Server Specifications](https://modelcontextprotocol.io/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/Notion%20MCP/Notion%20MCP.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Notion MCP Server (노션 워크스페이스 연동 MCP 서버)
- **관련 분야/카테고리**: Model Context Protocol / Productivity Integration / Knowledge Management
- **한 줄 요약**: LLM 에이전트가 사용자의 Notion 워크스페이스 내 페이지(Pages), 데이터베이스(Databases), 블록(Blocks)을 읽고 쓰며 생성할 수 있도록 노션 API를 표준 MCP 인터페이스로 래핑한 서버

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **사내 지식베이스 및 개인 노션과의 분리**:
  - 에이전트가 사용자의 기존 노션 문서 및 프로젝트 백로그 DB를 읽거나 신규 작성된 개념 정리를 자동으로 등록하지 못하는 문제 존재
- **노션 복잡한 API 구조 접근성 문제**:
  - 노션의 Rich Text, Block, Page Property 파싱 및 JSON 스키마 생성이 까다로움
- **Notion MCP 도입을 통한 해결**:
  - `API-post-page`, `API-patch-block-children`, `API-query-data-source` 등 규격화된 툴 인터페이스를 제공하여 에이전트가 손쉽게 노션 DB 관리 수행

## 3. 핵심 원리 및 메커니즘 (How?)
- **Notion MCP의 3대 프리미티브 노출 아키텍처**:
  - **`API-query-data-source`**: 지정된 데이터베이스 ID의 레코드 항목 필터링 및 조회
  - **`API-post-page`**: 신규 페이지 문서 생성 및 DB 속성(Property) 등록
  - **`API-get-block-children`**: 특정 페이지 내부의 세부 블록(텍스트, 코드, 이미지) 마크다운 파싱 읽기

```text
[LLM Agent]
     │ (JSON-RPC call: API-post-page)
     ▼
┌───────────────────────────┐
│ Notion MCP Server         │ ─── (Notion API Bearer Token Authentication)
└───────────────────────────┘
     │
     ▼
┌───────────────────────────┐
│ User Notion Workspace     │ ─── (Create New Database Entry)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Database Property Mapping (DB 속성 매핑)**:
  - 노션 DB의 Select, Multi-select, Date, Title, URL 속성을 에이전트가 이해하는 표준 파이썬 딕셔너리로 양방향 변환
- **Markdown-to-Notion Block Conversion**:
  - 에이전트의 마크다운 텍스트를 노션의 계층적 Heading, Bulleted list, Code block으로 자동 변환 처리

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class NotionMCPServer:
    """Notion API를 MCP 표준 툴로 노출하는 파이썬 시뮬레이션 클래스"""

    def __init__(self, api_token: str) -> None:
        """노션 MCP 서버 초기화
        
        Args:
            api_token (str): 노션 시크릿 API 토큰
        """
        self.api_token: str = api_token

    def create_database_page(self, db_id: str, title: str, category: str, summary: str, url: str) -> Dict[str, Any]:
        """[MCP Tool: API-post-page] 노션 DB에 개념 정리 신규 항목을 추가하는 메서드
        
        Args:
            db_id (str): 대상 노션 데이터베이스 ID
            title (str): 개념 문서 제목
            category (str): 카테고리 (예: LLM, Harness Engineering)
            summary (str): 한 줄 요약
            url (str): GitHub 상대 파일 경로 또는 URL
            
        Returns:
            Dict[str, Any]: 생성된 페이지 상태 정보
        """
        # 노션 DB 속성 payload 생성 시뮬레이션
        payload = {
            "parent": {"database_id": db_id},
            "properties": {
                "Title": {"title": [{"text": {"content": title}}]},
                "Category": {"select": {"name": category}},
                "Summary": {"rich_text": [{"text": {"content": summary}}]},
                "GitHub URL": {"url": url}
            }
        }
        
        return {
            "success": True,
            "page_id": f"notion-page-{hash(title)}",
            "message": "노션 DB 항목 등록 완료",
            "payload": payload
        }


# 노션 MCP 서버 테스트
if __name__ == "__main__":
    server = NotionMCPServer(api_token="secret_notion_token_123")
    res = server.create_database_page(
        db_id="db_998877",
        title="Harness Engineering",
        category="Harness Engineering",
        summary="에이전트 샌드박스 런타임 및 평가 시스템 구축 기법",
        url="https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Harness%20Engineering/Harness%20Engineering.md"
    )
    print("노션 페이지 생성 결과:", json.dumps(res, indent=2, ensure_ascii=False))
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | 수동 노션 정리 | Notion MCP Server 연동 |
| :--- | :--- | :--- |
| **작성 자동화** | 사람이 직접 노션에 복사 및 DB 속성 입력 | 에이전트가 정리 마치는 즉시 자동 DB 등록 |
| **지식 동기화** | GitHub 레포와 노션 간 싱크 이탈 자주 발생 | GitHub 파일 URL 및 작성 날짜 실시간 동기화 |
| **속도** | 소요 시간 큼 | 수 초 만에 파이프라인 자동 완결 |

- **장점**:
  - 지식 정리 노트를 작성한 뒤 노션 지식베이스 DB로 자동 등록 지원
- **한계점**:
  - Notion Integration API 인증 토큰 및 Database ID 설정 필요

## 7. 활용 사례 및 응용
- **Automatic Knowledge Base Manager**:
  - 개념 정리를 마칠 때마다 노션 "공부/개념" 페이지 DB에 항목을 자동으로 등록하는 워크플로우
