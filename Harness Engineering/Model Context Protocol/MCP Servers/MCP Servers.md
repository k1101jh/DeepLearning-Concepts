# MCP Servers (대표적인 MCP 서버 종류 및 활용법)

---
Reference:
- [Official Model Context Protocol Servers Registry & GitHub](https://github.com/modelcontextprotocol/servers)
- [Anthropic MCP Integration Guides](https://modelcontextprotocol.io/introduction)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/MCP%20Servers/MCP%20Servers.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: MCP Servers (대표적인 Model Context Protocol 서버 모음 및 활용)
- **관련 분야/카테고리**: Model Context Protocol / AI Tool Integrations / Open Standards
- **한 줄 요약**: LLM 에이전트가 외부 개발 환경, 데이터베이스, 웹, 파일 시스템에 보안 격리된 표준 통신 방식으로 접근할 수 있도록 돕는 검증된 주요 MCP 서버 모음

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **서드파티 커넥터 개발의 파편화 및 중복**:
  - 개발 도구, DB, 웹 브라우저 연동 시 개별 에이전트 앱마다 커스텀 API 코드를 반복적으로 구현해야 하는 문제 존재
- **보안 및 접근 권한 통제의 어려움**:
  - LLM에게 DB나 파일 시스템 전체 접근 권한을 임의 부여할 경우 심각한 보안 사고 발생 위험
- **MCP Servers 표준 모음 도입을 통한 해결**:
  - 오픈소스 커뮤니티 및 Anthropic이 표준 검증한 MCP 서버를 `stdio` 또는 `SSE` 바인딩으로 손쉽게 연결하여 보안 격리된 컨텍스트 및 툴 제공

## 3. 실무 대표 주요 MCP 서버 모음 (Key MCP Servers)

각 MCP 서버의 독립 상세 개념 문서:
- **[Context7 MCP Server (라이브러리 문서 주입 MCP)](../Context7%20MCP/Context7%20MCP.md)**: 최신 프레임워크 API 문서를 실시간 주입하여 **LLM 구버전 API 환각을 차단**하는 서버
- **[Playwright MCP Server (브라우저 자동화 MCP)](../Playwright%20MCP/Playwright%20MCP.md)**: 헤드리스 브라우저 조작, DOM 클릭, 폼 제출 및 **시각적 스크린샷 캡처** 지원 서버
- **[Notion MCP Server (노션 워크스페이스 연동 MCP)](../Notion%20MCP/Notion%20MCP.md)**: 노션 페이지, 데이터베이스, 블록을 **에이전트 지식베이스 DB로 자동 등록/조회**하는 서버
- **[Memory MCP Server (장기 지식 그래프 메모리 MCP)](../Memory%20MCP/Memory%20MCP.md)**: 세션을 뛰어넘어 **지식 그래프(Knowledge Graph) 형태로 영구 보존**하는 장기 기억 서버

---

### 1) GitHub MCP Server (`@modelcontextprotocol/server-github`)
- **개념**: GitHub API를 MCP 표준 툴로 노출하는 서버
- **제공 기능**:
  - `Tools`: 레포지토리 파일 읽기/쓰기, PR(Pull Request) 생성 및 리뷰, 이슈 검색 및 댓글 추가
  - `Resources`: 특정 레포지토리 커밋 이력 및 분기 상태 조회
- **활용 사례**: 에이전트가 코드 수정 후 자동으로 GitHub PR을 작성하고 이슈를 업데이트하는 자동화

### 2) Postgres / Database MCP Server (`@modelcontextprotocol/server-postgres`)
- **개념**: relational 데이터베이스의 테이블 스키마와 쿼리 기능을 안전하게 제공하는 서버
- **제공 기능**:
  - `Tools`: 인라인 SQL 쿼리 실행 (단, 하네스 정책에 따라 Read-only 제한 가능)
  - `Resources`: 데이터베이스 테이블 스키마 및 인덱스 정보 URIs (`postgres://database/schema`)
- **활용 사례**: 데이터 분석 에이전트가 DB 구조를 파악하고 자연어를 SQL로 전환하여 결과 리포트를 생성

### 3) Sequential Thinking MCP Server (`@modelcontextprotocol/server-sequential-thinking`)
- **개념**: 에이전트가 복잡한 수학이나 아키텍처 문제를 풀 때 동적으로 생각 단계(Thought Sequence)를 기록, 수정, 가지치기(Pruning)할 수 있도록 지원하는 추론 전용 MCP 서버
- **제공 기능**:
  - `Tools`: `sequentialthinking` (현재 생각 번호, 기존 생각 수정 여부, 가설 검증 단계 전달)
- **활용 사례**: 복잡한 버그 원인 분석 및 다단계 알고리즘 설계 시 에이전트의 사고 정밀도 향상

### 4) Filesystem MCP Server (`@modelcontextprotocol/server-filesystem`)
- **개념**: 호스트의 특정 지정된 디렉토리(Allowed Directories)만 샌드박싱하여 파일 읽기/쓰기를 허용하는 서버
- **제공 기능**:
  - `Tools`: `read_file`, `write_file`, `list_directory`, `search_files`
- **활용 사례**: IDE 에이전트가 허용된 프로젝트 폴더 내부 문서 및 코드를 안전하게 편집

### 5) Fetch & Browser MCP Server (`@modelcontextprotocol/server-fetch` / `puppeteer`)
- **개념**: 웹 페이지 URL 컨텍스트 추출 및 헤드리스 브라우저 클릭/타이핑을 제공하는 웹 연동 서버
- **제공 기능**:
  - `Tools`: `fetch` (HTML을 마크다운으로 변환 추출), `navigate`, `click`, `screenshot`
- **활용 사례**: 실시간 뉴스/문서 수집 및 웹 애플리케이션 E2E 테스트 자동화

## 4. MCP 서버 구성 및 바인딩 설정 예시 (JSON Config)

`claude_desktop_config.json` 또는 에이전트 설정 파일에 MCP 서버를 stdio 프로토콜로 등록하는 규격 예시:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class MCPServerRegistry:
    """여러 MCP 서버를 등록하고 사용 가능한 도구 목록을 통합 통합 관리하는 레지스트리 파이썬 클래스"""

    def __init__(self) -> None:
        """레지스트리 초기화"""
        self.servers: Dict[str, Dict[str, Any]] = {}

    def register_mcp_server(self, server_id: str, command: str, args: List[str]) -> None:
        """MCP 서버 바인딩 정보 등록 메서드
        
        Args:
            server_id (str): MCP 서버 식별자
            command (str): 실행 커맨드 (예: npx, python)
            args (List[str]): 실행 인자 리스트
        """
        self.servers[server_id] = {
            "command": command,
            "args": args,
            "status": "ready"
        }

    def generate_host_configuration(self) -> Dict[str, Any]:
        """Host 애플리케이션용 mcpServers JSON 설정 구조를 생성하는 메서드
        
        Returns:
            Dict[str, Any]: 표준 MCP Host 딕셔너리 설정
        """
        config: Dict[str, Any] = {"mcpServers": {}}
        for s_id, info in self.servers.items():
            config["mcpServers"][s_id] = {
                "command": info["command"],
                "args": info["args"]
            }
        return config


# MCP 레지스트리 테스트
if __name__ == "__main__":
    registry = MCPServerRegistry()
    registry.register_mcp_server("github", "npx", ["-y", "@modelcontextprotocol/server-github"])
    registry.register_mcp_server("postgres", "npx", ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"])
    
    host_config = registry.generate_host_configuration()
    print("생성된 MCP Host 설정:\n", json.dumps(host_config, indent=2))
```

## 6. 장단점 및 기존 개념과의 비교

| MCP Server 종류 | 주요 제공 컨텍스트 | 주 사용 툴 및 특징 |
| :--- | :--- | :--- |
| **GitHub MCP** | 커밋, PR, 이슈, 레포 파일 | `create_issue`, `create_pull_request` |
| **Postgres MCP** | DB 스키마, 인덱스, 레코드 | `query_database` (Read-only 권한 제어) |
| **Sequential Thinking** | 사고 과정 단계 (Thought Trace) | `sequentialthinking` (추론 정밀도 향상) |
| **Filesystem MCP** | 호스트 격리 디렉토리 파일 | `read_file`, `write_file` (보안 샌드박싱) |
| **Fetch MCP** | 웹 URL 콘텐츠 | `fetch_markdown` (웹 텍스트 추출) |

- **장점**:
  - `npx -y` 기반으로 검증된 MCP 서버를 즉시 설치 및 확장 사용 가능
  - 로컬 프로세스 격리를 통해 보안 안전성 보장
- **한계점**:
  - MCP 서버 수가 늘어남에 따라 Host의 프로세스 관리 및 툴 스키마 선택 오버헤드 발생

## 7. 활용 사례 및 응용
- **IDE AI Assistant (Cursor / Claude Desktop)**:
  - GitHub MCP + Filesystem MCP + Postgres MCP를 동시 연동하여 풀스택 개발 지원
