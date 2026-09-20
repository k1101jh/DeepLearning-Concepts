# Model Context Protocol (MCP)

---
Reference:
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [Official MCP Specification Documentation](https://modelcontextprotocol.io/)
- [Model Context Protocol GitHub Repository](https://github.com/modelcontextprotocol)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/Model%20Context%20Protocol.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Model Context Protocol (MCP)
- **관련 분야/카테고리**: AI Open Standard / Agentic Protocol / Context Integration
- **한 줄 요약**: LLM 애플리케이션(Host)과 외부 데이터, 도구, 시스템 간의 상호운용성을 제공하는 오픈소스 표준 비동기 통신 프로토콜

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **$N \times M$ 연동 파편화 문제 (Custom Integration Spaghetti)**:
  - 기존에는 개별 LLM 애플리케이션(Claude Desktop, VS Code 커서 등 $N$개)이 각 데이터 연동 툴(GitHub, PostgreSQL, Slack 등 $M$개)마다 커스텀 커넥터를 일일이 커스텀 개발해야 함
  - 연동 대상 증가 시 시스템 연동 복잡도가 $O(N \times M)$으로 급증함
- **MCP 도입을 통한 문제 해결**:
  - 클라이언트-서버 단일 표준 규격을 정의하여 복잡도를 $O(N + M)$ 수준으로 감소
  - 보안 격리된 1:1 바인딩 구조로 안전한 컨텍스트 제공 파이프라인 형성

## 3. 핵심 원리 및 메커니즘 (How?)
- **Host-Client-Server 아키텍처 구조**:
  - **MCP Host**: 사용자 인터페이스 및 AI 에이전트를 구동하는 주체 (예: Claude Desktop, IDE)
  - **MCP Client**: Host 내부에서 특정 MCP Server와의 1:1 메시지 통신 상태를 관리하는 커넥터
  - **MCP Server**: 데이터, 툴, 프로토콜 리소스를 외부에 노출하는 경량 서비스 프로세스
  - **Transport Layer**: `stdio` (로컬 프로세스 표준 입출력) 또는 `SSE` (Server-Sent Events / HTTP) 기반 메시지 전달

- **JSON-RPC 2.0 기반 메시지 요청/응답 구조**:
  - 요청 메시지 수식 표현 규격:
    $$Request = \{ \text{"jsonrpc"}: "2.0", \text{"id"}: 1, \text{"method"}: \text{"tools/call"}, \text{"params"}: \{ \dots \} \}$$

## 4. 핵심 세부 개념 및 부가 설명
- **3대 핵심 프리미티브 (Core Primitives)**:
  - **Tools**: LLM이 실행 명령을 내려 외부 연동 작업을 수행할 수 있는 호출 가능 함수 (Executable Functions)
  - **Resources**: LLM이 읽기 전용으로 조회할 수 있는 외부 데이터 및 파일 (URIs 기반 읽기 컨텍스트)
  - **Prompts**: 사용자가 재사용 가능하도록 사전 정의된 상호작용 템플릿 및 가이드라인
- **Transport Types (전송 프로토콜 유형)**:
  - **stdio Transport**: 동일 머신 내 subprocess 기반으로 표준 입력/출력을 주고받는 고성능 로컬 통신
  - **SSE Transport**: 원격 HTTP 서버 상에서 비동기 이벤트를 스트리밍받는 네트워크 통신

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List, Optional


class MCPServer:
    """JSON-RPC 2.0 규격을 준수하는 최소 구조의 파이썬 MCP 서버 시뮬레이션 클래스"""

    def __init__(self, server_name: str) -> None:
        """MCP 서버 객체 초기화
        
        Args:
            server_name (str): MCP 서버 식별 명칭
        """
        self.server_name: str = server_name
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, handler: Any) -> None:
        """MCP Tool 프리미티브 등록 메서드
        
        Args:
            name (str): 툴 이름
            description (str): 툴 역할 설명
            handler (Any): 실행 핸들러 함수
        """
        self.tools[name] = {
            "description": description,
            "handler": handler
        }

    def handle_json_rpc_request(self, request_raw: str) -> str:
        """JSON-RPC 2.0 요청을 수신하여 디코딩하고 응답을 생성하는 메인 통신 메서드
        
        Args:
            request_raw (str): JSON-RPC 규격의 문자열
            
        Returns:
            str: JSON-RPC 규격의 응답 문자열
        """
        try:
            req: Dict[str, Any] = json.loads(request_raw)
            req_id: Optional[int] = req.get("id")
            method: str = req.get("method", "")
            params: Dict[str, Any] = req.get("params", {})

            # 1. tools/list 요청 처리
            if method == "tools/list":
                tool_list = [
                    {"name": k, "description": v["description"]}
                    for k, v in self.tools.items()
                ]
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": tool_list}
                })

            # 2. tools/call 요청 처리
            elif method == "tools/call":
                tool_name: str = params.get("name", "")
                arguments: Dict[str, Any] = params.get("arguments", {})

                if tool_name not in self.tools:
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": "도구를 찾을 수 없음"}
                    })

                # 도구 핸들러 호출
                result_content = self.tools[tool_name]["handler"](**arguments)
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": str(result_content)}]}
                })

            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "지원하지 않는 메서드"}
                })

        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"내부 파싱 오류: {str(e)}"}
            })


# MCP 서버 및 통신 테스트
if __name__ == "__main__":
    server = MCPServer(server_name="MathMCP")
    
    # 덧셈 툴 등록
    def add_numbers(a: int, b: int) -> int:
        return a + b
        
    server.register_tool("add", "두 정수를 더하는 도구", add_numbers)

    # 요청 메세지 전송 시뮬레이션
    sample_request = json.dumps({
        "jsonrpc": "2.0",

        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "add",
            "arguments": {"a": 15, "b": 27}
        }
    })

    response = server.handle_json_rpc_request(sample_request)
    print("MCP Server JSON-RPC 응답:", response)
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | 기존 Custom API Direct Calling | Model Context Protocol (MCP) |
| :--- | :--- | :--- |
| **통합 구조** | N×M 방식 개별 API 커넥터 파편화 | 1:1 비동기 오픈 표준 프로토콜 (O(N+M)) |
| **보안 메커니즘** | 각 앱별 별도 인증 처리 | Host 레벨 사용자 승인 및 stdio/SSE 통제 |
| **재사용성** | 앱마다 커스텀 개발 필요 | 생성된 MCP 서버를 타 MCP Host에서도 즉시 재사용 |
| **주요 인터페이스** | API Endpoint 규격 제각각 | Prompts, Resources, Tools 3대 프리미티브 통일 |

- **장점**:
  - LLM 생태계의 도구 및 데이터 연동 표준화로 파편화 해소
  - 로컬 프로세스 격리를 통한 높은 보안성 제공
- **한계점**:
  - JSON-RPC 2.0 직렬화/역직렬화 오버헤드 및 네트워크 레이턴시 고려 필요

## 7. 활용 사례 및 응용
- **대표적인 MCP 서버 모음 및 활용**:
  - **[MCP Servers (대표적인 MCP 서버 종류 및 활용법)](./MCP%20Servers/MCP%20Servers.md)**: GitHub, Postgres DB, Sequential Thinking, Filesystem, Fetch MCP 등 검증된 5대 주요 MCP 서버 및 JSON 바인딩 가이드 정리
- **Claude Desktop / Cursor IDE / Windsurf**:
  - 로컬 파일 시스템, Postgres DB, GitHub 레포지토리를 MCP 서버로 바인딩하여 컨텍스트 참조
- **MCP Inspector**:
  - MCP 서버의 Tools, Resources, Prompts를 독립적으로 검증하는 디버깅 도구 활용
