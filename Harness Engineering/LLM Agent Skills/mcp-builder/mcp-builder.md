# MCP Builder Skill (MCP 서버 작성 스킬)

---
Reference:
- [Official Model Context Protocol Specification & SDKs](https://modelcontextprotocol.io/)
- [Python FastMCP Framework Documentation](https://github.com/jlowin/fastmcp)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/mcp-builder/mcp-builder.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: MCP Builder Skill (MCP 서버 구현 및 디버깅 스킬)
- **관련 분야/카테고리**: LLM Agent Skill / MCP Server Development / FastMCP / Protocol Engineering
- **한 줄 요약**: 외부 API 및 시스템을 MCP 오픈 프로토콜 표준으로 노출하는 신규 커스텀 MCP 서버(Python FastMCP / TypeScript MCP SDK)를 올바른 프리미티브 스키마 구조로 신속 개발하도록 지침을 제공하는 스킬

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **MCP 표준 통신 규격의 파악 어려움**:
  - 개발자가 직접 MCP 서버 구현 시 JSON-RPC 2.0 프로토콜 직렬화 규칙 및 3대 프리미티브(Tools, Resources, Prompts) 핸들러 바인딩 구조 이해 부족으로 오류 발생
- **디버깅 및 테스트 검증 복잡성**:
  - `stdio` 또는 `SSE` 통신 상에서 발생하는 에러 메시지를 포착하기 어렵고 MCP Inspector 검증 절차가 까다로움
- **MCP Builder 스킬 도입을 통한 해결**:
  - Python `FastMCP` 및 TypeScript `MCP SDK` 기반의 보일러플레이트 코드, 스키마 정의 규격 및 MCP Inspector 검증 절차를 지침으로 통합 제공

## 3. 핵심 원리 및 메커니즘 (How?)
- **MCP Builder 4단계 구축 워크플로우**:
  1. **Primitive Architecture Setup**: Tools(실행), Resources(읽기 전용), Prompts(템플릿) 역할 분담
  2. **Schema & Decorator Binding**: Python FastMCP `@mcp.tool()` 데코레이터 및 docstring/typing 자동 파싱 적용
  3. **Transport Protocol Selection**: 로컬 subprocess 통신용 `stdio` 또는 네트워크 통신용 `SSE` 결정
  4. **MCP Inspector Verification**: `npx @modelcontextprotocol/inspector` 도구로 바인딩 검증

```text
[External API / Database]
           │
           ▼
┌───────────────────────────┐
│ FastMCP Server Code       │ ─── (@mcp.tool() & @mcp.resource())
└───────────────────────────┘
           │
           ▼
┌───────────────────────────┐
│ MCP Inspector Testing     │ ─── (JSON-RPC 2.0 stdio Verification)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **FastMCP Framework (파이썬 FastMCP)**:
  - 기존 장황한 MCP SDK 코드를 FastAPI 스타일의 데코레이터(`@mcp.tool()`) 기반으로 수 줄 만에 구축할 수 있는 현대적 Python MCP 프레임워크
- **Dynamic Docstring-to-Schema Parsing**:
  - 파이썬 함수의 한글 docstring과 타입 힌팅(typing)을 읽어 들여 LLM이 이해하는 JSON Schema 파라미터 묘사로 자동 변환하는 구조

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, Callable


class FastMCPServerBuilder:
    """Python FastMCP 기반 커스텀 MCP 서버 보일러플레이트를 생성하고 검증하는 파이썬 클래스"""

    def __init__(self, server_name: str) -> None:
        """빌더 초기화
        
        Args:
            server_name (str): 생성할 MCP 서버 명칭
        """
        self.server_name: str = server_name
        self.tools_code: list[str] = []

    def add_tool_code(self, fn_name: str, docstring: str, args_type: str, return_type: str) -> None:
        """MCP Tool 함수 코드를 데코레이터 형태로 자동 생성하는 메서드
        
        Args:
            fn_name (str): 툴 함수 이름
            docstring (str): 한글 설명 docstring
            args_type (str): 인자 정의 문자열
            return_type (str): 반환 타입
        """
        tool_snippet = (
            f"@mcp.tool()\n"
            f"def {fn_name}({args_type}) -> {return_type}:\n"
            f'    """{docstring}"""\n'
            f"    # TODO: 툴 실행 로직 구현\n"
            f"    return f'성공: {{{args_type.split(\":\")[0]}}}'\n"
        )
        self.tools_code.append(tool_snippet)

    def generate_full_server_code(self) -> str:
        """전체 Python FastMCP 서버 파이썬 실행 코드를 문자열로 반환하는 메서드
        
        Returns:
            str: 완결된 파이썬 MCP 서버 실행 코드
        """
        header = (
            "from fastmcp import FastMCP\n\n"
             f'mcp = FastMCP("{self.server_name}")\n\n'
        )
        body = "\n".join(self.tools_code)
        footer = '\nif __name__ == "__main__":\n    mcp.run(transport="stdio")\n'
        
        return header + body + footer


# FastMCP 빌더 테스트
if __name__ == "__main__":
    builder = FastMCPServerBuilder("CustomWeatherMCP")
    builder.add_tool_code(
        fn_name="get_weather",
        docstring="지정된 도시의 날씨 정보를 조회하는 MCP 툴",
        args_type="city: str",
        return_type="str"
    )
    
    generated_code = builder.generate_full_server_code()
    print("생성된 FastMCP 서버 코드:\n")
    print(generated_code)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | Raw JSON-RPC 저수준 구현 | FastMCP 기반 MCP Builder Skill 적용 |
| :--- | :--- | :--- |
| **코드 라인 수** | 100~200줄 이상의 직렬화 제어 필요 | 데코레이터 기반 10~20줄로 단축 |
| **스키마 유지보수** | JSON Schema 수동 작성으로 에러 잦음 | 파이썬 docstring / typing에서 자동 파싱 |
| **디버깅 용이성** | 통신 패킷 분석 어려움 | MCP Inspector 통합 인터페이스 검증 지원 |

- **장점**:
  - 외부 서비스 및 API를 빠른 시간 안에 안전한 표준 MCP 서버로 래핑하여 에이전트에 바인딩
- **한계점**:
  - `fastmcp` 파이썬 패키지 의존성 필요

## 7. 활용 사례 및 응용
- **Custom Enterprise API Integration**:
  - 사내 레거시 API 및 마이크로서비스를 MCP 서버로 표준화하여 Claude Desktop / Cursor 에이전트에 공급
