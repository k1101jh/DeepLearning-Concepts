# Playwright MCP Server (브라우저 자동화 MCP)

---
Reference:
- [Official Playwright MCP Server Repository](https://github.com/modelcontextprotocol/servers)
- [Playwright Web Automation Framework](https://playwright.dev/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/Playwright%20MCP/Playwright%20MCP.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Playwright MCP Server (플레이라이트 브라우저 자동화 MCP 서버)
- **관련 분야/카테고리**: Model Context Protocol / Browser Automation / E2E Testing / Web Scraping
- **한 줄 요약**: LLM 에이전트가 헤드리스 브라우저(Playwright)를 통해 동적 웹 페이지 접속, DOM 클릭/타이핑, 폼 제출 및 Visual 스크린샷 캡처를 제어할 수 있도록 돕는 연동 MCP 서버

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **정적 HTTP Fetch의 한계 (JavaScript 렌더링 문제)**:
  - 단순 HTTP `curl`이나 `fetch` 도구는 React, Vue 등 자바스크립트 싱글 페이지 애플리케이션(SPA)의 동적 렌더링 콘텐츠를 스크래핑할 수 없음
- **사용자 시나리오 웹 조작 불가능성**:
  - 회원가입, 로그인 폼 채우기, 버튼 클릭 등 복잡한 브라우저 인터랙션 제어 수단 부재
- **Playwright MCP 도입을 통한 해결**:
  - 브라우저 조작 기능(`navigate`, `click`, `fill`, `screenshot` 등)을 표준 MCP Tool 스키마로 에이전트에 노출하여 자유로운 웹 자동화 구현

## 3. 핵심 원리 및 메커니즘 (How?)
- **Playwright MCP 툴 제어 아키텍처**:
  - **`playwright_navigate`**: 목표 웹 URL에 접속하고 JS 렌더링 완료 대기
  - **`playwright_click`**: CSS Selector 기반 버튼 또는 링크 클릭
  - **`playwright_fill`**: 텍스트 상자 및 폼 요소에 키보드 입력
  - **`playwright_screenshot`**: 현재 화면 시각적 스크린샷 이미지를 Base64/파일로 반환

```text
[LLM Agent]
     │ (Call JSON-RPC method: tools/call "playwright_click")
     ▼
┌───────────────────────────┐
│ Playwright MCP Server     │ ─── (Execute Headless Browser DOM Interaction)
└───────────────────────────┘
     │
     ▼
┌───────────────────────────┐
│ Target Web Application    │ ─── (DOM Rendered & Return Screenshot)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Headless & Headed Mode (헤드리스/헤디드 모드)**:
  - 백그라운드 무소음 실행(Headless)뿐만 아니라 개발자 시각 확인을 위한 디버깅 브라우저 렌더링 지원
- **DOM Accessibility Tree Scrape**:
  - 무거운 raw HTML 대신 에이전트가 읽기 쉬운 접근성 트리(Accessibility Tree) 형태로 화면 요소 전송

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class PlaywrightMCPServer:
    """Playwright 브라우저 조작 도구를 MCP 표준 인터페이스로 노출하는 파이썬 시뮬레이션 클래스"""

    def __init__(self) -> None:
        """브라우저 컨텍스트 상태 초기화"""
        self.current_url: str = ""
        self.page_title: str = ""

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """[MCP Tools/Call] 브라우저 자동화 툴 호출 처리 메서드
        
        Args:
            tool_name (str): MCP 툴 이름
            arguments (Dict[str, Any]): 툴 인자
            
        Returns:
            Dict[str, Any]: 실행 결과 딕셔너리
        """
        if tool_name == "playwright_navigate":
            url = arguments.get("url", "")
            self.current_url = url
            self.page_title = f"Title for {url}"
            return {"success": True, "url": self.current_url, "title": self.page_title}
            
        elif tool_name == "playwright_click":
            selector = arguments.get("selector", "")
            return {"success": True, "clicked_selector": selector, "status": "DOM element clicked"}

        elif tool_name == "playwright_screenshot":
            return {"success": True, "screenshot_format": "png", "path": "/artifacts/screenshot.png"}

        return {"success": False, "error": "지원하지 않는 도구"}


# Playwright MCP 서버 테스트
if __name__ == "__main__":
    server = PlaywrightMCPServer()
    res1 = server.handle_tool_call("playwright_navigate", {"url": "https://example.com"})
    print("페이지 이동 결과:", res1)
    
    res2 = server.handle_tool_call("playwright_click", {"selector": "#login-button"})
    print("클릭 결과:", res2)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 단순 Fetch/HTML Parser | Playwright MCP Server |
| :--- | :--- | :--- |
| **자바스크립트 렌더링** | 불가능 (정적 HTML만 획득) | 가능 (React, Vue 등 SPA 완벽 지원) |
| **인터랙션 (클릭/입력)** | 불가능 | 폼 제출, 버튼 클릭, 드래그 등 자유로움 |
| **시각적 스크린샷** | 지원 안 됨 | 스크린샷 캡처를 통한 비주얼 판별 지원 |

- **장점**:
  - LLM 에이전트가 사람처럼 웹 사이트를 자율 탐색하고 복잡한 웹 업무를 자동화
- **한계점**:
  - Chromium/Firefox 브라우저 런타임 구동에 따른 메모리 사용량 존재

## 7. 활용 사례 및 응용
- **E2E Web Agent & Scraping**:
  - 동적 로그인 필요 사이트에서의 데이터 수집 및 웹 애플리케이션 QA 자동 검증
