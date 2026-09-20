# WebApp Testing Skill (웹앱 테스트 스킬)

---
Reference:
- [Playwright Official Documentation](https://playwright.dev/)
- [Vercel & Playwright Agent Testing Workflows](https://github.com/vercel-labs/skills)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/webapp-testing/webapp-testing.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: WebApp Testing Skill (웹앱 테스트 스킬)
- **관련 분야/카테고리**: LLM Agent Skill / E2E Testing / Web Automation / Playwright
- **한 줄 요약**: 에이전트가 헤드리스 브라우저(Playwright 등)를 조작하여 웹 앱의 UI 요소 조작, 폼 제출, 네트워크 콘솔 로그 감시 및 Visual regression 스크린샷 검증을 수행하는 자동화 스킬

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **생성된 프론트엔드 코드의 검증 불가능성**:
  - 기존 에이전트는 코드 작성 후 실제 브라우저 상에서 인터랙션(버튼 클릭, 폼 전송)이 제대로 동작하는지 자체적으로 검증 불가능
- **런타임 자바스크립트 콘솔 에러 누락**:
  - 코드는 정상 컴파일되나 브라우저 실행 시 `Uncaught TypeError`나 네트워크 CORS 에러가 발생하는지 확인 불가
- **WebApp Testing 스킬 도입을 통한 해결**:
  - Playwright 기반 브라우저 조작 툴과 스크린샷 시각적 검증 파이프라인을 주입하여 작성된 웹 앱을 자율 검증 및 자기 수정(Self-Correction)하도록 구현

## 3. 핵심 원리 및 메커니즘 (How?)
- **WebApp Testing 4단계 검증 파이프라인**:
  1. **Page Navigation & Render**: 로컬 dev 서버 URL 접속 및 렌더링 확인
  2. **DOM Element Scrape**: 주요 버튼, 입력 필드, 모달 요소의 고유 ID/Selector 추출
  3. **Interaction Execution**: 클릭, 키보드 입력, 드래그 등 사용자 시나리오 수행
  4. **Log & Screenshot Capture**: 콘솔 에러 로그 수집 및 Visual Screenshot 캡처 검증

```text
[Generated Web App]
       │
       ▼
┌───────────────────────────┐
│ Playwright Browser Engine │ ─── (Navigate & Action Execution)
└───────────────────────────┘
       │
       ▼
┌───────────────────────────┐
│ Console Logs & Screenshot │ ─── (Self-Correction Feedback Loop)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Visual Assertion (시각적 검증)**:
  - 브라우저 스크린샷 캡처 이미지를 획득하여 요소의 겹침(Layout Shift)이나 깨짐 현상을 멀티모달 LLM이 직접 판별
- **Console & Network Interceptor**:
  - `console.error` 및 4xx/5xx HTTP 실패 상태를 실시간 기록하여 실패 이유를 에이전트에게 힌트로 피드백

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class WebAppTestingHarness:
    """Playwright 기반 웹앱 자동 테스트 시뮬레이션 및 에이전트 피드백 생성 클래스"""

    def __init__(self, base_url: str = "http://localhost:3000") -> None:
        """테스트 하네스 초기화
        
        Args:
            base_url (str): 테스트할 웹 앱 기본 URL
        """
        self.base_url: str = base_url

    def run_e2e_scenario(self, test_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """브라우저 자동화 액션 목록을 실행하고 검증 결과를 반환하는 메서드
        
        Args:
            test_actions (List[Dict[str, Any]]): 클릭, 입력 등의 액션 목록
            
        Returns:
            Dict[str, Any]: 테스트 성공 여부, 콘솔 로그, 스크린샷 경로 딕셔너리
        """
        console_logs: List[str] = []
        errors: List[str] = []

        # 액션 순회 시뮬레이션
        for act in test_actions:
            action_type = act.get("type")
            selector = act.get("selector")
            
            if action_type == "click" and selector == "#invalid-btn":
                errors.append(f"ElementNotFound: '{selector}' 요소를 찾을 수 없음")
                console_logs.append("TypeError: Cannot read properties of null")

        is_passed = len(errors) == 0
        return {
            "success": is_passed,
            "actions_executed": len(test_actions),
            "console_errors": console_logs,
            "errors": errors,
            "screenshot_path": "/artifacts/test_screenshot.png"
        }


# 웹앱 테스트 스킬 하네스 검증
if __name__ == "__main__":
    harness = WebAppTestingHarness()
    actions = [
        {"type": "navigate", "url": "http://localhost:3000"},
        {"type": "click", "selector": "#submit-btn"},
        {"type": "click", "selector": "#invalid-btn"}
    ]
    result = harness.run_e2e_scenario(actions)
    print("웹앱 E2E 테스트 결과:", json.dumps(result, indent=2, ensure_ascii=False))
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 코드 눈으로 확인 (Static Analysis) | WebApp Testing Skill 적용 |
| :--- | :--- | :--- |
| **검증 범위** | 문법 에러 유무 파악만 가능 | 실제 브라우저 DOM 조작 및 런타임 콘솔 확인 |
| **시각적 레이아웃** | 레이아웃 깨짐 판별 불가 | 스크린샷 캡처를 통한 비주얼 디버깅 가능 |
| **자동 복구** | 수동 문제 해결 필요 | 콘솔 에러 로그를 바탕으로 자동 자기 수정 |

- **장점**:
  - 웹 애플리케이션의 런타임 신뢰성 및 E2E 테스트 자동화 달성
- **한계점**:
  - 헤드리스 브라우저 실행에 따른 추가 연산 및 메모리 오버헤드 존재

## 7. 활용 사례 및 응용
- **Frontend Agent Frameworks (Claude Code / OpenInterpreter)**:
  - 생성된 웹 애플리케이션을 브라우저에서 자동 실행하고 디버깅하는 E2E 검증 파이프라인
