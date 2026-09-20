# Frontend Design Skill (프론트엔드 디자인 스킬)

---
Reference:
- [Vercel AI SDK & UI Component Guidelines](https://sdk.vercel.ai/)
- [shadcn/ui & Tailwind CSS Design System Pattern](https://ui.shadcn.com/)
- [Anthropic Official Skills Repository](https://github.com/anthropics/skills)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/frontend-design/frontend-design.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Frontend Design Skill (프론트엔드 디자인 스킬)
- **관련 분야/카테고리**: LLM Agent Skill / UI-UX Design / Component System / Web Development
- **한 줄 요약**: 에이전트가 템플릿화된 기본 스타일에서 벗어나 모던 디자인 토큰, 테마 시스템, 반응형 레이아웃 및 미세 애니메이션을 적용하여 고품질 웹 UI를 작성하도록 통제하는 스킬

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **LLM 기본 출력 UI의 밋밋함 및 단조로움**:
  - 기본적인 프롬프트만으로 코드 작성 시 원색 위주의 스타일링, 기본 브라우저 폰트, 반응형 지원 미흡 등 품질이 저하된 코드 생성
- **디자인 토큰 및 가이드라인 미준수**:
  - 프로젝트의 색상 팔레트, 현대적 타이포그래피, 반응형 그리드 시스템을 무시한 채 일회성 CSS를 무분별하게 생성하는 문제 존재
- **Frontend Design 스킬 도입을 통한 해결**:
  - 디자인 토큰(Design Tokens), 컴포넌트 캡슐화, Tailwind/shadcn-ui 표준 규격을 지침으로 주입하여 프로덕션 레벨의 UI 코드를 일관되게 생성

## 3. 핵심 원리 및 메커니즘 (How?)
- **스킬 지침 4대 설계 요구사항 (Design System Rules)**:
  1. **Color System**: 단순 원색 금지, HSL 기반의 테이크아웃 테마 색상 및 다크모드 대응 팔레트 구성
  2. **Typography Hierarchy**: Google Fonts(Inter, Outfit 등) 기반의 계층적 폰트 시스템 정의
  3. **Micro-Interactions**: Hover 효과, Smooth Transition, Glassmorphic 효과 적용
  4. **Component Isolation**: 재사용 가능한 원자적 컴포넌트(Button, Card, Modal) 구조화

```text
[User UI Requirements]
       │
       ▼
┌───────────────────────────┐
│ Frontend Design Skill     │ ─── (Inject Design Tokens & Component Rules)
└───────────────────────────┘
       │
       ▼
┌───────────────────────────┐
│ Dynamic UI Component Code │ ─── (React + Tailwind CSS + Micro-Animations)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Progressive UI Enhancement (점진적 UI 향상)**:
  - 기본 시맨틱 HTML5 구조를 먼저 갖춘 뒤 스타일링 레이어를 덧씌우는 안전성 중심 설계 방식
- **Placeholder Elimination (더미 이미지 대체)**:
  - 텍스트 렌더링에 필요한 이미지가 있을 경우 더미 URL 대신 SVG 생성 도구나 generate_image 스킬을 연결하여 실제 시동 코드로 완성

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any, List


class FrontendDesignSkillHarness:
    """에이전트에게 모던 프론트엔드 디자인 시스템 규칙을 주입하고 UI 코드를 검증하는 파이썬 클래스"""

    DESIGN_TOKENS_INSTRUCTION: str = (
        "[SKILL: FRONTEND-DESIGN]\n"
        "UI 스타일링 원칙:\n"
        "1. 원색(plain red, blue) 절대 사용 금지. HSL 기반 테마 팔레트 사용.\n"
        "2. Inter/Outfit 등 모던 타이포그래피 폰트 계층 필수 명시.\n"
        "3. Hover 애니메이션 및 micro-interactionTransition 효과 필수 적용."
    )

    def __init__(self) -> None:
        """하네스 객체 초기화"""
        pass

    def inject_frontend_skill(self, system_prompt: str) -> str:
        """에이전트 프롬프트에 Frontend Design 스킬 지침을 동적 주입하는 메서드
        
        Args:
            system_prompt (str): 기존 시스템 프롬프트
            
        Returns:
            str: 스킬 지침이 포함된 강화 프롬프트
        """
        return f"{system_prompt}\n\n{self.DESIGN_TOKENS_INSTRUCTION}"

    def audit_ui_code(self, html_css_code: str) -> Dict[str, Any]:
        """생성된 HTML/CSS 코드의 모던 디자인 시스템 준수 여부를 검증하는 시뮬레이션 메서드
        
        Args:
            html_css_code (str): 에이전트가 생성한 UI 코드
            
        Returns:
            Dict[str, Any]: 검증 점수 및 보완 사항
        """
        violations: List[str] = []
        
        # 단순 기본 원색 스타일 검사
        if "color: red" in html_css_code or "bg-red-500" in html_css_code:
            violations.append("원색 기본 파랑/빨강 사용 감지 - 커스텀 HSL 톤 적용 권장")
            
        if "transition" not in html_css_code and "hover:" not in html_css_code:
            violations.append("Micro-interaction (Hover/Transition) 효과 누락 감지")

        score = max(0, 100 - (len(violations) * 30))
        return {
            "score": score,
            "passed": score >= 70,
            "violations": violations
        }


# 프론트엔드 디자인 스킬 테스트
if __name__ == "__main__":
    harness = FrontendDesignSkillHarness()
    sample_code = "<button style='color: red;'>Click Me</button>"
    result = harness.audit_ui_code(sample_code)
    print("UI 코드 검증 통계:", result)
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 일반 생성 코드 | Frontend Design Skill 적용 코드 |
| :--- | :--- | :--- |
| **디자인 감성** | 브라우저 기본 및 원색 위주 단조로움 | HSL 커스텀 테마, 다크모드, 완성도 높은 Visual |
| **인터랙션** | 정적(Static) 버튼 및 텍스트 | Hover, Active, Smooth Transition 미세 애니메이션 |
| **컴포넌트 구조** | 단일 파일에 스타일 난립 | 디자인 토큰 및 원자적 컴포넌트 캡슐화 |

- **장점**:
  - 에이전트가 프로덕션 출시 가능한 고품질 UI 코드를 일관되게 작성
- **한계점**:
  - CSS 토큰 규칙 주입에 따른 프롬프트 길이 증가

## 7. 활용 사례 및 응용
- **v0.dev / Claude Artifacts UI Builder**:
  - React, Tailwind, shadcn/ui 기반 아티팩트 및 대시보드 컴포넌트 자동 생성
