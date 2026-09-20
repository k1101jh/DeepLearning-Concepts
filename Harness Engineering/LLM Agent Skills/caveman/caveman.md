# Caveman Skill (케이원인 스타일 프롬프팅 스킬)

---
Reference:
- [Agent Skills Ecosystem & Token Optimization Techniques](https://github.com/vercel-labs/skills)
- [Caveman Prompting: Minimizing Agent Prose Tokens](https://github.com/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/caveman/caveman.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Caveman Skill (케이브맨 스킬)
- **관련 분야/카테고리**: LLM Agent Skill / Prompt Optimization / Token Reduction
- **한 줄 요약**: AI 에이전트의 대화 서론, 미사여구, 경어체를 철저히 배제하고 기호와 단문 중심의 전보 스타일(Telegraphic Style)로 응답하도록 제어하여 출력 토큰 소모량을 최적화하는 스킬

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **Agent의 과도한 텍스트 생성 오버헤드**:
  - LLM 에이전트가 작업 수행 시 "네, 말씀하신 작업을 수행하겠습니다...", "감사합니다", "결과는 다음과 같습니다" 등의 정중하고 길어진 응답으로 인해 불필요한 출력 토큰(Completion Tokens) 대량 소비
- **추론 비용 및 레이턴시 증가**:
  - 생성 토큰 수가 늘어남에 따라 API 비용 증가 및 사용자 대기 시간(Latency) 상승
- **Caveman 스킬 도입을 통한 해결**:
  - "Why use many tokens when few tokens do trick" 철학을 적용하여 에이전트의 말하기 방식(Agent Prose)을 극단적으로 경량화

## 3. 핵심 원리 및 메커니즘 (How?)
- **Caveman Prompting 규칙**:
  - 모든 친절한 인삿말, 사과, 서론, 결론 문장 완전 제거
  - 기호(`→`, `=`, `vs`, `w/`, `w/o`), 명사형 키워드, 단문 위주 전보체(Telegraphic Style) 사용
  - 코드 블록이나 필수 오류 메시지는 훼손하지 않되 설명 텍스트만 최소화
- **토큰 감소율 수식 표현**:
  - 기존 생성 토큰 수 $T_{raw}$ 대비 Caveman 적용 토큰 수 $T_{cave}$의 토큰 절감률(Saving Ratio $R$):
    $$R = \left( 1 - \frac{T_{cave}}{T_{raw}} \right) \times 100 \quad (\%) \quad (R \approx 40\% \sim 70\%)$$

## 4. 핵심 세부 개념 및 부가 설명
- **Target: Agent's Prose (말하기 스타일 표적 제어)**:
  - 생성되는 코드의 로직이나 품질은 변경하지 않고, 오직 에이전트의 텍스트 설명 부분만 압축
- **Telegraphic & Fragment Formatting (전보체 구조화)**:
  - `설정 완료. 포트 8080 오픈 → 서버 가동 중`과 같이 직관적인 단어 연쇄로 표현
- **Ponytail 스킬과의 시너지 연동**:
  - Ponytail(코드 생성 최소화)과 Caveman(설명 텍스트 최소화)을 조합하여 코드 및 프롬프트 토큰 양쪽 모두 최적화

## 5. 코드 구현 예시 (PyTorch / Python)

```python
from typing import Dict, Any


class CavemanPromptHarness:
    """에이전트 시스템 프롬프트에 Caveman 스타일 지침을 주입하고 출력을 후처리하는 파이썬 클래스"""

    CAVEMAN_INSTRUCTION: str = (
        "[SKILL: CAVEMAN]\n"
        "말하기 스타일 규칙:\n"
        "1. 경어체, 인사말, 서론, 결론, 사과 문장 완전 금지.\n"
        "2. 기호(->, =, vs, w/), 명사형 전보체, 단문만 사용.\n"
        "3. 예: '파일 수정 완료 -> 테스트 통과'. 복잡한 설명 배제하고 핵심만 출력."
    )

    def __init__(self) -> None:
        """하네스 객체 초기화"""
        pass

    def apply_caveman_system_prompt(self, base_system_prompt: str) -> str:
        """기본 시스템 프롬프트에 Caveman 스킬 지침을 주입하는 메서드
        
        Args:
            base_system_prompt (str): 기존 에이전트 시스템 프롬프트
            
        Returns:
            str: Caveman 지침이 결합된 강화 프롬프트
        """
        return f"{base_system_prompt}\n\n{self.CAVEMAN_INSTRUCTION}"

    def estimate_token_savings(self, original_text: str, caveman_text: str) -> Dict[str, Any]:
        """Caveman 적용 전후 텍스트 길이 및 절감률 추정 메서드
        
        Args:
            original_text (str): 기존 길어진 에이전트 응답
            caveman_text (str): Caveman 적용 압축 응답
            
        Returns:
            Dict[str, Any]: 길이 비교 및 절감률 정보
        """
        len_orig = len(original_text.split())
        len_cave = len(caveman_text.split())
        saving = (1.0 - (len_cave / max(1, len_orig))) * 100.0
        
        return {
            "original_words": len_orig,
            "caveman_words": len_cave,
            "saving_ratio_pct": round(saving, 2)
        }


# Caveman 하네스 테스트
if __name__ == "__main__":
    harness = CavemanPromptHarness()
    
    orig = "안녕하세요! 요청하신 데이터베이스 연결 설정을 정상적으로 완료하였습니다. 이제 포트 5432에서 응답을 대기하고 있습니다."
    cave = "DB 연결 완료 -> 포트 5432 대기 중"
    
    stats = harness.estimate_token_savings(orig, cave)
    print("Caveman 토큰 절감 통계:", stats)
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | 일반 Agent 응답 | Caveman Skill 응답 |
| :--- | :--- | :--- |
| **어조 (Tone)** | 정중함, 서론/결론 포함 친절체 | 극단적 명사형 전보체 (기호 활용) |
| **토큰 소모량** | 큼 (높은 API 비용) | 작음 (40~70% 토큰 절감) |
| **응답 레이턴시** | 생성 시간 길음 | 즉시 생성 완료 (빠른 반응속도) |
| **적용 영역** | 대고객 상담 서비스 | CLI 에이전트, 개발자용 빠른 반복 작업 |

- **장점**:
  - 에이전트의 텍스트 생성 토큰 소모량 및 API 가동 비용 획기적 절감
  - 가독성이 높고 핵심 상태만 빠르게 파악 가능
- **한계점**:
  - 친절한 설명이 필요한 초보자용 대화형 AI에는 부적합

## 7. 활용 사례 및 응용
- **Claude Code CLI / Terminal Agent**:
  - 개발자의 터미널 명령 수행 시 빠른 반응 속도 확보를 위한 Caveman 모드 적용
