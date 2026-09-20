# LLM Agent Skills (LLM 에이전트 스킬)

---
Reference:
- [OpenAI Function Calling & Tool Use Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Claude Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [Vercel Labs: Agent Skills Specification & CLI (`npx skills`)](https://github.com/vercel-labs/skills)
- [GitHub CLI Skill Management & Open Agent Skills](https://github.com/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/LLM%20Agent%20Skills/LLM%20Agent%20Skills.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: LLM Agent Skills (LLM 에이전트 스킬 및 툴 바인딩)
- **관련 분야/카테고리**: Agentic Capabilities / Tool Calling / Dynamic Skill Discovery / Skill Package Management
- **한 줄 요약**: 특정 목적 과제를 수행하기 위해 지침(Instruction), 도구 스키마(Tool Schema), 컨텍스트 자원 및 CLI 패키지 관리 기법을 하나로 캡슐화한 동적 모듈화 단위

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **단순 Function Calling의 모듈화 부재**:
  - 기존 툴 호출(Function Calling)은 개별 API 함수 스키마 전송에 그쳐, 해당 툴을 사용하기 위한 고도화된 프롬프트 지침 및 비즈니스 로직 캡슐화 불가능
- **컨텍스트 창(Context Window) 과부하 문제**:
  - 사용 가능한 모든 도구 및 스키마를 상시 프롬프트에 주입할 경우 프롬프트 토큰 비효율 및 추론 정확도 저하 발생
- **스킬 배포 및 공유 표준의 부재**:
  - 프로젝트마다 스킬 정의 방식이 제각각이어서 외부 커뮤니티의 검증된 스킬을 재사용하기 어려움
- **스킬(Skill) 및 CLI 매니저 도입을 통한 문제 해결**:
  - 도구, 설명 지침, 가드레일을 하나의 독립 디렉토리/모듈로 단위화
  - `npx` 기반 스킬 패키지 매니저(`npx skills`, `npx skillz` 등)를 도입하여 외부 스킬의 간편한 설치, 버전 관리 및 동적 로딩 구현

## 3. 핵심 원리 및 메커니즘 (How?)
- **Skill의 3대 핵심 캡슐화 구성요소**:
  - **SKILL.md (Instructions)**: 스킬의 사용 시점, 입력 가이드, 사용 규칙을 명시한 메타데이터 (YAML Frontmatter + Markdown Body)
  - **Tool Schemas (Functions)**: LLM이 호출 가능한 파라미터 구조 정의
  - **Supporting Resources**: 스킬 실행 시 필요한 템플릿, 스크립트, 레퍼런스 가이드

- **Dynamic Skill Loading & Activation 3단계 메커니즘**:
  1. **Discovery (탐색)**: 앱 시작 시 `.agents/skills/` 또는 글로벌 경로를 스캔하여 메타데이터(YAML frontmatter)만 콤팩트 주입
  2. **Matching (매칭)**: 사용자 요청 $Q$와 스킬 메타데이터 간 유사도 $Sim(Q, s_i)$가 임계값 $\tau$ 이상인 스킬 탐색:
     $$C_{active} = \{ s_i \in S \mid Sim(Q, s_i) \ge \tau \}$$
  3. **Activation (활성화)**: 매칭된 스킬의 상세 `SKILL.md` 지침 및 연관 파이썬/CLI 도구를 LLM 실행 컨텍스트에 동적 추가

```text
[User Request Q]
       │
       ▼
┌───────────────────────────┐
│ Dynamic Skill Registry    │ ─── (npx / CLI Package Manager)
└───────────────────────────┘
       │ Active Skills (C_active)
       ▼
┌───────────────────────────┐
│ LLM Context Window        │ ─── (Progressive Context Activation)
└───────────────────────────┘
```

## 4. npx 기반 스킬 패키지 관리 기법 (npx Skill Management)

### npx를 활용한 스킬 패키징 및 라이프사이클
`npx` 및 Node 기반 CLI 패키지 매니저(`npx skills`, `npx skillz`, `npx skillpm` 등)를 활용하여 외부 GitHub 저장소나 중앙 레지스트리로부터 에이전트 스킬을 관리하는 기법

1. **스킬 검색 및 프로젝트 추가 (Skill Add)**:
   - 외부 레포지토리의 에이전트 스킬을 현재 프로젝트 환경으로 원클릭 다운로드 및 바인딩
   ```bash
   npx skills add vercel-labs/agent-skills/browser-testing
   npx skillz install https://github.com/user/custom-skill-repo
   ```
2. **스킬 템플릿 신규 보일러플레이트 생성 (Skill Init)**:
   - 규격화된 `SKILL.md` 및 폴더 구조를 자동 생성
   ```bash
   npx skills init my-custom-skill
   ```
3. **스킬 검증 및 스키마 채점 (Skill Lint / Audit)**:
   - 작성된 스킬 문서의 YAML 프론트매터 및 구문 정확도를 CLI로 검증
   ```bash
   npx skills lint ./skills/my-custom-skill
   ```

### 하위 스킬 디렉토리 확장 가이드 (Extensibility Guide)
본 `LLM Agent Skills` 디렉토리 하위에는 개별 유용한 스킬들을 독립 디렉토리 및 개념 문서로 추가하여 체계적으로 관리합니다.

- **[Caveman Skill (케이브맨 프롬프팅 스킬)](./caveman/caveman.md)**: AI 에이전트의 대화 미사여구를 제거하고 기호/전보체 중심 응답을 유도하여 **출력 토큰을 40~70% 절감**하는 스킬
- **[Ponytail Skill (폰테일 코드 최소화 스킬)](./ponytail/ponytail.md)**: 3단계 의사결정 사다리(YAGNI, 기존 코드 재사용, 네이티브 라이브러리)를 적용하여 **코드 오버엔지니어링과 불필요한 라인을 50~90% 절감**하는 스킬
- **[Frontend Design Skill (프론트엔드 디자인 스킬)](./frontend-design/frontend-design.md)**: 디자인 토큰, HSL 테마, 반응형 레이아웃 및 미세 애니메이션을 적용하여 **고품질 웹 UI를 생성**하는 스킬
- **[WebApp Testing Skill (웹앱 테스트 스킬)](./webapp-testing/webapp-testing.md)**: Playwright 기반 브라우저 E2E 자동 조작, 콘솔 에러 수집 및 **시각적 스크린샷 자율 검증** 스킬
- **[Doc Co-authoring Skill (문서 공동 작성 스킬)](./doc-coauthoring/doc-coauthoring.md)**: 개요 확정 $\rightarrow$ 표적 질문 $\rightarrow$ 섹션별 단편 작성을 통해 **고품질 기술 문서를 사용자와 협업하여 완성**하는 스킬
- **[MCP Builder Skill (MCP 서버 작성 스킬)](./mcp-builder/mcp-builder.md)**: 파이썬 FastMCP 데코레이터 기반으로 **신규 MCP 커스텀 서버를 손쉽게 구현하고 디버깅**하는 스킬

```text
LLM Agent Skills/
├── LLM Agent Skills.md (본 개요 문서)
├── caveman/
│   └── caveman.md (Caveman 스킬 상세)
├── ponytail/
│   └── ponytail.md (Ponytail 스킬 상세)
├── frontend-design/
│   └── frontend-design.md (Frontend Design 스킬 상세)
├── webapp-testing/
│   └── webapp-testing.md (WebApp Testing 스킬 상세)
├── doc-coauthoring/
│   └── doc-coauthoring.md (Doc Co-authoring 스킬 상세)
└── mcp-builder/
    └── mcp-builder.md (MCP Builder 스킬 상세)
```

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import os
import json
import subprocess
from typing import Dict, Any, List, Optional


class NPXSkillManager:
    """npx 기반 CLI 명령을 호출하여 외부 에이전트 스킬을 동적으로 관리 및 로딩하는 파이썬 클래스"""

    def __init__(self, target_dir: str = "./.agents/skills") -> None:
        """스킬 매니저 초기화
        
        Args:
            target_dir (str): 스킬이 다운로드되어 설치될 저장 경로
        """
        self.target_dir: str = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def install_skill_via_npx(self, repo_url: str) -> Dict[str, Any]:
        """npx skills CLI 명령을 사용하여 지정된 저장소의 스킬을 다운로드 설치하는 메서드
        
        Args:
            repo_url (str): 외부 스킬 GitHub 저장소 주소
            
        Returns:
            Dict[str, Any]: 설치 수행 결과 정보 딕셔너리
        """
        try:
            # npx skills add 명령 실행 시뮬레이션 (npx -y 사용 규칙 준수)
            command = ["npx", "-y", "skills", "add", repo_url, "--target", self.target_dir]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return {"success": True, "message": f"스킬 설치 완료: {repo_url}", "output": result.stdout}
            else:
                return {"success": False, "message": f"npx 설치 오류: {result.stderr}", "output": ""}
        except Exception as e:
            return {"success": False, "message": f"실행 예외 발생: {str(e)}", "output": ""}

    def parse_skill_metadata(self, skill_folder_path: str) -> Optional[Dict[str, str]]:
        """설치된 스킬 폴더 내 SKILL.md의 프론트매터 메타데이터를 추출 파싱하는 메서드
        
        Args:
            skill_folder_path (str): 대상 스킬 디렉토리 경로
            
        Returns:
            Optional[Dict[str, str]]: 파싱된 메타데이터 (이름, 설명 등)
        """
        skill_md_path = os.path.join(skill_folder_path, "SKILL.md")
        if not os.path.exists(skill_md_path):
            return None

        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 간단한 파싱 구문 (YAML Frontmatter 분리 시뮬레이션)
        lines = content.splitlines()
        metadata: Dict[str, str] = {}
        in_frontmatter = False
        
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()

        return metadata


# npx 스킬 매니저 테스트
if __name__ == "__main__":
    manager = NPXSkillManager()
    print("npx 스킬 매니저 초기화 완료:", manager.target_dir)
```

## 6. 장단점 및 기존 개념과의 비교

| 비교 항목 | Raw Function Calling | Dynamic Agent Skill | npx Managed Skill |
| :--- | :--- | :--- | :--- |
| **구성 요소** | 단일 JSON 스키마 | 지침 + 스키마 + 자원 모듈 | CLI 버전 관리 + 원격 배포 표준 패키지 |
| **설치/배포 방식** | 하드코딩 개발 필요 | 디렉토리 복사 | `npx skills add` 커맨드 원클릭 설치 |
| **토큰 효율성** | 상시 주입으로 토큰 소모 큼 | 요구 시점 동적 탐색 | 메타데이터 상시 등록 + 필요 시 활성화 |
| **보안/감사** | 내부 코드 직접 통합 | 서드파티 코드 검증 어려움 | CLI 기반 패키지 린팅 및 검증 절차 도입 |

- **장점**:
  - `npx`를 이용한 오픈소스 에이전트 스킬의 높은 재사용성 및 생산성 확보
  - 버전 핀(Version Pinning) 및 공급망 보안(Supply Chain Security) 검증 용이
- **한계점**:
  - 원격 CLI 패키지 다운로드 시 네트워크 의존성 및 보안 권한 검증 필요

## 7. 활용 사례 및 응용
- **Vercel Labs `skills` CLI**:
  - `npx skills add`를 통한 프론트엔드/백엔드 테스트용 에이전트 스킬 다운로드
- **Antigravity IDE & Claude Code**:
  - 프로젝트 내 `.agents/skills/` 디렉토리를 감지하여 npx 패키지로 관리되는 외부 스킬 자동 동기화
