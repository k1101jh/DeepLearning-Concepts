# Ecosystem Registries (스킬 및 MCP 탐색 플랫폼 모음)

---
Reference:
- [Anthropic Official Skills Repository](https://github.com/anthropics/skills)
- [Glama MCP Explorer & Gateway](https://glama.ai/mcp)
- [Skills.sh: Agent Skills Directory & CLI](https://skills.sh)
- [Official MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [Awesome LLM Skills Repository](https://github.com/Prat011/awesome-llm-skills)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Ecosystem%20Registries/Ecosystem%20Registries.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Ecosystem Registries & Explorers (에이전트 스킬 및 MCP 탐색 플랫폼 모음)
- **관련 분야/카테고리**: Agent Ecosystem / Open Standards / Package Management / Security Audit
- **한 줄 요약**: AI 에이전트의 확장 기능(Agent Skills)과 도구 연동 커넥터(MCP Servers)를 탐색, 검증 및 원클릭으로 바인딩할 수 있는 공식 저장소 및 커뮤니티 허브 플랫폼 종합 가이드

## 2. 등장 배경 및 필요성 (Why?)
- **스킬 및 MCP 연동 도구의 수량 급증**:
  - 2026년 기준 20,000개 이상의 MCP 서버와 수만 개의 에이전트 스킬이 오픈소스 생태계에 등장하며 체계적인 탐색 플랫폼 필요성 대두
- **서드파티 악성 코드 및 보안 위협 (AgentBaiting)**:
  - 검증되지 않은 스킬이나 MCP 서버 설치 시 호스트 파일 시스템 유출, 악성 커맨드 실행 등의 보안 리스크 존재
- **탐색 플랫폼 도입을 통한 문제 해결**:
  - 검증된 공식 오픈소스 레포지토리 및 안전 평가 채점(Safety Score)이 도입된 허브 사이트를 사용하여 검증된 자원만 선택 활용

---

## 3. Agent Skills 주요 탐색 플랫폼 & 저장소

### 1) Anthropic Official Skills Repository (`anthropics/skills`)
- **URL**: `https://github.com/anthropics/skills`
- **특징**: Anthropic이 공식 유지관리하는 대표 에이전트 스킬 저장소
- **주요 내용**: `SKILL.md` 규격 표준 구조를 준수하는 공식 스킬 예시 라이브러리 (문서 자동화, 리팩토링, 디자인 시스템 등)

### 2) Skills.sh (Vercel Labs)
- **URL**: `https://skills.sh`
- **특징**: Vercel Labs에서 제공하는 오픈소스 에이전트 스킬 디렉토리 및 CLI 플랫폼
- **주요 기능**: `npx skills add <author>/<skill>` 커맨드를 이용해 다양한 인프라 및 개발 스킬을 원클릭 바인딩

### 3) Awesome LLM Skills
- **URL**: `https://github.com/Prat011/awesome-llm-skills`
- **특징**: 개발자 커뮤니티에서 기여한 분야별 에이전트 스킬 큐레이션 저장소
- **분류**: 코딩 리뷰어, TDD 워크플로우, 데이터 시각화(D3.js), 문서 변환 스킬 모음

### 4) AGNT.gg
- **URL**: `https://agnt.gg`
- **특징**: 에이전트 스킬 디렉터리 및 도메인별(DevOps, 코딩, 데이터 분석 등) 가이드 제공 플랫폼

### 5) OpenAgentSkill
- **URL**: `https://openagentskill.com`
- **특징**: 스킬 작성자의 신뢰도(Trust Score) 및 코드 안전성을 산출하여 안전한 스킬 탐색을 돕는 가이드 사이트

---

## 4. MCP Servers 주요 탐색 플랫폼 & 저장소

### 1) Glama MCP Explorer
- **URL**: `https://glama.ai/mcp`
- **특징**: 현재 가장 거대한 규모의 MCP 서버 검색, 웹 인스펙터(Inspector) 디버깅 및 안전 점수 제공 플랫폼
- **주요 기능**: 브라우저 상에서 MCP 서버의 Tools, Resources, Prompts 인터페이스를 사전 조작 테스트 가능

### 2) Official MCP Servers Registry (Linux Foundation)
- **URL**: `https://github.com/modelcontextprotocol/servers`
- **특징**: MCP 작업 그룹이 직접 관리하는 참조 구현체(Reference Implementations) 서버 모음
- **포함 서버**: GitHub, Postgres, Filesystem, Memory, Sequential Thinking 등 검증된 코어 서버

### 3) Awesome MCP Servers
- **URL**: `https://github.com/punkpeye/awesome-mcp-servers`
- **특징**: 카테고리별(DB, 통신, 클라우드, 개발 도구 등) 커뮤니티 검증 MCP 서버 링크 모음

### 4) MCPBundles
- **URL**: `https://mcpbundles.com/`
- **특징**: 엔터프라이즈급 인증(OAuth 2.0 등) 및 보안이 사전 검증된 프로덕션용 MCP 패키지 모음 사이트

---

## 5. 서드파티 스킬 및 MCP 보안 검증 가이드라인 (Security Best Practices)

1. **소프트웨어 공급망 보안 검증 (Supply Chain Verification)**:
   - 출처가 모호한 레포지토리의 스킬/MCP 설치 지양하고, `anthropics/skills`, `modelcontextprotocol/servers` 등 검증된 계정 우선 사용
2. **SKILL.md 및 툴 쉘 스크립트 사전 오디팅**:
   - `npx skills add` 또는 MCP 바인딩 전 `SKILL.md` 내부 지침 및 파이썬/Node 실행 스크립트 코드 전수 확인
3. **최소 권한의 원칙 (Principle of Least Privilege)**:
   - Filesystem MCP 또는 Postgres MCP 사용 시 특정 디렉토리/Read-Only 쿼리 권한만 최소 범위로 부여

---

## 6. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class EcosystemRegistryExplorer:
    """공식 및 커뮤니티 에이전트 스킬/MCP 레지스트리 URL 정보를 관리하고 권장성을 평가하는 파이썬 클래스"""

    def __init__(self) -> None:
        """레지스트리 목록 초기화"""
        self.skill_registries: List[Dict[str, str]] = [
            {"name": "Anthropic Official Skills", "url": "https://github.com/anthropics/skills", "official": "True"},
            {"name": "Skills.sh (Vercel)", "url": "https://skills.sh", "official": "True"},
            {"name": "Awesome LLM Skills", "url": "https://github.com/Prat011/awesome-llm-skills", "official": "False"}
        ]
        self.mcp_registries: List[Dict[str, str]] = [
            {"name": "Official MCP Servers", "url": "https://github.com/modelcontextprotocol/servers", "official": "True"},
            {"name": "Glama MCP Explorer", "url": "https://glama.ai/mcp", "official": "True"},
            {"name": "Awesome MCP Servers", "url": "https://github.com/punkpeye/awesome-mcp-servers", "official": "False"}
        ]

    def get_official_sources(self) -> Dict[str, List[Dict[str, str]]]:
        """공식 검증된 레지스트리 소스만 필터링하여 반환하는 메서드
        
        Returns:
            Dict[str, List[Dict[str, str]]]: 스킬 및 MCP 공식 소스 목록 딕셔너리
        """
        official_skills = [s for s in self.skill_registries if s["official"] == "True"]
        official_mcps = [m for m in self.mcp_registries if m["official"] == "True"]
        
        return {
            "official_skill_registries": official_skills,
            "official_mcp_registries": official_mcps
        }


# 탐색기 테스트
if __name__ == "__main__":
    explorer = EcosystemRegistryExplorer()
    official_data = explorer.get_official_sources()
    print("공식 검증 레지스트리 목록:\n", json.dumps(official_data, indent=2, ensure_ascii=False))
```

## 7. 장단점 및 탐색 플랫폼 비교

| 탐색 플랫폼 | 연동 대상 | 주요 기능 | 추천 사용 대상 |
| :--- | :--- | :--- | :--- |
| **`anthropics/skills`** | Agent Skill | Anthropic 공식 레퍼런스 스킬 | Claude Code / 스킬 직접 개발자 |
| **`skills.sh`** | Agent Skill | `npx skills add` CLI 원클릭 관리 | Vercel & Node 에이전트 개발자 |
| **`glama.ai/mcp`** | MCP Server | 브라우저 인스펙터 및 안전 채점 | MCP 서버 탐색 및 브라우저 테스트 |
| **`modelcontextprotocol/servers`** | MCP Server | Linux Foundation 코어 참조 구현체 | 엔터프라이즈 및 기본 MCP 연결 |

---
