# Memory MCP Server (장기 메모리 MCP)

---
Reference:
- [Official Memory MCP Server Repository](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
- [Knowledge Graph-based Long-term Agent Memory Architecture](https://modelcontextprotocol.io/)
- GitHub Web Repository: `https://github.com/k1101jh/DeepLearning-Concepts/blob/main/Harness%20Engineering/Model%20Context%20Protocol/Memory%20MCP/Memory%20MCP.md`

---

## 1. 개념 정보 및 한 줄 요약
- **개념명**: Memory MCP Server (장기 지식 그래프 메모리 MCP 서버)
- **관련 분야/카테고리**: Model Context Protocol / Long-term Memory / Knowledge Graph / Agent Persistence
- **한 줄 요약**: 단일 대화 세션을 넘어 사용자의 선호도, 엔티티(Entities), 관계(Relations) 및 관찰(Observations) 지식을 지식 그래프로 영구 보존하고 조회하는 장기 메모리 전용 MCP 서버

## 2. 등장 배경 및 해결하려는 문제 (Why?)
- **세션 종료 시 메모리 휘발 문제 (State Amnesia)**:
  - 일반 LLM 대화는 컨텍스트 윈도우가 초기화되면 이전 세션에서 파악한 사용자의 기호, 환경 설정, 주요 지식 맥락이 모두 손실됨
- **장기 대화에 따른 프롬프트 비효율**:
  - 이전 전체 대화 이력을 상시 프롬프트로 유지하면 토큰 비용 및 처리 지연이 극심해짐
- **Memory MCP 도입을 통한 해결**:
  - 중요한 정보만 엔티티 및 지식 그래프(Knowledge Graph) 형태로 추출하여 데이터베이스에 영구 보존 후 필요할 때만 인덱싱 조회

## 3. 핵심 원리 및 메커니즘 (How?)
- **Knowledge Graph 기반 4대 메모리 툴 인터페이스**:
  - **`create_entities`**: 인물, 프로젝트, 기술명 등 엔티티(Entities) 노드 생성
  - **`create_relations`**: 엔티티 간의 관계(Edge: `A -> depends_on -> B`) 생성
  - **`add_observations`**: 엔티티에 대한 세부 관찰 사실(Observation) 기록
  - **`read_graph`**: 지식 그래프 전체 구조 및 연결 상태 조회

```text
[User Dialogue Input]
          │
          ▼
┌───────────────────────────┐
│ Memory MCP Server         │ ─── (Extract Entities & Relations)
└───────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Knowledge Graph Storage   │ ─── (Persist: User -> prefers -> PyTorch)
└───────────────────────────┘
```

## 4. 핵심 세부 개념 및 부가 설명
- **Entity-Relation Graph Structure (엔티티-관계 그래프)**:
  - 텍스트 덩어리 대신 `(User, prefers, Python)`, `(Project, uses, FastMCP)`와 같이 트리 구성으로 기억을 압축하여 높은 정확도 제공
- **Cross-Session Persistence (이종 세션 간 지속성)**:
  - 세션이 바뀌더라도 에이전트가 동일한 Memory MCP 서버에 접근하여 사용자 개인화 맥락 유지

## 5. 코드 구현 예시 (PyTorch / Python)

```python
import json
from typing import Dict, Any, List


class MemoryMCPServer:
    """Knowledge Graph 기반 지식 보존 Memory MCP 서버 파이썬 시뮬레이션 클래스"""

    def __init__(self) -> None:
        """메모리 그래프 초기화"""
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, str]] = []

    def create_entity(self, name: str, entity_type: str, observations: List[str]) -> Dict[str, Any]:
        """[MCP Tool: create_entities] 신규 엔티티 및 관찰 내용 기록 메서드
        
        Args:
            name (str): 엔티티 이름
            entity_type (str): 엔티티 유형 (Person, Tech, Project 등)
            observations (str): 세부 관찰 사실 리스트
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        self.entities[name] = {
            "type": entity_type,
            "observations": observations
        }
        return {"success": True, "entity": name, "status": "엔티티 생성 완료"}

    def create_relation(self, from_entity: str, relation: str, to_entity: str) -> Dict[str, Any]:
        """[MCP Tool: create_relations] 엔티티 간 관계 형성 메서드
        
        Args:
            from_entity (str): 출발 엔티티
            relation (str): 관계 명칭 (prefers, uses 등)
            to_entity (str): 도착 엔티티
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        edge = {"from": from_entity, "relation": relation, "to": to_entity}
        self.relations.append(edge)
        return {"success": True, "relation": edge}


# Memory MCP 서버 테스트
if __name__ == "__main__":
    server = MemoryMCPServer()
    server.create_entity("User", "Person", ["파이썬과 PyTorch 선호함", "한글 docstring 작성 규칙 준수"])
    server.create_entity("Harness Engineering", "Concept", ["에이전트 샌드박스 런타임 기술"])
    server.create_relation("User", "studies", "Harness Engineering")
    
    print("Memory MCP 엔티티:", json.dumps(server.entities, indent=2, ensure_ascii=False))
    print("Memory MCP 관계 그래프:", json.dumps(server.relations, indent=2, ensure_ascii=False))
```

## 6. 장단점 및 기존 방식과의 비교

| 비교 항목 | 일반 대화 이력 (Chat History) | Memory MCP Server |
| :--- | :--- | :--- |
| **기억 지속성** | 세션 종료 시 휘발됨 | DB/그래프로 세션을 넘어 영구 보존 |
| **토큰 효율성** | 전체 대화 텍스트 상시 주입으로 토큰 소모 큼 | 압축된 지식 그래프 노드만 콤팩트 주입 |
| **개인화 추천** | 매 대화마다 기향점 재설명 필요 | 사용자의 기호 및 프로필 자동 기억 |

- **장점**:
  - 에이전트 간 지속 가능한 장기 기억(Long-term Memory) 시스템 형성
- **한계점**:
  - 엔티티 추출을 위한 백그라운드 LLM 연산 1회 추가 필요

## 7. 활용 사례 및 응용
- **Personalized AI Co-pilot**:
  - 사용자의 개발 스타일, 선호하는 라이브러리, 커스텀 규칙을 대화 세션을 뛰어넘어 장기 보존
