# DeepLearning Concepts 🧠

딥러닝 핵심 개념, 이론, 메커니즘 및 PyTorch 구현 노트를 구조화하여 정리하는 저장소입니다.

---

## 📚 카테고리별 개념 목록

### 🏋️ Training & Fine-Tuning (학습 기법 및 스케줄링)
- [EMA (Exponential Moving Average)](./Training/EMA/EMA.md)
- [Learning Rate Warmup](./Training/Warmup/Warmup.md)
- [AMP (Automatic Mixed Precision)](./Training/AMP/AMP.md)
- [LoRA (Low-Rank Adaptation - 경량 미세조정)](./Training/LoRA/LoRA.md)

### ⚙️ Optimizer (최적화 알고리즘)
- [AdamW](./Optimizer/AdamW/AdamW.md)
- [Optimizer 기초 및 발전 과정](./Basics/Optimizer/Optimizer.md)

### 🧱 Basics & Spatial AI (딥러닝 기초 및 공간 지능)
- [Reparameterization (Structural Reparameterization - 구조적 재파라미터화)](./Basics/Reparameterization/Reparameterization.md)
- [World Model (세계 모델 / 공간 시뮬레이터)](./Basics/World%20Model/World%20Model.md)
- [Activation Function (활성화 함수 - ReLU, GELU, SiLU 등)](./Basics/Activation%20Function/Activation%20Function.md)
- [Data Augmentation (데이터 증강)](./Basics/Augmentation/Augmentation.md)
- [Knowledge Distillation (지식 증류)](./Basics/Distillation/Distillation.md)
- [Federated Learning (연합 학습)](./Basics/Federated%20Learning/Federated%20Learning.md)
- [Loss Function (손실 함수)](./Basics/Loss%20function/Loss.md)
- [CTC Loss](./Basics/Loss%20function/CTC%20Loss.md)
- [Normalization (정규화 기법 - BatchNorm, LayerNorm 등)](./Basics/Normalization/Normalization.md)
- [Pruning (가지치기)](./Basics/Pruning/Pruning.md)
- [모델 경량화 (Model Compression)](./Basics/모델%20경량화/모델%20경량화.md)

### ⚡ Transformer & Sequence Architecture (시퀀스 및 아키텍처)
- [Mamba (Selective State Space Model - $O(N)$ 시퀀스 모델)](./LLM/Mamba/Mamba.md)
- [Attention Mechanism](./Attention/Attention.md)
- [Transformer (트랜스포머 아키텍처)](./Transformer/Transformer/Transformer.md)

### 🎨 Generative Models & 3D (생성 모델 및 3D 비전)
- [3D Gaussian Splatting (3DGS)](./Vision/3D%20Gaussian%20Splatting/3D%20Gaussian%20Splatting.md)
- [Diffusion Model (확산 모델)](./Diffusion%20Model/Diffusion%20Model/Diffusion%20Model.md)
- [GAN (Generative Adversarial Network)](./GAN/GAN/GAN.md)
- [VAE (Variational Autoencoder)](./VAE/VAE/VAE.md)

### 💬 LLM & Multi-Modal / Robotics (대형 언어 모델 및 로보틱스)
- [RAG (Retrieval-Augmented Generation / GraphRAG - 검색 증강 생성)](./LLM/RAG/RAG.md)
- [VLA (Vision-Language-Action Model - 시각-언어-행동 모델)](./LLM/VLA/VLA.md)
- [VLM (Vision-Language Model - 시각-언어 모델)](./LLM/VLM/VLM.md)
- [InstructGPT](./LLM/InstructGPT/InstructGPT.md)
- [LLM 파라미터 및 경량화 설정](./LLM/LLM%20파라미터/LLM%20파라미터.md)
- [Semantic Router](./LLM/Semantic%20Router/Semantic%20Router.md)

### 🛠️ Harness Engineering & Agent System (하네스 엔지니어링 및 에이전트 시스템)
- [Harness Engineering (하네스 엔지니어링 개념 및 샌드박스/평가 체계)](./Harness%20Engineering/Harness%20Engineering/Harness%20Engineering.md)
  - [Human-in-the-Loop (HITL 승인 하네스 - 고위험 액션 사용자 승인 인터셉트)](./Harness%20Engineering/Harness%20Engineering/Human-in-the-Loop/Human-in-the-Loop.md)
  - [Tool RAG (동적 도구 검색 하네스 - 쿼리 벡터 탐색 Top-K 툴 바인딩)](./Harness%20Engineering/Harness%20Engineering/Tool%20RAG/Tool%20RAG.md)
  - [Multi-Agent Orchestration (다중 에이전트 오케스트레이션 - Supervisor-Worker 상태 그래프)](./Harness%20Engineering/Harness%20Engineering/Multi-Agent%20Orchestration/Multi-Agent%20Orchestration.md)
  - [Agentic Self-Correction (자가 수정 루프 하네스 - 단위 테스트 연동 CRITIC 자율 수정)](./Harness%20Engineering/Harness%20Engineering/Agentic%20Self-Correction/Agentic%20Self-Correction.md)
  - [Context Window Compression (컨텍스트 창 압축 하네스 - Head-Tail 프루닝 및 토큰 오버플로우 방지)](./Harness%20Engineering/Harness%20Engineering/Context%20Window%20Compression/Context%20Window%20Compression.md)
- [Model Context Protocol (MCP - 표준 컨텍스트 연동 프로토콜)](./Harness%20Engineering/Model%20Context%20Protocol/Model%20Context%20Protocol.md)
  - [MCP Servers (대표적인 MCP 서버 종류 및 활용법)](./Harness%20Engineering/Model%20Context%20Protocol/MCP%20Servers/MCP%20Servers.md)
  - [Context7 MCP Server (라이브러리 문서 주입 - API 환각 차단)](./Harness%20Engineering/Model%20Context%20Protocol/Context7%20MCP/Context7%20MCP.md)
  - [Playwright MCP Server (브라우저 자동화 - DOM 조작 및 E2E 테스트)](./Harness%20Engineering/Model%20Context%20Protocol/Playwright%20MCP/Playwright%20MCP.md)
  - [Notion MCP Server (노션 워크스페이스 연동 - 지식베이스 DB 자동 등록)](./Harness%20Engineering/Model%20Context%20Protocol/Notion%20MCP/Notion%20MCP.md)
  - [Memory MCP Server (장기 지식 그래프 메모리 - 크로스 세션 영구 보존)](./Harness%20Engineering/Model%20Context%20Protocol/Memory%20MCP/Memory%20MCP.md)
- [LLM Utilization Methods (LLM 활용 방법론 및 프롬프팅/에이전틱 패턴)](./Harness%20Engineering/LLM%20Utilization%20Methods/LLM%20Utilization%20Methods.md)
  - [Prompt Caching (프롬프트 캐싱 및 접두사 공유 - 입력 토큰 비용 50~90% 절감)](./Harness%20Engineering/LLM%20Utilization%20Methods/Prompt%20Caching/Prompt%20Caching.md)
- [LLM Agent Skills (LLM 에이전트 스킬 및 Dynamic Discovery)](./Harness%20Engineering/LLM%20Agent%20Skills/LLM%20Agent%20Skills.md)
  - [Caveman Skill (케이브맨 프롬프팅 스킬 - 토큰 40~70% 절감)](./Harness%20Engineering/LLM%20Agent%20Skills/caveman/caveman.md)
  - [Ponytail Skill (폰테일 코드 최소화 스킬 - 코드 라인 50~90% 절감)](./Harness%20Engineering/LLM%20Agent%20Skills/ponytail/ponytail.md)
  - [Frontend Design Skill (프론트엔드 디자인 스킬 - 모던 UI/UX 생성)](./Harness%20Engineering/LLM%20Agent%20Skills/frontend-design/frontend-design.md)
  - [WebApp Testing Skill (웹앱 테스트 스킬 - Playwright 브라우저 E2E 자동 검증)](./Harness%20Engineering/LLM%20Agent%20Skills/webapp-testing/webapp-testing.md)
  - [Doc Co-authoring Skill (문서 공동 작성 스킬 - 계층적 기술 문서 협업)](./Harness%20Engineering/LLM%20Agent%20Skills/doc-coauthoring/doc-coauthoring.md)
  - [MCP Builder Skill (MCP 서버 작성 스킬 - FastMCP 기반 커스텀 구축)](./Harness%20Engineering/LLM%20Agent%20Skills/mcp-builder/mcp-builder.md)
- [Ecosystem Registries (스킬 및 MCP 탐색 플랫폼 모음)](./Harness%20Engineering/Ecosystem%20Registries/Ecosystem%20Registries.md)

### 👁️ Computer Vision & SLAM (컴퓨터 비전 및 위치 추적)
- [Visual SLAM (비주얼 동시적 위치추정 및 지도작성)](./Vision/Visual%20SLAM/Visual%20SLAM.md)
- [Vision Transformer (ViT)](./Vision/Vision%20Transformer/Vision%20Transformer.md)
- [Human Pose Estimation](./Vision/Human%20Pose%20Estimation/Human%20Pose%20Estimation.md)
- [Mutual Nearest Neighbor](./Vision/Mutual%20Nearest%20Neighbor/Mutual%20Nearest%20Neighbor.md)
- [Object Detection](./Vision/Object%20Detection/Object%20Detection.md)
- [Optical Character Recognition (OCR)](./Vision/Optical%20Character%20Recognition/Optical%20Character%20Recognition.md)
- [Segmentation](./Vision/Segmentation/Segmentation.md)
- [Perspective n Points (VPS)](./Vision/Vision%20Positioning%20System/Perspective%20n%20Points.md)

### ⚡ Model Efficiency & Quantization
- [Quantization (양자화 - GPTQ, AWQ, BitNet, QLoRA)](./Quantization/Quantization/Quantization.md)
- [Data Distillation](./Data/Data%20Distillation/Data%20Distillation.md)

### 🎮 Reinforcement Learning (강화학습)
- [PPO (Proximal Policy Optimization)](./Reinforcement%20Learning/PPO/PPO.md)

### 🛠️ PyTorch & Library (라이브러리 및 실전 연산)
- [PyTorch Contiguous](./Pytorch/Contiguous/Contiguous.md)
- [Optuna (하이퍼파라미터 최적화)](./Library/Optuna/Optuna.md)
- [SAHI (Slicing Aided Hyper Inference)](./Library/SAHI/SAHI.md)
- [THOP (PyTorch-OpCounter)](./Library/THOP/THOP.md)

---
*새로운 딥러닝 개념 정리 노트들이 계속해서 추가 및 업데이트됩니다.*
