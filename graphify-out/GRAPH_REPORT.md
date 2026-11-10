# Graph Report - nava  (2026-11-10)

## Corpus Check
- Corpus is ~34,684 words - fits in a single context window. You may not need a graph.

## Summary
- 1057 nodes · 2063 edges · 61 communities (38 shown, 23 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 79 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Model Registry & Downloader
- TTS Microservice API
- Docker Compose Deployment
- Testing Harness & Fixtures
- Docker CI/CD Workflow
- Async Client SDK
- STT Engine Abstractions
- HTTP Request Limiting
- Text Processing Pipeline Tests
- System Architecture & Specs
- TTS Engine Abstractions
- Persian Text Normalizer
- End-to-End Service Tests
- Voice Asset Management
- GitHub CI Matrix
- Persian Inverse Text Normalization
- Shenava STT Engine
- Endpoint Integration Tests
- Unified Server CLI
- TTS REST Endpoints
- Async SDK Session Lifecycle
- Model Management Interface
- Engine Directory Resolvers
- Persian Ezafe Engine
- SDK Unit Tests
- WebSocket Integration Tests
- Streaming Client SDK
- Custom Replacement Dictionary
- Synthetic PCM Generator
- Albert Ezafe Model
- STT REST Tests
- Synchronous Client SDK
- Streaming ASR Adapters
- GitHub Release Notes
- Nemotron Streaming Engine
- Ava TTS Engine
- GitHub Release Automation
- Pocket TTS Voice Cloning
- Rule-Based Ezafe Fallback
- STT REST Endpoints
- CI Action Secrets
- PyPI Deployment Workflow
- Piper TTS Engine
- Voice Cloning Endpoints
- PyPI Release Environment
- Matcha TTS Flow Matching
- STT WebSocket Endpoint
- Python Test Matrix
- GitHub Release Deployment
- Hardware Auto-Configuration
- Docker Image Build Scripts
- Release Automation Script
- Model Downloader Script
- Curl Test Suite
- Audio Spec Architecture
- Nava Package Root
- SDK Distribution Metadata

## God Nodes (most connected - your core abstractions)
1. `ModelNotInstalledError` - 29 edges
2. `AsyncNavaClient` - 22 edges
3. `inverse_normalize_text()` - 21 edges
4. `NavaClient` - 20 edges
5. `BaseTTSEngine` - 20 edges
6. `STTResult` - 19 edges
7. `preprocess_tts_text()` - 19 edges
8. `BaseSTTEngine` - 18 edges
9. `EngineManager` - 18 edges
10. `ShenavaStreamingEngine` - 17 edges

## Surprising Connections (you probably didn't know these)
- `CPU Zero-Shot Voice Cloning` --semantically_similar_to--> `Pocket Engine Voice Prompts`  [INFERRED] [semantically similar]
  README.md → docs/USER_MANUAL.md
- `Nava TTS Microservice Architecture` --implements--> `Nava TTS Engine Suite`  [INFERRED]
  docs/ARCHITECTURE.md → README.md
- `Nava TTS Microservice Architecture` --references--> `TTS Service Package Dependencies`  [INFERRED]
  docs/ARCHITECTURE.md → nava-tts/requirements.txt
- `infer()` --calls--> `get_stt_engine()`  [EXTRACTED]
  nava-stt/app/main.py → nava/engines/factory.py
- `__getattr__()` --indirect_call--> `client()`  [INFERRED]
  nava/__init__.py → tests/unit/test_server_endpoints.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Persian Speech & Text Normalization Pipeline** — readme_nava_platform, readme_g2p_ezafe_pipeline, readme_itn, nava_sdk_readme_persian_g2p_ezafe [INFERRED 0.85]
- **Nava CPU Microservices Architecture** — docs_architecture_system_overview, docs_architecture_nava_stt_service, docs_architecture_nava_tts_service, readme_docker_deployment [INFERRED 0.95]
- **Nava Synchronous & Asynchronous Client Suite** — nava_sdk_readme_nava_client, nava_sdk_readme_async_nava_client, docs_user_manual_websocket_protocol [INFERRED 0.85]

## Communities (61 total, 23 thin omitted)

### Community 0 - "Model Registry & Downloader"
Cohesion: 0.06
Nodes (20): compute_sha256(), download_file(), _download_http_fallback(), download_model(), download_profile(), main(), get_model(), get_profile_models() (+12 more)

### Community 1 - "TTS Microservice API"
Cohesion: 0.05
Nodes (22): get_tts_engine(), cpu(), G2P, gap_bytes(), health(), lifespan(), _PatchedSessionOptions, Piper (+14 more)

### Community 2 - "Docker Compose Deployment"
Cohesion: 0.05
Nodes (55): context, dockerfile, ALL_PROXY, HF_HUB_OFFLINE, HF_TOKEN, HTTP_PROXY, HTTPS_PROXY, MKL_NUM_THREADS (+47 more)

### Community 3 - "Testing Harness & Fixtures"
Cohesion: 0.06
Nodes (17): _alive(), bodylimit(), _load_module(), stt_settings(), textpipe(), tts_settings(), engines(), http() (+9 more)

### Community 4 - "Docker CI/CD Workflow"
Cohesion: 0.04
Nodes (46): name, runs-on, steps, strategy, concurrency, cancel-in-progress, group, context (+38 more)

### Community 5 - "Async Client SDK"
Cohesion: 0.08
Nodes (9): AsyncNavaClient, check(), NavaClient, NavaServiceError, _recv_or_raise(), _send_chunked(), SpeechResult, TranscriptionResult (+1 more)

### Community 6 - "STT Engine Abstractions"
Cohesion: 0.11
Nodes (8): BaseStreamingSTTEngine, BaseSTTEngine, STTResult, EngineManager, get_stt_engine(), FasterWhisperEngine, _parse_pcm(), WhisperSherpaEngine

### Community 7 - "HTTP Request Limiting"
Cohesion: 0.09
Nodes (11): read_limited(), TooLargeError, Engine, health(), infer(), lifespan(), Result, transcribe() (+3 more)

### Community 8 - "Text Processing Pipeline Tests"
Cohesion: 0.07
Nodes (5): TestChunkPocket, TestFitTokenBudget, TestLexicon, TestNormalize, TestSplitSentences

### Community 9 - "System Architecture & Specs"
Cohesion: 0.07
Nodes (25): Read-Only Model Storage Layout, Nava STT Microservice Architecture, Nava TTS Microservice Architecture, Infrastructure Hardware Profiles, Lexicon & Pronunciation Overrides, Pocket Engine Voice Prompts, WebSocket Streaming Protocol Specification, Asynchronous AsyncNavaClient SDK (+17 more)

### Community 10 - "TTS Engine Abstractions"
Cohesion: 0.14
Nodes (4): BaseTTSEngine, TranscriptionSegment, TTSResult, _split_into_sentences()

### Community 11 - "Persian Text Normalizer"
Cohesion: 0.14
Nodes (13): normalize_currencies(), _repl(), normalize_dates(), _repl(), normalize_numbers(), _repl(), normalize_percentages(), _repl() (+5 more)

### Community 12 - "End-to-End Service Tests"
Cohesion: 0.22
Nodes (13): check_health_and_models(), create_synthetic_beep(), print_err(), print_header(), print_ok(), print_warn(), resample_pcm16(), run_all_tests() (+5 more)

### Community 13 - "Voice Asset Management"
Cohesion: 0.12
Nodes (3): get_voice_manager(), VoiceProfileManager, TestVoiceCloning

### Community 14 - "GitHub CI Matrix"
Cohesion: 0.09
Nodes (21): jobs, test, os, python-version, name, on, pull_request, push (+13 more)

### Community 15 - "Persian Inverse Text Normalization"
Cohesion: 0.15
Nodes (4): inverse_normalize_text(), parse_number_tokens(), PersianITN, TestPersianITN

### Community 16 - "Shenava STT Engine"
Cohesion: 0.17
Nodes (5): ensure_nemo_ctc_metadata(), ShenavaStreamingEngine, test_ensure_nemo_ctc_metadata_tokens_rewrite_and_meta(), test_shenava_streaming_buffered_ctc_fallback(), test_stt_websocket_aliases_and_eos()

### Community 17 - "Endpoint Integration Tests"
Cohesion: 0.13
Nodes (7): __getattr__(), client(), TestModelsEndpoints, TestServerHealth, TestSTTEndpoints, TestTTSEndpoints, TestWebSockets

### Community 18 - "Unified Server CLI"
Cohesion: 0.19
Nodes (5): create_app(), lifespan(), main(), main(), run_cmd()

### Community 19 - "TTS REST Endpoints"
Cohesion: 0.16
Nodes (4): generate_speech(), SpeechRequest, websocket_tts_endpoint(), main()

### Community 20 - "Async SDK Session Lifecycle"
Cohesion: 0.18
Nodes (3): AsyncNavaClient, SpeechResult, TranscriptionResult

### Community 23 - "Persian Ezafe Engine"
Cohesion: 0.20
Nodes (3): PersianEzafeProcessor, apply_custom_dictionary(), PersianG2PPipeline

### Community 24 - "SDK Unit Tests"
Cohesion: 0.23
Nodes (8): _mock(), test_async_transcribe_and_synthesize(), test_chunked_ws_send_splits_large_payloads(), test_client_reuses_one_http_pool_and_closes(), test_sync_synthesize_parses_headers(), test_sync_transcribe_parses_result(), handler(), test_sync_transcribe_raises_on_error_status()

### Community 25 - "WebSocket Integration Tests"
Cohesion: 0.30
Nodes (11): stt_ws(), test_sdk_stream_speech_ws_yields_audio(), test_sdk_transcribe_ws_chunked_frames(), test_stt_websocket_error_on_empty_utterance(), test_stt_websocket_full_protocol(), test_stt_websocket_oversized_cumulative_audio_closes_1009(), test_stt_websocket_rejects_non_json_control(), test_tts_websocket_invalid_json_reports_error() (+3 more)

### Community 26 - "Streaming Client SDK"
Cohesion: 0.18
Nodes (5): NavaServiceError, _recv_or_raise(), _send_chunked(), ws_url(), TestWsUrl

### Community 28 - "Synthetic PCM Generator"
Cohesion: 0.22
Nodes (4): _make_synthetic_pcm(), make_wav_bytes(), TestVoiceCloningEndpoint, make_test_wav()

### Community 30 - "STT REST Tests"
Cohesion: 0.16
Nodes (5): make_pcm(), http(), test_413_aborts_during_chunked_streaming(), test_transcribe_returns_result_shape(), test_transcribe_with_query_params()

### Community 31 - "Synchronous Client SDK"
Cohesion: 0.22
Nodes (4): NavaClient, test_async_client_end_to_end(), test_sync_client_end_to_end(), test_sync_client_maps_service_errors()

### Community 32 - "Streaming ASR Adapters"
Cohesion: 0.21
Nodes (3): _apply_replacements(), _parse_pcm(), ShenavaRNNTEngine

### Community 33 - "GitHub Release Notes"
Cohesion: 0.17
Nodes (12): with, cache, draft, files, generate_release_notes, name, packages-dir, password (+4 more)

### Community 36 - "GitHub Release Automation"
Cohesion: 0.20
Nodes (9): name, needs, runs-on, jobs, build, name, permissions, contents (+1 more)

### Community 39 - "Rule-Based Ezafe Fallback"
Cohesion: 0.24
Nodes (3): insert_ezafe_connector(), rule_based_ezafe_predict(), TestEzafeInsertion

### Community 41 - "CI Action Secrets"
Cohesion: 0.33
Nodes (9): steps, GITHUB_TOKEN, steps, steps, env, name, run, uses (+1 more)

### Community 42 - "PyPI Deployment Workflow"
Cohesion: 0.22
Nodes (9): publish_to_pypi, on, push, workflow_dispatch, default, description, type, tags (+1 more)

### Community 45 - "PyPI Release Environment"
Cohesion: 0.25
Nodes (8): name, url, publish-pypi, environment, if, name, needs, runs-on

### Community 48 - "Python Test Matrix"
Cohesion: 0.33
Nodes (6): test, python-version, matrix, name, runs-on, strategy

### Community 49 - "GitHub Release Deployment"
Cohesion: 0.40
Nodes (5): if, name, needs, runs-on, github-release

### Community 50 - "Hardware Auto-Configuration"
Cohesion: 0.83
Nodes (3): detect_cores(), detect_ram_gb(), configure_infra.sh script

## Knowledge Gaps
- **18 isolated node(s):** `nava-fa-speech`, `nava-fa-speech`, `Nava Model Downloader CLI`, `Nava Docker Microservices Deployment`, `Read-Only Model Storage Layout` (+13 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 448 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `inverse_normalize_text()` connect `Persian Inverse Text Normalization` to `Streaming ASR Adapters`, `Nemotron Streaming Engine`, `STT Engine Abstractions`, `Shenava STT Engine`, `Engine Directory Resolvers`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `ModelNotInstalledError` (e.g. with `NemotronStreamingEngine` and `ShenavaCTCEngine`) actually correct?**
  _`ModelNotInstalledError` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `nava-fa-speech`, `nava-fa-speech`, `Nava Model Downloader CLI` to the rest of the system?**
  _18 weakly-connected nodes found - possible missing edges._
- **Should `Model Registry & Downloader` be split into smaller, more focused modules?**
  _Cohesion score 0.056265984654731455 - nodes in this community are weakly interconnected._
- **Why does `ModelNotInstalledError` connect `Nemotron Streaming Engine` to `Streaming ASR Adapters`, `STT Engine Abstractions`, `Pocket TTS Voice Cloning`, `TTS Engine Abstractions`, `Piper TTS Engine`, `Shenava STT Engine`, `Engine Directory Resolvers`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `TTS Microservice API` be split into smaller, more focused modules?**
  _Cohesion score 0.05288207297726071 - nodes in this community are weakly interconnected._
- **Why does `CustomDictionaryManager` connect `Custom Replacement Dictionary` to `Rule-Based Ezafe Fallback`, `Persian Ezafe Engine`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._