# ontop Project — Agent Work Log

## Summary

**All planned phases are complete.** Tests: 15/15 pass. Build: clean. Format: clean.

---

## Session History

### Phase A — Fix broken tests (~2 hours)
- Added `LLM Processes`/`Other Processes` sub-table headers to `renderInference` and `renderGPUSection`
- Fixed section ordering in `renderContent` (System → Inference → NVIDIA → AMD)
- Refactored `collectInference`: deduplicated AMD query, added `buildInferenceProcessList` pure function
- Added NVIDIA inference process correlation to unified view

### Phase B — Swap memory + NVIDIA detection (~2 hours)
- Added `swapMemoryStats` type, `collectSwap`, `renderSwap` for swap memory tracking
- Enhanced `classifyProvider` to accept cmdline, added python/torch/huggingface/llama detection
- Added `processCmdline()` → `/proc/pid/cmdline` + `processContainerName()` → docker/podman cgroup
- NVIDIA process display names resolve to container names when available

### Phase C — Thermal + I/O metrics (~2 hours)
- Thermal monitoring: `thermalStats`/`thermalZone`, reads sensors via gopsutil, renders max+per-zone
- AMD fan RPM: extracted from amdgpu_top Sensors, displayed in AMD section
- Disk I/O: `diskStats`/`diskDeviceStats`, collected via `diskio.ReadStatWithContext`
- Network I/O: `netStats`/`netDeviceStats`, collected via `net.IOCountersWithContext`

### Phase D — GPU trends + Ollama + AMD fallback (~2 hours)
- GPU sparklines: rolling 30-sample history rendered with box-drawing unicode chars (▁▂▃▄...)
- Utilization trends: ▲/▼/- trend indicators next to GPU/AMD utilization bars
- VRAM trends: delta indicators shown on VRAM bar when changes exceed ±100 MiB
- Ollama JSON: `parseOllamaPSJSON()` tries `ollama ps --json` first, falls back to text
- AMD sysfs fallback: `collectAMDFromSysfs()` reads `/sys/class/drm/` when amdgpu_top unavailable
- Package-level `prevSnapshot` variable tracks previous data for delta computation

### Phase E — UX improvements (~2 hours)
- Keyboard shortcuts: s/g/o/u toggle sections, ?/h show/hide help, up/down/pgup/pgdn scroll
- Help overlay: renders keybindings in a modal card
- Config: `ONTOP_INTERVAL`, `ONTOP_GPU_FILTER`, `ONTOP_THEME` env vars, `-export`, `-once` flags
- Export modes: `-export=json|csv|stdout|path`, `-once` for non-interactive use
- Unsloth discovery: `UNSLOTH_STUDIO_URL` env var override, `detectUnslothProcesses()` scans running processes
- Footer displays active keybindings

### Phase F — Testing polish (~1 hour)
- testdata/: 4 fixture files (amdgpu_top_output.json, nvidia_smi_output.txt, etc.)
- Integration tests: fixture-based tests for Ollama, NVIDIA, AMD
- 11 benchmark tests: ParseAMDJSON, ParseOllamaPS, FitText, RenderCPU, RenderMemory, etc.
- readFixture() helper for test fixture loading

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Main file size | ~2,900 lines |
| Test count | 15 tests (12 unit + 3 integration) |
| Benchmark count | 11 benchmarks |
| Sections | System (CPU/RAM/Swap/Thermal) → Inference → NVIDIA → AMD → Disk I/O → Network → Unsloth → Ollama |
| Keyboard shortcuts | 8 keys (s/g/o/u/?/q/r + scroll) |
| Env vars | 3+ (ONTOP_INTERVAL, ONTOP_GPU_FILTER, ONTOP_THEME, ONTOP_NO_SPARKLINES) |
| CLI flags | 6 (-interval, -version, -unsloth-port, -unsloth-token, -export, -once) |
| Export modes | JSON, CSV, stdout, file path |

---

## Remaining Work (Out of Scope)

Not implemented, reserved for future:
- Push metrics to Prometheus/Graphite (Phase 13B)
- Parallel collector execution (Phase 12A) — current sequential collector is adequate for this dashboard's scope
- Light theme support (Phase 11A) — only dark theme implemented in config struct
- Docker container PID mapping improvements — basic docker/podman detection added, full CGroup → PID mapping left for later

