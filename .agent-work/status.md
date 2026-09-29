# ontop Project — Agent Work Log

## Session History

### Phase A (Completed)
| Commit | Description |
|--------|-------------|
| (commit 1) | Fix rendering tests: add LLM/Other sub-table headers, fix section ordering |
| (commit 2) | Refactor collectInference: deduplicate AMD query, buildInferenceProcessList |

**Result:** All 12 tests pass.

### Phase B (Completed)
| Commit | Description |
|--------|-------------|
| 56b9d09 | Swap memory tracking + NVIDIA process cmdline/container detection |
| 8a077b4 | Update work status log |

**Result:** All 12 tests pass.

### Phase C (Completed)
| Commit | Description |
|--------|-------------|
| 617763c | Thermal monitoring, AMD fan RPM, disk I/O, network I/O |
**Result:** All 12 tests pass.

---

## Phase C Details
| Task | Implementation |
|------|---------------|
| 4A | **Thermal readings** — `thermalStats`/`thermalZone` types, `collectThermal` via `mem.SensorReadingsWithContext`, renders max temp bar + per-zone list in System card |
| 5A | **AMD fan RPM** — `FanPercent optFloat` field in `amdGPUStats`, parsed from `amdgpu_top` Sensors, displayed in renderAMDSection |
| 8A | **Disk I/O** — `diskStats`/`diskDeviceStats` types, collected via `diskio.ReadStatWithContext`, rendered in "Disk I/O" card |
| 9A | **Network I/O** — `netStats`/`netDeviceStats` types, collected via `net.IOCountersWithContext`, rendered in "Network I/O" card |
**Section order**: System → Inference → NVIDIA → AMD → Disk → Network → Unsloth → Ollama |

---

## Phase D — Next (Pending)

**Work items:**
| Task | Description | Status |
|------|-------------|--------|
| 10A | GPU utilization history (rolling window + sparkline) | Next |
| 10B | VRAM utilization trend (delta indicator) | Next |
| 7A | Ollama model parsing edge cases | Next |
| 3A | AMD fallback collection modes | Next |
| 4A | CPU thermal data (non-sensor fallback) | Next |

**Status:** Starting Phase D next.
