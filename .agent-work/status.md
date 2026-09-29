# ontop Project — Agent Work Log

## Session History

### Phase A (Completed)
| Commit | Description |
|--------|-------------|
| `ses_f110d222bffeJle6jFqBef0Vvd` | Fix rendering tests: add LLM/Other sub-table headers, fix section ordering |
| `ses_f11098b5bffe31Tg9Zw5VP6oRd` | Refactor collectInference: deduplicate AMD query, buildInferenceProcessList |

**Result:** All 12 tests pass.

### Phase B — Task 6A Swap Memory (Completed)
| Commit | Description |
|--------|-------------|
| (included in NVIDIA commit above) | Add swapMemoryStats struct, collectSwap, renderSwap |
- Added `swapMemoryStats` type after `memoryStats`
- Added `Swap` field to `snapshot` struct
- Added `collectSwap` function using `mem.SwapMemoryWithContext`
- Updated `collectMetrics` to collect swap
- Updated `renderSystem` to render swap section when available
- Added `renderSwap` function

### Phase B — Task 2B NVIDIA Process Detection (Completed)
| Commit | Description |
|--------|-------------| 
| `56b9d09` | Improve NVIDIA process detection with cmdline and container name resolution |
- `classifyProvider` signature: `(name, cmdline string) string`
- Detect Python/Torch/HuggingFace/LLM processes by cmdline matching
- New `processCmdline(pid)` helper reads `/proc/pid/cmdline`
- New `processContainerName(pid)` helper detects docker/podman containers
- Enhanced container name resolution in `collectGPUProcesses`
- Updated test calls

### Phase B — Summary
| Task | Description | Status |
|------|-------------|--------|
| 2A   | Augment NVIDIA process (idle procs, aggressive provider matching) | ✅ Done (merged into 2B commit) |
| 2B   | GPU process correlation improvements | ✅ Done |
| 6A   | Add swap memory tracking | ✅ Done |

**Test Results:**
- ✅ All 12 tests pass
- ✅ gofmt -l . clean

---

## Phase B Complete. Ready for Phase C or new work.

