# ontop Project — Agent Work Log

## Session: Phase B (Complete)

**Work:**
| Task | Description | Status |
|------|-------------|--------|
| 6A   | Add swap memory tracking to renderContent + renderSystem | ✅ Done |
| 2A   | Augment NVIDIA process collection | Next |
| 2B   | GPU process correlation improvements | Next |

**Test Results (post-6A):**
- ✅ All 12 tests pass
- ✅ gofmt -l . clean

---

## Session: Phase B — Task 2A (In Progress)

**Work (Task 2A):**
1. Improve provider detection in `collectGPUProcesses` — detect Python/Torch/transformers/cmdline patterns and containerized processes (docker/Podman)
2. Add `vram_grow` to `gpuProcess` struct to track VRAM delta over intervals
3. Keep track of prior GPU state between collect cycles to compute delta

Requirements from the plan:
- Match non-obvious processes: python/python3/uv → check cmdline for torch/transformers/ollama patterns
- Handle containers (docker/Podman) by mapping CGroups to GPU process tables
- Add `vram_grow` metric

Status: Starting 2A now.

--- STATUS