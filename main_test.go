package main

import (
	"strings"
	"testing"
	"time"

	"ontop/internal/collect"
	"ontop/internal/render"
)

func TestRenderContentOmitsUnavailableOptionalServices(t *testing.T) {
	content := render.RenderContent(snapshot{CollectedAt: time.Now()}, 80, nil)
	if strings.Contains(content, "NVIDIA GPU") || strings.Contains(content, "Unsloth Studio") || strings.Contains(content, "Ollama Processes") {
		t.Fatalf("optional service card rendered without metrics: %q", content)
	}
}

func TestRenderContentOrdering(t *testing.T) {
	s := snapshot{CollectedAt: time.Now()}
	s.Inference = []inferenceProcess{
		{PID: 200, Provider: "Ollama", Name: "ollama", VRAMMiB: optFloat{Value: 4096, OK: true}},
	}
	s.GPUs = []gpuStats{{Name: "Test GPU", Index: "0"}}
	s.AMDGPUs = []amdGPUStats{{Name: "Test AMD", Index: "0"}}

	content := render.RenderContent(s, 120, nil)

	idx := func(name string) int { return strings.Index(content, name) }
	wantOrder := func(before, after string) int {
		b, a := idx(before), idx(after)
		if b < 0 && a < 0 {
			return 0
		}
		if b < 0 {
			return -1
		}
		if a < 0 {
			return 1
		}
		return b - a
	}
	if wantOrder("NVIDIA GPU", "AMD GPU") > 0 || wantOrder("AMD GPU", "Memory") > 0 || wantOrder("Memory", "System") > 0 {
		t.Fatalf("unexpected section ordering: %q", content)
	}
}

func TestRenderGPUSectionGrouping(t *testing.T) {
	gpus := []gpuStats{
		{
			Name:  "Test GPU",
			Index: "0",
			Processes: []gpuProcess{
				{GPUUUID: "uuid-0", PID: 200, Provider: "vLLM", Name: "vllm", UsedMemoryMB: optFloat{Value: 8192, OK: true}},
				{GPUUUID: "uuid-0", PID: 50, Name: "chrome", UsedMemoryMB: optFloat{Value: 1024, OK: true}},
			},
		},
	}

	content := collect.RenderGPUSection(gpus, nil, 120, nil)
	if !strings.Contains(content, "NVIDIA GPU") {
		t.Fatalf("expected NVIDIA card: %q", content)
	}
	if !strings.Contains(content, "LLM Processes") || !strings.Contains(content, "Other Processes") {
		t.Fatalf("expected LLM and Other sub-tables: %q", content)
	}
	if !strings.Contains(content, "vllm") || !strings.Contains(content, "chrome") {
		t.Fatalf("expected both process groups rendered: %q", content)
	}
}

func TestRenderGPUSectionLLMOnly(t *testing.T) {
	gpus := []gpuStats{
		{
			Name:  "Test GPU",
			Index: "0",
			Processes: []gpuProcess{
				{GPUUUID: "uuid-0", PID: 200, Provider: "vLLM", Name: "vllm", UsedMemoryMB: optFloat{Value: 8192, OK: true}},
			},
		},
	}

	content := collect.RenderGPUSection(gpus, nil, 120, nil)
	if !strings.Contains(content, "NVIDIA GPU") || !strings.Contains(content, "LLM Processes") {
		t.Fatalf("expected NVIDIA card with LLM table only: %q", content)
	}
	if strings.Contains(content, "Other Processes") {
		t.Fatalf("did not expect Other Processes table when empty: %q", content)
	}
}

func TestMainPackageTypeAliases(t *testing.T) {
	var _ optFloat = collect.OptFloat{}
	var _ snapshot = collect.Snapshot{}
	var _ inferenceProcess = collect.InferenceProcess{}
	var _ cpuStats = collect.CpuStats{}
	var _ memoryStats = collect.MemoryStats{}
	var _ swapMemoryStats = collect.SwapMemoryStats{}
	var _ gpuStats = collect.GpuStats{}
	var _ gpuProcess = collect.GpuProcess{}
	var _ amdGPUStats = collect.AmdGpuStats{}
}


