package main

import (
	"strings"
	"testing"
	"time"

	"github.com/mattn/go-runewidth"
)

func TestParseAMDJSON(t *testing.T) {
	output := `{"devices":[{"Info":{"DeviceName":"AMD Radeon RX 9070 XT","DevicePath":{"pci":"0000:03:00.0"}},"gpu_activity":{"GFX":{"unit":"%","value":42}},"Sensors":{"Edge Temperature":{"unit":"C","value":55},"Average Power":{"unit":"W","value":120}},"VRAM":{"Total VRAM Usage":{"unit":"MiB","value":2048},"Total VRAM":{"unit":"MiB","value":16384}},"fdinfo":{"123":{"name":"ollama","usage":{"name":"ollama","usage":{"GTT":{"unit":"MiB","value":16},"VRAM":{"unit":"MiB","value":1024}}}}}}]}`

	gpus, err := parseAMDJSON(output)
	if err != nil {
		t.Fatalf("parseAMDJSON returned error: %v", err)
	}
	if len(gpus) != 1 {
		t.Fatalf("got %d GPUs, want 1", len(gpus))
	}
	gpu := gpus[0]
	if gpu.Name != "AMD Radeon RX 9070 XT" || gpu.PCI != "0000:03:00.0" {
		t.Fatalf("unexpected identity: %+v", gpu)
	}
	if !gpu.UtilPercent.OK || gpu.UtilPercent.Value != 42 || !gpu.MemoryUsed.OK || gpu.MemoryUsed.Value != 2048 {
		t.Fatalf("unexpected metrics: %+v", gpu)
	}
	if len(gpu.Processes) != 1 || gpu.Processes[0].Name != "ollama" || gpu.Processes[0].VRAMMiB.Value != 1024 || gpu.Processes[0].GTTMiB.Value != 16 {
		t.Fatalf("unexpected processes: %+v", gpu.Processes)
	}
}

func TestParseOllamaPS(t *testing.T) {
	output := `NAME ID SIZE PROCESSOR CONTEXT UNTIL
ornith-1.5:9b-opencode abcdef123456 5.0 GB 100% GPU 262144 4 minutes from now`

	models := parseOllamaPS(output)
	if len(models) != 1 {
		t.Fatalf("got %d models, want 1", len(models))
	}
	model := models[0]
	if model.Name != "ornith-1.5:9b-opencode" || model.Size != "5.0 GB" || model.Processor != "100% GPU" || model.Context != "262144" {
		t.Fatalf("unexpected model: %+v", model)
	}
}

func TestFitTextUsesDisplayWidth(t *testing.T) {
	text := fitText("模型名称 abc", 10)
	if width := runewidth.StringWidth(text); width > 10 {
		t.Fatalf("display width %d exceeds limit: %q", width, text)
	}
}

func TestRenderCoreRowWrapsInsteadOfHiding(t *testing.T) {
	row := renderCoreRow([]float64{10, 20, 30, 40, 50, 60}, 20)
	if strings.Contains(row, "hidden") || !strings.Contains(row, "C0") || !strings.Contains(row, "C5") {
		t.Fatalf("unexpected core row: %q", row)
	}
}

func TestRenderContentOmitsUnavailableOptionalServices(t *testing.T) {
	content := renderContent(snapshot{CollectedAt: time.Now()}, 80)
	if strings.Contains(content, "NVIDIA GPU") || strings.Contains(content, "Unsloth Studio") || strings.Contains(content, "Ollama Processes") {
		t.Fatalf("optional service card rendered without metrics: %q", content)
	}
}

func TestClassifyProvider(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"ollama", "Ollama"},
		{"OLLAMA", "Ollama"},
		{"  ollama  ", "Ollama"},
		{"unsloth", "Unsloth"},
		{"python-unsloth", "Unsloth"},
		{"Python-Unsloth", "Unsloth"},
		{"unsloth-trainer", "Unsloth"},
		{"vllm", "vLLM"},
		{"text-generation-server", "vLLM"},
		{"koboldcpp", "KoboldCPP"},
		{"xtt", "Local Inference"},
		{"voyager", "Local Inference"},
		{"sGLM", "Local Inference"},
		{"llama.cpp", "Local Inference"},
		{"llama-srv", "Local Inference"},
		{"chrome", ""},
		{"node", ""},
		{"unknown-binary-1234", ""},
	}
	for _, c := range cases {
		if got := classifyProvider(c.in, ""); got != c.want {
			t.Fatalf("classifyProvider(%q, %q) = %q, want %q", c.in, "", got, c.want)
		}
	}
}

func TestCollectInferenceSortsByVRAMDesc(t *testing.T) {
	output := `{"devices":[
		{"Info":{"DeviceName":"AMD Radeon RX 9070 XT","DevicePath":{"pci":"0000:03:00.0"}},"gpu_activity":{},"Sensors":{},"VRAM":{},"fdinfo":{
			"100":{"name":"ollama","usage":{"GTT":{"unit":"MiB","value":8},"VRAM":{"unit":"MiB","value":4096}}},
			"200":{"name":"python-unsloth","usage":{"GTT":{"unit":"MiB","value":16},"VRAM":{"unit":"MiB","value":12288}}}
		}}]}`

	gpus, err := parseAMDJSON(output)
	if err != nil {
		t.Fatalf("parseAMDJSON returned error: %v", err)
	}
	if len(gpus) != 1 || len(gpus[0].Processes) != 2 {
		t.Fatalf("unexpected GPUs/processes: %+v", gpus)
	}

	inference := buildInferenceProcessList(nil, gpus)
	if len(inference) != 2 {
		t.Fatalf("got %d inference processes, want 2", len(inference))
	}
	if !inference[0].VRAMMiB.OK || inference[0].VRAMMiB.Value != 12288 {
		t.Fatalf("expected highest VRAM first: %+v", inference[0])
	}
	if inference[0].Provider != "Unsloth" || inference[1].Provider != "Ollama" {
		t.Fatalf("unexpected providers/order: %+v", inference)
	}
	if inference[0].PID != 200 || inference[1].PID != 100 {
		t.Fatalf("unexpected PIDs/order: %+v", inference)
	}
}

func TestRenderInferenceEmptyCard(t *testing.T) {
	card := renderInference(nil, 80)
	if !strings.Contains(card, "Unified Inference") || !strings.Contains(card, "No models detected/loaded") {
		t.Fatalf("expected empty inference card: %q", card)
	}
}

func TestRenderInferenceGrouping(t *testing.T) {
	inference := []inferenceProcess{
		{PID: 200, Provider: "Unsloth", Name: "unsloth-trainer", VRAMMiB: optFloat{Value: 12288, OK: true}, GTTMiB: optFloat{Value: 16, OK: true}},
		{PID: 100, Provider: "Ollama", Name: "ollama", VRAMMiB: optFloat{Value: 4096, OK: true}, GTTMiB: optFloat{Value: 8, OK: true}},
		{PID: 50, Name: "chrome", VRAMMiB: optFloat{Value: 2048, OK: true}, GTTMiB: optFloat{}},
	}

	content := renderInference(inference, 120)
	if !strings.Contains(content, "Unified Inference") {
		t.Fatalf("expected inference card: %q", content)
	}
	if !strings.Contains(content, "LLM Processes") || !strings.Contains(content, "Other Processes") {
		t.Fatalf("expected LLM and Other sub-tables: %q", content)
	}
	if !strings.Contains(content, "Unsloth") || !strings.Contains(content, "Ollama") {
		t.Fatalf("expected provider labels in LLM table: %q", content)
	}
	if !strings.Contains(content, "chrome") {
		t.Fatalf("expected chrome in Other table: %q", content)
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

	content := renderGPUSection(gpus, nil, 120)
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

	content := renderGPUSection(gpus, nil, 120)
	if !strings.Contains(content, "NVIDIA GPU") || !strings.Contains(content, "LLM Processes") {
		t.Fatalf("expected NVIDIA card with LLM table only: %q", content)
	}
	if strings.Contains(content, "Other Processes") {
		t.Fatalf("did not expect Other Processes table when empty: %q", content)
	}
}

func TestRenderContentOrdering(t *testing.T) {
	s := snapshot{CollectedAt: time.Now()}
	s.Inference = []inferenceProcess{
		{PID: 200, Provider: "Ollama", Name: "ollama", VRAMMiB: optFloat{Value: 4096, OK: true}},
	}
	s.GPUs = []gpuStats{{Name: "Test GPU", Index: "0"}}

	content := renderContent(s, 120)

	idx := func(name string) int { return strings.Index(content, name) }
	if idx("System") > idx("Unified Inference") || idx("Unified Inference") > idx("NVIDIA GPU") || idx("NVIDIA GPU") > idx("AMD GPU") {
		t.Fatalf("unexpected section ordering: %q", content)
	}
}
