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
	if strings.Contains(content, "NVIDIA GPU") || strings.Contains(content, "AMD GPU") || strings.Contains(content, "Unsloth Studio") || strings.Contains(content, "Ollama Processes") {
		t.Fatalf("optional service card rendered without metrics: %q", content)
	}
}
