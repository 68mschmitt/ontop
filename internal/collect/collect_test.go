package collect

import (
	"strings"
	"testing"

	"github.com/mattn/go-runewidth"
)

func TestFitTextUsesDisplayWidth(t *testing.T) {
	text := FitText("模型名称 abc", 10)
	if width := runewidth.StringWidth(text); width > 10 {
		t.Fatalf("display width %d exceeds limit: %q", width, text)
	}
}

func TestRenderCoreRowWrapsInsteadOfHiding(t *testing.T) {
	row := RenderCoreRow([]float64{10, 20, 30, 40, 50, 60}, 20)
	if strings.Contains(row, "hidden") || !strings.Contains(row, "C0") || !strings.Contains(row, "C5") {
		t.Fatalf("unexpected core row: %q", row)
	}
}

func TestRenderGPUSectionGrouping(t *testing.T) {
	gpus := []GpuStats{
		{
			Name:  "Test GPU",
			Index: "0",
			Processes: []GpuProcess{
				{GPUUUID: "uuid-0", PID: 200, Provider: "vLLM", Name: "vllm", UsedMemoryMB: OptFloat{Value: 8192, OK: true}},
				{GPUUUID: "uuid-0", PID: 50, Name: "chrome", UsedMemoryMB: OptFloat{Value: 1024, OK: true}},
			},
		},
	}

	content := RenderGPUSection(gpus, nil, 120, nil)
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
	gpus := []GpuStats{
		{
			Name:  "Test GPU",
			Index: "0",
			Processes: []GpuProcess{
				{GPUUUID: "uuid-0", PID: 200, Provider: "vLLM", Name: "vllm", UsedMemoryMB: OptFloat{Value: 8192, OK: true}},
			},
		},
	}

	content := RenderGPUSection(gpus, nil, 120, nil)
	if !strings.Contains(content, "NVIDIA GPU") || !strings.Contains(content, "LLM Processes") {
		t.Fatalf("expected NVIDIA card with LLM table only: %q", content)
	}
	if strings.Contains(content, "Other Processes") {
		t.Fatalf("did not expect Other Processes table when empty: %q", content)
	}
}

func BenchmarkFitText(b *testing.B) {
	for i := 0; i < b.N; i++ {
		FitText("模型名称 abc 测试", 30)
	}
}

func BenchmarkRenderCPU(b *testing.B) {
	stats := CpuStats{
		OK:      true,
		Total:   45.2,
		PerCore: []float64{35.1, 50.2, 44.5, 60.3, 25.7, 70.1},
	}
	for i := 0; i < b.N; i++ {
		RenderCPU(stats, nil, 160)
	}
}

func BenchmarkRenderMemory(b *testing.B) {
	stats := MemoryStats{
		OK:        true,
		Used:      12000000000,
		Total:     32000000000,
		Available: 8000000000,
		Percent:   37.5,
	}
	for i := 0; i < b.N; i++ {
		RenderMemory(stats, nil, 120)
	}
}

func BenchmarkRenderCoreCell(b *testing.B) {
	for i := 0; i < b.N; i++ {
		RenderCoreCell(0, 42.5, 160)
	}
}

func BenchmarkRenderBar(b *testing.B) {
	for i := 0; i < b.N; i++ {
		RenderBar(75.0, 20)
	}
}

func BenchmarkHumanBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		HumanBytes(1234567890)
	}
}
