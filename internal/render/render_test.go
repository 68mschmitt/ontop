package render

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"

	"ontop/internal/collect"
)

func TestStatusLabelKeepsFixedWidth(t *testing.T) {
	idle := statusLabel(false, "")
	collecting := statusLabel(true, "⠋")

	if !strings.Contains(idle, "idle") {
		t.Fatalf("idle label missing: %q", idle)
	}
	if !strings.Contains(collecting, "collecting") {
		t.Fatalf("collecting label missing: %q", collecting)
	}
	if lipgloss.Width(idle) != lipgloss.Width(collecting) {
		t.Fatalf("status label width changes: idle=%d collecting=%d", lipgloss.Width(idle), lipgloss.Width(collecting))
	}
	if lipgloss.Width(collecting) != statusWidth {
		t.Fatalf("status label width = %d, want %d", lipgloss.Width(collecting), statusWidth)
	}
}

func TestRenderContentOmitsUnavailableOptionalServices(t *testing.T) {
	content := RenderContent(collect.Snapshot{CollectedAt: time.Now()}, 80, nil)
	if strings.Contains(content, "NVIDIA GPU") || strings.Contains(content, "Unsloth Studio") || strings.Contains(content, "Ollama Processes") {
		t.Fatalf("optional service card rendered without metrics: %q", content)
	}
}

func TestRenderContentOrdering(t *testing.T) {
	s := collect.Snapshot{
		CollectedAt: time.Now(),
		Inference: []collect.InferenceProcess{
			{PID: 200, Provider: "Ollama", Name: "ollama", VRAMMiB: collect.OptFloat{Value: 4096, OK: true}},
		},
		GPUs:    []collect.GpuStats{{Name: "Test GPU", Index: "0"}},
		AMDGPUs: []collect.AmdGpuStats{{Name: "Test AMD", Index: "0"}},
	}

	content := RenderContent(s, 120, nil)

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
