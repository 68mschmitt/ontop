package collect

import "strings"

type section interface {
	Name() string
	Render(width int) string
	Enabled() bool
}

func makeSections(s Snapshot, disabled map[string]bool) []section {
	return []section{
		&ollamaModelSection{snapshot: s, disabled: disabled},
		&gpuSection{snapshot: s, disabled: disabled},
		&amdSection{snapshot: s, disabled: disabled},
		&memorySection{snapshot: s, disabled: disabled},
		&unslothSection{snapshot: s, disabled: disabled},
		&ollamaProcessSection{snapshot: s, disabled: disabled},
		&systemSection{snapshot: s, disabled: disabled},
		&diskSection{snapshot: s, disabled: disabled},
		&netSection{snapshot: s, disabled: disabled},
	}
}

func RenderContentSections(s Snapshot, width int, disabledSections map[string]bool) string {
	if s.CollectedAt.IsZero() {
		return RenderCard("Status", mutedStyle.Render("Collecting initial metrics..."), width)
	}

	var parts []string
	var notices string
	if len(s.Warnings) > 0 {
		if w := RenderWarnings(s.Warnings, width); w != "" {
			notices = w
		}
	}
	for _, sec := range makeSections(s, disabledSections) {
		if sec.Enabled() {
			content := sec.Render(width)
			if strings.TrimSpace(content) != "" {
				parts = append(parts, content)
			}
		}
	}
	if notices != "" {
		parts = append(parts, notices)
	}

	return strings.Join(parts, "\n\n")
}

type systemSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *systemSection) Name() string        { return "System" }
func (s *systemSection) Enabled() bool       { return !(s.disabled != nil && s.disabled["System"]) }
func (s *systemSection) Render(w int) string { return RenderSystem(s.snapshot, w, s.disabled) }

type gpuSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *gpuSection) Name() string  { return "NVIDIA GPU" }
func (s *gpuSection) Enabled() bool { return !(s.disabled != nil && s.disabled["GPU"]) }
func (s *gpuSection) Render(w int) string {
	return RenderGPUSection(s.snapshot.GPUs, s.snapshot.GPUSparkline, w, s.disabled)
}

type amdSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *amdSection) Name() string        { return "AMD GPU" }
func (s *amdSection) Enabled() bool       { return !(s.disabled != nil && s.disabled["GPU"]) }
func (s *amdSection) Render(w int) string { return RenderAMDSection(s.snapshot.AMDGPUs, w, s.disabled) }

type diskSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *diskSection) Name() string        { return "Disk I/O" }
func (s *diskSection) Enabled() bool       { return !(s.disabled != nil && s.disabled["Disk"]) }
func (s *diskSection) Render(w int) string { return RenderDiskCard(s.snapshot.Disk, w, s.disabled) }

type netSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *netSection) Name() string        { return "Network I/O" }
func (s *netSection) Enabled() bool       { return !(s.disabled != nil && s.disabled["Network"]) }
func (s *netSection) Render(w int) string { return RenderNetCard(s.snapshot.Net, w, s.disabled) }

type unslothSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *unslothSection) Name() string  { return "Unsloth Studio" }
func (s *unslothSection) Enabled() bool { return !(s.disabled != nil && s.disabled["Unsloth"]) }
func (s *unslothSection) Render(w int) string {
	return RenderUnslothStudio(s.snapshot.UnslothStudio, w, s.disabled)
}

type ollamaProcessSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *ollamaProcessSection) Name() string  { return "Ollama Processes" }
func (s *ollamaProcessSection) Enabled() bool { return !(s.disabled != nil && s.disabled["Ollama"]) }
func (s *ollamaProcessSection) Render(w int) string {
	return RenderOllamaProcesses(s.snapshot.OllamaProcesses, w, s.disabled)
}

type ollamaModelSection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *ollamaModelSection) Name() string  { return "Ollama Models" }
func (s *ollamaModelSection) Enabled() bool { return !(s.disabled != nil && s.disabled["Ollama"]) }
func (s *ollamaModelSection) Render(w int) string {
	return RenderOllamaPS(s.snapshot.OllamaPS, w, s.disabled)
}

type memorySection struct {
	snapshot Snapshot
	disabled map[string]bool
}

func (s *memorySection) Name() string  { return "Memory" }
func (s *memorySection) Enabled() bool { return !(s.disabled != nil && s.disabled["Memory"]) }
func (s *memorySection) Render(w int) string {
	return RenderMemoryCard(s.snapshot.Memory, s.snapshot.Swap, s.snapshot.RAMHistory, w)
}
