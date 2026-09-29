package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	netio "github.com/shirou/gopsutil/v3/net"
	"github.com/shirou/gopsutil/v3/process"
)

const defaultInterval = time.Second
const version = "dev"

// Global CLI config for collectors that need it.
var studioPort = 8888
var studioToken = ""

type optFloat struct {
	Value float64
	OK    bool
}

type snapshot struct {
	CollectedAt      time.Time
	CPU              cpuStats
	Memory           memoryStats
	Swap             swapMemoryStats
	Thermal          thermalStats
	GPUs             []gpuStats
	AMDGPUs          []amdGPUStats
	Disk             diskStats
	Net              netStats
	Inference        []inferenceProcess
	OllamaProcesses  []ollamaProcess
	OllamaPS         commandOutput
	UnslothStudio    unslothStudioStats
	GPUSparkline     map[string][]float64
	CPUHistory       []float64
	RAMHistory       []float64
	Warnings         []string
	CollectionMillis int64
}

type inferenceProcess struct {
	PID      int32
	Provider string
	Name     string
	VRAMMiB  optFloat
	GTTMiB   optFloat
}

type cpuStats struct {
	OK      bool
	Total   float64
	PerCore []float64
}

type memoryStats struct {
	OK        bool
	Used      uint64
	Total     uint64
	Available uint64
	Percent   float64
}

type swapMemoryStats struct {
	OK    bool
	Used  uint64
	Total uint64
}

type thermalStats struct {
	OK    bool
	Zone  []thermalZone
	Total optFloat
}

type thermalZone struct {
	Index       int
	Type        string
	Temperature optFloat
}

type diskStats struct {
	OK       bool
	Devices  []diskDeviceStats
	Warnings []string
	Rates    []diskDeviceRates
}

type diskDeviceStats struct {
	Name       string
	ReadBytes  uint64
	WriteBytes uint64
	ReadIOss   uint64
	WriteIOSS  uint64
}

type diskDeviceRates struct {
	Name     string
	ReadBps  float64
	WriteBps float64
}

type netStats struct {
	OK      bool
	Devices []netDeviceStats
	Rates   []netDeviceRates
}

type netDeviceStats struct {
	Name        string
	BytesSent   uint64
	BytesRecv   uint64
	PacketsSent uint64
	PacketsRecv uint64
}

type netDeviceRates struct {
	Name         string
	BytesSentBps float64
	BytesRecvBps float64
}

type gpuStats struct {
	Index       string
	UUID        string
	Name        string
	UtilPercent optFloat
	MemoryUsed  optFloat
	MemoryTotal optFloat
	Temperature optFloat
	PowerDraw   optFloat
	PowerLimit  optFloat
	FanPercent  optFloat
	Processes   []gpuProcess
	UtilTrend   string
	UtilDelta   float64
	VRAMTrend   string
	VRAMDelta   float64
}

type gpuProcess struct {
	GPUUUID      string
	PID          int32
	Provider     string
	Name         string
	UsedMemoryMB optFloat
}

type amdGPUStats struct {
	Index       string
	PCI         string
	Name        string
	UtilPercent optFloat
	MemoryUsed  optFloat
	MemoryTotal optFloat
	Temperature optFloat
	PowerDraw   optFloat
	FanPercent  optFloat
	Processes   []amdGPUProcess
	UtilTrend   string
	UtilDelta   float64
}

type amdGPUProcess struct {
	PID      int32
	Provider string
	Name     string
	VRAMMiB  optFloat
	GTTMiB   optFloat
}

type ollamaProcess struct {
	PID           int32
	Name          string
	Command       string
	CPUPercent    float64
	MemoryPercent float64
	RSS           uint64
	Runtime       time.Duration
	RuntimeOK     bool
}

type commandOutput struct {
	Output  string
	Error   string
	Missing bool
	Models  []ollamaModel
}

type ollamaModel struct {
	Name         string
	ID           string
	Size         string
	Processor    string
	Context      string
	Until        string
	PromptTokens int
	CtxTokens    int
}

type unslothStudioStats struct {
	Connected           bool
	ActiveModel         string
	ModelIdentifier     string
	GGUFVariant         string
	IsVision            bool
	IsAudio             bool
	SupportsReasoning   bool
	LoadedModels        []string
	LoadingModels       []string
	ContextLength       int
	MaxContextLength    int
	NativeContextLength int
	CurrentContext      int
	ConcurrentSessions  int
	TokensPerSecond     optFloat
	SpeculativeType     string
	TensorParallel      bool
	TrainStatus         string
	TrainStep           int
	TrainLoss           float64
	TrainLr             float64
	LoadPhase           string
	LoadBytes           int64
	LoadTotal           int64
	Error               string
}

// classifyProvider maps a process name and optional cmdline to a specific
// provider label. Returns "" for unrecognized names (rendered as "other").
var prevSnapshot snapshot
var prevDiskIO map[string]diskDeviceStats
var prevNetIO map[string]netDeviceStats
var CurrentTheme Theme = themes["dark"]

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func classifyProvider(name string, cmdline string) string {
	lower := strings.ToLower(strings.TrimSpace(name))
	cmdLower := strings.ToLower(strings.TrimSpace(cmdline))

	text := lower + " " + cmdLower

	switch {
	case strings.Contains(text, "python") && (strings.Contains(text, "transform") || strings.Contains(text, "torch") || strings.Contains(text, "pytorch") || strings.Contains(text, "huggingface") || strings.Contains(text, "llama") || strings.Contains(text, "bert") || strings.Contains(text, "gpt")):
		return "PyTorch"
	case strings.Contains(text, "python") && (strings.Contains(text, "ollama")):
		return "Ollama"
	case strings.Contains(text, "python") && (strings.Contains(text, "unsloth")):
		return "Unsloth"
	case strings.Contains(text, "python") && (strings.Contains(text, "text-generation") || strings.Contains(text, "vllm")):
		return "vLLM"
	case strings.Contains(text, "python") && (strings.Contains(text, "kobold")):
		return "KoboldCPP"
	case strings.Contains(text, "ollama") && !strings.Contains(text, "python"):
		return "Ollama"
	case strings.Contains(text, "unsloth"):
		return "Unsloth"
	case strings.Contains(text, "vllm") || strings.Contains(text, "text-generation"):
		return "vLLM"
	case strings.Contains(text, "kobold"):
		return "KoboldCPP"
	case strings.Contains(text, "xtt") || strings.Contains(text, "voyager") || strings.Contains(text, "sglm") || strings.Contains(text, "llama.cpp") || strings.Contains(text, "llama-srv"):
		return "Local Inference"
	default:
		return ""
	}
}

type tickMsg time.Time

type metricsMsg struct {
	Snapshot snapshot
}

type keyMap struct {
	Quit    key.Binding
	Refresh key.Binding
}

var keys = keyMap{
	Quit: key.NewBinding(
		key.WithKeys("q", "ctrl+c"),
		key.WithHelp("q/ctrl+c", "quit"),
	),
	Refresh: key.NewBinding(
		key.WithKeys("r"),
		key.WithHelp("r", "refresh"),
	),
}

type config struct {
	Interval     time.Duration
	GPUFilter    int
	Theme        string
	NoSparklines bool
	ShowNotices  bool
}

type model struct {
	interval         time.Duration
	width            int
	height           int
	ready            bool
	loading          bool
	viewport         viewport.Model
	snapshot         snapshot
	showHelp         bool
	disabledSections map[string]bool
	cfg              config
	spinnerIndex     int
	flashActive      bool
	flashEnd         int64
	notification     string
	notificationEnd  int64
}

var (
	accentColor = lipgloss.Color("#7DFFB3")
	mutedColor  = lipgloss.Color("#88909C")
	warnColor   = lipgloss.Color("#FFD166")
	dangerColor = lipgloss.Color("#FF6B6B")
	panelColor  = lipgloss.Color("#323846")
	textColor   = lipgloss.Color("#F3F5F7")

	headerStyleTemplate = lipgloss.NewStyle().
				Padding(0, 1).
				Border(lipgloss.NormalBorder(), false, false, true, false).
				BorderForeground(panelColor)
	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	mutedStyle = lipgloss.NewStyle().Foreground(mutedColor)
	valueStyle = lipgloss.NewStyle().Bold(true).Foreground(textColor)
	warnStyle  = lipgloss.NewStyle().Foreground(warnColor)
	cardStyle  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(panelColor).
			Padding(0, 1)
	sectionTitleStyle   = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	labelStyle          = lipgloss.NewStyle().Foreground(mutedColor)
	footerStyle         = lipgloss.NewStyle().Foreground(mutedColor).Padding(0, 1)
	barOKStyle          = lipgloss.NewStyle().Foreground(accentColor)
	barWarnStyle        = lipgloss.NewStyle().Foreground(warnColor)
	barDangerStyle      = lipgloss.NewStyle().Foreground(dangerColor)
	barEmptyStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("#3A3F4B"))
	sevOKStyle          = lipgloss.NewStyle().Foreground(accentColor)
	sevWarnColorStyle   = lipgloss.NewStyle().Foreground(warnColor)
	sevDangerColorStyle = lipgloss.NewStyle().Foreground(dangerColor)
)

func main() {
	interval := flag.Duration("interval", defaultInterval, "refresh interval such as 500ms, 1s, or 2s")
	showVersion := flag.Bool("version", false, "print version and exit")
	exportTarget := flag.String("export", "", "export target: stdout, json, csv, or file path")
	onceMode := flag.Bool("once", false, "single snapshot and exit")
	themeFlag := flag.String("theme", "", "theme to use (dark, midnight, monokai)")
	listThemesFlag := flag.Bool("list-themes", false, "list available themes")
	flag.IntVar(&studioPort, "unsloth-port", 8888, "Unsloth Studio API port")
	flag.StringVar(&studioToken, "unsloth-token", "", "Unsloth Studio API Bearer token (or set UNSLOTH_STUDIO_TOKEN)")
	flag.Parse()

	if *listThemesFlag {
		avail := listThemes()
		fmt.Println("Available themes:", strings.Join(avail, ", "))
		return
	}

	if *showVersion {
		fmt.Println("ontop " + version)
		return
	}

	if *themeFlag != "" {
		if t, ok := themes[*themeFlag]; ok {
			applyTheme(t)
			CurrentTheme = t
		} else {
			fmt.Fprintf(os.Stderr, "unknown theme: %s (available: %v)\n", *themeFlag, listThemes())
			os.Exit(1)
		}
	}

	if envTheme := os.Getenv("ONTOP_THEME"); envTheme != "" {
		if t, ok := themes[envTheme]; ok {
			applyTheme(t)
			CurrentTheme = t
		}
	}

	// Support env var override for Unsloth Studio URL before import.
	_ = os.Getenv("UNSLOTH_STUDIO_URL")

	envInterval := os.Getenv("ONTOP_INTERVAL")
	if envInterval != "" {
		if parsed, err := time.ParseDuration(envInterval); err == nil {
			interval = &parsed
		}
	}

	// Support env var as fallback for token.
	if studioToken == "" {
		studioToken = os.Getenv("UNSLOTH_STUDIO_TOKEN")
	}

	if *interval <= 0 {
		fmt.Fprintln(os.Stderr, "interval must be greater than zero")
		os.Exit(2)
	}

	if *exportTarget != "" || *onceMode {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		s := collectMetrics(ctx)

		if *onceMode && len(s.GPUs) > 0 {
			// Second collection to populate sparklines with real delta data.
			<-time.After(500 * time.Millisecond)
			s = collectMetrics(ctx)
		}

		switch *exportTarget {
		case "json":
			bytes, err := json.MarshalIndent(s, "", "  ")
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to marshal JSON: %v\n", err)
				os.Exit(1)
			}
			fmt.Println(string(bytes))
		case "csv":
			writeCSV(s)
			if *onceMode || *exportTarget == "" {
				if *onceMode && *exportTarget != "" {
					return
				}
			}
		default:
			if len(*exportTarget) > 0 && *exportTarget != "stdout" {
				f, err := os.Create(*exportTarget)
				if err != nil {
					fmt.Fprintf(os.Stderr, "failed to create output file: %v\n", err)
					os.Exit(1)
				}
				defer f.Close()
				_, _ = f.WriteString(renderContent(s, 200, nil))
				fmt.Printf("exported to %s\n", *exportTarget)
				return
			}
			fmt.Println(renderContent(s, 200, nil))
		}
		return
	}

	p := tea.NewProgram(newModel(*interval, config{}), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to run dashboard: %v\n", err)
		os.Exit(1)
	}
}

func writeCSV(s snapshot) {
	w := csv.NewWriter(os.Stdout)
	defer w.Flush()

	w.Write([]string{"timestamp", "cpu_total", "cpu_per_core", "ram_used", "ram_total", "ram_percent", "swap_used", "swap_total"})

	cores := make([]string, len(s.CPU.PerCore))
	for i, v := range s.CPU.PerCore {
		cores[i] = fmt.Sprintf("%.1f%%", v)
	}
	w.Write([]string{
		s.CollectedAt.Format(time.RFC3339),
		fmt.Sprintf("%.1f", s.CPU.Total),
		strings.Join(cores, ";"),
		fmt.Sprintf("%d", s.Memory.Used),
		fmt.Sprintf("%d", s.Memory.Total),
		fmt.Sprintf("%.1f", s.Memory.Percent),
		fmt.Sprintf("%d", s.Swap.Used),
		fmt.Sprintf("%d", s.Swap.Total),
	})
}

func newModel(interval time.Duration, cfg config) model {
	vp := viewport.New(100, 24)
	vp.Style = lipgloss.NewStyle()

	return model{
		interval:         interval,
		loading:          true,
		viewport:         vp,
		disabledSections: make(map[string]bool),
		cfg:              cfg,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(collectMetricsCmd(), tickCmd(m.interval))
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.showHelp {
			m.showHelp = false
			return m, nil
		}
		switch {
		case key.Matches(msg, keys.Quit):
			return m, tea.Quit
		case key.Matches(msg, keys.Refresh):
			if !m.loading {
				m.viewport.GotoTop()
				m.loading = true
				cmds = append(cmds, collectMetricsCmd())
			}
		}
		if msg.Type == tea.KeyRunes && len(msg.Runes) > 0 {
			switch msg.Runes[0] {
			case 'g':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["GPU"]
				m.disabledSections["GPU"] = enabled
				m.notification = fmt.Sprintf("GPU section %s", mapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 's':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["System"]
				m.disabledSections["System"] = enabled
				m.notification = fmt.Sprintf("System section %s", mapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'm':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Memory"]
				m.disabledSections["Memory"] = enabled
				m.notification = fmt.Sprintf("Memory section %s", mapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'u':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Unsloth"]
				m.disabledSections["Unsloth"] = enabled
				m.notification = fmt.Sprintf("Unsloth section %s", mapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'o':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Ollama"]
				m.disabledSections["Ollama"] = enabled
				m.notification = fmt.Sprintf("Ollama section %s", mapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case '?', 'h':
				m.showHelp = !m.showHelp
			}
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true
		m.flashActive = true
		m.flashEnd = time.Now().Add(300 * time.Millisecond).UnixNano()
		m.updateViewport()
	case metricsMsg:
		m.snapshot = msg.Snapshot
		m.loading = false
		m.updateViewport()
	case tickMsg:
		m.spinnerIndex = (m.spinnerIndex + 1) % len(spinnerFrames)
		if time.Now().UnixNano() >= m.notificationEnd {
			m.notification = ""
		}
		cmds = append(cmds, tickCmd(m.interval))
		if !m.loading {
			m.loading = true
			cmds = append(cmds, collectMetricsCmd())
		}
	}

	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

func (m model) View() string {
	width := m.width
	if width <= 0 {
		width = 100
	}

	if !m.ready {
		return "Collecting initial metrics..."
	}

	if m.showHelp {
		return renderHelpOverlay(width, m.disabledSections)
	}

	flash := m.flashActive
	if flash && time.Now().UnixNano() >= m.flashEnd {
		m.flashActive = false
		flash = false
	}

	frame := ""
	if m.loading {
		frame = spinnerFrames[m.spinnerIndex]
	}
	header := renderHeader(m.snapshot, m.interval, m.loading, flash, frame, m.width)
	footer := renderFooter(width, m.disabledSections)
	vp := m.viewport
	vp.Height = maxInt(1, m.height-lipgloss.Height(header)-lipgloss.Height(footer)-1)

	var notif string
	if m.notification != "" && time.Now().UnixNano() < m.notificationEnd {
		notif = "\n" + lipgloss.NewStyle().Foreground(accentColor).Render(m.notification)
	} else if m.notification != "" {
		m.notification = ""
	}

	content := header + "\n" + vp.View() + "\n" + footer
	if notif != "" {
		content = content + "\n" + notif
	}
	return content
}

func (m *model) updateViewport() {
	if !m.ready {
		return
	}

	m.viewport.Width = maxInt(20, m.width)
	m.viewport.Height = maxInt(1, m.height-5)
	m.viewport.SetContent(renderContent(m.snapshot, m.viewport.Width, m.disabledSections))
}

func tickCmd(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func collectMetricsCmd() tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
		defer cancel()

		return metricsMsg{Snapshot: collectMetrics(ctx)}
	}
}

func collectMetrics(ctx context.Context) snapshot {
	started := time.Now()
	s := snapshot{CollectedAt: started}

	prevDisk := make(map[string]diskDeviceStats)
	if prevDiskIO != nil {
		prevDisk = make(map[string]diskDeviceStats, len(prevDiskIO))
		for k, v := range prevDiskIO {
			prevDisk[k] = v
		}
	}
	prevNet := make(map[string]netDeviceStats)
	if prevNetIO != nil {
		prevNet = make(map[string]netDeviceStats, len(prevNetIO))
		for k, v := range prevNetIO {
			prevNet[k] = v
		}
	}

	prevCollectedAt := prevSnapshot.CollectedAt
	var elapsed float64
	if !prevCollectedAt.IsZero() {
		elapsed = s.CollectedAt.Sub(prevCollectedAt).Seconds()
	}

	s.CPU = collectCPU(ctx, &s.Warnings)
	s.Memory = collectMemory(ctx, &s.Warnings)
	s.Swap = collectSwap(ctx, &s.Warnings)
	s.Thermal = collectThermal(ctx, &s.Warnings)
	s.Disk = collectDisk(ctx, &s.Warnings)
	s.Net = collectNet(ctx, &s.Warnings)
	s.GPUs = collectNVIDIA(ctx, &s.Warnings, prevSnapshot.GPUs)
	s.AMDGPUs = collectAMD(ctx, &s.Warnings, prevSnapshot.AMDGPUs)
	s.GPUSparkline = mergeGPUSparkline(prevSnapshot.GPUSparkline, s.GPUs)

	if elapsed > 0.5 {
		for _, curr := range s.Disk.Devices {
			if prev, ok := prevDisk[curr.Name]; ok {
				dt := float64(curr.ReadBytes-prev.ReadBytes) / elapsed
				wt := float64(curr.WriteBytes-prev.WriteBytes) / elapsed
				s.Disk.Rates = append(s.Disk.Rates, diskDeviceRates{
					Name:     curr.Name,
					ReadBps:  math.Max(dt, 0),
					WriteBps: math.Max(wt, 0),
				})
			}
		}
		for _, curr := range s.Net.Devices {
			if prev, ok := prevNet[curr.Name]; ok {
				st := float64(curr.BytesSent-prev.BytesSent) / elapsed
				rt := float64(curr.BytesRecv-prev.BytesRecv) / elapsed
				s.Net.Rates = append(s.Net.Rates, netDeviceRates{
					Name:         curr.Name,
					BytesSentBps: math.Max(st, 0),
					BytesRecvBps: math.Max(rt, 0),
				})
			}
		}
	}

	diskForPrev := make([]diskDeviceStats, len(s.Disk.Devices))
	copy(diskForPrev, s.Disk.Devices)
	netForPrev := make([]netDeviceStats, len(s.Net.Devices))
	copy(netForPrev, s.Net.Devices)
	prevDiskIO = make(map[string]diskDeviceStats)
	for _, d := range diskForPrev {
		prevDiskIO[d.Name] = d
	}
	prevNetIO = make(map[string]netDeviceStats)
	for _, d := range netForPrev {
		prevNetIO[d.Name] = d
	}

	var historyCPU []float64
	if len(prevSnapshot.CPUHistory) > 0 {
		historyCPU = make([]float64, len(prevSnapshot.CPUHistory))
		copy(historyCPU, prevSnapshot.CPUHistory)
	}
	if s.CPU.OK && len(s.CPU.PerCore) > 0 {
		total := s.CPU.Total
		history := historyCPU
		if len(history) == 0 {
			history = append(history, 0)
		}
		history = append(history, total)
		if len(history) > 30 {
			history = history[1:]
		}
		s.CPUHistory = history
	}

	var historyRAM []float64
	if len(prevSnapshot.RAMHistory) > 0 {
		historyRAM = make([]float64, len(prevSnapshot.RAMHistory))
		copy(historyRAM, prevSnapshot.RAMHistory)
	}
	if s.Memory.OK {
		percent := s.Memory.Percent
		history := historyRAM
		if len(history) == 0 {
			history = append(history, 0)
		}
		history = append(history, percent)
		if len(history) > 30 {
			history = history[1:]
		}
		s.RAMHistory = history
	}

	s.Inference = collectInference(ctx, s.GPUs, s.AMDGPUs, &s.Warnings)
	s.OllamaProcesses = collectOllamaProcesses(ctx, &s.Warnings)
	s.OllamaPS = collectOllamaPS(ctx)
	s.UnslothStudio = collectUnslothStudio(ctx, &s.Warnings)
	s.CollectionMillis = time.Since(started).Milliseconds()

	prevSnapshot = s

	return s
}

func collectCPU(ctx context.Context, warnings *[]string) cpuStats {
	stats := cpuStats{}
	total, err := cpu.PercentWithContext(ctx, 0, false)
	if err != nil {
		addWarning(warnings, "CPU metrics unavailable: "+cleanError(err.Error()))
		return stats
	}
	if len(total) > 0 {
		stats.Total = total[0]
		stats.OK = true
	}

	perCore, err := cpu.PercentWithContext(ctx, 0, true)
	if err != nil {
		addWarning(warnings, "per-core CPU metrics unavailable: "+cleanError(err.Error()))
		return stats
	}
	stats.PerCore = perCore

	return stats
}

func collectMemory(ctx context.Context, warnings *[]string) memoryStats {
	vm, err := mem.VirtualMemoryWithContext(ctx)
	if err != nil {
		addWarning(warnings, "RAM metrics unavailable: "+cleanError(err.Error()))
		return memoryStats{}
	}

	return memoryStats{
		OK:        true,
		Used:      vm.Used,
		Total:     vm.Total,
		Available: vm.Available,
		Percent:   vm.UsedPercent,
	}
}

func collectSwap(ctx context.Context, warnings *[]string) swapMemoryStats {
	vm, err := mem.SwapMemoryWithContext(ctx)
	if err != nil {
		addWarning(warnings, "swap metrics unavailable: "+cleanError(err.Error()))
		return swapMemoryStats{}
	}

	return swapMemoryStats{
		OK:    true,
		Used:  vm.Used,
		Total: vm.Total,
	}
}

func collectThermal(ctx context.Context, warnings *[]string) thermalStats {
	stats := thermalStats{}
	thermalPath := "/sys/class/thermal"

	dir, err := os.Open(thermalPath)
	if err != nil {
		addWarning(warnings, "thermal sensors unavailable")
		return stats
	}
	defer dir.Close()

	entries, err := dir.ReadDir(-1)
	if err != nil || len(entries) == 0 {
		addWarning(warnings, "thermal sensors unavailable")
		return stats
	}

	stats.OK = true
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		zonePath := thermalPath + "/" + entry.Name()
		typeFile := zonePath + "/type"
		tempFile := zonePath + "/temp"

		typeData, err := os.ReadFile(typeFile)
		if err != nil {
			continue
		}
		zoneType := strings.TrimSpace(string(typeData))
		if zoneType == "" {
			continue
		}

		tempData, err := os.ReadFile(tempFile)
		if err != nil {
			continue
		}
		var tempMilli float64
		_, err = fmt.Sscanf(string(tempData), "%f", &tempMilli)
		if err != nil {
			continue
		}

		tempCelsius := tempMilli / 1000.0
		stats.Zone = append(stats.Zone, thermalZone{
			Index:       len(stats.Zone),
			Type:        fmt.Sprintf("%s (%s)", zoneType, entry.Name()),
			Temperature: optFloat{Value: tempCelsius, OK: true},
		})
		if tempCelsius > stats.Total.Value {
			stats.Total = optFloat{Value: tempCelsius, OK: true}
		}
	}

	if len(stats.Zone) == 0 {
		stats.OK = false
	}
	return stats
}

func collectDisk(ctx context.Context, warnings *[]string) diskStats {
	stats := diskStats{}
	devices, err := disk.IOCountersWithContext(ctx)
	if err != nil {
		addWarning(warnings, "disk I/O metrics unavailable: "+cleanError(err.Error()))
		return stats
	}
	stats.OK = true
	for name, dev := range devices {
		stats.Devices = append(stats.Devices, diskDeviceStats{
			Name:       name,
			ReadBytes:  dev.ReadBytes,
			WriteBytes: dev.WriteBytes,
			ReadIOss:   dev.ReadCount,
			WriteIOSS:  dev.WriteCount,
		})
	}
	if len(stats.Devices) == 0 {
		stats.OK = false
	}
	return stats
}

func collectNet(ctx context.Context, warnings *[]string) netStats {
	stats := netStats{}
	devices, err := netio.IOCountersWithContext(ctx, false)
	if err != nil {
		addWarning(warnings, "network I/O metrics unavailable: "+cleanError(err.Error()))
		return stats
	}
	stats.OK = true
	for _, dev := range devices {
		stats.Devices = append(stats.Devices, netDeviceStats{
			Name:        dev.Name,
			BytesSent:   dev.BytesSent,
			BytesRecv:   dev.BytesRecv,
			PacketsSent: dev.PacketsSent,
			PacketsRecv: dev.PacketsRecv,
		})
	}
	return stats
}

func collectNVIDIA(ctx context.Context, warnings *[]string, prevGpus []gpuStats) []gpuStats {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return nil
	}

	fields := "index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed"
	stdout, stderr, err := runCommand(ctx, "nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		addWarning(warnings, "nvidia-smi GPU query failed: "+cleanCommandError(err, stderr))
		return nil
	}

	gpus, err := parseGPUCSV(stdout)
	if err != nil {
		addWarning(warnings, "could not parse nvidia-smi GPU metrics: "+cleanError(err.Error()))
		return nil
	}

	processes, processWarning := collectGPUProcesses(ctx)
	if processWarning != "" {
		addWarning(warnings, processWarning)
	}

	byUUID := make(map[string]int, len(gpus))
	for i := range gpus {
		byUUID[gpus[i].UUID] = i
	}
	for _, p := range processes {
		if idx, ok := byUUID[p.GPUUUID]; ok {
			gpus[idx].Processes = append(gpus[idx].Processes, p)
		}
	}

	addGPUUtilDelta(prevGpus, gpus)
	addVRAMDeltas(prevGpus, gpus)

	return gpus
}

func collectAMD(ctx context.Context, warnings *[]string, prevGpus []amdGPUStats) []amdGPUStats {
	if _, err := exec.LookPath("amdgpu_top"); err != nil {
		stats := collectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			addAMDUtilDelta(nil, stats)
		}
		return stats
	}

	stdout, stderr, err := runCommand(ctx, "amdgpu_top", "--json", "--no-pc", "-n", "1", "-s", "100")
	if err != nil {
		addWarning(warnings, "amdgpu_top query failed: "+cleanCommandError(err, stderr))
		stats := collectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			addAMDUtilDelta(prevGpus, stats)
		}
		return stats
	}

	gpus, err := parseAMDJSON(stdout)
	if err != nil {
		addWarning(warnings, "could not parse amdgpu_top metrics: "+cleanError(err.Error()))
		stats := collectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			addAMDUtilDelta(prevGpus, stats)
		}
		return stats
	}
	addAMDUtilDelta(prevGpus, gpus)
	return gpus
}

// buildInferenceProcessList correlates GPU processes from already-collected
// NVIDIA and AMD data into a unified sorted inference process list.
func buildInferenceProcessList(nvidia []gpuStats, amd []amdGPUStats) []inferenceProcess {
	inference := make([]inferenceProcess, 0)

	for _, gpu := range nvidia {
		for _, p := range gpu.Processes {
			if !p.UsedMemoryMB.OK {
				continue
			}
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			inference = append(inference, inferenceProcess{
				PID:      p.PID,
				Provider: p.Provider,
				Name:     name,
				VRAMMiB:  p.UsedMemoryMB,
				GTTMiB:   optFloat{},
			})
		}
	}

	for _, gpu := range amd {
		for _, p := range gpu.Processes {
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			inference = append(inference, inferenceProcess{
				PID:      p.PID,
				Provider: p.Provider,
				Name:     name,
				VRAMMiB:  p.VRAMMiB,
				GTTMiB:   p.GTTMiB,
			})
		}
	}

	sort.SliceStable(inference, func(i, j int) bool {
		if inference[i].VRAMMiB.OK != inference[j].VRAMMiB.OK {
			return inference[i].VRAMMiB.OK
		}
		return inference[i].VRAMMiB.Value > inference[j].VRAMMiB.Value
	})

	return inference
}

func collectInference(ctx context.Context, nvidia []gpuStats, amd []amdGPUStats, warnings *[]string) []inferenceProcess {
	return buildInferenceProcessList(nvidia, amd)
}

func parseGPUCSV(output string) ([]gpuStats, error) {
	records, err := readCSV(output)
	if err != nil {
		return nil, err
	}

	gpus := make([]gpuStats, 0, len(records))
	for _, row := range records {
		if len(row) < 10 {
			return nil, fmt.Errorf("expected 10 GPU fields, got %d", len(row))
		}

		gpus = append(gpus, gpuStats{
			Index:       strings.TrimSpace(row[0]),
			Name:        strings.TrimSpace(row[1]),
			UUID:        strings.TrimSpace(row[2]),
			UtilPercent: parseOptFloat(row[3]),
			MemoryUsed:  parseOptFloat(row[4]),
			MemoryTotal: parseOptFloat(row[5]),
			Temperature: parseOptFloat(row[6]),
			PowerDraw:   parseOptFloat(row[7]),
			PowerLimit:  parseOptFloat(row[8]),
			FanPercent:  parseOptFloat(row[9]),
		})
	}

	return gpus, nil
}

type amdTopDocument struct {
	Devices []amdTopDevice `json:"devices"`
}

type amdTopDevice struct {
	Info struct {
		DeviceName string `json:"DeviceName"`
		DevicePath struct {
			PCI string `json:"pci"`
		} `json:"DevicePath"`
	} `json:"Info"`
	GPUActivity map[string]json.RawMessage `json:"gpu_activity"`
	Sensors     map[string]json.RawMessage `json:"Sensors"`
	VRAM        map[string]json.RawMessage `json:"VRAM"`
	FDInfo      map[string]json.RawMessage `json:"fdinfo"`
}

func parseAMDJSON(output string) ([]amdGPUStats, error) {
	var document amdTopDocument
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &document); err != nil {
		return nil, err
	}

	gpus := make([]amdGPUStats, 0, len(document.Devices))
	for i, device := range document.Devices {
		gpu := amdGPUStats{
			Index:       strconv.Itoa(i),
			PCI:         device.Info.DevicePath.PCI,
			Name:        device.Info.DeviceName,
			UtilPercent: amdMetric(device.GPUActivity, "GFX"),
			MemoryUsed:  amdMetric(device.VRAM, "Total VRAM Usage"),
			MemoryTotal: amdMetric(device.VRAM, "Total VRAM"),
			Temperature: firstAMDMetric(device.Sensors, "Junction Temperature", "Edge Temperature"),
			PowerDraw:   firstAMDMetric(device.Sensors, "Average Power", "GFX Power", "Input Power"),
			FanPercent:  firstAMDMetric(device.Sensors, "Fan Speed", "Fan"),
			Processes:   parseAMDProcesses(device.FDInfo),
		}
		if gpu.Name == "" {
			gpu.Name = "AMD GPU"
		}
		gpus = append(gpus, gpu)
	}

	return gpus, nil
}

func parseAMDProcesses(raw map[string]json.RawMessage) []amdGPUProcess {
	processes := make([]amdGPUProcess, 0, len(raw))
	for pidString, processRaw := range raw {
		pid, err := strconv.ParseInt(pidString, 10, 32)
		if err != nil {
			continue
		}

		name, vram, gtt := parseAMDProcess(processRaw)
		processes = append(processes, amdGPUProcess{
			PID:      int32(pid),
			Provider: classifyProvider(name, ""),
			Name:     name,
			VRAMMiB:  vram,
			GTTMiB:   gtt,
		})
	}

	sort.Slice(processes, func(i, j int) bool {
		return processes[i].PID < processes[j].PID
	})
	return processes
}

func parseAMDProcess(raw json.RawMessage) (string, optFloat, optFloat) {
	var current map[string]json.RawMessage
	if json.Unmarshal(raw, &current) != nil {
		return "", optFloat{}, optFloat{}
	}

	name := ""
	if value, ok := current["name"]; ok {
		_ = json.Unmarshal(value, &name)
	}

	for depth := 0; depth < 4; depth++ {
		if value, ok := current["VRAM"]; ok {
			return name, amdMetricValue(value), amdNestedMetric(current, "GTT")
		}
		value, ok := current["usage"]
		if !ok || json.Unmarshal(value, &current) != nil {
			break
		}
	}

	return name, optFloat{}, optFloat{}
}

func amdNestedMetric(metrics map[string]json.RawMessage, key string) optFloat {
	value, ok := metrics[key]
	if !ok {
		return optFloat{}
	}
	return amdMetricValue(value)
}

func amdMetric(metrics map[string]json.RawMessage, key string) optFloat {
	value, ok := metrics[key]
	if !ok {
		return optFloat{}
	}
	return amdMetricValue(value)
}

func firstAMDMetric(metrics map[string]json.RawMessage, keys ...string) optFloat {
	for _, key := range keys {
		if value := amdMetric(metrics, key); value.OK {
			return value
		}
	}
	return optFloat{}
}

func amdMetricValue(raw json.RawMessage) optFloat {
	var value struct {
		Value *float64 `json:"value"`
	}
	if json.Unmarshal(raw, &value) == nil && value.Value != nil {
		return optFloat{Value: *value.Value, OK: true}
	}

	var number float64
	if json.Unmarshal(raw, &number) == nil {
		return optFloat{Value: number, OK: true}
	}
	return optFloat{}
}

// processCmdline reads the full command line for a PID from /proc.
func processCmdline(pid int32) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return ""
	}
	s := strings.ReplaceAll(string(data), "\x00", " ")
	return strings.TrimSpace(s)
}

// processContainerName resolves a PID to its container name if it's a container.
func processContainerName(pid int32) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.Contains(line, "docker") || strings.Contains(line, "podman") {
			for _, part := range strings.Split(line, " ") {
				for _, prefix := range []string{"docker/", "podman/"} {
					if strings.HasPrefix(part, prefix) {
						return part[len(prefix):]
					}
				}
			}
		}
	}
	return ""
}

func collectGPUProcesses(ctx context.Context) ([]gpuProcess, string) {
	fields := "gpu_uuid,pid,process_name,used_memory"
	stdout, stderr, err := runCommand(ctx, "nvidia-smi", "--query-compute-apps="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		text := strings.ToLower(stdout + " " + stderr + " " + err.Error())
		if strings.Contains(text, "no running") || strings.Contains(text, "not supported") {
			return nil, ""
		}
		return nil, "nvidia-smi process query failed: " + cleanCommandError(err, stderr)
	}

	records, err := readCSV(stdout)
	if err != nil {
		return nil, "could not parse nvidia-smi process metrics: " + cleanError(err.Error())
	}

	processes := make([]gpuProcess, 0, len(records))
	for _, row := range records {
		if len(row) < 4 {
			continue
		}

		pid64, err := strconv.ParseInt(strings.TrimSpace(row[1]), 10, 32)
		if err != nil {
			continue
		}

		processName := strings.TrimSpace(row[2])
		cmdline := processCmdline(int32(pid64))
		containerName := processContainerName(int32(pid64))
		displayName := processName
		if containerName != "" {
			displayName = containerName
		}
		processes = append(processes, gpuProcess{
			GPUUUID:      strings.TrimSpace(row[0]),
			PID:          int32(pid64),
			Provider:     classifyProvider(processName, cmdline),
			Name:         displayName,
			UsedMemoryMB: parseOptFloat(row[3]),
		})
	}

	sort.Slice(processes, func(i, j int) bool {
		if processes[i].GPUUUID == processes[j].GPUUUID {
			return processes[i].PID < processes[j].PID
		}
		return processes[i].GPUUUID < processes[j].GPUUUID
	})

	return processes, ""
}

func collectOllamaProcesses(ctx context.Context, warnings *[]string) []ollamaProcess {
	procs, err := process.ProcessesWithContext(ctx)
	if err != nil {
		addWarning(warnings, "process list unavailable: "+cleanError(err.Error()))
		return nil
	}

	now := time.Now()
	ollamaProcs := make([]ollamaProcess, 0)
	for _, p := range procs {
		name, _ := p.NameWithContext(ctx)
		cmdline, _ := p.CmdlineWithContext(ctx)
		identity := strings.ToLower(name + " " + cmdline)
		if !strings.Contains(identity, "ollama") {
			continue
		}

		item := ollamaProcess{
			PID:     p.Pid,
			Name:    strings.TrimSpace(name),
			Command: strings.TrimSpace(cmdline),
		}

		if item.Command == "" {
			item.Command = item.Name
		}

		if cpuPct, err := p.CPUPercentWithContext(ctx); err == nil {
			item.CPUPercent = cpuPct
		}
		if memPct, err := p.MemoryPercentWithContext(ctx); err == nil {
			item.MemoryPercent = float64(memPct)
		}
		if mi, err := p.MemoryInfoWithContext(ctx); err == nil && mi != nil {
			item.RSS = mi.RSS
		}
		if createMS, err := p.CreateTimeWithContext(ctx); err == nil && createMS > 0 {
			started := time.UnixMilli(createMS)
			if started.Before(now) {
				item.Runtime = now.Sub(started)
				item.RuntimeOK = true
			}
		}

		ollamaProcs = append(ollamaProcs, item)
	}

	sort.Slice(ollamaProcs, func(i, j int) bool {
		if ollamaProcs[i].CPUPercent == ollamaProcs[j].CPUPercent {
			return ollamaProcs[i].PID < ollamaProcs[j].PID
		}
		return ollamaProcs[i].CPUPercent > ollamaProcs[j].CPUPercent
	})

	return ollamaProcs
}

func collectOllamaPS(ctx context.Context) commandOutput {
	if _, err := exec.LookPath("ollama"); err != nil {
		return commandOutput{Missing: true, Error: "ollama command not found"}
	}

	stdout, stderr, err := runCommand(ctx, "ollama", "ps", "--json")
	if err == nil {
		output := strings.TrimSpace(stdout)
		models := parseOllamaPSJSON(output)
		return commandOutput{Output: output, Models: models}
	}

	textStdout, _, textErr := runCommand(ctx, "ollama", "ps")
	if textErr != nil {
		return commandOutput{Error: cleanCommandError(err, stderr)}
	}

	output := strings.TrimSpace(textStdout)
	return commandOutput{Output: output, Models: parseOllamaPS(output)}
}

func parseOllamaPSJSON(output string) []ollamaModel {
	var models []ollamaModel

	type ollamaLine struct {
		Name          string  `json:"name"`
		Id            string  `json:"id"`
		Size          float64 `json:"size"`
		SizeVRAM      float64 `json:"size_vram"`
		TransferRate  string  `json:"transfer_rate"`
		ContextLength int     `json:"context_length"`
		PromptTokens  int     `json:"prompt_tokens"`
		OutputTokens  int     `json:"output_tokens"`
	}

	var lines []ollamaLine
	if err := json.Unmarshal([]byte(output), &lines); err != nil {
		return nil
	}

	for _, l := range lines {
		sizeGB := l.Size / 1e9
		models = append(models, ollamaModel{
			Name:         l.Name,
			ID:           l.Id,
			Size:         fmt.Sprintf("%.1f GB", sizeGB),
			PromptTokens: l.PromptTokens,
			CtxTokens:    l.ContextLength,
		})
	}

	return models
}

func parseOllamaPS(output string) []ollamaModel {
	var models []ollamaModel
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 || strings.EqualFold(fields[0], "name") {
			continue
		}

		processorStart := -1
		contextIndex := -1
		for i := 2; i < len(fields); i++ {
			if strings.Contains(fields[i], "%") {
				processorStart = i
				break
			}
		}
		if processorStart < 0 {
			continue
		}
		for i := processorStart + 1; i < len(fields); i++ {
			contextField := strings.ToUpper(strings.TrimSuffix(fields[i], ","))
			if _, err := strconv.ParseInt(contextField, 10, 64); err == nil || strings.HasSuffix(contextField, "K") || strings.HasSuffix(contextField, "M") {
				contextIndex = i
				break
			}
		}
		if contextIndex < 0 {
			continue
		}

		models = append(models, ollamaModel{
			Name:      fields[0],
			ID:        fields[1],
			Size:      strings.Join(fields[2:processorStart], " "),
			Processor: strings.Join(fields[processorStart:contextIndex], " "),
			Context:   fields[contextIndex],
			Until:     strings.Join(fields[contextIndex+1:], " "),
		})
	}
	return models
}

func collectUnslothStudio(ctx context.Context, warnings *[]string) unslothStudioStats {
	base := discoverStudioBase(ctx)
	if base == "" {
		return unslothStudioStats{}
	}

	stats := unslothStudioStats{Connected: true}

	// If no token, we can still show connected status.
	if studioToken == "" {
		return stats
	}

	// Inference status.
	statusURL := base + "/api/inference/status"
	statusResp, err := studioHTTPGet(ctx, statusURL, studioToken)
	if err != nil {
		stats.Error = "inference status unavailable: " + err.Error()
	} else {
		defer statusResp.Body.Close()
		if statusResp.StatusCode == http.StatusOK {
			stats.parseInferenceStatus(statusResp.Body)
		} else {
			stats.Error = fmt.Sprintf("inference status %d", statusResp.StatusCode)
		}
	}

	// Training status.
	trainURL := base + "/api/train/status"
	trainResp, err := studioHTTPGet(ctx, trainURL, studioToken)
	if err != nil {
		// Non-fatal; training may simply not be running.
	} else {
		defer trainResp.Body.Close()
		if trainResp.StatusCode == http.StatusOK {
			stats.parseTrainStatus(trainResp.Body)
		}
	}

	// Load progress.
	loadURL := base + "/api/inference/load-progress"
	loadResp, err := studioHTTPGet(ctx, loadURL, studioToken)
	if err != nil {
		// Non-fatal.
	} else {
		defer loadResp.Body.Close()
		if loadResp.StatusCode == http.StatusOK {
			stats.parseLoadProgress(loadResp.Body)
		}
	}

	return stats
}

func discoverStudioBase(ctx context.Context) string {
	if configured := strings.TrimRight(strings.TrimSpace(os.Getenv("UNSLOTH_STUDIO_URL")), "/"); configured != "" {
		probeCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
		defer cancel()
		if studioHealthOK(probeCtx, configured) {
			return configured
		}
		return ""
	}

	if detected := detectUnslothProcesses(); detected != "" {
		return detected
	}

	ports := []int{studioPort}
	if studioPort == 8888 || studioPort == -1 {
		ports = []int{8888, 8000, 3000, 8080}
	}
	for _, port := range ports {
		base := fmt.Sprintf("http://127.0.0.1:%d", port)
		if studioHealthOK(ctx, base) {
			return base
		}
	}
	return ""
}

func detectUnslothProcesses() string {
	procs, err := process.Processes()
	if err != nil {
		return ""
	}
	for _, p := range procs {
		name, _ := p.Name()
		if strings.Contains(strings.ToLower(name), "unsloth") {
			cmdline, _ := p.Cmdline()
			for _, arg := range strings.Fields(cmdline) {
				if strings.HasPrefix(arg, "--port=") {
					port := strings.TrimPrefix(arg, "--port=")
					if portNum, err := strconv.Atoi(port); err == nil {
						return fmt.Sprintf("http://localhost:%d", portNum)
					}
				}
			}
			return "http://localhost:8888"
		}
	}
	return ""
}

func studioHealthOK(ctx context.Context, base string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer cancel()
	resp, err := studioHTTPGet(probeCtx, base+"/api/health", "")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (s *unslothStudioStats) parseInferenceStatus(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var b bool
	var num float64
	var arr []string

	if v, ok := raw["active_model"]; ok && json.Unmarshal(v, &str) == nil {
		s.ActiveModel = str
	}
	if v, ok := raw["model_identifier"]; ok && json.Unmarshal(v, &str) == nil {
		s.ModelIdentifier = str
	}
	if v, ok := raw["gguf_variant"]; ok && json.Unmarshal(v, &str) == nil {
		s.GGUFVariant = str
	}
	if v, ok := raw["is_vision"]; ok && json.Unmarshal(v, &b) == nil {
		s.IsVision = b
	}
	if v, ok := raw["is_audio"]; ok && json.Unmarshal(v, &b) == nil {
		s.IsAudio = b
	}
	if v, ok := raw["supports_reasoning"]; ok && json.Unmarshal(v, &b) == nil {
		s.SupportsReasoning = b
	}
	if v, ok := raw["loaded"]; ok && json.Unmarshal(v, &arr) == nil {
		s.LoadedModels = arr
	}
	if v, ok := raw["loading"]; ok && json.Unmarshal(v, &arr) == nil {
		s.LoadingModels = arr
	}
	if v, ok := raw["context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.ContextLength = int(num)
	}
	if v, ok := raw["max_context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.MaxContextLength = int(num)
	}
	if v, ok := raw["native_context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.NativeContextLength = int(num)
	}
	if num, ok := findJSONNumber(raw, "current_context", "current_context_length", "context_used", "context_tokens"); ok {
		s.CurrentContext = int(num)
	}
	if num, ok := findJSONNumber(raw, "concurrent_sessions", "active_sessions", "active_requests", "num_active_requests", "parallel_sessions", "parallel_requests"); ok {
		s.ConcurrentSessions = int(num)
	}
	if num, ok := findJSONNumber(raw, "tokens_per_second", "tokens_sec", "throughput", "generation_tokens_per_second"); ok {
		s.TokensPerSecond = optFloat{Value: num, OK: true}
	}
	if v, ok := raw["speculative_type"]; ok && json.Unmarshal(v, &str) == nil {
		s.SpeculativeType = str
	}
	if v, ok := raw["tensor_parallel"]; ok && json.Unmarshal(v, &b) == nil {
		s.TensorParallel = b
	}
}

func findJSONNumber(raw map[string]json.RawMessage, keys ...string) (float64, bool) {
	for _, key := range keys {
		if value, ok := raw[key]; ok {
			var number float64
			if json.Unmarshal(value, &number) == nil {
				return number, true
			}
		}
	}
	for _, containerKey := range []string{"metrics", "stats", "inference"} {
		value, ok := raw[containerKey]
		if !ok {
			continue
		}
		var nested map[string]json.RawMessage
		if json.Unmarshal(value, &nested) == nil {
			if number, ok := findJSONNumber(nested, keys...); ok {
				return number, true
			}
		}
	}
	return 0, false
}

func (s *unslothStudioStats) parseTrainStatus(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var num float64

	if v, ok := raw["status"]; ok && json.Unmarshal(v, &str) == nil {
		s.TrainStatus = str
	}
	if v, ok := raw["current_step"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainStep = int(num)
	}
	if v, ok := raw["current_loss"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainLoss = num
	}
	if v, ok := raw["current_lr"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainLr = num
	}
}

func (s *unslothStudioStats) parseLoadProgress(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var num float64

	if v, ok := raw["phase"]; ok && json.Unmarshal(v, &str) == nil {
		s.LoadPhase = str
	}
	if v, ok := raw["bytes_loaded"]; ok && json.Unmarshal(v, &num) == nil {
		s.LoadBytes = int64(num)
	}
	if v, ok := raw["bytes_total"]; ok && json.Unmarshal(v, &num) == nil {
		s.LoadTotal = int64(num)
	}
}

func studioHTTPGet(ctx context.Context, url string, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return http.DefaultClient.Do(req)
}

func runCommand(ctx context.Context, name string, args ...string) (string, string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func readCSV(output string) ([][]string, error) {
	output = strings.TrimSpace(output)
	if output == "" {
		return nil, nil
	}

	r := csv.NewReader(strings.NewReader(output))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1

	return r.ReadAll()
}

func parseOptFloat(raw string) optFloat {
	value := strings.TrimSpace(raw)
	lower := strings.ToLower(value)
	if value == "" || lower == "n/a" || lower == "na" || strings.Contains(lower, "not supported") || strings.Contains(lower, "not available") {
		return optFloat{}
	}

	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return optFloat{}
	}
	return optFloat{Value: parsed, OK: true}
}

func renderHeader(s snapshot, interval time.Duration, loading bool, flashActive bool, currentFrame string, width int) string {
	updated := "waiting for first sample"
	if !s.CollectedAt.IsZero() {
		updated = "updated " + s.CollectedAt.Format("15:04:05")
	}

	var statusStr string
	if loading {
		statusStr = lipgloss.NewStyle().Foreground(accentColor).Render(currentFrame + " collecting")
	} else {
		statusStr = lipgloss.NewStyle().Foreground(accentColor).Render("idle")
	}

	meta := fmt.Sprintf("%s | interval %s | %s", updated, trimDuration(interval), statusStr)
	if s.CollectionMillis > 0 {
		meta += fmt.Sprintf(" | collected in %dms", s.CollectionMillis)
	}

	header := lipgloss.JoinHorizontal(lipgloss.Center,
		titleStyle.Render("Ollama Machine Monitor"),
		"  ",
		mutedStyle.Render(meta),
	)

	borderCol := panelColor
	if flashActive {
		borderCol = accentColor
	}

	style := lipgloss.NewStyle().
		Padding(0, 1).
		Border(lipgloss.NormalBorder(), false, false, true, false).
		BorderForeground(borderCol)
	return style.Width(maxInt(20, width-2)).Render(header)
}

func renderFooter(width int, disabledSections map[string]bool) string {
	var toggles []string
	sections := []struct {
		key    string
		name   string
		active bool
	}{
		{"s", "System", !isEnabled(disabledSections, "System")},
		{"g", "GPU", !isEnabled(disabledSections, "GPU")},
		{"m", "Memory", !isEnabled(disabledSections, "Memory")},
		{"u", "Unsloth", !isEnabled(disabledSections, "Unsloth")},
		{"o", "Ollama", !isEnabled(disabledSections, "Ollama")},
	}
	enabledStyle := lipgloss.NewStyle().Foreground(accentColor)
	disabledStyle := lipgloss.NewStyle().Foreground(mutedColor)
	for _, sec := range sections {
		if sec.active {
			toggles = append(toggles, fmt.Sprintf("%s %s", sec.key, enabledStyle.Render(sec.name)))
		} else {
			toggles = append(toggles, fmt.Sprintf("%s %s", sec.key, disabledStyle.Render(sec.name)))
		}
	}
	toggleText := strings.Join(toggles, "  ")
	keysText := "q quit | ctrl+c quit | r refresh | up/down scroll | pgup/pgdn page | ? help"
	text := toggleText + "      " + keysText
	return footerStyle.Width(maxInt(20, width-2)).Render(fitText(text, maxInt(10, width-4)))
}

func isEnabled(disabledSections map[string]bool, name string) bool {
	if disabledSections == nil {
		return true
	}
	return !disabledSections[name]
}

func renderHelpOverlay(width int, disabledSections map[string]bool) string {
	lines := []string{
		"q / Ctrl+C   quit",
		"r            refresh",
		"s            toggle System (CPU)",
		"g            toggle GPU (NVIDIA/AMD)",
		"m            toggle Memory (RAM/Swap)",
		"u            toggle Unsloth",
		"o            toggle Ollama",
		"h/?          hide this help",
		"",
		"Navigation:",
		"up/down      scroll",
		"pgup/pgdn    page scroll",
	}
	body := strings.Join(lines, "\n")
	card1 := renderCard("Key Bindings", mutedStyle.Render(body), width)
	card2 := renderCard("Press any key to close", mutedStyle.Render(""), width)

	var stateLines []string
	sections := []struct {
		name   string
		active bool
	}{
		{"System", isEnabled(disabledSections, "System")},
		{"GPU", isEnabled(disabledSections, "GPU")},
		{"Memory", isEnabled(disabledSections, "Memory")},
		{"Unsloth", isEnabled(disabledSections, "Unsloth")},
		{"Ollama", isEnabled(disabledSections, "Ollama")},
	}
	enabledStyle := lipgloss.NewStyle().Foreground(accentColor)
	disabledStyle := lipgloss.NewStyle().Foreground(mutedColor)
	stateParts := make([]string, 0, len(sections))
	for _, sec := range sections {
		pair := sec.name + ": "
		if sec.active {
			pair += enabledStyle.Render("[✓]")
		} else {
			pair += disabledStyle.Render("[✗]")
		}
		stateParts = append(stateParts, pair)
	}
	stateLines = append(stateLines, "Sections:  [✓] = enabled  [✗] = disabled", "")
	stateLines = append(stateLines, strings.Join(stateParts, "  "))

	return card1 + "\n\n" + card2 + "\n\n" + renderCard("Section State", strings.Join(stateLines, "\n"), width)
}

func renderContent(s snapshot, width int, disabledSections map[string]bool) string {
	return renderContentSections(s, width, disabledSections)
}

func renderWarnings(warnings []string, width int) string {
	seen := make(map[string]bool, len(warnings))
	lines := make([]string, 0, len(warnings))
	for _, warning := range warnings {
		warning = strings.TrimSpace(warning)
		if warning == "" || seen[warning] {
			continue
		}
		seen[warning] = true
		lines = append(lines, warnStyle.Render("! ")+fitText(warning, maxInt(10, width-8)))
	}
	if len(lines) == 0 {
		return ""
	}
	return renderCard("Notices", strings.Join(lines, "\n"), width)
}

func renderSystem(s snapshot, width int, disabledSections map[string]bool) string {
	if disabledSections != nil && disabledSections["System"] {
		return ""
	}
	cpuBody := renderCPU(s.CPU, s.CPUHistory, width)

	var body string
	innerWidth := maxInt(20, width-6)
	thermBody := renderThermal(s.Thermal, width)
	hasThermal := s.Thermal.OK && len(s.Thermal.Zone) > 0 && thermBody != ""

	if innerWidth >= 50 && hasThermal {
		cpuCol := lipgloss.NewStyle().Width(innerWidth / 2).Render(sectionTitleStyle.Render("CPU") + "\n" + cpuBody)
		thermCol := lipgloss.NewStyle().Width(innerWidth / 2).Render(sectionTitleStyle.Render("Thermal") + "\n" + thermBody)
		body = lipgloss.JoinHorizontal(lipgloss.Top, cpuCol, thermCol)
	} else if hasThermal {
		body = sectionTitleStyle.Render("CPU") + "\n" + cpuBody + "\n\n" + sectionTitleStyle.Render("Thermal") + "\n" + thermBody
	} else {
		body = sectionTitleStyle.Render("CPU") + "\n" + cpuBody
	}
	return renderGroup("System", body, width)
}

func renderCPU(stats cpuStats, history []float64, width int) string {
	if !stats.OK {
		return mutedStyle.Render("CPU metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	lines := []string{
		metricLine("Total", stats.Total, fmt.Sprintf("%.1f%%", stats.Total), innerWidth),
	}

	if len(history) > 0 {
		spark := renderSparkline(history, minInt(len(history), innerWidth-10))
		lines = append(lines, mutedStyle.Render("trend: "+spark))
	}

	return strings.Join(lines, "\n")
}

func renderMemory(stats memoryStats, history []float64, width int) string {
	if !stats.OK {
		return mutedStyle.Render("RAM metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	used := fmt.Sprintf("%s / %s (available %s)", humanBytes(stats.Used), humanBytes(stats.Total), humanBytes(stats.Available))
	lines := []string{metricLine("RAM", stats.Percent, fmt.Sprintf("%.1f%%  %s", stats.Percent, used), innerWidth)}

	if len(history) > 0 {
		spark := renderSparkline(history, minInt(len(history), 20))
		lines = append(lines, mutedStyle.Render("trend: "+spark))
	}

	return strings.Join(lines, "\n")
}

func renderSwap(stats swapMemoryStats, width int) string {
	if !stats.OK {
		return mutedStyle.Render("Swap metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	used := fmt.Sprintf("%s / %s", humanBytes(stats.Used), humanBytes(stats.Total))
	percent := 0.0
	if stats.Total > 0 {
		percent = float64(stats.Used) / float64(stats.Total) * 100
	}
	return metricLine("Swap", percent, fmt.Sprintf("%.1f%%  %s", percent, used), innerWidth)
}

func renderMemoryCard(mem memoryStats, swap swapMemoryStats, history []float64, width int) string {
	if !mem.OK {
		return renderCard("Memory", mutedStyle.Render("RAM metrics unavailable."), width)
	}

	innerWidth := maxInt(20, width-6)
	lines := []string{metricLine("RAM", mem.Percent, fmt.Sprintf("free %s", humanBytes(mem.Available)), innerWidth)}

	if swap.OK && swap.Total > 0 {
		swapUsed := fmt.Sprintf("%s / %s", humanBytes(swap.Used), humanBytes(swap.Total))
		swapPercent := float64(swap.Used) / float64(swap.Total) * 100
		lines = append(lines, "")
		lines = append(lines, metricLine("Swap", swapPercent, fmt.Sprintf("%s", swapUsed), innerWidth))
	}
	if len(history) > 0 {
		lines = append(lines, "")
		lines = append(lines, mutedStyle.Render("trend: "+renderSparkline(history, minInt(len(history), maxInt(10, innerWidth-10)))))
	}

	return renderCard("Memory", strings.Join(lines, "\n"), width)
}

func renderThermal(stats thermalStats, width int) string {
	if !stats.Total.OK {
		return ""
	}

	innerWidth := maxInt(20, width-6)
	tempPercent := stats.Total.Value
	if tempPercent > 100 {
		tempPercent = 100.0
	}
	degrees := fmt.Sprintf("%.0f\u00b0C", stats.Total.Value)
	return metricLine("Temp", tempPercent, degrees, innerWidth)
}

func renderDiskCard(stats diskStats, width int, disabledSections map[string]bool) string {
	if !stats.OK {
		return ""
	}
	if disabledSections != nil && disabledSections["Disk"] {
		return ""
	}
	var text string
	if len(stats.Devices) == 0 {
		text = mutedStyle.Render("Disk I/O metrics unavailable.")
	} else {
		text = renderDisk(stats, width)
	}
	return renderCard("Disk I/O", text, width)
}

func renderDisk(stats diskStats, width int) string {
	innerWidth := maxInt(20, width-6)
	var lines []string

	barWidth := maxInt(10, innerWidth-70)

	for i, dev := range stats.Devices {
		readStr := humanBytes(dev.ReadBytes)
		writeStr := humanBytes(dev.WriteBytes)

		totalIO := float64(dev.ReadIOss + dev.WriteIOSS)
		lines = append(lines, labelStyle.Render(fmt.Sprintf("%s", dev.Name)))
		if len(stats.Rates) > i && stats.Rates[i].ReadBps > 0 {
			r := stats.Rates[i]
			if totalIO > 0 {
				pct := float64(dev.WriteIOSS) / float64(totalIO) * 100
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", renderBar(pct, barWidth), readStr, writeStr, humanBytesRate(r.ReadBps), humanBytesRate(r.WriteBps)))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", barEmptyStyle.Render(strings.Repeat("-", barWidth)), readStr, writeStr, humanBytesRate(r.ReadBps), humanBytesRate(r.WriteBps)))
			}
		} else {
			if totalIO > 0 {
				pct := float64(dev.WriteIOSS) / float64(totalIO) * 100
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", renderBar(pct, barWidth), readStr, writeStr))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", barEmptyStyle.Render(strings.Repeat("-", barWidth)), readStr, writeStr))
			}
		}
	}

	return strings.Join(lines, "\n")
}

func renderNetCard(stats netStats, width int, disabledSections map[string]bool) string {
	if !stats.OK {
		return ""
	}
	if disabledSections != nil && disabledSections["Network"] {
		return ""
	}
	var text string
	if len(stats.Devices) == 0 {
		text = mutedStyle.Render("Network I/O metrics unavailable.")
	} else {
		text = renderNet(stats, width)
	}
	return renderCard("Network I/O", text, width)
}

func renderNet(stats netStats, width int) string {
	if !stats.OK {
		return mutedStyle.Render("Network I/O metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	var lines []string

	for i, dev := range stats.Devices {
		barWidth := maxInt(10, innerWidth-60)
		totalBytes := dev.BytesSent + dev.BytesRecv
		pct := float64(dev.BytesRecv) / float64(totalBytes) * 100
		if totalBytes > 0 && pct > 100 {
			pct = 100
		}
		bar := renderBar(pct, barWidth)
		sentStr := humanBytes(dev.BytesSent)
		recvStr := humanBytes(dev.BytesRecv)

		lines = append(lines, labelStyle.Render(fmt.Sprintf("%s", dev.Name)))
		if len(stats.Rates) > i && stats.Rates[i].BytesSentBps > 0 {
			r := stats.Rates[i]
			lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", bar, recvStr, sentStr, humanBytesRate(r.BytesRecvBps), humanBytesRate(r.BytesSentBps)))
		} else {
			lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", bar, recvStr, sentStr))
		}
	}

	return strings.Join(lines, "\n")
}

func llmProcesses(processes []gpuProcess) []gpuProcess {
	out := make([]gpuProcess, 0)
	for _, p := range processes {
		if p.Provider != "" {
			out = append(out, p)
		}
	}
	return out
}

func otherProcesses(processes []gpuProcess) []gpuProcess {
	out := make([]gpuProcess, 0)
	for _, p := range processes {
		if p.Provider == "" {
			out = append(out, p)
		}
	}
	return out
}

func renderGPUSection(gpus []gpuStats, sparkline map[string][]float64, width int, disabledSections map[string]bool) string {
	if len(gpus) == 0 {
		return ""
	}
	if disabledSections != nil && disabledSections["GPU"] {
		return ""
	}

	lines := make([]string, 0)
	innerWidth := maxInt(20, width-6)
	for i, gpu := range gpus {
		if i > 0 {
			lines = append(lines, "")
		}

		title := fmt.Sprintf("GPU %s", gpu.Index)
		if gpu.Name != "" {
			title += "  " + gpu.Name
		}
		lines = append(lines, valueStyle.Render(fitText(title, innerWidth)))

		vramPercent := 0.0
		if gpu.MemoryUsed.OK && gpu.MemoryTotal.OK && gpu.MemoryTotal.Value > 0 {
			vramPercent = gpu.MemoryUsed.Value / gpu.MemoryTotal.Value * 100
		}
		vramStr := fmt.Sprintf("%s / %s (%.1f%%)", optMemoryString(gpu.MemoryUsed), optMemoryString(gpu.MemoryTotal), vramPercent)
		if gpu.VRAMTrend != "" && gpu.VRAMDelta != 0 {
			sign := ""
			if gpu.VRAMDelta > 0 {
				sign = "+"
			}
			vramStr = gpu.VRAMTrend + " " + sign + humanMiB(gpu.VRAMDelta) + "  " + vramStr
		}
		lines = append(lines, metricLine("VRAM", vramPercent, vramStr, innerWidth))

		utilStr := optPercentString(gpu.UtilPercent)
		if gpu.UtilTrend != "" {
			utilStr = gpu.UtilTrend + " " + utilStr
		}
		lines = append(lines, metricLine("Util", optPercentValue(gpu.UtilPercent), utilStr, innerWidth))

		if history := sparkline[gpu.Index]; len(history) > 0 {
			lines = append(lines, mutedStyle.Render("trend: "+renderSparkline(history, minInt(innerWidth-10, len(history)))))
		}

		details := []string{
			"Temp " + optTemperatureString(gpu.Temperature),
			"Power " + optPowerString(gpu.PowerDraw, gpu.PowerLimit),
		}
		lines = append(lines, mutedStyle.Render(strings.Join(details, "   ")))

		llm := llmProcesses(gpu.Processes)
		other := otherProcesses(gpu.Processes)

		if len(llm) > 0 {
			lines = append(lines, labelStyle.Render("LLM Processes"))
			lines = append(lines, renderGPUProcessTable(llm, innerWidth))
		}
		if len(other) > 0 {
			lines = append(lines, labelStyle.Render("Other Processes"))
			lines = append(lines, renderGPUProcessTable(other, innerWidth))
		}
	}

	return renderCard("NVIDIA GPU", strings.Join(lines, "\n"), width)
}

func llmAMDProcesses(processes []amdGPUProcess) []amdGPUProcess {
	out := make([]amdGPUProcess, 0)
	for _, p := range processes {
		if p.Provider != "" {
			out = append(out, p)
		}
	}
	return out
}

func otherAMDProcesses(processes []amdGPUProcess) []amdGPUProcess {
	out := make([]amdGPUProcess, 0)
	for _, p := range processes {
		if p.Provider == "" {
			out = append(out, p)
		}
	}
	return out
}

func renderAMDSection(gpus []amdGPUStats, width int, disabledSections map[string]bool) string {
	if len(gpus) == 0 {
		return ""
	}
	if disabledSections != nil && disabledSections["GPU"] {
		return ""
	}

	innerWidth := maxInt(20, width-6)
	lines := make([]string, 0, len(gpus)*6)
	for i, gpu := range gpus {
		if i > 0 {
			lines = append(lines, "")
		}

		title := fmt.Sprintf("GPU %s", gpu.Index)
		if gpu.Name != "" {
			title += "  " + gpu.Name
		}
		if gpu.PCI != "" {
			title += "  " + gpu.PCI
		}
		lines = append(lines, valueStyle.Render(fitText(title, innerWidth)))

		vramPercent := 0.0
		if gpu.MemoryUsed.OK && gpu.MemoryTotal.OK && gpu.MemoryTotal.Value > 0 {
			vramPercent = gpu.MemoryUsed.Value / gpu.MemoryTotal.Value * 100
		}
		vramValue := fmt.Sprintf("%s / %s (%.1f%%)", optMemoryString(gpu.MemoryUsed), optMemoryString(gpu.MemoryTotal), vramPercent)
		lines = append(lines, metricLine("VRAM", vramPercent, vramValue, innerWidth))

		utilStr := optPercentString(gpu.UtilPercent)
		if gpu.UtilTrend != "" {
			utilStr = gpu.UtilTrend + " " + utilStr
		}
		lines = append(lines, metricLine("Util", optPercentValue(gpu.UtilPercent), utilStr, innerWidth))

		lines = append(lines, mutedStyle.Render("Temp "+optTemperatureString(gpu.Temperature)+"   Power "+optPowerString(gpu.PowerDraw, optFloat{})))
	}

	return renderCard("AMD GPU", strings.Join(lines, "\n"), width)
}

func renderGPUProcessTable(processes []gpuProcess, width int) string {
	if width < 38 {
		processWidth := maxInt(8, width-18)
		lines := []string{labelStyle.Render(fmt.Sprintf("%-7s %-9s %s", "PID", "VRAM", "PROCESS"))}
		for _, p := range processes {
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			vramStr := optMemoryString(p.UsedMemoryMB)
			var percent float64
			if p.UsedMemoryMB.OK {
				percent = inferVRAMPercent(p.UsedMemoryMB.Value, p.UsedMemoryMB.OK)
			}
			lines = append(lines, fmt.Sprintf("%-7d %-9s %s", p.PID, coloredValue(vramStr, percent), fitText(name, processWidth)))
		}
		return strings.Join(lines, "\n")
	}

	processWidth := maxInt(12, width-19)
	lines := []string{labelStyle.Render(fmt.Sprintf("%-7s %-9s %s", "PID", "VRAM", "PROCESS"))}
	for _, p := range processes {
		name := p.Name
		if name == "" {
			name = "unknown"
		}
		vramStr := optMemoryString(p.UsedMemoryMB)
		var percent float64
		if p.UsedMemoryMB.OK {
			percent = inferVRAMPercent(p.UsedMemoryMB.Value, p.UsedMemoryMB.OK)
		}
		line := fmt.Sprintf("%-7d %-9s %s", p.PID, coloredValue(vramStr, percent), fitText(name, processWidth))
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func renderAMDProcessTable(processes []amdGPUProcess, width int) string {
	if width < 48 {
		processWidth := maxInt(8, width-18)
		lines := []string{labelStyle.Render(fmt.Sprintf("%-7s %-9s %s", "PID", "VRAM", "PROCESS"))}
		for _, p := range processes {
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			vramStr := optMemoryString(p.VRAMMiB)
			var percent float64
			if p.VRAMMiB.OK {
				percent = inferVRAMPercent(p.VRAMMiB.Value, p.VRAMMiB.OK)
			}
			lines = append(lines, fmt.Sprintf("%-7d %-9s %s", p.PID, coloredValue(vramStr, percent), fitText(name, processWidth)))
		}
		return strings.Join(lines, "\n")
	}

	processWidth := maxInt(12, width-29)
	lines := []string{labelStyle.Render(fmt.Sprintf("%-7s %-9s %-9s %s", "PID", "VRAM", "GTT", "PROCESS"))}
	for _, p := range processes {
		name := p.Name
		if name == "" {
			name = "unknown"
		}
		vramStr := optMemoryString(p.VRAMMiB)
		gttStr := optMemoryString(p.GTTMiB)
		var vramPercent, gttPercent float64
		if p.VRAMMiB.OK {
			vramPercent = inferVRAMPercent(p.VRAMMiB.Value, p.VRAMMiB.OK)
		}
		if p.GTTMiB.OK {
			gttPercent = inferVRAMPercent(p.GTTMiB.Value, p.GTTMiB.OK)
		}
		lines = append(lines, fmt.Sprintf("%-7d %-9s %-9s %s", p.PID, coloredValue(vramStr, vramPercent), coloredValue(gttStr, gttPercent), fitText(name, processWidth)))
	}
	return strings.Join(lines, "\n")
}

func renderOllamaProcesses(processes []ollamaProcess, width int, disabledSections map[string]bool) string {
	if len(processes) == 0 {
		return ""
	}
	if disabledSections != nil && disabledSections["Ollama"] {
		return ""
	}

	innerWidth := maxInt(20, width-6)
	lines := make([]string, 0, len(processes)+1)

	if innerWidth < 76 {
		cmdWidth := maxInt(10, innerWidth-26)
		lines = append(lines, labelStyle.Render(fmt.Sprintf("%-7s %-8s %-8s %s", "PID", "CPU", "RSS", "COMMAND")))
		for _, p := range processes {
			line := fmt.Sprintf("%-7d %-8s %-8s %s",
				p.PID,
				fmt.Sprintf("%.1f%%", p.CPUPercent),
				humanBytes(p.RSS),
				fitText(commandLabel(p), cmdWidth),
			)
			lines = append(lines, line)
		}
		return renderCard("Ollama Processes", strings.Join(lines, "\n"), width)
	}

	cmdWidth := maxInt(12, innerWidth-55)
	lines = append(lines, labelStyle.Render(fmt.Sprintf("%-7s %7s %8s %8s %-9s %s", "PID", "CPU", "RAM", "RSS", "RUNTIME", "COMMAND")))
	for _, p := range processes {
		line := fmt.Sprintf("%-7d %7s %8s %8s %-9s %s",
			p.PID,
			fmt.Sprintf("%.1f%%", p.CPUPercent),
			fmt.Sprintf("%.1f%%", p.MemoryPercent),
			humanBytes(p.RSS),
			runtimeLabel(p),
			fitText(commandLabel(p), cmdWidth),
		)
		lines = append(lines, line)
	}

	return renderCard("Ollama Processes", strings.Join(lines, "\n"), width)
}

func renderOllamaPS(output commandOutput, width int, disabledSections map[string]bool) string {
	if output.Missing || output.Error != "" || strings.TrimSpace(output.Output) == "" {
		return ""
	}
	if disabledSections != nil && disabledSections["Ollama"] {
		return ""
	}

	innerWidth := maxInt(20, width-6)
	if len(output.Models) > 0 {
		if innerWidth < 64 {
			nameWidth := maxInt(10, innerWidth-18)
			lines := []string{labelStyle.Render(fitText("MODEL  CONTEXT", innerWidth))}
			var totalCtx, totalPrompt int
			for _, model := range output.Models {
				totalCtx += model.CtxTokens
				totalPrompt += model.PromptTokens
				line := fmt.Sprintf("%-*s %s", nameWidth, fitText(model.Name, nameWidth), model.Context)
				lines = append(lines, fitText(line, innerWidth))
			}
			if totalCtx > 0 || totalPrompt > 0 {
				lines = append(lines, "")
				var tokenParts []string
				if totalCtx > 0 {
					tokenParts = append(tokenParts, fmt.Sprintf("tokens %s", formatTokenCount(totalCtx)))
				}
				if totalPrompt > 0 {
					tokenParts = append(tokenParts, fmt.Sprintf("prompt %s", formatTokenCount(totalPrompt)))
				}
				lines = append(lines, valueStyle.Render("  "+strings.Join(tokenParts, "  ")))
			}
			return renderCard("Ollama Models", strings.Join(lines, "\n"), width)
		}

		nameWidth := clampInt(innerWidth/3, 12, 30)
		lines := []string{labelStyle.Render(fmt.Sprintf("%-*s %-14s %-12s %-8s %s", nameWidth, "MODEL", "SIZE", "PROCESSOR", "CTX", "UNTIL"))}
		var totalCtx, totalPrompt int
		for _, model := range output.Models {
			totalCtx += model.CtxTokens
			totalPrompt += model.PromptTokens
			line := fmt.Sprintf("%-*s %-14s %-12s %-8s %s",
				nameWidth,
				fitText(model.Name, nameWidth),
				fitText(model.Size, 14),
				fitText(model.Processor, 12),
				model.Context,
				fitText(model.Until, maxInt(8, innerWidth-nameWidth-39)),
			)
			lines = append(lines, fitText(line, innerWidth))
		}
		if totalCtx > 0 || totalPrompt > 0 {
			lines = append(lines, "")
			var tokenParts []string
			if totalCtx > 0 {
				tokenParts = append(tokenParts, fmt.Sprintf("tokens %s", formatTokenCount(totalCtx)))
			}
			if totalPrompt > 0 {
				tokenParts = append(tokenParts, fmt.Sprintf("prompt %s", formatTokenCount(totalPrompt)))
			}
			lines = append(lines, valueStyle.Render("  "+strings.Join(tokenParts, "  ")))
		}
		return renderCard("Ollama Models", strings.Join(lines, "\n"), width)
	}

	lines := strings.Split(strings.TrimRight(output.Output, "\n"), "\n")
	for i := range lines {
		lines[i] = fitText(lines[i], innerWidth)
	}

	return renderCard("Ollama Models", strings.Join(lines, "\n"), width)
}

func renderUnslothStudio(s unslothStudioStats, width int, disabledSections map[string]bool) string {
	innerWidth := maxInt(20, width-6)

	// Not connected at all.
	if !s.Connected {
		return ""
	}
	if disabledSections != nil && disabledSections["Unsloth"] {
		return ""
	}

	lines := make([]string, 0, 12)

	// Active model line.
	if s.ActiveModel != "" {
		modelLine := valueStyle.Render(fitText(s.ActiveModel, innerWidth-2))
		if s.GGUFVariant != "" {
			modelLine += mutedStyle.Render("  " + s.GGUFVariant)
		}
		lines = append(lines, modelLine)
	} else {
		lines = append(lines, mutedStyle.Render("No active model"))
	}

	// Flags row.
	flags := make([]string, 0, 4)
	if s.IsVision {
		flags = append(flags, labelStyle.Render("vision"))
	}
	if s.IsAudio {
		flags = append(flags, labelStyle.Render("audio"))
	}
	if s.SupportsReasoning {
		flags = append(flags, labelStyle.Render("reasoning"))
	}
	if s.TensorParallel {
		flags = append(flags, labelStyle.Render("tensor-parallel"))
	}
	if len(flags) > 0 {
		lines = append(lines, strings.Join(flags, " "))
	}

	// Context length.
	if s.ContextLength > 0 {
		ctxStr := fmt.Sprintf("ctx %s", formatTokenCount(s.ContextLength))
		if s.MaxContextLength > 0 && s.MaxContextLength != s.ContextLength {
			ctxStr += " / " + formatTokenCount(s.MaxContextLength)
		}
		if s.NativeContextLength > 0 && s.NativeContextLength != s.MaxContextLength {
			ctxStr += " (native " + formatTokenCount(s.NativeContextLength) + ")"
		}
		lines = append(lines, labelStyle.Render(ctxStr))
	}
	if s.CurrentContext > 0 {
		lines = append(lines, labelStyle.Render("used "+formatTokenCount(s.CurrentContext)))
	}
	if s.ConcurrentSessions > 0 {
		lines = append(lines, labelStyle.Render(fmt.Sprintf("parallel sessions %d", s.ConcurrentSessions)))
	}
	if s.TokensPerSecond.OK {
		lines = append(lines, labelStyle.Render(fmt.Sprintf("throughput %.1f tok/s", s.TokensPerSecond.Value)))
	}

	// Speculative decoding.
	if s.SpeculativeType != "" {
		lines = append(lines, labelStyle.Render("spec "+s.SpeculativeType))
	}

	// Loading models.
	if len(s.LoadingModels) > 0 {
		for _, m := range s.LoadingModels {
			lines = append(lines, warnStyle.Render("loading: ")+fitText(m, innerWidth-10))
		}
	}

	// Load progress bar.
	if s.LoadPhase != "" && s.LoadTotal > 0 {
		pct := float64(s.LoadBytes) / float64(s.LoadTotal) * 100
		barLabel := fmt.Sprintf("%s / %s", humanBytes(uint64(s.LoadBytes)), humanBytes(uint64(s.LoadTotal)))
		lines = append(lines, metricLine("Load", pct, barLabel, innerWidth))
	}

	// Loaded models list.
	if len(s.LoadedModels) > 0 && s.ActiveModel == "" {
		lines = append(lines, "")
		for _, m := range s.LoadedModels {
			lines = append(lines, labelStyle.Render("  ")+fitText(m, innerWidth-4))
		}
	}

	// Training section.
	if s.TrainStatus != "" {
		lines = append(lines, "")
		lines = append(lines, sectionTitleStyle.Render("Training"))
		statusStyle := mutedStyle
		if s.TrainStatus == "running" {
			statusStyle = barOKStyle
		} else if s.TrainStatus == "error" {
			statusStyle = barDangerStyle
		} else if s.TrainStatus == "stopped" {
			statusStyle = warnStyle
		}
		lines = append(lines, statusStyle.Render(s.TrainStatus))
		if s.TrainStep > 0 {
			trainLine := fmt.Sprintf("step %d", s.TrainStep)
			if s.TrainLoss > 0 {
				trainLine += fmt.Sprintf("  loss %.4f", s.TrainLoss)
			}
			if s.TrainLr > 0 {
				trainLine += fmt.Sprintf("  lr %.2e", s.TrainLr)
			}
			lines = append(lines, labelStyle.Render(trainLine))
		}
	}

	// Error at bottom.
	if s.Error != "" {
		lines = append(lines, "")
		lines = append(lines, warnStyle.Render("! ")+fitText(s.Error, innerWidth-4))
	}

	return renderCard("Unsloth Studio", strings.Join(lines, "\n"), width)
}

func formatTokenCount(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%dK", n/1000)
	default:
		return strconv.Itoa(n)
	}
}

func renderGroup(title, body string, width int) string {
	innerWidth := maxInt(20, width-4)
	header := sectionTitleStyle.Render(" ━ " + title + " ━ ")
	s := cardStyle.Width(innerWidth)
	return s.Render(lipgloss.JoinVertical(lipgloss.Left, header, body))
}

func renderCard(title, body string, width int) string {
	innerWidth := maxInt(20, width-4)
	content := sectionTitleStyle.Render(title)
	if strings.TrimSpace(body) != "" {
		content += "\n" + body
	}
	return cardStyle.Width(innerWidth).Render(content)
}

func metricLine(label string, percent float64, value string, width int) string {
	labelWidth := 8
	value = fitText(value, maxInt(8, width-labelWidth-2))
	valueWidth := runewidth.StringWidth(value)
	barWidth := width - labelWidth - valueWidth - 2
	if barWidth < 3 {
		return fmt.Sprintf("%s %s", labelStyle.Width(labelWidth).Render(label), value)
	}
	barWidth = clampInt(barWidth, 3, 28)

	return fmt.Sprintf("%s %s %s",
		labelStyle.Width(labelWidth).Render(label),
		renderBar(percent, barWidth),
		value,
	)
}

func renderCoreCell(index int, percent float64, width int) string {
	barWidth := clampInt(width-13, 6, 14)
	filled := int(percent / 100 * float64(barWidth))
	filled = clampInt(filled, 0, barWidth)
	style := barOKStyle
	if percent >= 90 {
		style = barDangerStyle
	} else if percent >= 75 {
		style = barWarnStyle
	}
	coreLabel := fmt.Sprintf("C%-2d", index)
	return fmt.Sprintf("%s %s %5.1f%%", coreLabel, style.Render(strings.Repeat("█", filled)), percent)
}

func renderCoreRow(perCore []float64, width int) string {
	if len(perCore) == 0 || width <= 0 {
		return ""
	}

	compact := make([]string, len(perCore))
	for i, percent := range perCore {
		compact[i] = fmt.Sprintf("C%d:%.0f%%", i, percent)
	}
	cells := make([]string, len(perCore))
	for i, percent := range perCore {
		cells[i] = renderCoreCell(i, percent, 18)
	}
	if runewidth.StringWidth(strings.Join(cells, " ")) <= width {
		return strings.Join(cells, " ")
	}

	lines := make([]string, 0, len(compact))
	line := ""
	for _, cell := range compact {
		candidate := cell
		if line != "" {
			candidate = line + " " + cell
		}
		if line != "" && runewidth.StringWidth(candidate) > width {
			lines = append(lines, line)
			line = cell
			continue
		}
		line = candidate
	}
	if line != "" {
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func renderBar(percent float64, width int) string {
	percent = clampFloat(percent, 0, 100)
	filled := int(math.Round(percent / 100 * float64(width)))
	filled = clampInt(filled, 0, width)

	style := barOKStyle
	if percent >= 90 {
		style = barDangerStyle
	} else if percent >= 75 {
		style = barWarnStyle
	}

	full := "█"
	empty := "░"
	return style.Render(strings.Repeat(full, filled)) + barEmptyStyle.Render(strings.Repeat(empty, width-filled))
}

func commandLabel(p ollamaProcess) string {
	if p.Command != "" {
		return p.Command
	}
	if p.Name != "" {
		return p.Name
	}
	return "unknown"
}

func runtimeLabel(p ollamaProcess) string {
	if !p.RuntimeOK {
		return "n/a"
	}
	return shortDuration(p.Runtime)
}

func optPercentValue(value optFloat) float64 {
	if !value.OK {
		return 0
	}
	return value.Value
}

func optPercentString(value optFloat) string {
	if !value.OK {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", value.Value)
}

func optTemperatureString(value optFloat) string {
	if !value.OK {
		return "n/a"
	}
	return fmt.Sprintf("%.0f C", value.Value)
}

func optPowerString(draw optFloat, limit optFloat) string {
	if draw.OK && limit.OK {
		return fmt.Sprintf("%.0f / %.0f W", draw.Value, limit.Value)
	}
	if draw.OK {
		return fmt.Sprintf("%.0f W", draw.Value)
	}
	return "n/a"
}

func optMemoryString(value optFloat) string {
	if !value.OK {
		return "n/a"
	}
	return humanMiB(value.Value)
}

func humanMiB(value float64) string {
	if value >= 1024 {
		return fmt.Sprintf("%.1f GiB", value/1024)
	}
	return fmt.Sprintf("%.0f MiB", value)
}

func humanBytes(value uint64) string {
	const unit = 1024
	if value < unit {
		return fmt.Sprintf("%d B", value)
	}

	div := uint64(unit)
	exp := 0
	for value/div >= unit && exp < 5 {
		div *= unit
		exp++
	}

	return fmt.Sprintf("%.1f %ciB", float64(value)/float64(div), "KMGTPE"[exp])
}

func humanBytesRate(bps float64) string {
	if bps < 100 {
		return fmt.Sprintf("%.0f B/s", bps)
	}
	const unit = 1024.0
	if bps < unit {
		return fmt.Sprintf("%.1f B/s", bps)
	}
	exp := 0
	value := bps
	for value >= unit && exp < 5 {
		value /= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB/s", value, "KMGTPE"[exp])
}

func shortDuration(d time.Duration) string {
	if d < 0 {
		return "n/a"
	}

	d = d.Round(time.Second)
	days := d / (24 * time.Hour)
	d -= days * 24 * time.Hour
	hours := d / time.Hour
	d -= hours * time.Hour
	minutes := d / time.Minute
	d -= minutes * time.Minute
	seconds := d / time.Second

	switch {
	case days > 0:
		return fmt.Sprintf("%dd%dh", days, hours)
	case hours > 0:
		return fmt.Sprintf("%dh%02dm", hours, minutes)
	case minutes > 0:
		return fmt.Sprintf("%dm%02ds", minutes, seconds)
	default:
		return fmt.Sprintf("%ds", seconds)
	}
}

func trimDuration(d time.Duration) string {
	if d%time.Second == 0 {
		return d.String()
	}
	return d.Round(time.Millisecond).String()
}

func cleanCommandError(err error, stderr string) string {
	message := strings.TrimSpace(stderr)
	if message == "" && err != nil {
		message = err.Error()
	}
	return cleanError(message)
}

func cleanError(message string) string {
	message = strings.Join(strings.Fields(message), " ")
	return fitText(message, 180)
}

func addWarning(warnings *[]string, message string) {
	message = strings.TrimSpace(message)
	if message == "" {
		return
	}
	*warnings = append(*warnings, message)
}

func severityStyle(percent float64) lipgloss.Style {
	switch {
	case percent >= 90:
		return sevDangerColorStyle
	case percent >= 75:
		return sevWarnColorStyle
	case percent >= 50:
		return sevOKStyle
	default:
		return lipgloss.NewStyle().Foreground(mutedColor)
	}
}

func coloredValue(text string, percent float64) string {
	return severityStyle(percent).Render(text)
}

func inferVRAMPercent(vramMiB float64, _ bool) float64 {
	if vramMiB >= 50*1024 {
		return 100
	}
	if vramMiB >= 25*1024 {
		return 75
	}
	if vramMiB >= 10*1024 {
		return 60
	}
	return 30
}

func fitText(text string, width int) string {
	if width <= 0 {
		return ""
	}
	if runewidth.StringWidth(text) <= width {
		return text
	}
	if width <= 3 {
		return strings.Repeat(".", width)
	}
	return runewidth.Truncate(text, width, "...")
}

func clampFloat(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func mapBoolString(b bool) string {
	if b {
		return "enabled"
	}
	return "disabled"
}

func addGPUUtilDelta(prev []gpuStats, new []gpuStats) {
	for i := range new {
		if i < len(prev) && prev[i].UtilPercent.OK && new[i].UtilPercent.OK {
			delta := new[i].UtilPercent.Value - prev[i].UtilPercent.Value
			new[i].UtilDelta = delta
			if delta > 2 {
				new[i].UtilTrend = "▲"
			} else if delta < -2 {
				new[i].UtilTrend = "▼"
			} else {
				new[i].UtilTrend = "-"
			}
		}
	}
}

func addVRAMDeltas(prev []gpuStats, new []gpuStats) {
	for i := range new {
		if i < len(prev) && prev[i].MemoryUsed.OK && new[i].MemoryUsed.OK {
			delta := new[i].MemoryUsed.Value - prev[i].MemoryUsed.Value
			new[i].VRAMDelta = delta
			if delta > 100 {
				new[i].VRAMTrend = "▲"
			} else if delta < -100 {
				new[i].VRAMTrend = "▼"
			} else {
				new[i].VRAMTrend = "-"
			}
		}
	}
}

func addAMDUtilDelta(prev []amdGPUStats, new []amdGPUStats) {
	for i := range new {
		if i < len(prev) && prev[i].UtilPercent.OK && new[i].UtilPercent.OK {
			delta := new[i].UtilPercent.Value - prev[i].UtilPercent.Value
			new[i].UtilDelta = delta
			if delta > 2 {
				new[i].UtilTrend = "▲"
			} else if delta < -2 {
				new[i].UtilTrend = "▼"
			} else {
				new[i].UtilTrend = "-"
			}
		}
	}
}

func mergeGPUSparkline(prev map[string][]float64, new []gpuStats) map[string][]float64 {
	result := make(map[string][]float64)

	for k, v := range prev {
		if len(v) > 0 {
			result[k] = v
		}
	}

	for _, gpu := range new {
		if gpu.UtilPercent.OK {
			history := result[gpu.Index]
			if len(history) == 0 {
				history = append(history, 0)
			}
			history = append(history, gpu.UtilPercent.Value)
			if len(history) > 30 {
				history = history[1:]
			}
			result[gpu.Index] = history
		}
	}

	return result
}

func renderSparkline(values []float64, width int) string {
	if len(values) == 0 {
		return strings.Repeat(".", width)
	}

	chars := "▁▂▃▄▅▆▇█"
	maxVal := 0.0
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}

	var result strings.Builder
	step := float64(len(values)) / float64(width)

	for i := 0; i < width && i < len(values); i++ {
		idx := int(float64(i) / step)
		if idx >= len(values) {
			idx = len(values) - 1
		}
		level := int(values[idx] / 100.0 * 8)
		level = clampInt(level, 0, 7)
		result.WriteByte(chars[level])
	}

	return result.String()
}

func collectAMDFromSysfs(warnings *[]string) []amdGPUStats {
	stats := make([]amdGPUStats, 0)

	dirEntries, err := os.ReadDir("/sys/class/drm")
	if err != nil {
		addWarning(warnings, "AMD GPU sysfs unavailable: "+cleanError(err.Error()))
		return stats
	}

	for _, d := range dirEntries {
		if !d.IsDir() || !strings.HasPrefix(d.Name(), "card") {
			continue
		}

		cardDir := "/sys/class/drm/" + d.Name()
		deviceDir := cardDir + "/device"

		if _, err := os.Stat(deviceDir); os.IsNotExist(err) {
			continue
		}

		gpu := amdGPUStats{
			Index: strings.TrimPrefix(d.Name(), "card"),
			Name:  "AMD GPU",
		}

		if pci, err := os.Readlink(deviceDir); err == nil {
			gpu.PCI = filepath.Base(pci)
		}

		hwmonDir := deviceDir + "/hwmon"
		hwmonEntries, err := os.ReadDir(hwmonDir)
		if err == nil {
			for _, e := range hwmonEntries {
				if !strings.HasPrefix(e.Name(), "hwmon") {
					continue
				}
				tempFile := hwmonDir + "/" + e.Name() + "/temp1_input"
				if temp, err := os.ReadFile(tempFile); err == nil {
					if val, err := strconv.ParseFloat(strings.TrimSpace(string(temp)), 64); err == nil {
						if val > 0 {
							gpu.Temperature = optFloat{Value: val / 1000.0, OK: true}
						}
					}
				}
			}
		}

		stats = append(stats, gpu)
	}

	return stats
}
