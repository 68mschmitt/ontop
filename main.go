package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
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
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	netio "github.com/shirou/gopsutil/v3/net"
	"github.com/shirou/gopsutil/v3/process"

	"ontop/internal/collect"
	"ontop/internal/parse"
)

const defaultInterval = time.Second
const version = "dev"

// Global CLI config for collectors that need it.
var studioPort = 8888
var studioToken = ""

var prevSnapshot snapshot
var prevDiskIO map[string]diskDeviceStats
var prevNetIO map[string]netDeviceStats
var CurrentTheme Theme = themes["dark"]

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}


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

func init() {
	collect.ApplyStyling(accentColor, mutedColor, warnColor, dangerColor, panelColor, textColor)
}

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

	gpus, err := parse.ParseGPUCSV(stdout)
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

	gpus, err := parse.ParseAMDJSON(stdout)
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
	records, err := parse.ReadCSV(output)
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
			UtilPercent: parse.ParseOptFloat(row[3]),
			MemoryUsed:  parse.ParseOptFloat(row[4]),
			MemoryTotal: parse.ParseOptFloat(row[5]),
			Temperature: parse.ParseOptFloat(row[6]),
			PowerDraw:   parse.ParseOptFloat(row[7]),
			PowerLimit:  parse.ParseOptFloat(row[8]),
			FanPercent:  parse.ParseOptFloat(row[9]),
		})
	}

	return gpus, nil
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

	records, err := parse.ReadCSV(stdout)
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
			Provider:     parse.ClassifyProvider(processName, cmdline),
			Name:         displayName,
			UsedMemoryMB: parse.ParseOptFloat(row[3]),
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
		models := parse.ParseOllamaPSJSON(output)
		return commandOutput{Output: output, Models: models}
	}

	textStdout, _, textErr := runCommand(ctx, "ollama", "ps")
	if textErr != nil {
		return commandOutput{Error: cleanCommandError(err, stderr)}
	}

	output := strings.TrimSpace(textStdout)
	return commandOutput{Output: output, Models: parse.ParseOllamaPS(output)}
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
			stats.ParseInferenceStatus(statusResp.Body)
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
			stats.ParseTrainStatus(trainResp.Body)
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
			stats.ParseLoadProgress(loadResp.Body)
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
	return footerStyle.Width(maxInt(20, width-2)).Render(collect.FitText(text, maxInt(10, width-4)))
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
	card1 := collect.RenderCard("Key Bindings", mutedStyle.Render(body), width)
	card2 := collect.RenderCard("Press any key to close", mutedStyle.Render(""), width)

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

	return card1 + "\n\n" + card2 + "\n\n" + collect.RenderCard("Section State", strings.Join(stateLines, "\n"), width)
}

func renderContent(s snapshot, width int, disabledSections map[string]bool) string {
	return collect.RenderContentSections(s, width, disabledSections)
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

func mapBoolString(b bool) string {
	if b {
		return "enabled"
	}
	return "disabled"
}

func addWarning(warnings *[]string, message string) {
	message = strings.TrimSpace(message)
	if message == "" {
		return
	}
	*warnings = append(*warnings, message)
}

func cleanError(message string) string {
	message = strings.Join(strings.Fields(message), " ")
	return message
}

func cleanCommandError(err error, stderr string) string {
	message := strings.TrimSpace(stderr)
	if message == "" && err != nil {
		message = err.Error()
	}
	return cleanError(message)
}

func trimDuration(d time.Duration) string {
	if d%time.Second == 0 {
		return d.String()
	}
	return d.Round(time.Millisecond).String()
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
