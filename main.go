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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

const defaultInterval = time.Second

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
	GPUs             []gpuStats
	OllamaProcesses  []ollamaProcess
	OllamaPS         commandOutput
	UnslothStudio    unslothStudioStats
	Warnings         []string
	CollectionMillis int64
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
}

type gpuProcess struct {
	GPUUUID      string
	PID          int32
	Name         string
	UsedMemoryMB optFloat
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

type model struct {
	interval time.Duration
	width    int
	height   int
	ready    bool
	loading  bool
	viewport viewport.Model
	snapshot snapshot
}

var (
	accentColor = lipgloss.Color("#7DFFB3")
	mutedColor  = lipgloss.Color("#6F7785")
	warnColor   = lipgloss.Color("#FFD166")
	dangerColor = lipgloss.Color("#FF6B6B")
	panelColor  = lipgloss.Color("#323846")
	textColor   = lipgloss.Color("#F3F5F7")

	headerStyle = lipgloss.NewStyle().
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
	sectionTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	labelStyle        = lipgloss.NewStyle().Foreground(mutedColor)
	footerStyle       = lipgloss.NewStyle().Foreground(mutedColor).Padding(0, 1)
	barOKStyle        = lipgloss.NewStyle().Foreground(accentColor)
	barWarnStyle      = lipgloss.NewStyle().Foreground(warnColor)
	barDangerStyle    = lipgloss.NewStyle().Foreground(dangerColor)
	barEmptyStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("#3A3F4B"))
)

func main() {
	interval := flag.Duration("interval", defaultInterval, "refresh interval such as 500ms, 1s, or 2s")
	flag.IntVar(&studioPort, "unsloth-port", 8888, "Unsloth Studio API port")
	flag.StringVar(&studioToken, "unsloth-token", "", "Unsloth Studio API Bearer token (or set UNSLOTH_STUDIO_TOKEN)")
	flag.Parse()

	// Support env var as fallback for token.
	if studioToken == "" {
		studioToken = os.Getenv("UNSLOTH_STUDIO_TOKEN")
	}

	if *interval <= 0 {
		fmt.Fprintln(os.Stderr, "interval must be greater than zero")
		os.Exit(2)
	}

	p := tea.NewProgram(newModel(*interval), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to run dashboard: %v\n", err)
		os.Exit(1)
	}
}

func newModel(interval time.Duration) model {
	vp := viewport.New(100, 24)
	vp.Style = lipgloss.NewStyle()

	return model{
		interval: interval,
		loading:  true,
		viewport: vp,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(collectMetricsCmd(), tickCmd(m.interval))
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, keys.Quit):
			return m, tea.Quit
		case key.Matches(msg, keys.Refresh):
			if !m.loading {
				m.loading = true
				cmds = append(cmds, collectMetricsCmd())
			}
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true
		m.updateViewport()
	case metricsMsg:
		m.snapshot = msg.Snapshot
		m.loading = false
		m.updateViewport()
	case tickMsg:
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

	header := renderHeader(m.snapshot, m.interval, m.loading, width)
	footer := renderFooter(width)
	vp := m.viewport
	vp.Height = maxInt(1, m.height-lipgloss.Height(header)-lipgloss.Height(footer)-1)

	return header + "\n" + vp.View() + "\n" + footer
}

func (m *model) updateViewport() {
	if !m.ready {
		return
	}

	m.viewport.Width = maxInt(20, m.width)
	m.viewport.Height = maxInt(1, m.height-5)
	m.viewport.SetContent(renderContent(m.snapshot, m.viewport.Width))
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

	s.CPU = collectCPU(ctx, &s.Warnings)
	s.Memory = collectMemory(ctx, &s.Warnings)
	s.GPUs = collectNVIDIA(ctx, &s.Warnings)
	s.OllamaProcesses = collectOllamaProcesses(ctx, &s.Warnings)
	s.OllamaPS = collectOllamaPS(ctx)
	s.UnslothStudio = collectUnslothStudio(ctx, &s.Warnings)
	s.CollectionMillis = time.Since(started).Milliseconds()

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

func collectNVIDIA(ctx context.Context, warnings *[]string) []gpuStats {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		addWarning(warnings, "nvidia-smi not found; NVIDIA GPU metrics unavailable")
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

	return gpus
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

		processes = append(processes, gpuProcess{
			GPUUUID:      strings.TrimSpace(row[0]),
			PID:          int32(pid64),
			Name:         strings.TrimSpace(row[2]),
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

	stdout, stderr, err := runCommand(ctx, "ollama", "ps")
	if err != nil {
		return commandOutput{Error: cleanCommandError(err, stderr)}
	}

	return commandOutput{Output: strings.TrimSpace(stdout)}
}

func collectUnslothStudio(ctx context.Context, warnings *[]string) unslothStudioStats {
	base := fmt.Sprintf("http://127.0.0.1:%d", studioPort)

	// Health check (unauthenticated).
	healthURL := base + "/api/health"
	resp, err := studioHTTPGet(ctx, healthURL, "")
	if err != nil {
		addWarning(warnings, fmt.Sprintf("Unsloth Studio not reachable at %s", base))
		return unslothStudioStats{Error: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		addWarning(warnings, fmt.Sprintf("Unsloth Studio health check returned %d", resp.StatusCode))
		return unslothStudioStats{Error: fmt.Sprintf("health %d", resp.StatusCode)}
	}

	stats := unslothStudioStats{Connected: true}

	// If no token, we can still show connected status.
	if studioToken == "" {
		addWarning(warnings, "Unsloth Studio: set UNSLOTH_STUDIO_TOKEN or -unsloth-token for inference metrics")
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
	if v, ok := raw["speculative_type"]; ok && json.Unmarshal(v, &str) == nil {
		s.SpeculativeType = str
	}
	if v, ok := raw["tensor_parallel"]; ok && json.Unmarshal(v, &b) == nil {
		s.TensorParallel = b
	}
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

func renderHeader(s snapshot, interval time.Duration, loading bool, width int) string {
	updated := "waiting for first sample"
	if !s.CollectedAt.IsZero() {
		updated = "updated " + s.CollectedAt.Format("15:04:05")
	}

	statusStr := "loading..."
	if !loading {
		statusStr = "idle"
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

	return headerStyle.Width(maxInt(20, width-2)).Render(header)
}

func renderFooter(width int) string {
	text := "q quit | ctrl+c quit | r refresh | up/down scroll | pgup/pgdn page"
	return footerStyle.Width(maxInt(20, width-2)).Render(fitText(text, maxInt(10, width-4)))
}

func renderContent(s snapshot, width int) string {
	if s.CollectedAt.IsZero() {
		return renderCard("Status", mutedStyle.Render("Collecting initial metrics..."), width)
	}

	sections := make([]string, 0, 6)
	if len(s.Warnings) > 0 {
		sections = append(sections, renderWarnings(s.Warnings, width))
	}
	sections = append(sections, renderSystem(s, width))
	sections = append(sections, renderGPUSection(s.GPUs, width))
	sections = append(sections, renderUnslothStudio(s.UnslothStudio, width))
	sections = append(sections, renderOllamaProcesses(s.OllamaProcesses, width))
	sections = append(sections, renderOllamaPS(s.OllamaPS, width))

	return strings.Join(sections, "\n\n")
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

func renderSystem(s snapshot, width int) string {
	cpuBody := renderCPU(s.CPU, width)
	ramBody := renderMemory(s.Memory, width)

	body := sectionTitleStyle.Render("CPU") + "\n" + cpuBody + "\n\n" + sectionTitleStyle.Render("RAM") + "\n" + ramBody
	return renderCard("System", body, width)
}

func renderCPU(stats cpuStats, width int) string {
	if !stats.OK {
		return mutedStyle.Render("CPU metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	lines := []string{
		metricLine("Total", stats.Total, fmt.Sprintf("%.1f%%", stats.Total), innerWidth),
	}

	if len(stats.PerCore) == 0 {
		return strings.Join(lines, "\n")
	}

	lines = append(lines, renderCoreRow(stats.PerCore, innerWidth))

	return strings.Join(lines, "\n")
}

func renderMemory(stats memoryStats, width int) string {
	if !stats.OK {
		return mutedStyle.Render("RAM metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	used := fmt.Sprintf("%s / %s (available %s)", humanBytes(stats.Used), humanBytes(stats.Total), humanBytes(stats.Available))
	return metricLine("RAM", stats.Percent, fmt.Sprintf("%.1f%%  %s", stats.Percent, used), innerWidth)
}

func renderGPUSection(gpus []gpuStats, width int) string {
	if len(gpus) == 0 {
		return renderCard("NVIDIA GPU", mutedStyle.Render("No NVIDIA GPU metrics available."), width)
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
		lines = append(lines, metricLine("Util", optPercentValue(gpu.UtilPercent), optPercentString(gpu.UtilPercent), innerWidth))

		vramPercent := 0.0
		if gpu.MemoryUsed.OK && gpu.MemoryTotal.OK && gpu.MemoryTotal.Value > 0 {
			vramPercent = gpu.MemoryUsed.Value / gpu.MemoryTotal.Value * 100
		}
		vramValue := fmt.Sprintf("%s / %s (%.1f%%)", optMemoryString(gpu.MemoryUsed), optMemoryString(gpu.MemoryTotal), vramPercent)
		lines = append(lines, metricLine("VRAM", vramPercent, vramValue, innerWidth))

		details := []string{
			"Temp " + optTemperatureString(gpu.Temperature),
			"Power " + optPowerString(gpu.PowerDraw, gpu.PowerLimit),
			"Fan " + optPercentString(gpu.FanPercent),
		}
		lines = append(lines, mutedStyle.Render(strings.Join(details, "   ")))

		if len(gpu.Processes) == 0 {
			lines = append(lines, mutedStyle.Render("No active compute processes."))
			continue
		}
		lines = append(lines, renderGPUProcessTable(gpu.Processes, innerWidth))
	}

	return renderCard("NVIDIA GPU", strings.Join(lines, "\n"), width)
}

func renderGPUProcessTable(processes []gpuProcess, width int) string {
	processWidth := maxInt(12, width-19)
	lines := []string{labelStyle.Render(fmt.Sprintf("%-7s %-9s %s", "PID", "VRAM", "PROCESS"))}
	for _, p := range processes {
		name := p.Name
		if name == "" {
			name = "unknown"
		}
		line := fmt.Sprintf("%-7d %-9s %s", p.PID, optMemoryString(p.UsedMemoryMB), fitText(name, processWidth))
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func renderOllamaProcesses(processes []ollamaProcess, width int) string {
	if len(processes) == 0 {
		return renderCard("Ollama Processes", mutedStyle.Render("No Ollama-related processes found."), width)
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

func renderOllamaPS(output commandOutput, width int) string {
	if output.Missing {
		return renderCard("Ollama Models", mutedStyle.Render(output.Error), width)
	}
	if output.Error != "" {
		return renderCard("Ollama Models", warnStyle.Render("ollama ps unavailable: ")+fitText(output.Error, maxInt(10, width-24)), width)
	}
	if strings.TrimSpace(output.Output) == "" {
		return renderCard("Ollama Models", mutedStyle.Render("No loaded models reported by ollama ps."), width)
	}

	innerWidth := maxInt(20, width-6)
	lines := strings.Split(strings.TrimRight(output.Output, "\n"), "\n")
	for i := range lines {
		lines[i] = fitText(lines[i], innerWidth)
	}

	return renderCard("Ollama Models", strings.Join(lines, "\n"), width)
}

func renderUnslothStudio(s unslothStudioStats, width int) string {
	innerWidth := maxInt(20, width-6)

	// Not connected at all.
	if !s.Connected {
		return renderCard("Unsloth Studio", mutedStyle.Render("Not reachable on :"+strconv.Itoa(studioPort)), width)
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
	valueWidth := lipgloss.Width(value)
	barWidth := clampInt(width-labelWidth-valueWidth-5, 8, 28)

	return fmt.Sprintf("%s %s %s",
		labelStyle.Width(labelWidth).Render(label),
		renderBar(percent, barWidth),
		value,
	)
}

func renderCoreCell(index int, percent float64, width int) string {
	barWidth := clampInt(width-13, 6, 14)
	return fmt.Sprintf("C%-2d %s %5.1f%%", index, renderBar(percent, barWidth), percent)
}

func renderCoreRow(perCore []float64, width int) string {
	cells := make([]string, len(perCore))
	for i, percent := range perCore {
		cells[i] = renderCoreCell(i, percent, 18)
	}
	row := strings.Join(cells, " ")
	if lipgloss.Width(row) <= width {
		return row
	}

	for i, percent := range perCore {
		cells[i] = fmt.Sprintf("C%d:%.0f%%", i, percent)
	}
	row = strings.Join(cells, " ")
	if lipgloss.Width(row) <= width {
		return row
	}

	for i, percent := range perCore {
		cells[i] = fmt.Sprintf("%d:%.0f", i, percent)
	}
	row = strings.Join(cells, " ")
	if lipgloss.Width(row) <= width {
		return row
	}

	return mutedStyle.Render("Per-core CPU hidden; terminal is too narrow for one row.")
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

	return "[" + style.Render(strings.Repeat("#", filled)) + barEmptyStyle.Render(strings.Repeat("-", width-filled)) + "]"
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

func fitText(text string, width int) string {
	if width <= 0 {
		return ""
	}
	if lipgloss.Width(text) <= width {
		return text
	}
	if width <= 3 {
		return strings.Repeat(".", width)
	}

	runes := []rune(text)
	if len(runes) <= width {
		return text
	}
	return string(runes[:width-3]) + "..."
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
