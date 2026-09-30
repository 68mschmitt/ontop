package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"

	"os"

	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"ontop/internal/collect"
	"ontop/internal/render"
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

		s := collect.CollectMetrics(ctx)

		if *onceMode && len(s.GPUs) > 0 {
			// Second collection to populate sparklines with real delta data.
			<-time.After(500 * time.Millisecond)
			s = collect.CollectMetrics(ctx)
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
				_, _ = f.WriteString(render.RenderContent(s, 200, nil))
				fmt.Printf("exported to %s\n", *exportTarget)
				return
			}
			fmt.Println(render.RenderContent(s, 200, nil))
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
	return tea.Batch(collect.CollectMetricsCmd(), collect.TickCmd(m.interval))
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
				cmds = append(cmds, collect.CollectMetricsCmd())
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
				m.notification = fmt.Sprintf("GPU section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 's':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["System"]
				m.disabledSections["System"] = enabled
				m.notification = fmt.Sprintf("System section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'm':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Memory"]
				m.disabledSections["Memory"] = enabled
				m.notification = fmt.Sprintf("Memory section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'u':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Unsloth"]
				m.disabledSections["Unsloth"] = enabled
				m.notification = fmt.Sprintf("Unsloth section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'o':
				if m.disabledSections == nil {
					m.disabledSections = make(map[string]bool)
				}
				enabled := !m.disabledSections["Ollama"]
				m.disabledSections["Ollama"] = enabled
				m.notification = fmt.Sprintf("Ollama section %s", collect.MapBoolString(enabled))
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
		cmds = append(cmds, collect.TickCmd(m.interval))
		if !m.loading {
			m.loading = true
			cmds = append(cmds, collect.CollectMetricsCmd())
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
		return render.RenderHelpOverlay(width, m.disabledSections)
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
	header := render.RenderHeader(m.snapshot, m.interval, m.loading, flash, frame, m.width)
	footer := render.RenderFooter(width, m.disabledSections)
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
	m.viewport.SetContent(render.RenderContent(m.snapshot, m.viewport.Width, m.disabledSections))
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
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
