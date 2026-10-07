package ui

import (
	"fmt"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"ontop/internal/collect"
)

type RenderFuncs struct {
	RenderContentFn   func(s collect.Snapshot, width int, disabledSections map[string]bool) string
	RenderHeaderFn    func(s collect.Snapshot, interval time.Duration, loading bool, flash bool, frame string, width int) string
	RenderFooterFn    func(width int, disabledSections map[string]bool) string
	RenderHelpOverlay func(width int, disabledSections map[string]bool) string
}

type UIConfig struct {
	Interval     time.Duration
	GPUFilter    int
	Theme        string
	NoSparklines bool
	ShowNotices  bool
}

type Model struct {
	interval         time.Duration
	render           RenderFuncs
	width            int
	height           int
	ready            bool
	loading          bool
	viewport         viewport.Model
	snapshot         collect.Snapshot
	showHelp         bool
	disabledSections map[string]bool
	cfg              UIConfig
	spinnerIndex     int
	flashActive      bool
	flashEnd         int64
	notification     string
	notificationEnd  int64
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func NewModel(interval time.Duration, cfg UIConfig, rf RenderFuncs) *Model {
	vp := viewport.New(100, 24)
	vp.Style = lipgloss.NewStyle()

	return &Model{
		interval:         interval,
		loading:          true,
		viewport:         vp,
		disabledSections: make(map[string]bool),
		cfg:              cfg,
		render:           rf,
	}
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(collect.CollectMetricsCmd(), collect.TickCmd(m.interval))
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.showHelp {
			m.showHelp = false
			return m, nil
		}
		switch {
		case msg.Type == tea.KeyCtrlC || msg.String() == "q":
			return m, tea.Quit
		case msg.Type == tea.KeyRunes && len(msg.Runes) > 0 && string(msg.Runes) == "r":
			if !m.loading {
				m.viewport.GotoTop()
				m.loading = true
				cmds = append(cmds, collect.CollectMetricsCmd())
			}
		}
		if msg.Type == tea.KeyRunes && len(msg.Runes) > 0 {
			switch msg.Runes[0] {
			case 'g':
				enabled := !m.disabledSections["GPU"]
				m.disabledSections["GPU"] = enabled
				m.notification = fmt.Sprintf("GPU section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 's':
				enabled := !m.disabledSections["System"]
				m.disabledSections["System"] = enabled
				m.notification = fmt.Sprintf("System section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'm':
				enabled := !m.disabledSections["Memory"]
				m.disabledSections["Memory"] = enabled
				m.notification = fmt.Sprintf("Memory section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'u':
				enabled := !m.disabledSections["Unsloth"]
				m.disabledSections["Unsloth"] = enabled
				m.notification = fmt.Sprintf("Unsloth section %s", collect.MapBoolString(enabled))
				m.notificationEnd = time.Now().Add(1 * time.Second).UnixNano()
			case 'o':
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
	case collect.MetricsMessage:
		m.snapshot = msg.Snapshot
		m.loading = false
		m.updateViewport()
	case collect.TickMsg:
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

func (m *Model) View() string {
	width := m.width
	if width <= 0 {
		width = 100
	}

	if !m.ready {
		return "Collecting initial metrics..."
	}

	if m.showHelp {
		return m.render.RenderHelpOverlay(width, m.disabledSections)
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
	header := m.render.RenderHeaderFn(m.snapshot, m.interval, m.loading, flash, frame, m.width)
	footer := m.render.RenderFooterFn(width, m.disabledSections)
	vp := m.viewport
	vp.Height = maxInt(1, m.height-lipgloss.Height(header)-lipgloss.Height(footer)-1)

	var notif string
	if m.notification != "" && time.Now().UnixNano() < m.notificationEnd {
		notif = "\n" + lipgloss.NewStyle().Foreground(collect.AccentColor()).Render(m.notification)
	} else if m.notification != "" {
		m.notification = ""
	}

	content := header + "\n" + vp.View() + "\n" + footer
	if notif != "" {
		content = content + "\n" + notif
	}
	return content
}

func (m *Model) updateViewport() {
	if !m.ready {
		return
	}

	m.viewport.Width = maxInt(20, m.width)
	m.viewport.Height = maxInt(1, m.height-5)
	m.viewport.SetContent(m.render.RenderContentFn(m.snapshot, m.viewport.Width, m.disabledSections))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
