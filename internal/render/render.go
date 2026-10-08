package render

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"ontop/internal/collect"
)

// statusWidth keeps the "idle" and "collecting" labels the same display width
// so the centered header does not shift when the status changes.
const statusWidth = 12

func statusLabel(collecting bool, frame string) string {
	if collecting {
		return fmt.Sprintf("%-*s", statusWidth, frame+" collecting")
	}
	return fmt.Sprintf("%-*s", statusWidth, "idle")
}

func RenderHeader(s collect.Snapshot, interval time.Duration, collecting bool, flashActive bool, currentFrame string, width int) string {
	updated := "waiting for first sample"
	if !s.CollectedAt.IsZero() {
		updated = "updated " + s.CollectedAt.Format("15:04:05")
	}

	statusStr := lipgloss.NewStyle().Foreground(collect.AccentColor()).Render(statusLabel(collecting, currentFrame))

	meta := fmt.Sprintf("%s | interval %s | %s", updated, trimDuration(interval), statusStr)
	if s.CollectionMillis > 0 {
		meta += fmt.Sprintf(" | collected in %dms", s.CollectionMillis)
	}

	header := lipgloss.JoinHorizontal(lipgloss.Center,
		collect.TitleStyle().Render("Ollama Machine Monitor"),
		"  ",
		collect.MutedStyle().Render(meta),
	)

	borderCol := collect.PanelColor()
	if flashActive {
		borderCol = collect.AccentColor()
	}

	style := lipgloss.NewStyle().
		Padding(0, 1).
		Border(lipgloss.NormalBorder(), false, false, true, false).
		BorderForeground(borderCol)
	return style.Width(maxInt(20, width-2)).Render(header)
}

func RenderFooter(width int, disabledSections map[string]bool) string {
	var toggles []string
	sections := []struct {
		key    string
		name   string
		active bool
	}{
		{"s", "System", !IsEnabled(disabledSections, "System")},
		{"g", "GPU", !IsEnabled(disabledSections, "GPU")},
		{"m", "Memory", !IsEnabled(disabledSections, "Memory")},
		{"u", "Unsloth", !IsEnabled(disabledSections, "Unsloth")},
		{"o", "Ollama", !IsEnabled(disabledSections, "Ollama")},
	}
	enabledStyle := lipgloss.NewStyle().Foreground(collect.AccentColor())
	disabledStyle := lipgloss.NewStyle().Foreground(collect.MutedColor())
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
	return collect.FooterStyle().Width(maxInt(20, width-2)).Render(collect.FitText(text, maxInt(10, width-4)))
}

func IsEnabled(disabledSections map[string]bool, name string) bool {
	if disabledSections == nil {
		return true
	}
	return !disabledSections[name]
}

func RenderHelpOverlay(width int, disabledSections map[string]bool) string {
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
	card1 := collect.RenderCard("Key Bindings", collect.MutedStyle().Render(body), width)
	card2 := collect.RenderCard("Press any key to close", collect.MutedStyle().Render(""), width)

	var stateLines []string
	sections := []struct {
		name   string
		active bool
	}{
		{"System", IsEnabled(disabledSections, "System")},
		{"GPU", IsEnabled(disabledSections, "GPU")},
		{"Memory", IsEnabled(disabledSections, "Memory")},
		{"Unsloth", IsEnabled(disabledSections, "Unsloth")},
		{"Ollama", IsEnabled(disabledSections, "Ollama")},
	}
	enabledStyle := lipgloss.NewStyle().Foreground(collect.AccentColor())
	disabledStyle := lipgloss.NewStyle().Foreground(collect.MutedColor())
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

func RenderContent(s collect.Snapshot, width int, disabledSections map[string]bool) string {
	return collect.RenderContentSections(s, width, disabledSections)
}

func trimDuration(d time.Duration) string {
	if d%time.Second == 0 {
		return d.String()
	}
	return d.Round(time.Millisecond).String()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
