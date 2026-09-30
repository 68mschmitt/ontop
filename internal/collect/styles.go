package collect

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

// Style variables - initialized from main.
var (
	accentColor   lipgloss.Color
	mutedColor    lipgloss.Color
	warnColor     lipgloss.Color
	dangerColor   lipgloss.Color
	panelColor    lipgloss.Color
	textColor     lipgloss.Color
	StylesApplied bool

	titleStyle          = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	mutedStyle          = lipgloss.NewStyle().Foreground(mutedColor)
	valueStyle          = lipgloss.NewStyle().Bold(true).Foreground(textColor)
	warnStyle           = lipgloss.NewStyle().Foreground(warnColor)
	cardStyle           = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(panelColor).Padding(0, 1)
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
	headerStyleTemplate = lipgloss.NewStyle().Padding(0, 1).Border(lipgloss.NormalBorder(), false, false, true, false).BorderForeground(panelColor)
)

// Initialize styles from global color values.
func ApplyStyling(accent, muted, warn, danger, panel, text lipgloss.Color) {
	accentColor = accent
	mutedColor = muted
	warnColor = warn
	dangerColor = danger
	panelColor = panel
	textColor = text
	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	mutedStyle = lipgloss.NewStyle().Foreground(mutedColor)
	valueStyle = lipgloss.NewStyle().Bold(true).Foreground(textColor)
	warnStyle = lipgloss.NewStyle().Foreground(warnColor)
	cardStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(panelColor).Padding(0, 1)
	sectionTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(accentColor)
	labelStyle = lipgloss.NewStyle().Foreground(mutedColor)
	footerStyle = lipgloss.NewStyle().Foreground(mutedColor).Padding(0, 1)
	barOKStyle = lipgloss.NewStyle().Foreground(accentColor)
	barWarnStyle = lipgloss.NewStyle().Foreground(warnColor)
	barDangerStyle = lipgloss.NewStyle().Foreground(dangerColor)
	barEmptyStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#3A3F4B"))
	sevOKStyle = lipgloss.NewStyle().Foreground(accentColor)
	sevWarnColorStyle = lipgloss.NewStyle().Foreground(warnColor)
	sevDangerColorStyle = lipgloss.NewStyle().Foreground(dangerColor)
	headerStyleTemplate = lipgloss.NewStyle().Padding(0, 1).Border(lipgloss.NormalBorder(), false, false, true, false).BorderForeground(panelColor)
	StylesApplied = true
}

func ApplyMainStyles(accent, muted, warn, danger, panel, text lipgloss.Color) {
	ApplyStyling(accent, muted, warn, danger, panel, text)
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

func FitText(text string, width int) string {
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

func AccentColor() lipgloss.Color {
	return accentColor
}

func MutedColor() lipgloss.Color {
	return mutedColor
}

func WarnColor() lipgloss.Color {
	return warnColor
}

func DangerColor() lipgloss.Color {
	return dangerColor
}

func PanelColor() lipgloss.Color {
	return panelColor
}

func TextColor() lipgloss.Color {
	return textColor
}

func TitleStyle() lipgloss.Style {
	return titleStyle
}

func MutedStyle() lipgloss.Style {
	return mutedStyle
}

func ValueStyle() lipgloss.Style {
	return valueStyle
}

func WarnStyle() lipgloss.Style {
	return warnStyle
}

func FooterStyle() lipgloss.Style {
	return footerStyle
}

func SectionTitleStyle() lipgloss.Style {
	return sectionTitleStyle
}

func HeaderStyleTemplate() lipgloss.Style {
	return headerStyleTemplate
}
