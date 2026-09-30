package main

import (
	"sort"

	"github.com/charmbracelet/lipgloss"

	"ontop/internal/collect"
)

type Theme struct {
	Name       string
	Accent     string
	Muted      string
	Warn       string
	Danger     string
	Panel      string
	Text       string
	Background string
}

var themes = map[string]Theme{
	"dark": {
		Name:   "Dark",
		Accent: "#7DFFB3",
		Muted:  "#88909C",
		Warn:   "#FFD166",
		Danger: "#FF6B6B",
		Panel:  "#323846",
		Text:   "#F3F5F7",
	},
	"midnight": {
		Name:   "Midnight Blue",
		Accent: "#5C9CFF",
		Muted:  "#5C6F8C",
		Warn:   "#FFB75E",
		Danger: "#FF6B6B",
		Panel:  "#1E2333",
		Text:   "#D0D8F0",
	},
	"monokai": {
		Name:   "Monokai",
		Accent: "#A6E22E",
		Muted:  "#75715E",
		Warn:   "#E6DB74",
		Danger: "#F92672",
		Panel:  "#2E2D30",
		Text:   "#F8F8F2",
	},
}

var currentTheme = themes["dark"]

func applyTheme(t Theme) {
	currentTheme = t
	accentColor = lipgloss.Color(t.Accent)
	mutedColor = lipgloss.Color(t.Muted)
	warnColor = lipgloss.Color(t.Warn)
	dangerColor = lipgloss.Color(t.Danger)
	panelColor = lipgloss.Color(t.Panel)
	textColor = lipgloss.Color(t.Text)
	collect.ApplyStyling(accentColor, mutedColor, warnColor, dangerColor, panelColor, textColor)
}

func listThemes() []string {
	keys := make([]string, 0, len(themes))
	for k := range themes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
