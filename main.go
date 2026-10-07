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

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"ontop/internal/collect"
	"ontop/internal/render"
	"ontop/internal/ui"

	// Ensure parse package init() runs to register AMD parser callback.
	_ "ontop/internal/parse"
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

var (
	accentColor = lipgloss.Color("#7DFFB3")
	mutedColor  = lipgloss.Color("#88909C")
	warnColor   = lipgloss.Color("#FFD166")
	dangerColor = lipgloss.Color("#FF6B6B")
	panelColor  = lipgloss.Color("#323846")
	textColor   = lipgloss.Color("#F3F5F7")
)

func applyStyling() {
	collect.ApplyMainStyles(accentColor, mutedColor, warnColor, dangerColor, panelColor, textColor)
}

func init() {
	applyStyling()
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

	// UNSLOTH_STUDIO_URL is read by the collector during discovery.
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

	collect.Configure(collect.Config{
		StudioPort:  studioPort,
		StudioToken: studioToken,
	})

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

	renderFuncs := ui.RenderFuncs{
		RenderContentFn:   render.RenderContent,
		RenderHeaderFn:    render.RenderHeader,
		RenderFooterFn:    render.RenderFooter,
		RenderHelpOverlay: render.RenderHelpOverlay,
	}

	model := ui.NewModel(*interval, ui.UIConfig{}, renderFuncs)
	p := tea.NewProgram(model, tea.WithAltScreen())
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
