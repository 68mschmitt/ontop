package collect

import (
	"context"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

type tickMsg time.Time

type metricsMsg struct {
	Snapshot Snapshot
}

func TickCmd(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func CollectMetricsCmd() tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
		defer cancel()

		return metricsMsg{Snapshot: CollectMetrics(ctx)}
	}
}
