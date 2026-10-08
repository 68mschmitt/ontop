package collect

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

func RenderWarnings(warnings []string, width int) string {
	seen := make(map[string]bool, len(warnings))
	lines := make([]string, 0, len(warnings))
	for _, warning := range warnings {
		warning = strings.TrimSpace(warning)
		if warning == "" || seen[warning] {
			continue
		}
		seen[warning] = true
		lines = append(lines, warnStyle.Render("! ")+FitText(warning, maxInt(10, width-8)))
	}
	if len(lines) == 0 {
		return ""
	}
	return RenderCard("Notices", strings.Join(lines, "\n"), width)
}

func RenderSystem(s Snapshot, width int, disabledSections map[string]bool) string {
	if disabledSections != nil && disabledSections["System"] {
		return ""
	}
	cpuBody := RenderCPU(s.CPU, s.CPUHistory, width)

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

func RenderCPU(stats CpuStats, history []float64, width int) string {
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

func renderThermal(stats ThermalStats, width int) string {
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

func RenderMemory(stats MemoryStats, history []float64, width int) string {
	if !stats.OK {
		return mutedStyle.Render("RAM metrics unavailable.")
	}

	innerWidth := maxInt(20, width-6)
	used := fmt.Sprintf("%s / %s (available %s)", HumanBytes(stats.Used), HumanBytes(stats.Total), HumanBytes(stats.Available))
	lines := []string{metricLine("RAM", stats.Percent, fmt.Sprintf("%.1f%%  %s", stats.Percent, used), innerWidth)}

	if len(history) > 0 {
		spark := renderSparkline(history, minInt(len(history), 20))
		lines = append(lines, mutedStyle.Render("trend: "+spark))
	}

	return strings.Join(lines, "\n")
}

func RenderMemoryCard(mem MemoryStats, swap SwapMemoryStats, history []float64, width int) string {
	if !mem.OK {
		return RenderCard("Memory", mutedStyle.Render("RAM metrics unavailable."), width)
	}

	innerWidth := maxInt(20, width-6)
	lines := []string{metricLine("RAM", mem.Percent, fmt.Sprintf("free %s", HumanBytes(mem.Available)), innerWidth)}

	if swap.OK && swap.Total > 0 {
		swapUsed := fmt.Sprintf("%s / %s", HumanBytes(swap.Used), HumanBytes(swap.Total))
		swapPercent := float64(swap.Used) / float64(swap.Total) * 100
		lines = append(lines, "")
		lines = append(lines, metricLine("Swap", swapPercent, fmt.Sprintf("%s", swapUsed), innerWidth))
	}
	if len(history) > 0 {
		lines = append(lines, "")
		lines = append(lines, mutedStyle.Render("trend: "+renderSparkline(history, minInt(len(history), maxInt(10, innerWidth-10)))))
	}

	return RenderCard("Memory", strings.Join(lines, "\n"), width)
}

func RenderGPUSection(gpus []GpuStats, sparkline map[string][]float64, width int, disabledSections map[string]bool) string {
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
		lines = append(lines, valueStyle.Render(FitText(title, innerWidth)))

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

		llm := gpuProcessesByProvider(gpu.Processes, true)
		other := gpuProcessesByProvider(gpu.Processes, false)

		if len(llm) > 0 {
			lines = append(lines, labelStyle.Render("LLM Processes"))
			lines = append(lines, renderGPUProcessTable(llm, innerWidth))
		}
		if len(other) > 0 {
			lines = append(lines, labelStyle.Render("Other Processes"))
			lines = append(lines, renderGPUProcessTable(other, innerWidth))
		}
	}

	return RenderCard("NVIDIA GPU", strings.Join(lines, "\n"), width)
}

func RenderAMDSection(gpus []AmdGpuStats, width int, disabledSections map[string]bool) string {
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
		lines = append(lines, valueStyle.Render(FitText(title, innerWidth)))

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

		lines = append(lines, mutedStyle.Render("Temp "+optTemperatureString(gpu.Temperature)+"   Power "+optPowerString(gpu.PowerDraw, OptFloat{})))
	}

	return RenderCard("AMD GPU", strings.Join(lines, "\n"), width)
}

func RenderDiskCard(stats DiskStats, width int, disabledSections map[string]bool) string {
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
	return RenderCard("Disk I/O", text, width)
}

func RenderNetCard(stats NetStats, width int, disabledSections map[string]bool) string {
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
	return RenderCard("Network I/O", text, width)
}

func RenderOllamaProcesses(processes []OllamaProcess, width int, disabledSections map[string]bool) string {
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
				HumanBytes(p.RSS),
				FitText(commandLabel(p), cmdWidth),
			)
			lines = append(lines, line)
		}
		return RenderCard("Ollama Processes", strings.Join(lines, "\n"), width)
	}

	cmdWidth := maxInt(12, innerWidth-55)
	lines = append(lines, labelStyle.Render(fmt.Sprintf("%-7s %7s %8s %8s %-9s %s", "PID", "CPU", "RAM", "RSS", "RUNTIME", "COMMAND")))
	for _, p := range processes {
		line := fmt.Sprintf("%-7d %7s %8s %8s %-9s %s",
			p.PID,
			fmt.Sprintf("%.1f%%", p.CPUPercent),
			fmt.Sprintf("%.1f%%", p.MemoryPercent),
			HumanBytes(p.RSS),
			runtimeLabel(p),
			FitText(commandLabel(p), cmdWidth),
		)
		lines = append(lines, line)
	}

	return RenderCard("Ollama Processes", strings.Join(lines, "\n"), width)
}

func RenderOllamaPS(output CommandOutput, width int, disabledSections map[string]bool) string {
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
			lines := []string{labelStyle.Render(FitText("MODEL  CONTEXT", innerWidth))}
			var totalCtx, totalPrompt int
			for _, model := range output.Models {
				totalCtx += model.CtxTokens
				totalPrompt += model.PromptTokens
				line := fmt.Sprintf("%-*s %s", nameWidth, FitText(model.Name, nameWidth), model.Context)
				lines = append(lines, FitText(line, innerWidth))
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
			return RenderCard("Ollama Models", strings.Join(lines, "\n"), width)
		}

		nameWidth := clampInt(innerWidth/3, 12, 30)
		lines := []string{labelStyle.Render(fmt.Sprintf("%-*s %-14s %-12s %-8s %s", nameWidth, "MODEL", "SIZE", "PROCESSOR", "CTX", "UNTIL"))}
		var totalCtx, totalPrompt int
		for _, model := range output.Models {
			totalCtx += model.CtxTokens
			totalPrompt += model.PromptTokens
			line := fmt.Sprintf("%-*s %-14s %-12s %-8s %s",
				nameWidth,
				FitText(model.Name, nameWidth),
				FitText(model.Size, 14),
				FitText(model.Processor, 12),
				model.Context,
				FitText(model.Until, maxInt(8, innerWidth-nameWidth-39)),
			)
			lines = append(lines, FitText(line, innerWidth))
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
		return RenderCard("Ollama Models", strings.Join(lines, "\n"), width)
	}

	lines := strings.Split(strings.TrimRight(output.Output, "\n"), "\n")
	for i := range lines {
		lines[i] = FitText(lines[i], innerWidth)
	}

	return RenderCard("Ollama Models", strings.Join(lines, "\n"), width)
}

func RenderUnslothStudio(s UnslothStudioStats, width int, disabledSections map[string]bool) string {
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
		modelLine := valueStyle.Render(FitText(s.ActiveModel, innerWidth-2))
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
			lines = append(lines, warnStyle.Render("loading: ")+FitText(m, innerWidth-10))
		}
	}

	// Load progress bar.
	if s.LoadPhase != "" && s.LoadTotal > 0 {
		pct := float64(s.LoadBytes) / float64(s.LoadTotal) * 100
		barLabel := fmt.Sprintf("%s / %s", HumanBytes(uint64(s.LoadBytes)), HumanBytes(uint64(s.LoadTotal)))
		lines = append(lines, metricLine("Load", pct, barLabel, innerWidth))
	}

	// Loaded models list.
	if len(s.LoadedModels) > 0 && s.ActiveModel == "" {
		lines = append(lines, "")
		for _, m := range s.LoadedModels {
			lines = append(lines, labelStyle.Render("  ")+FitText(m, innerWidth-4))
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
		lines = append(lines, warnStyle.Render("! ")+FitText(s.Error, innerWidth-4))
	}

	if s.ActiveModel == "" && len(s.LoadingModels) == 0 && s.LoadPhase == "" && s.TrainStatus == "" && s.Error == "" {
		return ""
	}

	return RenderCard("Unsloth Studio", strings.Join(lines, "\n"), width)
}

func renderGroup(title, body string, width int) string {
	innerWidth := maxInt(20, width-4)
	header := sectionTitleStyle.Render(" ━ " + title + " ━ ")
	s := cardStyle.Width(innerWidth)
	return s.Render(lipgloss.JoinVertical(lipgloss.Left, header, body))
}

func RenderCard(title, body string, width int) string {
	innerWidth := maxInt(20, width-4)
	content := sectionTitleStyle.Render(title)
	if strings.TrimSpace(body) != "" {
		content += "\n" + body
	}
	return cardStyle.Width(innerWidth).Render(content)
}

func metricLine(label string, percent float64, value string, width int) string {
	labelWidth := 8
	value = FitText(value, maxInt(8, width-labelWidth-2))
	valueWidth := runewidth.StringWidth(value)
	barWidth := width - labelWidth - valueWidth - 2
	if barWidth < 3 {
		return fmt.Sprintf("%s %s", labelStyle.Width(labelWidth).Render(label), value)
	}
	barWidth = clampInt(barWidth, 3, 28)

	return fmt.Sprintf("%s %s %s",
		labelStyle.Width(labelWidth).Render(label),
		RenderBar(percent, barWidth),
		value,
	)
}

func RenderCoreCell(index int, percent float64, width int) string {
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

func RenderCoreRow(perCore []float64, width int) string {
	if len(perCore) == 0 || width <= 0 {
		return ""
	}

	compact := make([]string, len(perCore))
	for i, percent := range perCore {
		compact[i] = fmt.Sprintf("C%d:%.0f%%", i, percent)
	}
	cells := make([]string, len(perCore))
	for i, percent := range perCore {
		cells[i] = RenderCoreCell(i, percent, 18)
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

func RenderBar(percent float64, width int) string {
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

func renderDisk(stats DiskStats, width int) string {
	innerWidth := maxInt(20, width-6)
	var lines []string

	barWidth := maxInt(10, innerWidth-70)

	for i, dev := range stats.Devices {
		readStr := HumanBytes(dev.ReadBytes)
		writeStr := HumanBytes(dev.WriteBytes)

		totalIO := float64(dev.ReadIOss + dev.WriteIOSS)
		lines = append(lines, labelStyle.Render(fmt.Sprintf("%s", dev.Name)))
		if len(stats.Rates) > i && stats.Rates[i].ReadBps > 0 {
			r := stats.Rates[i]
			if totalIO > 0 {
				pct := float64(dev.WriteIOSS) / float64(totalIO) * 100
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", RenderBar(pct, barWidth), readStr, writeStr, HumanBytesRate(r.ReadBps), HumanBytesRate(r.WriteBps)))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", barEmptyStyle.Render(strings.Repeat("-", barWidth)), readStr, writeStr, HumanBytesRate(r.ReadBps), HumanBytesRate(r.WriteBps)))
			}
		} else {
			if totalIO > 0 {
				pct := float64(dev.WriteIOSS) / float64(totalIO) * 100
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", RenderBar(pct, barWidth), readStr, writeStr))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", barEmptyStyle.Render(strings.Repeat("-", barWidth)), readStr, writeStr))
			}
		}
	}

	return strings.Join(lines, "\n")
}

func renderNet(stats NetStats, width int) string {
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
		bar := RenderBar(pct, barWidth)
		sentStr := HumanBytes(dev.BytesSent)
		recvStr := HumanBytes(dev.BytesRecv)

		lines = append(lines, labelStyle.Render(fmt.Sprintf("%s", dev.Name)))
		if len(stats.Rates) > i && stats.Rates[i].BytesSentBps > 0 {
			r := stats.Rates[i]
			lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s  \u2193 %s/s  \u2191 %s/s", bar, recvStr, sentStr, HumanBytesRate(r.BytesRecvBps), HumanBytesRate(r.BytesSentBps)))
		} else {
			lines = append(lines, fmt.Sprintf("  %s  \u2193 %s  \u2191 %s", bar, recvStr, sentStr))
		}
	}

	return strings.Join(lines, "\n")
}

func renderGPUProcessTable(processes []GpuProcess, width int) string {
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
			lines = append(lines, fmt.Sprintf("%-7d %-9s %s", p.PID, coloredValue(vramStr, percent), FitText(name, processWidth)))
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
		line := fmt.Sprintf("%-7d %-9s %s", p.PID, coloredValue(vramStr, percent), FitText(name, processWidth))
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func renderSparkline(values []float64, width int) string {
	if len(values) == 0 {
		return strings.Repeat(".", width)
	}

	chars := []rune("▁▂▃▄▅▆▇█")

	var result strings.Builder
	step := float64(len(values)) / float64(width)

	for i := 0; i < width && i < len(values); i++ {
		idx := int(float64(i) / step)
		if idx >= len(values) {
			idx = len(values) - 1
		}
		level := int(values[idx] / 100.0 * 8)
		level = clampInt(level, 0, len(chars)-1)
		result.WriteRune(chars[level])
	}

	return result.String()
}

// Helper functions

func gpuProcessesByProvider(processes []GpuProcess, isLLM bool) []GpuProcess {
	out := make([]GpuProcess, 0)
	for _, p := range processes {
		matches := p.Provider != ""
		if isLLM {
			if matches {
				out = append(out, p)
			}
		} else {
			if !matches {
				out = append(out, p)
			}
		}
	}
	return out
}

func commandLabel(p OllamaProcess) string {
	if p.Command != "" {
		return p.Command
	}
	if p.Name != "" {
		return p.Name
	}
	return "unknown"
}

func runtimeLabel(p OllamaProcess) string {
	if !p.RuntimeOK {
		return "n/a"
	}
	return shortDuration(p.Runtime)
}

func optPercentValue(value OptFloat) float64 {
	if !value.OK {
		return 0
	}
	return value.Value
}

func optPercentString(value OptFloat) string {
	if !value.OK {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", value.Value)
}

func optTemperatureString(value OptFloat) string {
	if !value.OK {
		return "n/a"
	}
	return fmt.Sprintf("%.0f C", value.Value)
}

func optPowerString(draw OptFloat, limit OptFloat) string {
	if draw.OK && limit.OK {
		return fmt.Sprintf("%.0f / %.0f W", draw.Value, limit.Value)
	}
	if draw.OK {
		return fmt.Sprintf("%.0f W", draw.Value)
	}
	return "n/a"
}

func optMemoryString(value OptFloat) string {
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

func HumanBytes(value uint64) string {
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

func HumanBytesRate(bps float64) string {
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

func coloredValue(text string, percent float64) string {
	return severityStyle(percent).Render(text)
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
