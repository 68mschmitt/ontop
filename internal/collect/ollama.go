package collect

import (
	"context"
	"encoding/json"
	stdfmt "fmt"
	"sort"
	stdstrconv "strconv"
	stdstrings "strings"
	"time"

	"github.com/shirou/gopsutil/v3/process"
)

func CollectOllamaProcesses(ctx context.Context, warnings *[]string) []OllamaProcess {
	procs, err := process.ProcessesWithContext(ctx)
	if err != nil {
		AddWarning(warnings, "process list unavailable: "+CleanError(err.Error()))
		return nil
	}

	now := time.Now()
	ollamaProcs := make([]OllamaProcess, 0)
	for _, p := range procs {
		name, _ := p.NameWithContext(ctx)
		cmdline, _ := p.CmdlineWithContext(ctx)
		identity := stdstrings.ToLower(name + " " + cmdline)
		if !stdstrings.Contains(identity, "ollama") {
			continue
		}

		item := OllamaProcess{
			PID:     p.Pid,
			Name:    stdstrings.TrimSpace(name),
			Command: stdstrings.TrimSpace(cmdline),
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

func CollectOllamaPS(ctx context.Context) CommandOutput {
	_, err := execLookPath("ollama")
	if err != nil {
		return CommandOutput{Missing: true, Error: "ollama command not found"}
	}

	stdout, stderr, err := RunCommand(ctx, "ollama", "ps", "--json")
	if err == nil {
		output := stdstrings.TrimSpace(stdout)
		models := parseOllamaPSJSON(output)
		return CommandOutput{Output: output, Models: models}
	}

	textStdout, _, textErr := RunCommand(ctx, "ollama", "ps")
	if textErr != nil {
		return CommandOutput{Error: CleanCommandError(err, stderr)}
	}

	output := stdstrings.TrimSpace(textStdout)
	return CommandOutput{Output: output, Models: parseOllamaPS(output)}
}

func parseOllamaPSJSON(output string) []OllamaModel {
	var models []OllamaModel

	type ollamaLine struct {
		Name          string  `json:"name"`
		Id            string  `json:"id"`
		Size          float64 `json:"size"`
		SizeVRAM      float64 `json:"size_vram"`
		TransferRate  string  `json:"transfer_rate"`
		ContextLength int     `json:"context_length"`
		PromptTokens  int     `json:"prompt_tokens"`
		OutputTokens  int     `json:"output_tokens"`
	}

	var lines []ollamaLine
	if err := json.Unmarshal([]byte(output), &lines); err != nil {
		return nil
	}

	for _, l := range lines {
		sizeGB := l.Size / 1e9
		models = append(models, OllamaModel{
			Name:         l.Name,
			ID:           l.Id,
			Size:         stdfmt.Sprintf("%.1f GB", sizeGB),
			PromptTokens: l.PromptTokens,
			CtxTokens:    l.ContextLength,
		})
	}

	return models
}

func parseOllamaPS(output string) []OllamaModel {
	var models []OllamaModel
	for _, line := range stdstrings.Split(stdstrings.TrimSpace(output), "\n") {
		fields := stdstrings.Fields(line)
		if len(fields) < 4 || stdstrings.EqualFold(fields[0], "name") {
			continue
		}

		processorStart := -1
		contextIndex := -1
		for i := 2; i < len(fields); i++ {
			if stdstrings.Contains(fields[i], "%") {
				processorStart = i
				break
			}
		}
		if processorStart < 0 {
			continue
		}
		for i := processorStart + 1; i < len(fields); i++ {
			contextField := stdstrings.ToUpper(stdstrings.TrimSuffix(fields[i], ","))
			if _, err := stdstrconv.ParseInt(contextField, 10, 64); err == nil || stdstrings.HasSuffix(contextField, "K") || stdstrings.HasSuffix(contextField, "M") {
				contextIndex = i
				break
			}
		}
		if contextIndex < 0 {
			continue
		}

		models = append(models, OllamaModel{
			Name:      fields[0],
			ID:        fields[1],
			Size:      stdstrings.Join(fields[2:processorStart], " "),
			Processor: stdstrings.Join(fields[processorStart:contextIndex], " "),
			Context:   fields[contextIndex],
			Until:     stdstrings.Join(fields[contextIndex+1:], " "),
		})
	}
	return models
}
