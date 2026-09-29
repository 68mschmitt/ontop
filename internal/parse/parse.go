package parse

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"ontop/internal/collect"
)

// ClassifyProvider maps a process name and optional cmdline to a specific
// provider label. Returns "" for unrecognized names (rendered as "other").
func ClassifyProvider(name string, cmdline string) string {
	lower := strings.ToLower(strings.TrimSpace(name))
	cmdLower := strings.ToLower(strings.TrimSpace(cmdline))

	text := lower + " " + cmdLower

	switch {
	case strings.Contains(text, "python") && (strings.Contains(text, "transform") || strings.Contains(text, "torch") || strings.Contains(text, "pytorch") || strings.Contains(text, "huggingface") || strings.Contains(text, "llama") || strings.Contains(text, "bert") || strings.Contains(text, "gpt")):
		return "PyTorch"
	case strings.Contains(text, "python") && (strings.Contains(text, "ollama")):
		return "Ollama"
	case strings.Contains(text, "python") && (strings.Contains(text, "unsloth")):
		return "Unsloth"
	case strings.Contains(text, "python") && (strings.Contains(text, "text-generation") || strings.Contains(text, "vllm")):
		return "vLLM"
	case strings.Contains(text, "python") && (strings.Contains(text, "kobold")):
		return "KoboldCPP"
	case strings.Contains(text, "ollama") && !strings.Contains(text, "python"):
		return "Ollama"
	case strings.Contains(text, "unsloth"):
		return "Unsloth"
	case strings.Contains(text, "vllm") || strings.Contains(text, "text-generation"):
		return "vLLM"
	case strings.Contains(text, "kobold"):
		return "KoboldCPP"
	case strings.Contains(text, "xtt") || strings.Contains(text, "voyager") || strings.Contains(text, "sglm") || strings.Contains(text, "llama.cpp") || strings.Contains(text, "llama-srv"):
		return "Local Inference"
	default:
		return ""
	}
}

func ReadCSV(output string) ([][]string, error) {
	output = strings.TrimSpace(output)
	if output == "" {
		return nil, nil
	}

	r := csv.NewReader(strings.NewReader(output))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1

	return r.ReadAll()
}

func ParseOptFloat(raw string) collect.OptFloat {
	value := strings.TrimSpace(raw)
	lower := strings.ToLower(value)
	if value == "" || lower == "n/a" || lower == "na" || strings.Contains(lower, "not supported") || strings.Contains(lower, "not available") {
		return collect.OptFloat{}
	}

	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return collect.OptFloat{}
	}
	return collect.OptFloat{Value: parsed, OK: true}
}

func ParseGPUCSV(output string) ([]collect.GpuStats, error) {
	records, err := ReadCSV(output)
	if err != nil {
		return nil, err
	}

	gpus := make([]collect.GpuStats, 0, len(records))
	for _, row := range records {
		if len(row) < 10 {
			return nil, fmt.Errorf("expected 10 GPU fields, got %d", len(row))
		}

		gpus = append(gpus, collect.GpuStats{
			Index:       strings.TrimSpace(row[0]),
			Name:        strings.TrimSpace(row[1]),
			UUID:        strings.TrimSpace(row[2]),
			UtilPercent: ParseOptFloat(row[3]),
			MemoryUsed:  ParseOptFloat(row[4]),
			MemoryTotal: ParseOptFloat(row[5]),
			Temperature: ParseOptFloat(row[6]),
			PowerDraw:   ParseOptFloat(row[7]),
			PowerLimit:  ParseOptFloat(row[8]),
			FanPercent:  ParseOptFloat(row[9]),
		})
	}

	return gpus, nil
}

func ParseAMDJSON(output string) ([]collect.AmdGpuStats, error) {
	var document AmdTopDocument
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &document); err != nil {
		return nil, err
	}

	gpus := make([]collect.AmdGpuStats, 0, len(document.Devices))
	for i, device := range document.Devices {
		gpu := collect.AmdGpuStats{
			Index:       strconv.Itoa(i),
			PCI:         device.Info.DevicePath.PCI,
			Name:        device.Info.DeviceName,
			UtilPercent: AMDMetric(device.GPUActivity, "GFX"),
			MemoryUsed:  AMDMetric(device.VRAM, "Total VRAM Usage"),
			MemoryTotal: AMDMetric(device.VRAM, "Total VRAM"),
			Temperature: FirstAMDMetric(device.Sensors, "Junction Temperature", "Edge Temperature"),
			PowerDraw:   FirstAMDMetric(device.Sensors, "Average Power", "GFX Power", "Input Power"),
			FanPercent:  FirstAMDMetric(device.Sensors, "Fan Speed", "Fan"),
			Processes:   ParseAMDProcesses(device.FDInfo),
		}
		if gpu.Name == "" {
			gpu.Name = "AMD GPU"
		}
		gpus = append(gpus, gpu)
	}

	return gpus, nil
}

func ParseAMDProcesses(raw map[string]json.RawMessage) []collect.AmdGpuProcess {
	processes := make([]collect.AmdGpuProcess, 0, len(raw))
	for pidString, processRaw := range raw {
		pid, err := strconv.ParseInt(pidString, 10, 32)
		if err != nil {
			continue
		}

		name, vram, gtt := ParseAMDProcess(processRaw)
		processes = append(processes, collect.AmdGpuProcess{
			PID:      int32(pid),
			Provider: ClassifyProvider(name, ""),
			Name:     name,
			VRAMMiB:  vram,
			GTTMiB:   gtt,
		})
	}

	sort.Slice(processes, func(i, j int) bool {
		return processes[i].PID < processes[j].PID
	})
	return processes
}

func ParseAMDProcess(raw json.RawMessage) (string, collect.OptFloat, collect.OptFloat) {
	var current map[string]json.RawMessage
	if json.Unmarshal(raw, &current) != nil {
		return "", collect.OptFloat{}, collect.OptFloat{}
	}

	name := ""
	if value, ok := current["name"]; ok {
		_ = json.Unmarshal(value, &name)
	}

	for depth := 0; depth < 4; depth++ {
		if value, ok := current["VRAM"]; ok {
			return name, AMDMetricValue(value), AMDNestedMetric(current, "GTT")
		}
		value, ok := current["usage"]
		if !ok || json.Unmarshal(value, &current) != nil {
			break
		}
	}

	return name, collect.OptFloat{}, collect.OptFloat{}
}

func AMDNestedMetric(metrics map[string]json.RawMessage, key string) collect.OptFloat {
	value, ok := metrics[key]
	if !ok {
		return collect.OptFloat{}
	}
	return AMDMetricValue(value)
}

func AMDMetric(metrics map[string]json.RawMessage, key string) collect.OptFloat {
	value, ok := metrics[key]
	if !ok {
		return collect.OptFloat{}
	}
	return AMDMetricValue(value)
}

func FirstAMDMetric(metrics map[string]json.RawMessage, keys ...string) collect.OptFloat {
	for _, key := range keys {
		if value := AMDMetric(metrics, key); value.OK {
			return value
		}
	}
	return collect.OptFloat{}
}

func AMDMetricValue(raw json.RawMessage) collect.OptFloat {
	var value struct {
		Value *float64 `json:"value"`
	}
	if json.Unmarshal(raw, &value) == nil && value.Value != nil {
		return collect.OptFloat{Value: *value.Value, OK: true}
	}

	var number float64
	if json.Unmarshal(raw, &number) == nil {
		return collect.OptFloat{Value: number, OK: true}
	}
	return collect.OptFloat{}
}

func ParseOllamaPSJSON(output string) []collect.OllamaModel {
	var models []collect.OllamaModel

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
		models = append(models, collect.OllamaModel{
			Name:         l.Name,
			ID:           l.Id,
			Size:         fmt.Sprintf("%.1f GB", sizeGB),
			PromptTokens: l.PromptTokens,
			CtxTokens:    l.ContextLength,
		})
	}

	return models
}

func ParseOllamaPS(output string) []collect.OllamaModel {
	var models []collect.OllamaModel
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 || strings.EqualFold(fields[0], "name") {
			continue
		}

		processorStart := -1
		contextIndex := -1
		for i := 2; i < len(fields); i++ {
			if strings.Contains(fields[i], "%") {
				processorStart = i
				break
			}
		}
		if processorStart < 0 {
			continue
		}
		for i := processorStart + 1; i < len(fields); i++ {
			contextField := strings.ToUpper(strings.TrimSuffix(fields[i], ","))
			if _, err := strconv.ParseInt(contextField, 10, 64); err == nil || strings.HasSuffix(contextField, "K") || strings.HasSuffix(contextField, "M") {
				contextIndex = i
				break
			}
		}
		if contextIndex < 0 {
			continue
		}

		models = append(models, collect.OllamaModel{
			Name:      fields[0],
			ID:        fields[1],
			Size:      strings.Join(fields[2:processorStart], " "),
			Processor: strings.Join(fields[processorStart:contextIndex], " "),
			Context:   fields[contextIndex],
			Until:     strings.Join(fields[contextIndex+1:], " "),
		})
	}
	return models
}
