package collect

import (
	"context"
	"encoding/csv"
	stdfmt "fmt"
	"sort"
	stdstrconv "strconv"
	stdstrings "strings"
)

func parseCSV(output string) ([][]string, error) {
	output = stdstrings.TrimSpace(output)
	if output == "" {
		return nil, nil
	}

	r := csv.NewReader(stdstrings.NewReader(output))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1

	return r.ReadAll()
}

func CollectNVIDIA(ctx context.Context, warnings *[]string, prevGpus []GpuStats) []GpuStats {
	_, err := execLookPath("nvidia-smi")
	if err != nil {
		return nil
	}

	fields := "index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed"
	stdout, stderr, err := RunCommand(ctx, "nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		AddWarning(warnings, "nvidia-smi GPU query failed: "+CleanCommandError(err, stderr))
		return nil
	}

	gpus, err := parseGPUCSVFields(stdout)
	if err != nil {
		AddWarning(warnings, "could not parse nvidia-smi GPU metrics: "+CleanError(err.Error()))
		return nil
	}

	processes, processWarning := CollectGPUProcesses(ctx)
	if processWarning != "" {
		AddWarning(warnings, processWarning)
	}

	byUUID := make(map[string]int, len(gpus))
	for i := range gpus {
		byUUID[gpus[i].UUID] = i
	}
	for _, p := range processes {
		if idx, ok := byUUID[p.GPUUUID]; ok {
			gpus[idx].Processes = append(gpus[idx].Processes, p)
		}
	}

	AddGPUUtilDelta(prevGpus, gpus)
	AddVRAMDeltas(prevGpus, gpus)

	return gpus
}

func CollectAMD(ctx context.Context, warnings *[]string, prevGpus []AmdGpuStats) []AmdGpuStats {
	_, err := execLookPath("amdgpu_top")
	if err != nil {
		stats := CollectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			AddAMDUtilDelta(nil, stats)
		}
		return stats
	}

	stdout, stderr, err := RunCommand(ctx, "amdgpu_top", "--json", "--no-pc", "-n", "1", "-s", "100")
	if err != nil {
		AddWarning(warnings, "amdgpu_top query failed: "+CleanCommandError(err, stderr))
		stats := CollectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			AddAMDUtilDelta(prevGpus, stats)
		}
		return stats
	}

	// Parse the amdgpu_top JSON output.
	gpus, err := parseAMDJSON(stdout)
	if err != nil {
		AddWarning(warnings, "amdgpu_top JSON parse failed: "+err.Error())
		stats := CollectAMDFromSysfs(warnings)
		if len(stats) > 0 {
			AddAMDUtilDelta(prevGpus, stats)
		}
		return stats
	}
	if len(gpus) > 0 {
		AddAMDUtilDelta(prevGpus, gpus)
	}
	return gpus
}

func BuildInferenceProcessList(nvidia []GpuStats, amd []AmdGpuStats) []InferenceProcess {
	inference := make([]InferenceProcess, 0)

	for _, gpu := range nvidia {
		for _, p := range gpu.Processes {
			if !p.UsedMemoryMB.OK {
				continue
			}
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			inference = append(inference, InferenceProcess{
				PID:      p.PID,
				Provider: p.Provider,
				Name:     name,
				VRAMMiB:  p.UsedMemoryMB,
				GTTMiB:   OptFloat{},
			})
		}
	}

	for _, gpu := range amd {
		for _, p := range gpu.Processes {
			name := p.Name
			if name == "" {
				name = "unknown"
			}
			inference = append(inference, InferenceProcess{
				PID:      p.PID,
				Provider: p.Provider,
				Name:     name,
				VRAMMiB:  p.VRAMMiB,
				GTTMiB:   p.GTTMiB,
			})
		}
	}

	sort.SliceStable(inference, func(i, j int) bool {
		if inference[i].VRAMMiB.OK != inference[j].VRAMMiB.OK {
			return inference[i].VRAMMiB.OK
		}
		return inference[i].VRAMMiB.Value > inference[j].VRAMMiB.Value
	})

	return inference
}

func CollectInference(ctx context.Context, nvidia []GpuStats, amd []AmdGpuStats, warnings *[]string) []InferenceProcess {
	return BuildInferenceProcessList(nvidia, amd)
}

func parseGPUCSVFields(output string) ([]GpuStats, error) {
	records, err := parseCSV(output)
	if err != nil {
		return nil, err
	}

	gpus := make([]GpuStats, 0, len(records))
	for _, row := range records {
		if len(row) < 10 {
			return nil, stdfmt.Errorf("expected 10 GPU fields, got %d", len(row))
		}

		util := parseOptFloat(row[3])
		memUsed := parseOptFloat(row[4])
		memTotal := parseOptFloat(row[5])
		temp := parseOptFloat(row[6])
		power := parseOptFloat(row[7])
		powerLimit := parseOptFloat(row[8])
		fan := parseOptFloat(row[9])

		gpus = append(gpus, GpuStats{
			Index:       stdstrings.TrimSpace(row[0]),
			Name:        stdstrings.TrimSpace(row[1]),
			UUID:        stdstrings.TrimSpace(row[2]),
			UtilPercent: util,
			MemoryUsed:  memUsed,
			MemoryTotal: memTotal,
			Temperature: temp,
			PowerDraw:   power,
			PowerLimit:  powerLimit,
			FanPercent:  fan,
		})
	}

	return gpus, nil
}

func parseOptFloat(raw string) OptFloat {
	value := stdstrings.TrimSpace(raw)
	lower := stdstrings.ToLower(value)
	if value == "" || lower == "n/a" || lower == "na" || stdstrings.Contains(lower, "not supported") || stdstrings.Contains(lower, "not available") {
		return OptFloat{}
	}

	parsed, err := stdstrconv.ParseFloat(value, 64)
	if err != nil {
		return OptFloat{}
	}
	return OptFloat{Value: parsed, OK: true}
}

func CollectGPUProcesses(ctx context.Context) ([]GpuProcess, string) {
	fields := "gpu_uuid,pid,process_name,used_memory"
	stdout, stderr, err := RunCommand(ctx, "nvidia-smi", "--query-compute-apps="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		text := stdstrings.ToLower(stdout + " " + stderr + " " + err.Error())
		if stdstrings.Contains(text, "no running") || stdstrings.Contains(text, "not supported") {
			return nil, ""
		}
		return nil, "nvidia-smi process query failed: " + CleanCommandError(err, stderr)
	}

	records, err := parseCSV(stdout)
	if err != nil {
		return nil, "could not parse nvidia-smi process metrics: " + CleanError(err.Error())
	}

	processes := make([]GpuProcess, 0, len(records))
	for _, row := range records {
		if len(row) < 4 {
			continue
		}

		pid64, err := stdstrconv.ParseInt(stdstrings.TrimSpace(row[1]), 10, 32)
		if err != nil {
			continue
		}

		processName := stdstrings.TrimSpace(row[2])
		cmdline := processCmdline(int32(pid64))
		containerName := processContainerName(int32(pid64))
		displayName := processName
		if containerName != "" {
			displayName = containerName
		}
		processes = append(processes, GpuProcess{
			GPUUUID:      stdstrings.TrimSpace(row[0]),
			PID:          int32(pid64),
			Provider:     classifyProvider(processName, cmdline),
			Name:         displayName,
			UsedMemoryMB: parseOptFloat(row[3]),
		})
	}

	sort.Slice(processes, func(i, j int) bool {
		if processes[i].GPUUUID == processes[j].GPUUUID {
			return processes[i].PID < processes[j].PID
		}
		return processes[i].GPUUUID < processes[j].GPUUUID
	})

	return processes, ""
}

func classifyProvider(name string, cmdline string) string {
	lower := stdstrings.ToLower(stdstrings.TrimSpace(name))
	cmdLower := stdstrings.ToLower(stdstrings.TrimSpace(cmdline))
	text := lower + " " + cmdLower

	switch {
	case stdstrings.Contains(text, "python") && (stdstrings.Contains(text, "transform") || stdstrings.Contains(text, "torch") || stdstrings.Contains(text, "pytorch") || stdstrings.Contains(text, "huggingface") || stdstrings.Contains(text, "llama") || stdstrings.Contains(text, "bert") || stdstrings.Contains(text, "gpt")):
		return "PyTorch"
	case stdstrings.Contains(text, "python") && stdstrings.Contains(text, "ollama"):
		return "Ollama"
	case stdstrings.Contains(text, "python") && stdstrings.Contains(text, "unsloth"):
		return "Unsloth"
	case stdstrings.Contains(text, "python") && (stdstrings.Contains(text, "text-generation") || stdstrings.Contains(text, "vllm")):
		return "vLLM"
	case stdstrings.Contains(text, "python") && stdstrings.Contains(text, "kobold"):
		return "KoboldCPP"
	case stdstrings.Contains(text, "ollama") && !stdstrings.Contains(text, "python"):
		return "Ollama"
	case stdstrings.Contains(text, "unsloth"):
		return "Unsloth"
	case stdstrings.Contains(text, "vllm") || stdstrings.Contains(text, "text-generation"):
		return "vLLM"
	case stdstrings.Contains(text, "kobold"):
		return "KoboldCPP"
	case stdstrings.Contains(text, "xtt") || stdstrings.Contains(text, "voyager") || stdstrings.Contains(text, "sglm") || stdstrings.Contains(text, "llama.cpp") || stdstrings.Contains(text, "llama-srv"):
		return "Local Inference"
	default:
		return ""
	}
}
