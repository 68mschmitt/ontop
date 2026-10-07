package collect

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// amdTopDocument is the top-level structure returned by `amdgpu_top --json`.
type amdTopDocument struct {
	Devices []amdTopDevice `json:"devices"`
}

// amdTopDevice is a single device in the amdgpu_top JSON output.
type amdTopDevice struct {
	Info struct {
		DeviceName string `json:"DeviceName"`
		DevicePath struct {
			PCI string `json:"pci"`
		} `json:"DevicePath"`
	} `json:"Info"`
	GPUActivity map[string]json.RawMessage `json:"gpu_activity"`
	Sensors     map[string]json.RawMessage `json:"Sensors"`
	VRAM        map[string]json.RawMessage `json:"VRAM"`
	FDInfo      map[string]json.RawMessage `json:"fdinfo"`
}

func parseAMDJSON(output string) ([]AmdGpuStats, error) {
	var document amdTopDocument
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &document); err != nil {
		return nil, err
	}

	gpus := make([]AmdGpuStats, 0, len(document.Devices))
	for i, device := range document.Devices {
		gpu := AmdGpuStats{
			Index:       strconv.Itoa(i),
			PCI:         device.Info.DevicePath.PCI,
			Name:        device.Info.DeviceName,
			UtilPercent: amdMetric(device.GPUActivity, "GFX"),
			MemoryUsed:  amdMetric(device.VRAM, "Total VRAM Usage"),
			MemoryTotal: amdMetric(device.VRAM, "Total VRAM"),
			Temperature: firstAMDMetric(device.Sensors, "Junction Temperature", "Edge Temperature"),
			PowerDraw:   firstAMDMetric(device.Sensors, "Average Power", "GFX Power", "Input Power"),
			FanPercent:  firstAMDMetric(device.Sensors, "Fan Speed", "Fan"),
			Processes:   parseAMDProcesses(device.FDInfo),
		}
		if gpu.Name == "" {
			gpu.Name = "AMD GPU"
		}
		gpus = append(gpus, gpu)
	}

	return gpus, nil
}

func parseAMDProcesses(raw map[string]json.RawMessage) []AmdGpuProcess {
	processes := make([]AmdGpuProcess, 0, len(raw))
	for pidString, processRaw := range raw {
		pid, err := strconv.ParseInt(pidString, 10, 32)
		if err != nil {
			continue
		}

		name, vram, gtt := parseAMDProcess(processRaw)
		processes = append(processes, AmdGpuProcess{
			PID:      int32(pid),
			Provider: classifyProvider(name, ""),
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

func parseAMDProcess(raw json.RawMessage) (string, OptFloat, OptFloat) {
	var current map[string]json.RawMessage
	if json.Unmarshal(raw, &current) != nil {
		return "", OptFloat{}, OptFloat{}
	}

	name := ""
	if value, ok := current["name"]; ok {
		_ = json.Unmarshal(value, &name)
	}

	for depth := 0; depth < 4; depth++ {
		if value, ok := current["VRAM"]; ok {
			return name, amdMetricValue(value), amdNestedMetric(current, "GTT")
		}
		value, ok := current["usage"]
		if !ok || json.Unmarshal(value, &current) != nil {
			break
		}
	}

	return name, OptFloat{}, OptFloat{}
}

func amdNestedMetric(metrics map[string]json.RawMessage, key string) OptFloat {
	value, ok := metrics[key]
	if !ok {
		return OptFloat{}
	}
	return amdMetricValue(value)
}

func amdMetric(metrics map[string]json.RawMessage, key string) OptFloat {
	value, ok := metrics[key]
	if !ok {
		return OptFloat{}
	}
	return amdMetricValue(value)
}

func firstAMDMetric(metrics map[string]json.RawMessage, keys ...string) OptFloat {
	for _, key := range keys {
		if value := amdMetric(metrics, key); value.OK {
			return value
		}
	}
	return OptFloat{}
}

func amdMetricValue(raw json.RawMessage) OptFloat {
	var value struct {
		Value *float64 `json:"value"`
	}
	if json.Unmarshal(raw, &value) == nil && value.Value != nil {
		return OptFloat{Value: *value.Value, OK: true}
	}

	var number float64
	if json.Unmarshal(raw, &number) == nil {
		return OptFloat{Value: number, OK: true}
	}
	return OptFloat{}
}
