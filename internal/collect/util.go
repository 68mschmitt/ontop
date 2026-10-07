package collect

import (
	"bytes"
	"context"
	"encoding/csv"
	stdfmt "fmt"
	stdos "os"
	stdexec "os/exec"
	stdstrconv "strconv"
	stdstrings "strings"
	"time"
)

var execLookPath = stdexec.LookPath

func CollectAMDFromSysfs(warnings *[]string) []AmdGpuStats {
	stats := make([]AmdGpuStats, 0)

	dirEntries, err := stdos.ReadDir("/sys/class/drm")
	if err != nil {
		AddWarning(warnings, "AMD GPU sysfs unavailable: "+CleanError(err.Error()))
		return stats
	}

	for _, d := range dirEntries {
		if !d.IsDir() || !stdstrings.HasPrefix(d.Name(), "card") {
			continue
		}

		cardDir := "/sys/class/drm/" + d.Name()
		deviceDir := cardDir + "/device"

		if _, err := stdos.Stat(deviceDir); stdos.IsNotExist(err) {
			continue
		}

		gpu := AmdGpuStats{
			Index: stdstrings.TrimPrefix(d.Name(), "card"),
			Name:  "AMD GPU",
		}

		if pci, err := stdos.Readlink(deviceDir); err == nil {
			gpu.PCI = filePathBase(pci)
		}

		hwmonDir := deviceDir + "/hwmon"
		hwmonEntries, err := stdos.ReadDir(hwmonDir)
		if err == nil {
			for _, e := range hwmonEntries {
				if !stdstrings.HasPrefix(e.Name(), "hwmon") {
					continue
				}
				tempFile := hwmonDir + "/" + e.Name() + "/temp1_input"
				if temp, err := stdos.ReadFile(tempFile); err == nil {
					if val, err := stdstrconv.ParseFloat(stdstrings.TrimSpace(string(temp)), 64); err == nil {
						if val > 0 {
							gpu.Temperature = OptFloat{Value: val / 1000.0, OK: true}
						}
					}
				}
			}
		}

		stats = append(stats, gpu)
	}

	return stats
}

func MapBoolString(b bool) string {
	if b {
		return "enabled"
	}
	return "disabled"
}

func AddWarning(warnings *[]string, message string) {
	message = stdstrings.TrimSpace(message)
	if message == "" {
		return
	}
	*warnings = append(*warnings, message)
}

func AddGPUUtilDelta(prev []GpuStats, new []GpuStats) {
	for i := range new {
		if i < len(prev) && prev[i].UtilPercent.OK && new[i].UtilPercent.OK {
			delta := new[i].UtilPercent.Value - prev[i].UtilPercent.Value
			new[i].UtilDelta = delta
			if delta > 2 {
				new[i].UtilTrend = "▲"
			} else if delta < -2 {
				new[i].UtilTrend = "▼"
			} else {
				new[i].UtilTrend = "-"
			}
		}
	}
}

func AddVRAMDeltas(prev []GpuStats, new []GpuStats) {
	for i := range new {
		if i < len(prev) && prev[i].MemoryUsed.OK && new[i].MemoryUsed.OK {
			delta := new[i].MemoryUsed.Value - prev[i].MemoryUsed.Value
			new[i].VRAMDelta = delta
			if delta > 100 {
				new[i].VRAMTrend = "▲"
			} else if delta < -100 {
				new[i].VRAMTrend = "▼"
			} else {
				new[i].VRAMTrend = "-"
			}
		}
	}
}

func AddAMDUtilDelta(prev []AmdGpuStats, new []AmdGpuStats) {
	for i := range new {
		if i < len(prev) && prev[i].UtilPercent.OK && new[i].UtilPercent.OK {
			delta := new[i].UtilPercent.Value - prev[i].UtilPercent.Value
			new[i].UtilDelta = delta
			if delta > 2 {
				new[i].UtilTrend = "▲"
			} else if delta < -2 {
				new[i].UtilTrend = "▼"
			} else {
				new[i].UtilTrend = "-"
			}
		}
	}
}

func MergeGPUSparkline(prev map[string][]float64, new []GpuStats) map[string][]float64 {
	result := make(map[string][]float64)

	for k, v := range prev {
		if len(v) > 0 {
			result[k] = v
		}
	}

	for _, gpu := range new {
		if gpu.UtilPercent.OK {
			history := result[gpu.Index]
			if len(history) == 0 {
				history = append(history, 0)
			}
			history = append(history, gpu.UtilPercent.Value)
			if len(history) > 30 {
				history = history[1:]
			}
			result[gpu.Index] = history
		}
	}

	return result
}

func CleanError(message string) string {
	message = stdstrings.Join(stdstrings.Fields(message), " ")
	return message
}

func CleanCommandError(err error, stderr string) string {
	message := stdstrings.TrimSpace(stderr)
	if message == "" && err != nil {
		message = err.Error()
	}
	return CleanError(message)
}

func RunCommand(ctx context.Context, name string, args ...string) (string, string, error) {
	cmd := stdexec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func processCmdline(pid int32) string {
	data, err := stdos.ReadFile(stdfmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return ""
	}
	s := stdstrings.ReplaceAll(string(data), "\x00", " ")
	return stdstrings.TrimSpace(s)
}

func processContainerName(pid int32) string {
	data, err := stdos.ReadFile(stdfmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return ""
	}
	for _, line := range stdstrings.Split(string(data), "\n") {
		if stdstrings.Contains(line, "docker") || stdstrings.Contains(line, "podman") {
			for _, part := range stdstrings.Split(line, " ") {
				for _, prefix := range []string{"docker/", "podman/"} {
					if stdstrings.HasPrefix(part, prefix) {
						return part[len(prefix):]
					}
				}
			}
		}
	}
	return ""
}

func WriteCSV(s Snapshot) {
	w := csv.NewWriter(stdos.Stdout)
	defer w.Flush()

	_ = w.Write([]string{"timestamp", "cpu_total", "cpu_per_core", "ram_used", "ram_total", "ram_percent", "swap_used", "swap_total"})

	cores := make([]string, len(s.CPU.PerCore))
	for i, v := range s.CPU.PerCore {
		cores[i] = stdfmt.Sprintf("%.1f%%", v)
	}
	_ = w.Write([]string{
		s.CollectedAt.Format(time.RFC3339),
		stdfmt.Sprintf("%.1f", s.CPU.Total),
		stdstrings.Join(cores, ";"),
		stdfmt.Sprintf("%d", s.Memory.Used),
		stdfmt.Sprintf("%d", s.Memory.Total),
		stdfmt.Sprintf("%.1f", s.Memory.Percent),
		stdfmt.Sprintf("%d", s.Swap.Used),
		stdfmt.Sprintf("%d", s.Swap.Total),
	})
}

func filePathBase(path string) string {
	return path[stdstrings.LastIndex(path, "/")+1:]
}
