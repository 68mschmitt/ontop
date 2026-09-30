package collect

import (
	"context"
	stdfmt "fmt"
	stdos "os"
	stdstrings "strings"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	netio "github.com/shirou/gopsutil/v3/net"
)

func CollectCPU(ctx context.Context, warnings *[]string) CpuStats {
	stats := CpuStats{}
	total, err := cpu.PercentWithContext(ctx, 0, false)
	if err != nil {
		AddWarning(warnings, "CPU metrics unavailable: "+CleanError(err.Error()))
		return stats
	}
	if len(total) > 0 {
		stats.Total = total[0]
		stats.OK = true
	}

	perCore, err := cpu.PercentWithContext(ctx, 0, true)
	if err != nil {
		AddWarning(warnings, "per-core CPU metrics unavailable: "+CleanError(err.Error()))
		return stats
	}
	stats.PerCore = perCore

	return stats
}

func CollectMemory(ctx context.Context, warnings *[]string) MemoryStats {
	vm, err := mem.VirtualMemoryWithContext(ctx)
	if err != nil {
		AddWarning(warnings, "RAM metrics unavailable: "+CleanError(err.Error()))
		return MemoryStats{}
	}

	return MemoryStats{
		OK:        true,
		Used:      vm.Used,
		Total:     vm.Total,
		Available: vm.Available,
		Percent:   vm.UsedPercent,
	}
}

func CollectSwap(ctx context.Context, warnings *[]string) SwapMemoryStats {
	vm, err := mem.SwapMemoryWithContext(ctx)
	if err != nil {
		AddWarning(warnings, "swap metrics unavailable: "+CleanError(err.Error()))
		return SwapMemoryStats{}
	}

	return SwapMemoryStats{
		OK:    true,
		Used:  vm.Used,
		Total: vm.Total,
	}
}

func CollectThermal(ctx context.Context, warnings *[]string) ThermalStats {
	stats := ThermalStats{}
	thermalPath := "/sys/class/thermal"

	dir, err := stdos.Open(thermalPath)
	if err != nil {
		AddWarning(warnings, "thermal sensors unavailable")
		return stats
	}
	defer dir.Close()

	entries, err := dir.ReadDir(-1)
	if err != nil || len(entries) == 0 {
		AddWarning(warnings, "thermal sensors unavailable")
		return stats
	}

	stats.OK = true
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		zonePath := thermalPath + "/" + entry.Name()
		typeFile := zonePath + "/type"
		tempFile := zonePath + "/temp"

		typeData, err := stdos.ReadFile(typeFile)
		if err != nil {
			continue
		}
		zoneType := stdstrings.TrimSpace(string(typeData))
		if zoneType == "" {
			continue
		}

		tempData, err := stdos.ReadFile(tempFile)
		if err != nil {
			continue
		}
		var tempMilli float64
		_, err = stdfmt.Sscanf(string(tempData), "%f", &tempMilli)
		if err != nil {
			continue
		}

		tempCelsius := tempMilli / 1000.0
		stats.Zone = append(stats.Zone, ThermalZone{
			Index:       len(stats.Zone),
			Type:        stdfmt.Sprintf("%s (%s)", zoneType, entry.Name()),
			Temperature: OptFloat{Value: tempCelsius, OK: true},
		})
		if tempCelsius > stats.Total.Value {
			stats.Total = OptFloat{Value: tempCelsius, OK: true}
		}
	}

	if len(stats.Zone) == 0 {
		stats.OK = false
	}
	return stats
}

func CollectDisk(ctx context.Context, warnings *[]string) DiskStats {
	stats := DiskStats{}
	devices, err := disk.IOCountersWithContext(ctx)
	if err != nil {
		AddWarning(warnings, "disk I/O metrics unavailable: "+CleanError(err.Error()))
		return stats
	}
	stats.OK = true
	for name, dev := range devices {
		stats.Devices = append(stats.Devices, DiskDeviceStats{
			Name:       name,
			ReadBytes:  dev.ReadBytes,
			WriteBytes: dev.WriteBytes,
			ReadIOss:   dev.ReadCount,
			WriteIOSS:  dev.WriteCount,
		})
	}
	if len(stats.Devices) == 0 {
		stats.OK = false
	}
	return stats
}

func CollectNet(ctx context.Context, warnings *[]string) NetStats {
	stats := NetStats{}
	devices, err := netio.IOCountersWithContext(ctx, false)
	if err != nil {
		AddWarning(warnings, "network I/O metrics unavailable: "+CleanError(err.Error()))
		return stats
	}
	stats.OK = true
	for _, dev := range devices {
		stats.Devices = append(stats.Devices, NetDeviceStats{
			Name:        dev.Name,
			BytesSent:   dev.BytesSent,
			BytesRecv:   dev.BytesRecv,
			PacketsSent: dev.PacketsSent,
			PacketsRecv: dev.PacketsRecv,
		})
	}
	return stats
}
