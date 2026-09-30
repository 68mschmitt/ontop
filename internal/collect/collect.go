package collect

import (
	"context"
	"math"
	"time"
)

var prevSnapshot Snapshot
var prevDiskIO map[string]DiskDeviceStats
var prevNetIO map[string]NetDeviceStats

var StudioPort = 8888
var StudioToken = ""

func CollectMetrics(ctx context.Context) Snapshot {
	started := time.Now()
	s := Snapshot{CollectedAt: started}

	prevDisk := make(map[string]DiskDeviceStats)
	if prevDiskIO != nil {
		prevDisk = make(map[string]DiskDeviceStats, len(prevDiskIO))
		for k, v := range prevDiskIO {
			prevDisk[k] = v
		}
	}
	prevNet := make(map[string]NetDeviceStats)
	if prevNetIO != nil {
		prevNet = make(map[string]NetDeviceStats, len(prevNetIO))
		for k, v := range prevNetIO {
			prevNet[k] = v
		}
	}

	prevCollectedAt := prevSnapshot.CollectedAt
	var elapsed float64
	if !prevCollectedAt.IsZero() {
		elapsed = s.CollectedAt.Sub(prevCollectedAt).Seconds()
	}

	s.CPU = CollectCPU(ctx, &s.Warnings)
	s.Memory = CollectMemory(ctx, &s.Warnings)
	s.Swap = CollectSwap(ctx, &s.Warnings)
	s.Thermal = CollectThermal(ctx, &s.Warnings)
	s.Disk = CollectDisk(ctx, &s.Warnings)
	s.Net = CollectNet(ctx, &s.Warnings)
	s.GPUs = CollectNVIDIA(ctx, &s.Warnings, prevSnapshot.GPUs)
	s.AMDGPUs = CollectAMD(ctx, &s.Warnings, prevSnapshot.AMDGPUs)
	s.GPUSparkline = MergeGPUSparkline(prevSnapshot.GPUSparkline, s.GPUs)

	if elapsed > 0.5 {
		for _, curr := range s.Disk.Devices {
			if prev, ok := prevDisk[curr.Name]; ok {
				dt := float64(curr.ReadBytes-prev.ReadBytes) / elapsed
				wt := float64(curr.WriteBytes-prev.WriteBytes) / elapsed
				s.Disk.Rates = append(s.Disk.Rates, DiskDeviceRates{
					Name:     curr.Name,
					ReadBps:  math.Max(dt, 0),
					WriteBps: math.Max(wt, 0),
				})
			}
		}
		for _, curr := range s.Net.Devices {
			if prev, ok := prevNet[curr.Name]; ok {
				st := float64(curr.BytesSent-prev.BytesSent) / elapsed
				rt := float64(curr.BytesRecv-prev.BytesRecv) / elapsed
				s.Net.Rates = append(s.Net.Rates, NetDeviceRates{
					Name:         curr.Name,
					BytesSentBps: math.Max(st, 0),
					BytesRecvBps: math.Max(rt, 0),
				})
			}
		}
	}

	diskForPrev := make([]DiskDeviceStats, len(s.Disk.Devices))
	copy(diskForPrev, s.Disk.Devices)
	netForPrev := make([]NetDeviceStats, len(s.Net.Devices))
	copy(netForPrev, s.Net.Devices)
	prevDiskIO = make(map[string]DiskDeviceStats)
	for _, d := range diskForPrev {
		prevDiskIO[d.Name] = d
	}
	prevNetIO = make(map[string]NetDeviceStats)
	for _, d := range netForPrev {
		prevNetIO[d.Name] = d
	}

	var historyCPU []float64
	if len(prevSnapshot.CPUHistory) > 0 {
		historyCPU = make([]float64, len(prevSnapshot.CPUHistory))
		copy(historyCPU, prevSnapshot.CPUHistory)
	}
	if s.CPU.OK && len(s.CPU.PerCore) > 0 {
		total := s.CPU.Total
		history := historyCPU
		if len(history) == 0 {
			history = append(history, 0)
		}
		history = append(history, total)
		if len(history) > 30 {
			history = history[1:]
		}
		s.CPUHistory = history
	}

	var historyRAM []float64
	if len(prevSnapshot.RAMHistory) > 0 {
		historyRAM = make([]float64, len(prevSnapshot.RAMHistory))
		copy(historyRAM, prevSnapshot.RAMHistory)
	}
	if s.Memory.OK {
		percent := s.Memory.Percent
		history := historyRAM
		if len(history) == 0 {
			history = append(history, 0)
		}
		history = append(history, percent)
		if len(history) > 30 {
			history = history[1:]
		}
		s.RAMHistory = history
	}

	s.Inference = CollectInference(ctx, s.GPUs, s.AMDGPUs, &s.Warnings)
	s.OllamaProcesses = CollectOllamaProcesses(ctx, &s.Warnings)
	s.OllamaPS = CollectOllamaPS(ctx)
	s.UnslothStudio = CollectUnslothStudio(ctx, &s.Warnings)
	s.CollectionMillis = time.Since(started).Milliseconds()

	prevSnapshot = s

	return s
}
