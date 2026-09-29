package collect

import (
	"time"
)

type OptFloat struct {
	Value float64
	OK    bool
}

type Snapshot struct {
	CollectedAt      time.Time
	CPU              CpuStats
	Memory           MemoryStats
	Swap             SwapMemoryStats
	Thermal          ThermalStats
	GPUs             []GpuStats
	AMDGPUs          []AmdGpuStats
	Disk             DiskStats
	Net              NetStats
	Inference        []InferenceProcess
	OllamaProcesses  []OllamaProcess
	OllamaPS         CommandOutput
	UnslothStudio    UnslothStudioStats
	GPUSparkline     map[string][]float64
	CPUHistory       []float64
	RAMHistory       []float64
	Warnings         []string
	CollectionMillis int64
}

type InferenceProcess struct {
	PID      int32
	Provider string
	Name     string
	VRAMMiB  OptFloat
	GTTMiB   OptFloat
}

type CpuStats struct {
	OK      bool
	Total   float64
	PerCore []float64
}

type MemoryStats struct {
	OK        bool
	Used      uint64
	Total     uint64
	Available uint64
	Percent   float64
}

type SwapMemoryStats struct {
	OK    bool
	Used  uint64
	Total uint64
}

type ThermalStats struct {
	OK    bool
	Zone  []ThermalZone
	Total OptFloat
}

type ThermalZone struct {
	Index       int
	Type        string
	Temperature OptFloat
}

type DiskStats struct {
	OK       bool
	Devices  []DiskDeviceStats
	Warnings []string
	Rates    []DiskDeviceRates
}

type DiskDeviceStats struct {
	Name       string
	ReadBytes  uint64
	WriteBytes uint64
	ReadIOss   uint64
	WriteIOSS  uint64
}

type DiskDeviceRates struct {
	Name     string
	ReadBps  float64
	WriteBps float64
}

type NetStats struct {
	OK      bool
	Devices []NetDeviceStats
	Rates   []NetDeviceRates
}

type NetDeviceStats struct {
	Name        string
	BytesSent   uint64
	BytesRecv   uint64
	PacketsSent uint64
	PacketsRecv uint64
}

type NetDeviceRates struct {
	Name         string
	BytesSentBps float64
	BytesRecvBps float64
}

type GpuStats struct {
	Index       string
	UUID        string
	Name        string
	UtilPercent OptFloat
	MemoryUsed  OptFloat
	MemoryTotal OptFloat
	Temperature OptFloat
	PowerDraw   OptFloat
	PowerLimit  OptFloat
	FanPercent  OptFloat
	Processes   []GpuProcess
	UtilTrend   string
	UtilDelta   float64
	VRAMTrend   string
	VRAMDelta   float64
}

type GpuProcess struct {
	GPUUUID      string
	PID          int32
	Provider     string
	Name         string
	UsedMemoryMB OptFloat
}

type AmdGpuStats struct {
	Index       string
	PCI         string
	Name        string
	UtilPercent OptFloat
	MemoryUsed  OptFloat
	MemoryTotal OptFloat
	Temperature OptFloat
	PowerDraw   OptFloat
	FanPercent  OptFloat
	Processes   []AmdGpuProcess
	UtilTrend   string
	UtilDelta   float64
}

type AmdGpuProcess struct {
	PID      int32
	Provider string
	Name     string
	VRAMMiB  OptFloat
	GTTMiB   OptFloat
}

type OllamaProcess struct {
	PID           int32
	Name          string
	Command       string
	CPUPercent    float64
	MemoryPercent float64
	RSS           uint64
	Runtime       time.Duration
	RuntimeOK     bool
}

type CommandOutput struct {
	Output  string
	Error   string
	Missing bool
	Models  []OllamaModel
}

type OllamaModel struct {
	Name         string
	ID           string
	Size         string
	Processor    string
	Context      string
	Until        string
	PromptTokens int
	CtxTokens    int
}

type UnslothStudioStats struct {
	Connected           bool
	ActiveModel         string
	ModelIdentifier     string
	GGUFVariant         string
	IsVision            bool
	IsAudio             bool
	SupportsReasoning   bool
	LoadedModels        []string
	LoadingModels       []string
	ContextLength       int
	MaxContextLength    int
	NativeContextLength int
	CurrentContext      int
	ConcurrentSessions  int
	TokensPerSecond     OptFloat
	SpeculativeType     string
	TensorParallel      bool
	TrainStatus         string
	TrainStep           int
	TrainLoss           float64
	TrainLr             float64
	LoadPhase           string
	LoadBytes           int64
	LoadTotal           int64
	Error               string
}
