package parse

import (
	"encoding/json"
)

// AmdTopDocument represents the top-level structure returned by amdgpu_top --json.
type AmdTopDocument struct {
	Devices []AmdTopDevice `json:"devices"`
}

// AmdTopDevice represents a single device in the amdgpu_top JSON output.
type AmdTopDevice struct {
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
