package parse

import (
	"os"
	"testing"
)

func readFixture(name string) string {
	data, err := os.ReadFile("../../testdata/" + name)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func TestParseAMDJSON(t *testing.T) {
	output := `{"devices":[{"Info":{"DeviceName":"AMD Radeon RX 9070 XT","DevicePath":{"pci":"0000:03:00.0"}},"gpu_activity":{"GFX":{"unit":"%","value":42}},"Sensors":{"Edge Temperature":{"unit":"C","value":55},"Average Power":{"unit":"W","value":120}},"VRAM":{"Total VRAM Usage":{"unit":"MiB","value":2048},"Total VRAM":{"unit":"MiB","value":16384}},"fdinfo":{"123":{"name":"ollama","usage":{"name":"ollama","usage":{"GTT":{"unit":"MiB","value":16},"VRAM":{"unit":"MiB","value":1024}}}}}}]}`

	gpus, err := ParseAMDJSON(output)
	if err != nil {
		t.Fatalf("ParseAMDJSON returned error: %v", err)
	}
	if len(gpus) != 1 {
		t.Fatalf("got %d GPUs, want 1", len(gpus))
	}
	gpu := gpus[0]
	if gpu.Name != "AMD Radeon RX 9070 XT" || gpu.PCI != "0000:03:00.0" {
		t.Fatalf("unexpected identity: %+v", gpu)
	}
	if !gpu.UtilPercent.OK || gpu.UtilPercent.Value != 42 || !gpu.MemoryUsed.OK || gpu.MemoryUsed.Value != 2048 {
		t.Fatalf("unexpected metrics: %+v", gpu)
	}
	if len(gpu.Processes) != 1 || gpu.Processes[0].Name != "ollama" || gpu.Processes[0].VRAMMiB.Value != 1024 || gpu.Processes[0].GTTMiB.Value != 16 {
		t.Fatalf("unexpected processes: %+v", gpu.Processes)
	}
}

func TestParseOllamaPS(t *testing.T) {
	output := `NAME ID SIZE PROCESSOR CONTEXT UNTIL
ornith-1.5:9b-opencode abcdef123456 5.0 GB 100% GPU 262144 4 minutes from now`

	models := ParseOllamaPS(output)
	if len(models) != 1 {
		t.Fatalf("got %d models, want 1", len(models))
	}
	model := models[0]
	if model.Name != "ornith-1.5:9b-opencode" || model.Size != "5.0 GB" || model.Processor != "100% GPU" || model.Context != "262144" {
		t.Fatalf("unexpected model: %+v", model)
	}
}

func TestParseOllamaPSJSONWithTokens(t *testing.T) {
	output := `[{"name":"qwen2.5:14b","id":"abc123","size":14000000000,"context_length":8192,"prompt_tokens":1520,"output_tokens":340},{"name":"llama3:8b","id":"def456","size":8000000000,"context_length":4096,"prompt_tokens":800,"output_tokens":120}]`

	models := ParseOllamaPSJSON(output)
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].CtxTokens != 8192 || models[0].PromptTokens != 1520 {
		t.Fatalf("unexpected token counts for model 0: ctx=%d want 8192, prompt=%d want 1520", models[0].CtxTokens, models[0].PromptTokens)
	}
	if models[1].CtxTokens != 4096 || models[1].PromptTokens != 800 {
		t.Fatalf("unexpected token counts for model 1: ctx=%d want 4096, prompt=%d want 800", models[1].CtxTokens, models[1].PromptTokens)
	}
}

func TestParseOllamaPSWithFixture(t *testing.T) {
	output := readFixture("ollama_ps_output.txt")
	models := ParseOllamaPS(output)
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].Name != "qwen2.5:14b" || models[0].Size != "8.1 GB" {
		t.Fatalf("unexpected model: %+v", models[0])
	}
}

func TestParseNVIDIASVMDetectsMultipleGPUs(t *testing.T) {
	output := readFixture("nvidia_smi_output.txt")
	gpus, err := ParseGPUCSV(output)
	if err != nil {
		t.Fatalf("ParseGPUCSV returned error: %v", err)
	}
	if len(gpus) != 2 {
		t.Fatalf("got %d GPUs, want 2", len(gpus))
	}
	if gpus[0].Name != "NVIDIA GeForce RTX 4090" {
		t.Fatalf("unexpected GPU 0: %v", gpus[0])
	}
}

func TestParseAMDProcessesFromJSON(t *testing.T) {
	output := readFixture("amdgpu_top_output.json")
	gpus, err := ParseAMDJSON(output)
	if err != nil {
		t.Fatalf("ParseAMDJSON returned error: %v", err)
	}
	if len(gpus) != 1 {
		t.Fatalf("got %d GPUs, want 1", len(gpus))
	}
	if len(gpus[0].Processes) != 2 {
		t.Fatalf("got %d processes, want 2", len(gpus[0].Processes))
	}
	found := make(map[string]bool)
	for _, p := range gpus[0].Processes {
		found[p.Name] = true
		found[p.Provider] = true
	}
	if !found["ollama"] {
		t.Fatalf("ollama not found in GPU processes")
	}
	if !found["vllm"] {
		t.Fatalf("vllm provider not found in GPU processes")
	}
}

func TestClassifyProvider(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"ollama", "Ollama"},
		{"OLLAMA", "Ollama"},
		{"  ollama  ", "Ollama"},
		{"unsloth", "Unsloth"},
		{"python-unsloth", "Unsloth"},
		{"Python-Unsloth", "Unsloth"},
		{"unsloth-trainer", "Unsloth"},
		{"vllm", "vLLM"},
		{"text-generation-server", "vLLM"},
		{"koboldcpp", "KoboldCPP"},
		{"xtt", "Local Inference"},
		{"voyager", "Local Inference"},
		{"sGLM", "Local Inference"},
		{"llama.cpp", "Local Inference"},
		{"llama-srv", "Local Inference"},
		{"chrome", ""},
		{"node", ""},
		{"unknown-binary-1234", ""},
	}
	for _, c := range cases {
		if got := ClassifyProvider(c.in, ""); got != c.want {
			t.Fatalf("ClassifyProvider(%q, %q) = %q, want %q", c.in, "", got, c.want)
		}
	}
}

func BenchmarkParseAMDJSON(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fixtures, _ := os.ReadFile("testdata/amdgpu_top_output.json")
		ParseAMDJSON(string(fixtures))
	}
}

func BenchmarkParseOllamaPS(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fixtures, _ := os.ReadFile("testdata/ollama_ps_output.txt")
		ParseOllamaPS(string(fixtures))
	}
}

func BenchmarkClassifyProvider(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ClassifyProvider("python", "python3 -m torch.nn.parallel.DistributedDataParallel --ollama")
	}
}

func BenchmarkParseOptFloat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseOptFloat("12345.67")
	}
}


