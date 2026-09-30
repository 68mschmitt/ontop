package collect

import (
	"context"
	"encoding/json"
	stdfmt "fmt"
	"io"
	stdnethttp "net/http"
	stdos "os"
	stdstrconv "strconv"
	stdstrings "strings"
	"time"

	"github.com/shirou/gopsutil/v3/process"
)

var studioPort = StudioPort
var studioToken = StudioToken

func CollectUnslothStudio(ctx context.Context, warnings *[]string) UnslothStudioStats {
	base := DiscoverStudioBase(ctx)
	if base == "" {
		return UnslothStudioStats{}
	}

	stats := UnslothStudioStats{Connected: true}

	if studioToken == "" {
		return stats
	}

	statusURL := base + "/api/inference/status"
	statusResp, err := studioHTTPGet(ctx, statusURL, studioToken)
	if err != nil {
		stats.Error = "inference status unavailable: " + err.Error()
	} else {
		defer statusResp.Body.Close()
		if statusResp.StatusCode == stdnethttp.StatusOK {
			stats.ParseInferenceStatus(statusResp.Body)
		} else {
			stats.Error = stdfmt.Sprintf("inference status %d", statusResp.StatusCode)
		}
	}

	trainURL := base + "/api/train/status"
	trainResp, err := studioHTTPGet(ctx, trainURL, studioToken)
	if err != nil {
		// Non-fatal; training may simply not be running.
	} else {
		defer trainResp.Body.Close()
		if trainResp.StatusCode == stdnethttp.StatusOK {
			stats.ParseTrainStatus(trainResp.Body)
		}
	}

	loadURL := base + "/api/inference/load-progress"
	loadResp, err := studioHTTPGet(ctx, loadURL, studioToken)
	if err != nil {
		// Non-fatal.
	} else {
		defer loadResp.Body.Close()
		if loadResp.StatusCode == stdnethttp.StatusOK {
			stats.ParseLoadProgress(loadResp.Body)
		}
	}

	return stats
}

func DiscoverStudioBase(ctx context.Context) string {
	configured := ""
	if env := stdos.Getenv("UNSLOTH_STUDIO_URL"); env != "" {
		configured = stdstrings.TrimRight(stdstrings.TrimSpace(env), "/")
	}
	if configured != "" {
		probeCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
		defer cancel()
		if StudioHealthOK(probeCtx, configured) {
			return configured
		}
		return ""
	}

	if detected := DetectUnslothProcesses(); detected != "" {
		return detected
	}

	ports := []int{studioPort}
	if studioPort == 8888 || studioPort == -1 {
		ports = []int{8888, 8000, 3000, 8080}
	}
	for _, port := range ports {
		base := stdfmt.Sprintf("http://127.0.0.1:%d", port)
		if StudioHealthOK(ctx, base) {
			return base
		}
	}
	return ""
}

func DetectUnslothProcesses() string {
	procs, err := process.Processes()
	if err != nil {
		return ""
	}
	for _, p := range procs {
		name, _ := p.Name()
		if stdstrings.Contains(stdstrings.ToLower(name), "unsloth") {
			cmdline, _ := p.Cmdline()
			for _, arg := range stdstrings.Fields(cmdline) {
				if stdstrings.HasPrefix(arg, "--port=") {
					port := stdstrings.TrimPrefix(arg, "--port=")
					if portNum, err := stdstrconv.Atoi(port); err == nil {
						return stdfmt.Sprintf("http://localhost:%d", portNum)
					}
				}
			}
			return "http://localhost:8888"
		}
	}
	return ""
}

func StudioHealthOK(ctx context.Context, base string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer cancel()
	resp, err := studioHTTPGet(probeCtx, base+"/api/health", "")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == stdnethttp.StatusOK
}

func studioHTTPGet(ctx context.Context, url string, token string) (*stdnethttp.Response, error) {
	req, err := stdnethttp.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return stdnethttp.DefaultClient.Do(req)
}

func (s *UnslothStudioStats) ParseInferenceStatus(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var b bool
	var num float64
	var arr []string

	if v, ok := raw["active_model"]; ok && json.Unmarshal(v, &str) == nil {
		s.ActiveModel = str
	}
	if v, ok := raw["model_identifier"]; ok && json.Unmarshal(v, &str) == nil {
		s.ModelIdentifier = str
	}
	if v, ok := raw["gguf_variant"]; ok && json.Unmarshal(v, &str) == nil {
		s.GGUFVariant = str
	}
	if v, ok := raw["is_vision"]; ok && json.Unmarshal(v, &b) == nil {
		s.IsVision = b
	}
	if v, ok := raw["is_audio"]; ok && json.Unmarshal(v, &b) == nil {
		s.IsAudio = b
	}
	if v, ok := raw["supports_reasoning"]; ok && json.Unmarshal(v, &b) == nil {
		s.SupportsReasoning = b
	}
	if v, ok := raw["loaded"]; ok && json.Unmarshal(v, &arr) == nil {
		s.LoadedModels = arr
	}
	if v, ok := raw["loading"]; ok && json.Unmarshal(v, &arr) == nil {
		s.LoadingModels = arr
	}
	if v, ok := raw["context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.ContextLength = int(num)
	}
	if v, ok := raw["max_context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.MaxContextLength = int(num)
	}
	if v, ok := raw["native_context_length"]; ok && json.Unmarshal(v, &num) == nil {
		s.NativeContextLength = int(num)
	}
	if num, ok := findJSONNumber(raw, "current_context", "current_context_length", "context_used", "context_tokens"); ok {
		s.CurrentContext = int(num)
	}
	if num, ok := findJSONNumber(raw, "concurrent_sessions", "active_sessions", "active_requests", "num_active_requests", "parallel_sessions", "parallel_requests"); ok {
		s.ConcurrentSessions = int(num)
	}
	if num, ok := findJSONNumber(raw, "tokens_per_second", "tokens_sec", "throughput", "generation_tokens_per_second"); ok {
		s.TokensPerSecond = OptFloat{Value: num, OK: true}
	}
	if v, ok := raw["speculative_type"]; ok && json.Unmarshal(v, &str) == nil {
		s.SpeculativeType = str
	}
	if v, ok := raw["tensor_parallel"]; ok && json.Unmarshal(v, &b) == nil {
		s.TensorParallel = b
	}
}

func findJSONNumber(raw map[string]json.RawMessage, keys ...string) (float64, bool) {
	for _, key := range keys {
		if value, ok := raw[key]; ok {
			var number float64
			if json.Unmarshal(value, &number) == nil {
				return number, true
			}
		}
	}
	for _, containerKey := range []string{"metrics", "stats", "inference"} {
		value, ok := raw[containerKey]
		if !ok {
			continue
		}
		var nested map[string]json.RawMessage
		if json.Unmarshal(value, &nested) == nil {
			if number, ok := findJSONNumber(nested, keys...); ok {
				return number, true
			}
		}
	}
	return 0, false
}

func (s *UnslothStudioStats) ParseTrainStatus(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var num float64

	if v, ok := raw["status"]; ok && json.Unmarshal(v, &str) == nil {
		s.TrainStatus = str
	}
	if v, ok := raw["current_step"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainStep = int(num)
	}
	if v, ok := raw["current_loss"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainLoss = num
	}
	if v, ok := raw["current_lr"]; ok && json.Unmarshal(v, &num) == nil {
		s.TrainLr = num
	}
}

func (s *UnslothStudioStats) ParseLoadProgress(body io.Reader) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return
	}

	var str string
	var num float64

	if v, ok := raw["phase"]; ok && json.Unmarshal(v, &str) == nil {
		s.LoadPhase = str
	}
	if v, ok := raw["bytes_loaded"]; ok && json.Unmarshal(v, &num) == nil {
		s.LoadBytes = int64(num)
	}
	if v, ok := raw["bytes_total"]; ok && json.Unmarshal(v, &num) == nil {
		s.LoadTotal = int64(num)
	}
}
