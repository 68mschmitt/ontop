package collect

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestConfigureDefaultsStudioPort(t *testing.T) {
	t.Cleanup(func() { Configure(Config{StudioPort: 8888}) })

	Configure(Config{})
	if config.StudioPort != 8888 {
		t.Fatalf("empty StudioPort should default to 8888, got %d", config.StudioPort)
	}

	Configure(Config{StudioPort: 9000, StudioToken: "abc"})
	if config.StudioPort != 9000 || config.StudioToken != "abc" {
		t.Fatalf("Configure did not store values: %+v", config)
	}
}

// TestCollectUnslothStudioSendsConfiguredToken guards the config plumbing:
// the port and token must reach the collector, and a healthy Studio must be
// queried even though discovery and the health probe need no token.
func TestCollectUnslothStudioSendsConfiguredToken(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/health":
			w.WriteHeader(http.StatusOK)
		case "/api/inference/status":
			requests++
			if got := r.Header.Get("Authorization"); got != "Bearer secret-token" {
				t.Errorf("Authorization = %q, want %q", got, "Bearer secret-token")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"active_model":"qwen2.5:14b","context_length":8192}`)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	t.Setenv("UNSLOTH_STUDIO_URL", server.URL)
	Configure(Config{StudioPort: 8888, StudioToken: "secret-token"})
	t.Cleanup(func() { Configure(Config{StudioPort: 8888}) })

	var warnings []string
	stats := CollectUnslothStudio(context.Background(), &warnings)

	if !stats.Connected {
		t.Fatalf("expected Connected stats, got %+v", stats)
	}
	if requests != 1 {
		t.Fatalf("inference status requested %d times, want 1", requests)
	}
	if stats.ActiveModel != "qwen2.5:14b" || stats.ContextLength != 8192 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}
