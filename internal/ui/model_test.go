package ui

import (
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"ontop/internal/collect"
)

func newTestModel() *Model {
	return NewModel(time.Second, RenderFuncs{
		RenderContentFn:   func(collect.Snapshot, int, map[string]bool) string { return "" },
		RenderHeaderFn:    func(collect.Snapshot, time.Duration, bool, bool, string, int) string { return "" },
		RenderFooterFn:    func(int, map[string]bool) string { return "" },
		RenderHelpOverlay: func(int, map[string]bool) string { return "" },
	})
}

func runeKey(r rune) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}}
}

func TestSectionToggles(t *testing.T) {
	cases := []struct {
		key     rune
		section string
	}{
		{'g', "GPU"},
		{'s', "System"},
		{'m', "Memory"},
		{'u', "Unsloth"},
		{'o', "Ollama"},
	}

	for _, c := range cases {
		m := newTestModel()

		if _, _ = m.Update(runeKey(c.key)); !m.disabledSections[c.section] {
			t.Fatalf("%q should disable the %s section", c.key, c.section)
		}
		if _, _ = m.Update(runeKey(c.key)); m.disabledSections[c.section] {
			t.Fatalf("%q should re-enable the %s section", c.key, c.section)
		}
	}
}

func TestQuitKeys(t *testing.T) {
	for _, msg := range []tea.KeyMsg{runeKey('q'), {Type: tea.KeyCtrlC}} {
		m := newTestModel()
		_, cmd := m.Update(msg)
		if cmd == nil {
			t.Fatalf("%q should return a command", msg.String())
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Fatalf("%q should quit", msg.String())
		}
	}
}

func TestHelpTogglesAndDismisses(t *testing.T) {
	m := newTestModel()

	if _, _ = m.Update(runeKey('?')); !m.showHelp {
		t.Fatal("? should show help")
	}
	// Any key dismisses the overlay while it is open.
	if _, _ = m.Update(runeKey('h')); m.showHelp {
		t.Fatal("a key press should dismiss help")
	}
	if _, _ = m.Update(runeKey('h')); !m.showHelp {
		t.Fatal("h should show help again")
	}
}

func TestWindowSizeMarksReady(t *testing.T) {
	m := newTestModel()

	if _, _ = m.Update(tea.WindowSizeMsg{Width: 120, Height: 40}); !m.ready {
		t.Fatal("window size message should mark the model ready")
	}
	if m.width != 120 || m.height != 40 {
		t.Fatalf("unexpected dimensions: %dx%d", m.width, m.height)
	}
}

func TestMetricsMessageClearsLoading(t *testing.T) {
	m := newTestModel()
	m.loading = true

	if _, _ = m.Update(collect.MetricsMessage{Snapshot: collect.Snapshot{CollectedAt: time.Now()}}); m.loading {
		t.Fatal("metrics message should clear the loading state")
	}
}

func TestCollectingIndicatorDebouncesFastCollections(t *testing.T) {
	m := newTestModel()

	if !m.collecting() {
		t.Fatal("the initial load should report collecting")
	}

	m.snapshot = collect.Snapshot{CollectedAt: time.Now()}
	m.loading = true
	m.collectStartedAt = time.Now()
	if m.collecting() {
		t.Fatal("a just-started collection should not flicker the status")
	}

	m.collectStartedAt = time.Now().Add(-2 * busyIndicatorDelay)
	if !m.collecting() {
		t.Fatal("a slow collection should report collecting")
	}

	m.loading = false
	if m.collecting() {
		t.Fatal("an idle model should report idle")
	}
}
