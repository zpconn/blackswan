package app

import (
	"blackswan-tui/internal/storage"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestDefaultsLoadedDoesNotOverwritePinnedConfig(t *testing.T) {
	t.Parallel()

	m := NewModelWithOptions(nil, nil, ModelOptions{
		InitialConfigJSON: `{"custom":true}`,
		InitialConfigPath: "/tmp/config.json",
	})
	initial := m.configEditor.Value()

	nextModel, _ := m.Update(defaultsLoadedMsg{
		defaults: map[string]any{"defaults_key": true},
	})
	next := nextModel.(Model)

	if next.configEditor.Value() != initial {
		t.Fatalf("expected pinned config to stay unchanged")
	}
	if next.statusText != "Defaults loaded. Existing config preserved." {
		t.Fatalf("unexpected status text: %q", next.statusText)
	}
}

func TestDefaultsLoadedPopulatesEditorWhenNotPinned(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	nextModel, _ := m.Update(defaultsLoadedMsg{
		defaults: map[string]any{"defaults_key": true},
	})
	next := nextModel.(Model)

	if !strings.Contains(next.configEditor.Value(), `"defaults_key": true`) {
		t.Fatalf("expected defaults JSON in editor, got: %q", next.configEditor.Value())
	}
}

func TestCtrlOPromptFlowLoadsConfigFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"risk":{"alpha":0.03}}`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	m := NewModel(nil, nil)

	openedModel, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	opened := openedModel.(Model)
	if !opened.showConfigPathPrompt {
		t.Fatalf("expected config path prompt to open")
	}

	opened.configPathInput.SetValue(path)
	submittedModel, cmd := opened.Update(tea.KeyMsg{Type: tea.KeyEnter})
	submitted := submittedModel.(Model)
	if cmd == nil {
		t.Fatalf("expected load command on enter")
	}
	if submitted.showConfigPathPrompt {
		t.Fatalf("expected prompt to close after enter")
	}

	msg := cmd()
	loadedModel, _ := submitted.Update(msg)
	loaded := loadedModel.(Model)
	if strings.TrimSpace(loaded.errorText) != "" {
		t.Fatalf("unexpected error text: %q", loaded.errorText)
	}
	if !loaded.configPinned {
		t.Fatalf("expected configPinned=true after successful file load")
	}
	if !strings.Contains(loaded.configEditor.Value(), `"alpha": 0.03`) {
		t.Fatalf("loaded editor content missing expected JSON: %q", loaded.configEditor.Value())
	}
}

func TestListJSONFilesInDirFiltersAndSorts(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "b.JSON"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	files, err := listJSONFilesInDir(dir)
	if err != nil {
		t.Fatalf("listJSONFilesInDir returned error: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 json files, got %d (%v)", len(files), files)
	}
	if files[0] != "a.json" || files[1] != "b.JSON" {
		t.Fatalf("unexpected json file ordering: %v", files)
	}
}

func TestConfigPromptArrowSelectionUpdatesInput(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.showConfigPathPrompt = true
	m.configPathChoices = []string{"a.json", "b.json", "c.json"}
	m.configPathChoiceCursor = 0
	m.configPathInput.SetValue("")

	nextModel, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	next := nextModel.(Model)

	if next.configPathChoiceCursor != 1 {
		t.Fatalf("expected cursor 1 after down, got %d", next.configPathChoiceCursor)
	}
	if strings.TrimSpace(next.configPathInput.Value()) != "b.json" {
		t.Fatalf("expected input to follow selected file, got %q", next.configPathInput.Value())
	}
}

func TestConfigPromptEnterLoadsSelectedWhenInputEmpty(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "picked.json")
	if err := os.WriteFile(path, []byte(`{"ok":true}`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	m := NewModel(nil, nil)
	m.showConfigPathPrompt = true
	m.configPathChoices = []string{path}
	m.configPathChoiceCursor = 0
	m.configPathInput.SetValue("")

	nextModel, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	next := nextModel.(Model)
	if cmd == nil {
		t.Fatalf("expected load command from enter")
	}
	if next.showConfigPathPrompt {
		t.Fatalf("expected prompt to close after enter")
	}

	msg := cmd()
	loaded, ok := msg.(configFileLoadedMsg)
	if !ok {
		t.Fatalf("expected configFileLoadedMsg, got %T", msg)
	}
	if loaded.err != nil {
		t.Fatalf("expected selected file to load successfully, got err=%v", loaded.err)
	}
	if loaded.path != path {
		t.Fatalf("expected loaded path %q, got %q", path, loaded.path)
	}
}

func TestCtrlOPromptEnterEmptyPathShowsError(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.showConfigPathPrompt = true
	m.configPathChoices = nil
	m.configPathInput.SetValue("")

	nextModel, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	next := nextModel.(Model)
	if next.showConfigPathPrompt {
		t.Fatalf("expected prompt closed after enter")
	}
	if next.errorText != "Config file path is required." {
		t.Fatalf("unexpected error text: %q", next.errorText)
	}
}

func TestConfigFileLoadedMsgClosesPromptOnSuccess(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	openedModel, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	opened := openedModel.(Model)
	if !opened.showConfigPathPrompt {
		t.Fatalf("expected config path prompt to open")
	}

	nextModel, _ := opened.Update(configFileLoadedMsg{
		path: "/tmp/config.json",
		text: "{\n  \"risk\": {\n    \"alpha\": 0.07\n  }\n}",
	})
	next := nextModel.(Model)

	if next.showConfigPathPrompt {
		t.Fatalf("expected prompt closed after configFileLoadedMsg success")
	}
	if !next.configPinned {
		t.Fatalf("expected config to be pinned after configFileLoadedMsg success")
	}
	if !strings.Contains(next.statusText, "Loaded config file: /tmp/config.json") {
		t.Fatalf("unexpected status text: %q", next.statusText)
	}
}

func TestConfigFileLoadedMsgClosesPromptOnError(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	openedModel, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	opened := openedModel.(Model)
	if !opened.showConfigPathPrompt {
		t.Fatalf("expected config path prompt to open")
	}

	nextModel, _ := opened.Update(configFileLoadedMsg{
		path: "/tmp/config.json",
		err:  errors.New("kaboom"),
	})
	next := nextModel.(Model)

	if next.showConfigPathPrompt {
		t.Fatalf("expected prompt closed after configFileLoadedMsg error")
	}
	if !strings.Contains(next.errorText, "Config file load failed: kaboom") {
		t.Fatalf("unexpected error text: %q", next.errorText)
	}
}

func TestViewStaysWithinWindowHeightWhenPromptVisible(t *testing.T) {
	t.Parallel()

	const width = 120
	const height = 30

	m := NewModel(nil, nil)
	sizedModel, _ := m.Update(tea.WindowSizeMsg{Width: width, Height: height})
	sized := sizedModel.(Model)
	openedModel, _ := sized.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	opened := openedModel.(Model)
	if !opened.showConfigPathPrompt {
		t.Fatalf("expected config path prompt to open")
	}

	view := opened.View()
	lineCount := strings.Count(view, "\n") + 1
	if lineCount > height {
		t.Fatalf("expected view line count <= window height (%d), got %d", height, lineCount)
	}
	if !strings.Contains(view, "Load Config File") {
		t.Fatalf("expected load-config panel title in view")
	}
	if !strings.Contains(view, "enter load | esc cancel") {
		t.Fatalf("expected load-config panel controls in view")
	}
	if !strings.Contains(view, ".json files in current directory:") {
		t.Fatalf("expected json file list header in prompt")
	}
}

func TestHighlightJSONKeysStylesObjectFields(t *testing.T) {
	t.Parallel()

	in := "{\n  \"alpha\": 0.1,\n  \"nested\": {\"beta\": true}\n}"
	out := highlightJSONKeys(in)

	if !strings.Contains(out, jsonKeyStyle.Render(`"alpha"`)) {
		t.Fatalf("expected alpha key to be styled, got: %q", out)
	}
	if !strings.Contains(out, jsonKeyStyle.Render(`"nested"`)) {
		t.Fatalf("expected nested key to be styled, got: %q", out)
	}
	if !strings.Contains(out, jsonKeyStyle.Render(`"beta"`)) {
		t.Fatalf("expected beta key to be styled, got: %q", out)
	}
	if !strings.Contains(out, "0.1") || !strings.Contains(out, "true") {
		t.Fatalf("expected values to remain present, got: %q", out)
	}
}

func TestHighlightJSONKeysHandlesEscapedQuotes(t *testing.T) {
	t.Parallel()

	in := "{\"a\\\"b\": 1, \"value\": \"x:y\"}"
	out := highlightJSONKeys(in)

	if !strings.Contains(out, jsonKeyStyle.Render(`"a\"b"`)) {
		t.Fatalf("expected escaped-quote key to be styled, got: %q", out)
	}
	if !strings.Contains(out, jsonKeyStyle.Render(`"value"`)) {
		t.Fatalf("expected value key to be styled, got: %q", out)
	}
}

func TestHighlightJSONKeysUsesDistinctStyleForObjectValuedKeys(t *testing.T) {
	t.Parallel()

	in := "{\n  \"portfolio\": {\n    \"initial_portfolio\": 1\n  },\n  \"enabled\": true\n}"
	out := highlightJSONKeys(in)
	matches := jsonKeyPattern.FindAllStringIndex(in, -1)
	if len(matches) != 3 {
		t.Fatalf("expected 3 key matches, got %d", len(matches))
	}
	if !jsonKeyHasObjectValue(in, matches[0][1]) {
		t.Fatalf("expected first key to be classified as object-valued")
	}
	if jsonKeyHasObjectValue(in, matches[1][1]) {
		t.Fatalf("expected nested scalar key to not be object-valued")
	}
	if jsonKeyHasObjectValue(in, matches[2][1]) {
		t.Fatalf("expected primitive key to not be object-valued")
	}

	if !strings.Contains(out, jsonObjectKeyStyle.Render(`"portfolio"`)) {
		t.Fatalf("expected object-valued key to use object-key style, got: %q", out)
	}
	if !strings.Contains(out, jsonKeyStyle.Render(`"enabled"`)) {
		t.Fatalf("expected primitive key to use normal key style, got: %q", out)
	}
}

func TestRenderUsageTrendRightAlignsWhenHistoryShort(t *testing.T) {
	t.Parallel()

	got := renderUsageTrend([]float64{50, 100}, 6)
	if !strings.HasPrefix(got, "....") {
		t.Fatalf("expected leading padding dots, got %q", got)
	}
	tailFromMinWidth := renderUsageTrend([]float64{50, 100}, 4)
	wantTail := tailFromMinWidth[len(tailFromMinWidth)-2:]
	if got[4:] != wantTail {
		t.Fatalf("expected right-aligned tail %q, got %q", wantTail, got[4:])
	}
}

func TestRenderUsageTrendUsesLatestWindow(t *testing.T) {
	t.Parallel()

	full := renderUsageTrend([]float64{0, 10, 20, 30, 40, 50}, 4)
	latestOnly := renderUsageTrend([]float64{20, 30, 40, 50}, 4)
	if full != latestOnly {
		t.Fatalf("expected trend to use latest window: full=%q latest=%q", full, latestOnly)
	}
}

func TestFocusTransitionsKeepStateAndRequestSample(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.cpuHasBaseline = true
	m.terminalFocused = true

	blurredModel, _ := m.Update(tea.BlurMsg{})
	blurred := blurredModel.(Model)
	if blurred.terminalFocused {
		t.Fatalf("expected terminal to be marked unfocused after BlurMsg")
	}
	if !blurred.cpuHasBaseline {
		t.Fatalf("expected cpu baseline to remain unchanged on blur")
	}

	focusedModel, cmd := blurred.Update(tea.FocusMsg{})
	focused := focusedModel.(Model)
	if !focused.terminalFocused {
		t.Fatalf("expected terminal to be marked focused after FocusMsg")
	}
	if !focused.cpuHasBaseline {
		t.Fatalf("expected cpu baseline to remain unchanged on focus")
	}
	if cmd == nil {
		t.Fatalf("expected focus to trigger immediate metrics sampling command")
	}
}

func TestSystemMetricsCatchupAppendsMultipleSamples(t *testing.T) {
	t.Parallel()

	base := time.Now()
	m := NewModel(nil, nil)
	m.cpuHasBaseline = true
	m.cpuLastTotal = 100
	m.cpuLastIdle = 40
	m.lastMetricsSampleAt = base

	msg := systemMetricsMsg{
		sampledAt: base.Add(4 * metricsPollInterval),
		snapshot: systemMetricsSnapshot{
			cpuTotalTicks: 500,
			cpuIdleTicks:  200,
			memAvailable:  true,
			memUsedPct:    33.0,
			gpuAvailable:  true,
			gpuUtilPct:    55.0,
			gpuMemUsedPct: 44.0,
			gpuProbeOK:    true,
		},
	}

	nextModel, _ := m.Update(msg)
	next := nextModel.(Model)
	if len(next.cpuHistory) != 4 {
		t.Fatalf("expected 4 cpu samples after catch-up, got %d", len(next.cpuHistory))
	}
	if len(next.memHistory) != 4 {
		t.Fatalf("expected 4 mem samples after catch-up, got %d", len(next.memHistory))
	}
	if len(next.gpuHistory) != 4 {
		t.Fatalf("expected 4 gpu samples after catch-up, got %d", len(next.gpuHistory))
	}
}

func TestHistorySelectionAutoScrollsViewport(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.history.Width = 120
	m.history.Height = 5
	m.historyItems = make([]storage.RunSummary, 12)
	for idx := range m.historyItems {
		m.historyItems[idx] = storage.RunSummary{
			SavedAt:       "2026-01-01T00:00:00Z",
			ObjectiveMode: "consensus",
		}
	}

	m.historyCursor = 0
	m.refreshHistoryView()
	if m.history.YOffset != 0 {
		t.Fatalf("expected top selection offset 0, got %d", m.history.YOffset)
	}

	m.historyCursor = 7
	m.refreshHistoryView()
	if m.history.YOffset != 4 {
		t.Fatalf("expected offset 4 for cursor 7 with height 5, got %d", m.history.YOffset)
	}

	m.historyCursor = 11
	m.refreshHistoryView()
	if m.history.YOffset != 7 {
		t.Fatalf("expected offset 7 for cursor 11 with height 5, got %d", m.history.YOffset)
	}

	m.historyCursor = 2
	m.refreshHistoryView()
	if m.history.YOffset != 1 {
		t.Fatalf("expected offset 1 when moving selection back up, got %d", m.history.YOffset)
	}
}

func TestHistorySelectionIsHighlightedInView(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.history.Width = 64
	m.history.Height = 5
	m.historyItems = []storage.RunSummary{
		{
			SavedAt:       "2026-01-01T00:00:00Z",
			ObjectiveMode: "consensus",
		},
		{
			SavedAt:       "2026-01-02T00:00:00Z",
			ObjectiveMode: "cvar_shortfall",
		},
	}
	m.historyCursor = 1
	m.refreshHistoryView()

	view := m.history.View()
	if !strings.Contains(view, "▶ ") {
		t.Fatalf("expected selected history row marker in view, got %q", view)
	}
	if !strings.Contains(view, "cvar_shortfall") {
		t.Fatalf("expected selected history row content in view, got %q", view)
	}
	plainSelected := "▶ "
	styledSelected := historySelectedLineStyle.Render(plainSelected)
	if styledSelected != plainSelected && !strings.Contains(view, styledSelected) {
		t.Fatalf("expected styled selected row in view, got %q", view)
	}
}

func TestHistoryAutoScrollAccountsForWrappedRows(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.history.Width = 20
	m.history.Height = 4
	m.historyItems = make([]storage.RunSummary, 8)
	for idx := range m.historyItems {
		m.historyItems[idx] = storage.RunSummary{
			SavedAt:       "2026-01-01T00:00:00Z",
			ObjectiveMode: "long-mode-name-to-force-wrap",
		}
	}

	m.historyCursor = 6
	m.refreshHistoryView()

	if m.history.YOffset <= 0 {
		t.Fatalf("expected positive y-offset for wrapped history content, got %d", m.history.YOffset)
	}
	if m.historyCursorBottomLine < m.history.YOffset {
		t.Fatalf("expected selected row bottom line to be in/after viewport top")
	}
	if m.historyCursorTopLine > m.history.YOffset+m.history.Height-1 {
		t.Fatalf("expected selected row top line to be in/before viewport bottom")
	}
}
