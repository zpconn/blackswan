package app

import (
	"blackswan-tui/internal/service"
	"blackswan-tui/internal/storage"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

var ansiEscapePattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

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

func TestResizePanelsKeepsAllPaneTitlesVisible(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	sizedModel, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	sized := sizedModel.(Model)

	view := stripANSI(sized.View())
	lineCount := strings.Count(view, "\n") + 1
	if lineCount > 30 {
		t.Fatalf("expected view line count <= window height (%d), got %d", 30, lineCount)
	}

	for _, title := range []string{
		"Config JSON",
		"System Monitor",
		"Month-to-Month Steps (So Far)",
		"Live Telemetry",
		"Results Explorer",
		"Run History",
	} {
		if !strings.Contains(view, title) {
			t.Fatalf("expected pane title %q to be visible in view", title)
		}
	}
}

func TestResizePanelsPrioritizesSystemMonitorVisibility(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	sizedModel, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	sized := sizedModel.(Model)

	if sized.resourceH < minMonitorPanelHeight {
		t.Fatalf("expected monitor panel height >= %d, got %d", minMonitorPanelHeight, sized.resourceH)
	}
	if !strings.Contains(stripANSI(sized.View()), "System Monitor") {
		t.Fatalf("expected system monitor panel to remain visible in compact layout")
	}
}

func TestViewAlwaysShowsComputeCounterPane(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	sizedModel, _ := m.Update(tea.WindowSizeMsg{Width: 110, Height: 24})
	sized := sizedModel.(Model)

	view := stripANSI(sized.View())
	if !strings.Contains(view, "Month-to-Month Steps (So Far)") {
		t.Fatalf("expected compute counter pane to always be visible, got %q", view)
	}
	if !strings.Contains(view, "Live Telemetry") {
		t.Fatalf("expected live telemetry pane to remain visible in standard window size, got %q", view)
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

func TestRenderUsageWaveformRightAlignsWhenHistoryShort(t *testing.T) {
	t.Parallel()

	got := []rune(stripANSI(renderUsageWaveform([]float64{50, 100}, 6, 0.85, cpuWavePalette)))
	if len(got) != 6 {
		t.Fatalf("expected waveform width 6, got %d", len(got))
	}
	if string(got[:4]) != strings.Repeat(string(waveformPadRune), 4) {
		t.Fatalf("expected leading waveform padding, got %q", string(got))
	}

	tailFromMinWidth := []rune(stripANSI(renderUsageWaveform([]float64{50, 100}, 4, 0.85, cpuWavePalette)))
	if string(got[4:]) != string(tailFromMinWidth[2:]) {
		t.Fatalf("expected right-aligned tail %q, got %q", string(tailFromMinWidth[2:]), string(got[4:]))
	}
}

func TestRenderUsageWaveformUsesLatestWindow(t *testing.T) {
	t.Parallel()

	full := stripANSI(renderUsageWaveform([]float64{0, 10, 20, 30, 40, 50}, 4, 0.4, cpuWavePalette))
	latestOnly := stripANSI(renderUsageWaveform([]float64{20, 30, 40, 50}, 4, 0.4, cpuWavePalette))
	if full != latestOnly {
		t.Fatalf("expected waveform to use latest window: full=%q latest=%q", full, latestOnly)
	}
}

func TestRenderUsageWaveformClampsValues(t *testing.T) {
	t.Parallel()

	clamped := stripANSI(renderUsageWaveform([]float64{-10, 120, 220}, 7, 0.5, cpuWavePalette))
	bounded := stripANSI(renderUsageWaveform([]float64{0, 100, 100}, 7, 0.5, cpuWavePalette))
	if clamped != bounded {
		t.Fatalf("expected waveform samples to clamp to [0,100], clamped=%q bounded=%q", clamped, bounded)
	}
}

func TestRenderUsageWaveformPhaseAffectsOutput(t *testing.T) {
	t.Parallel()

	samples := []float64{12, 38, 44, 58, 73, 29, 65, 82}
	first := stripANSI(renderUsageWaveform(samples, 16, 0.9, gpuWavePalette))
	again := stripANSI(renderUsageWaveform(samples, 16, 0.9, gpuWavePalette))
	shifted := stripANSI(renderUsageWaveform(samples, 16, 1.7, gpuWavePalette))
	if first != again {
		t.Fatalf("expected deterministic waveform at same phase, first=%q again=%q", first, again)
	}
	if first == shifted {
		t.Fatalf("expected waveform to change when phase changes, first=%q shifted=%q", first, shifted)
	}
}

func TestRenderUsageWaveformEmptyHistoryUsesBaseline(t *testing.T) {
	t.Parallel()

	got := stripANSI(renderUsageWaveform(nil, 7, 0, memWavePalette))
	want := strings.Repeat(string(waveformPadRune), 7)
	if got != want {
		t.Fatalf("expected baseline waveform %q, got %q", want, got)
	}
}

func TestRenderUsageWaveformRowsHighEnergyGetsUpperBand(t *testing.T) {
	t.Parallel()

	lowTop, _ := renderUsageWaveformRows([]float64{10, 12, 9, 15, 14}, 10, 0.6, cpuWavePalette)
	highTop, _ := renderUsageWaveformRows([]float64{92, 96, 98, 94, 99}, 10, 0.6, cpuWavePalette)

	if strings.TrimSpace(stripANSI(lowTop)) != "" {
		t.Fatalf("expected low-energy waveform to keep upper row empty, got %q", stripANSI(lowTop))
	}
	if strings.TrimSpace(stripANSI(highTop)) == "" {
		t.Fatalf("expected high-energy waveform to draw upper-row peaks, got %q", stripANSI(highTop))
	}
}

func TestFormatApproxCountUsesWordUnitsForLargeValues(t *testing.T) {
	t.Parallel()

	cases := []struct {
		value float64
		want  string
	}{
		{4_400_000, "4.4 million"},
		{4_400_000_000, "4.4 billion"},
		{14_500_000_000_000, "14.5 trillion"},
	}
	for _, tc := range cases {
		got := formatApproxCount(tc.value)
		if got != tc.want {
			t.Fatalf("formatApproxCount(%v) => %q, want %q", tc.value, got, tc.want)
		}
	}

	if got := formatApproxCount(6_000); got != "6.00K" {
		t.Fatalf("expected thousands formatting to remain compact, got %q", got)
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

func TestWaveTickAdvancesPhaseAndSchedulesNextTick(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	initial := m.wavePhase

	nextModel, cmd := m.Update(waveTickMsg{at: time.Now()})
	next := nextModel.(Model)
	if cmd == nil {
		t.Fatalf("expected wave tick to schedule next animation tick")
	}
	if next.wavePhase == initial {
		t.Fatalf("expected wave phase to advance, still %.2f", next.wavePhase)
	}
}

func TestTelemetryCounterTracksMonthUpdatesFromProgressPayload(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	cfg := map[string]any{
		"simulation": map[string]any{
			"n_sims":         1000,
			"horizon_months": 12,
		},
		"decision_grid": map[string]any{
			"num_points": 5,
		},
	}

	m.running = true
	m.currentRunID = "run-1"
	m.currentConfig = cfg
	m.resetTelemetryEstimate(cfg)

	nextModel, _ := m.Update(streamEventMsg{
		streamID: m.streamID,
		ok:       true,
		events: []service.StreamEvent{
			{
				Seq:   1,
				Event: "run_start",
				Payload: map[string]any{
					"n_sims":      1000,
					"n_fractions": 5,
				},
			},
			{
				Seq:   2,
				Event: "primary_pass_start",
				Payload: map[string]any{
					"n_sims":      1000,
					"n_fractions": 5,
				},
			},
			{
				Seq:   3,
				Event: "primary_pass_progress",
				Payload: map[string]any{
					"sims_completed":         100,
					"sims_total":             1000,
					"month_updates_computed": 6000,
				},
			},
		},
	})
	next := nextModel.(Model)
	if int64(next.monthUpdatesApprox) != 6000 {
		t.Fatalf("expected month updates 6000, got %.0f", next.monthUpdatesApprox)
	}
	if !strings.Contains(stripANSI(next.renderComputeCounter()), "6.00K") {
		t.Fatalf("expected compute counter pane content to include latest count, got %q", stripANSI(next.renderComputeCounter()))
	}

	finalModel, _ := next.Update(streamEventMsg{
		streamID: next.streamID,
		ok:       true,
		events: []service.StreamEvent{
			{
				Seq:   4,
				Event: "primary_pass_progress",
				Payload: map[string]any{
					"sims_completed":         250,
					"sims_total":             1000,
					"month_updates_computed": 15000,
				},
			},
		},
	})
	finalState := finalModel.(Model)
	if int64(finalState.monthUpdatesApprox) != 15000 {
		t.Fatalf("expected month updates 15000, got %.0f", finalState.monthUpdatesApprox)
	}
}

func TestTelemetryCounterTracksMonthUpdatesFromStatusPayload(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	cfg := map[string]any{
		"simulation": map[string]any{
			"n_sims":         1000,
			"horizon_months": 12,
		},
		"decision_grid": map[string]any{
			"num_points": 5,
		},
	}

	m.running = true
	m.currentRunID = "run-1"
	m.currentConfig = cfg
	m.resetTelemetryEstimate(cfg)

	nextModel, _ := m.Update(runStatusMsg{
		status: &service.RunStatus{
			RunID:                "run-1",
			Status:               "running",
			LatestSeq:            7,
			LatestEvent:          "primary_pass_progress",
			LatestEventTimestamp: "2026-02-24T18:30:00Z",
			LatestEventPayload: map[string]any{
				"sims_completed":         250,
				"sims_total":             1000,
				"month_updates_computed": 15000,
			},
		},
	})
	next := nextModel.(Model)
	if int64(next.monthUpdatesApprox) != 15000 {
		t.Fatalf("expected month updates 15000 from runStatus payload, got %.0f", next.monthUpdatesApprox)
	}
}

func TestRenderSystemMonitorUsesWaveformRows(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.resourceW = 54
	m.resourceH = 9
	m.wavePhase = 0.7
	m.cpuUsagePct = 42
	m.cpuHistory = []float64{18, 25, 30, 44, 51, 48}
	m.memAvailable = true
	m.memUsagePct = 67
	m.memHistory = []float64{38, 40, 43, 49, 57, 60}
	m.gpuAvailable = true
	m.gpuUsagePct = 74
	m.gpuHistory = []float64{45, 55, 62, 68, 74, 79}

	plain := stripANSI(m.renderSystemMonitor())
	if !regexp.MustCompile(`cpu~ [▁▂▃▄▅▆▇█]+`).MatchString(plain) {
		t.Fatalf("expected cpu waveform row, got %q", plain)
	}
	if !regexp.MustCompile(`gpu~ [▁▂▃▄▅▆▇█]+`).MatchString(plain) {
		t.Fatalf("expected gpu waveform row, got %q", plain)
	}
	if !regexp.MustCompile(`mem~ [▁▂▃▄▅▆▇█]+`).MatchString(plain) {
		t.Fatalf("expected mem waveform row, got %q", plain)
	}
}

func TestRenderSystemMonitorTallLayoutUsesTwoRowWaveforms(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.resourceW = 54
	m.resourceH = fullMonitorPanelHeight
	m.wavePhase = 0.7
	m.cpuUsagePct = 91
	m.cpuHistory = []float64{88, 90, 93, 95, 97, 96}
	m.memAvailable = true
	m.memUsagePct = 87
	m.memHistory = []float64{82, 84, 88, 90, 91, 89}
	m.gpuAvailable = true
	m.gpuUsagePct = 94
	m.gpuHistory = []float64{89, 92, 94, 96, 98, 97}

	lines := strings.Split(stripANSI(m.renderSystemMonitor()), "\n")
	if len(lines) < 9 {
		t.Fatalf("expected tall monitor layout with expanded waveform rows, got %d lines (%q)", len(lines), strings.Join(lines, "\n"))
	}

	var cpuRow int = -1
	for idx, line := range lines {
		if strings.HasPrefix(line, "cpu~ ") {
			cpuRow = idx
			break
		}
	}
	if cpuRow < 0 || cpuRow+1 >= len(lines) {
		t.Fatalf("expected cpu waveform row and continuation row in tall layout")
	}
	if !strings.HasPrefix(lines[cpuRow+1], waveformContinuationPrefix) {
		t.Fatalf("expected continuation prefix %q after cpu waveform row, got %q", waveformContinuationPrefix, lines[cpuRow+1])
	}
	if strings.TrimSpace(strings.TrimPrefix(lines[cpuRow], "cpu~ ")) == "" {
		t.Fatalf("expected high-energy cpu waveform upper row to be non-empty, got %q", lines[cpuRow])
	}
}

func TestRenderSystemMonitorCompactHeightKeepsGPUWaveRow(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.resourceW = 54
	m.resourceH = 5 // 4 visible body lines in monitor panel
	m.wavePhase = 0.9
	m.cpuUsagePct = 38
	m.cpuHistory = []float64{18, 23, 31, 40, 46, 52}
	m.gpuAvailable = true
	m.gpuUsagePct = 71
	m.gpuHistory = []float64{42, 55, 63, 67, 73, 79}
	m.memAvailable = true
	m.memUsagePct = 64
	m.memHistory = []float64{31, 35, 39, 44, 53, 58}

	plain := stripANSI(m.renderSystemMonitor())
	lines := strings.Split(plain, "\n")
	if len(lines) != 4 {
		t.Fatalf("expected compact monitor body with 4 lines, got %d (%q)", len(lines), plain)
	}
	if !strings.HasPrefix(lines[3], "gpu~ ") {
		t.Fatalf("expected gpu waveform line in compact view, got %q", lines[3])
	}
}

func TestRenderSystemMonitorKeepsGPUUnavailableMessage(t *testing.T) {
	t.Parallel()

	m := NewModel(nil, nil)
	m.resourceW = 54
	m.resourceH = 8
	m.cpuHistory = []float64{22, 24, 26}
	m.gpuAvailable = false
	m.gpuNote = "driver missing"

	plain := stripANSI(m.renderSystemMonitor())
	if !strings.Contains(plain, "gpu~ unavailable: driver missing") {
		t.Fatalf("expected gpu unavailable note in monitor, got %q", plain)
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

func stripANSI(raw string) string {
	return ansiEscapePattern.ReplaceAllString(raw, "")
}
