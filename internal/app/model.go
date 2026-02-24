package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"blackswan-tui/internal/service"
	"blackswan-tui/internal/storage"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	chromeBG        = lipgloss.Color("#05090C")
	panelBG         = lipgloss.Color("#0D141A")
	panelBorder     = lipgloss.Color("#2D6A80")
	accentPrimary   = lipgloss.Color("#50E3C2")
	accentSecondary = lipgloss.Color("#F6AE2D")
	mutedText       = lipgloss.Color("#8CA1AE")
	warningText     = lipgloss.Color("#FF6B6B")
)

var (
	headerStyle = lipgloss.NewStyle().
			Padding(0, 1).
			Bold(true).
			Foreground(accentPrimary)

	subHeaderStyle = lipgloss.NewStyle().
			Foreground(mutedText)

	statusStyle = lipgloss.NewStyle().
			Foreground(accentSecondary).
			Bold(true)

	errorStyle = lipgloss.NewStyle().
			Foreground(warningText).
			Bold(true)

	panelTitleStyle = lipgloss.NewStyle().
			Foreground(accentPrimary).
			Bold(true)

	panelStyle = lipgloss.NewStyle().
			Background(panelBG).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(panelBorder).
			Padding(0, 1)

	helpStyle = lipgloss.NewStyle().
			Foreground(mutedText)

	jsonKeyStyle = lipgloss.NewStyle().
			Foreground(accentPrimary).
			Bold(true)

	jsonObjectKeyStyle = lipgloss.NewStyle().
				Foreground(accentSecondary).
				Bold(true)

	historySelectedLineStyle = lipgloss.NewStyle().
					Foreground(accentPrimary).
					Background(lipgloss.Color("#16323D")).
					Bold(true)
)

var jsonKeyPattern = regexp.MustCompile(`"([^"\\]|\\.)*"\s*:`)

type defaultsLoadedMsg struct {
	defaults map[string]any
	err      error
}

type historyLoadedMsg struct {
	items []storage.RunSummary
	err   error
}

type configFileLoadedMsg struct {
	path   string
	config map[string]any
	text   string
	err    error
}

type runStartedMsg struct {
	runID  string
	config map[string]any
	err    error
}

type runStatusMsg struct {
	status *service.RunStatus
	err    error
}

type runCancelledMsg struct {
	err error
}

type streamEventMsg struct {
	streamID int64
	events   []service.StreamEvent
	ok       bool
}

type streamErrMsg struct {
	streamID int64
	err      error
	ok       bool
}

type bundleSavedMsg struct {
	summary storage.RunSummary
	err     error
}

type bundleLoadedMsg struct {
	bundle *storage.RunBundle
	err    error
}

type pollTickMsg struct {
	at time.Time
}

type systemMetricsSnapshot struct {
	cpuTotalTicks uint64
	cpuIdleTicks  uint64
	memAvailable  bool
	memUsedPct    float64
	gpuAvailable  bool
	gpuUtilPct    float64
	gpuMemUsedPct float64
	gpuLabel      string
	gpuNote       string
	gpuProbeOK    bool
}

type systemMetricsMsg struct {
	snapshot  systemMetricsSnapshot
	sampledAt time.Time
	err       error
}

type focusPane int

const (
	paneConfig focusPane = iota
	paneTelemetry
	paneResults
	paneHistory
)

const (
	metricsPollInterval      = 1200 * time.Millisecond
	maxMetricsCatchupSamples = 12
)

type ModelOptions struct {
	InitialConfigJSON string
	InitialConfigPath string
}

type Model struct {
	service *service.Manager
	store   *storage.Store

	ready  bool
	width  int
	height int

	configEditor    textarea.Model
	configPathInput textinput.Model
	telemetry       viewport.Model
	results         viewport.Model
	history         viewport.Model
	spinner         spinner.Model

	focusPane focusPane
	showHelp  bool

	statusText             string
	errorText              string
	configPinned           bool
	showConfigPathPrompt   bool
	lastConfigPath         string
	configPathChoices      []string
	configPathChoiceCursor int
	configPathChoiceOffset int

	running            bool
	currentRunID       string
	currentConfig      map[string]any
	latestStatus       *service.RunStatus
	streamCancel       context.CancelFunc
	streamChan         <-chan service.StreamEvent
	streamErrChan      <-chan error
	streamID           int64
	lastPolledEventSeq int64
	events             []service.StreamEvent
	finalizingRun      bool
	finalizingSince    time.Time
	pendingTerminal    *service.RunStatus

	historyItems            []storage.RunSummary
	historyCursor           int
	historyCursorTopLine    int
	historyCursorBottomLine int
	historyRenderedLines    int

	telemetryEntries    []string
	telemetryLines      []string
	telemetryAutoFollow bool

	cpuUsagePct         float64
	memUsagePct         float64
	memAvailable        bool
	gpuUsagePct         float64
	gpuMemUsagePct      float64
	gpuAvailable        bool
	gpuLabel            string
	gpuNote             string
	gpuProbeEnabled     bool
	cpuLastTotal        uint64
	cpuLastIdle         uint64
	cpuHasBaseline      bool
	cpuHistory          []float64
	memHistory          []float64
	gpuHistory          []float64
	terminalFocused     bool
	lastMetricsSampleAt time.Time

	configPanelW int
	configPanelH int
	resourceW    int
	resourceH    int
	telemetryW   int
	telemetryH   int
	resultsW     int
	resultsH     int
	historyW     int
	historyH     int
}

func NewModel(svc *service.Manager, store *storage.Store) Model {
	return NewModelWithOptions(svc, store, ModelOptions{})
}

func NewModelWithOptions(svc *service.Manager, store *storage.Store, opts ModelOptions) Model {
	cfgEditor := textarea.New()
	cfgEditor.CharLimit = 2_000_000
	cfgEditor.Prompt = ""
	cfgEditor.ShowLineNumbers = true
	cfgEditor.SetHeight(20)
	cfgEditor.SetWidth(70)
	cfgEditor.Focus()
	cfgEditor.Placeholder = "Waiting for defaults from local service..."

	telemetry := viewport.New(50, 20)
	telemetry.SetContent("Telemetry stream is idle.")

	results := viewport.New(50, 14)
	results.SetContent("No run yet. Press ctrl+r to launch a run.")

	history := viewport.New(40, 14)
	history.SetContent("No saved runs yet.")

	spin := spinner.New()
	spin.Spinner = spinner.MiniDot
	spin.Style = lipgloss.NewStyle().Foreground(accentSecondary)

	pathInput := textinput.New()
	pathInput.Prompt = "> "
	pathInput.Placeholder = "./config.json"
	pathInput.CharLimit = 2048
	pathInput.Width = 70

	model := Model{
		service:             svc,
		store:               store,
		configEditor:        cfgEditor,
		configPathInput:     pathInput,
		telemetry:           telemetry,
		results:             results,
		history:             history,
		spinner:             spin,
		focusPane:           paneConfig,
		showHelp:            true,
		statusText:          "Service connected. Loading defaults...",
		gpuProbeEnabled:     true,
		gpuNote:             "probing...",
		telemetryAutoFollow: true,
		configPanelW:        74,
		configPanelH:        22,
		resourceW:           54,
		resourceH:           8,
		telemetryW:          54,
		telemetryH:          22,
		resultsW:            54,
		resultsH:            16,
		historyW:            44,
		historyH:            16,
		terminalFocused:     true,
	}
	if strings.TrimSpace(opts.InitialConfigJSON) != "" {
		model.configEditor.SetValue(opts.InitialConfigJSON)
		model.configPinned = true
		model.lastConfigPath = strings.TrimSpace(opts.InitialConfigPath)
		if model.lastConfigPath != "" {
			model.statusText = "Service connected. Loaded startup config from " + model.lastConfigPath
		} else {
			model.statusText = "Service connected. Loaded startup config."
		}
	}
	return model
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		loadDefaultsCmd(m.service),
		loadHistoryCmd(m.store),
		sampleSystemMetricsCmd(m.gpuProbeEnabled),
		pollTickCmd(),
	)
}

func loadDefaultsCmd(svc *service.Manager) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		defaults, err := svc.Defaults(ctx)
		return defaultsLoadedMsg{defaults: defaults, err: err}
	}
}

func loadHistoryCmd(store *storage.Store) tea.Cmd {
	return func() tea.Msg {
		items, err := store.List(200)
		return historyLoadedMsg{items: items, err: err}
	}
}

func loadConfigFileCmd(path string) tea.Cmd {
	requestedPath := strings.TrimSpace(path)
	return func() tea.Msg {
		cfg, resolvedPath, err := LoadConfigFile(requestedPath)
		if err != nil {
			return configFileLoadedMsg{path: requestedPath, err: err}
		}
		text, err := FormatConfigJSON(cfg)
		if err != nil {
			return configFileLoadedMsg{path: resolvedPath, err: err}
		}
		return configFileLoadedMsg{
			path:   resolvedPath,
			config: cfg,
			text:   text,
		}
	}
}

func listJSONFilesInDir(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := strings.TrimSpace(entry.Name())
		if name == "" {
			continue
		}
		if strings.EqualFold(filepath.Ext(name), ".json") {
			files = append(files, name)
		}
	}

	sort.Slice(files, func(i, j int) bool {
		li := strings.ToLower(files[i])
		lj := strings.ToLower(files[j])
		if li == lj {
			return files[i] < files[j]
		}
		return li < lj
	})
	return files, nil
}

func (m *Model) configPathListVisibleRows() int {
	if m.height <= 0 {
		return 6
	}
	return clampInt(m.height/6, 4, 10)
}

func (m *Model) ensureConfigPathChoiceVisible() {
	if len(m.configPathChoices) == 0 {
		m.configPathChoiceCursor = 0
		m.configPathChoiceOffset = 0
		return
	}

	m.configPathChoiceCursor = clampInt(m.configPathChoiceCursor, 0, len(m.configPathChoices)-1)
	visibleRows := maxInt(1, m.configPathListVisibleRows())
	maxOffset := maxInt(0, len(m.configPathChoices)-visibleRows)
	m.configPathChoiceOffset = clampInt(m.configPathChoiceOffset, 0, maxOffset)

	if m.configPathChoiceCursor < m.configPathChoiceOffset {
		m.configPathChoiceOffset = m.configPathChoiceCursor
	}
	if m.configPathChoiceCursor >= m.configPathChoiceOffset+visibleRows {
		m.configPathChoiceOffset = m.configPathChoiceCursor - visibleRows + 1
	}
	m.configPathChoiceOffset = clampInt(m.configPathChoiceOffset, 0, maxOffset)
}

func (m *Model) setConfigPathChoiceCursor(cursor int) {
	if len(m.configPathChoices) == 0 {
		m.configPathChoiceCursor = 0
		m.configPathChoiceOffset = 0
		return
	}
	m.configPathChoiceCursor = clampInt(cursor, 0, len(m.configPathChoices)-1)
	m.ensureConfigPathChoiceVisible()
	selected := m.configPathChoices[m.configPathChoiceCursor]
	m.configPathInput.SetValue(selected)
	m.configPathInput.CursorEnd()
}

func (m *Model) syncConfigPathChoiceToInput() {
	if len(m.configPathChoices) == 0 {
		return
	}
	input := strings.TrimSpace(m.configPathInput.Value())
	if input == "" {
		return
	}

	inputBase := filepathBase(input)
	for idx, choice := range m.configPathChoices {
		if choice == input || choice == inputBase {
			m.configPathChoiceCursor = idx
			m.ensureConfigPathChoiceVisible()
			return
		}
	}
}

func (m *Model) refreshConfigPathChoices() error {
	choices, err := listJSONFilesInDir(".")
	if err != nil {
		m.configPathChoices = nil
		m.configPathChoiceCursor = 0
		m.configPathChoiceOffset = 0
		return err
	}

	m.configPathChoices = choices
	m.configPathChoiceCursor = 0
	m.configPathChoiceOffset = 0
	if len(choices) == 0 {
		return nil
	}

	desired := strings.TrimSpace(m.configPathInput.Value())
	if desired == "" {
		desired = strings.TrimSpace(m.lastConfigPath)
	}
	desiredBase := filepathBase(desired)
	for idx, choice := range choices {
		if choice == desired || choice == desiredBase {
			m.configPathChoiceCursor = idx
			break
		}
	}
	m.ensureConfigPathChoiceVisible()

	if strings.TrimSpace(m.configPathInput.Value()) == "" {
		m.configPathInput.SetValue(m.configPathChoices[m.configPathChoiceCursor])
		m.configPathInput.CursorEnd()
	}
	return nil
}

func (m Model) renderConfigPathChoices(visibleRows int) string {
	if len(m.configPathChoices) == 0 {
		return mutedTextStyle("No .json files in current directory.")
	}

	visibleRows = maxInt(1, visibleRows)
	start := clampInt(m.configPathChoiceOffset, 0, len(m.configPathChoices)-1)
	end := minInt(len(m.configPathChoices), start+visibleRows)
	if end-start < visibleRows && end > 0 {
		start = maxInt(0, end-visibleRows)
	}

	lines := make([]string, 0, (end-start)+1)
	for idx := start; idx < end; idx++ {
		prefix := "  "
		line := m.configPathChoices[idx]
		if idx == m.configPathChoiceCursor {
			prefix = "▶ "
			line = historySelectedLineStyle.Render(prefix + line)
		} else {
			line = prefix + line
		}
		lines = append(lines, line)
	}
	lines = append(lines, mutedTextStyle(fmt.Sprintf("Showing %d-%d of %d", start+1, end, len(m.configPathChoices))))
	return strings.Join(lines, "\n")
}

func mutedTextStyle(text string) string {
	return lipgloss.NewStyle().Foreground(mutedText).Render(text)
}

func startRunCmd(svc *service.Manager, config map[string]any) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		runID, err := svc.StartRun(ctx, config)
		return runStartedMsg{runID: runID, config: config, err: err}
	}
}

func cancelRunCmd(svc *service.Manager, runID string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := svc.CancelRun(ctx, runID)
		return runCancelledMsg{err: err}
	}
}

func fetchRunStatusCmd(svc *service.Manager, runID string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		status, err := svc.GetRun(ctx, runID)
		return runStatusMsg{status: status, err: err}
	}
}

func saveBundleCmd(store *storage.Store, runID string, config map[string]any, result map[string]any, events []service.StreamEvent) tea.Cmd {
	copyEvents := append([]service.StreamEvent(nil), events...)
	copyConfig := cloneMap(config)
	copyResult := cloneMap(result)
	return func() tea.Msg {
		summary, err := store.SaveRun(runID, copyConfig, copyResult, copyEvents)
		return bundleSavedMsg{summary: summary, err: err}
	}
}

func loadBundleCmd(store *storage.Store, directory string) tea.Cmd {
	return func() tea.Msg {
		bundle, err := store.LoadBundle(directory)
		return bundleLoadedMsg{bundle: bundle, err: err}
	}
}

func sampleSystemMetricsCmd(gpuProbeEnabled bool) tea.Cmd {
	return func() tea.Msg {
		sampledAt := time.Now()
		snapshot, err := collectSystemMetrics(gpuProbeEnabled)
		return systemMetricsMsg{snapshot: snapshot, sampledAt: sampledAt, err: err}
	}
}

func collectSystemMetrics(gpuProbeEnabled bool) (systemMetricsSnapshot, error) {
	snapshot := systemMetricsSnapshot{gpuProbeOK: gpuProbeEnabled}

	total, idle, err := readCPUTickTotals()
	if err != nil {
		return snapshot, err
	}
	snapshot.cpuTotalTicks = total
	snapshot.cpuIdleTicks = idle

	if memPct, err := readMemoryUsedPercent(); err == nil {
		snapshot.memAvailable = true
		snapshot.memUsedPct = memPct
	}

	if !gpuProbeEnabled {
		return snapshot, nil
	}

	gpuPct, gpuMemPct, gpuLabel, gpuNote, probeOK := readTotalGPUUsage()
	snapshot.gpuProbeOK = probeOK
	snapshot.gpuLabel = gpuLabel
	snapshot.gpuNote = gpuNote
	if gpuPct >= 0 {
		snapshot.gpuAvailable = true
		snapshot.gpuUtilPct = gpuPct
		snapshot.gpuMemUsedPct = gpuMemPct
	}
	return snapshot, nil
}

func readCPUTickTotals() (uint64, uint64, error) {
	blob, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, err
	}
	firstLine := ""
	lines := strings.SplitN(string(blob), "\n", 2)
	if len(lines) > 0 {
		firstLine = strings.TrimSpace(lines[0])
	}
	fields := strings.Fields(firstLine)
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0, 0, fmt.Errorf("unexpected /proc/stat cpu line")
	}

	var total uint64
	var idle uint64
	for idx := 1; idx < len(fields); idx++ {
		value, parseErr := strconv.ParseUint(fields[idx], 10, 64)
		if parseErr != nil {
			continue
		}
		total += value
		if idx == 4 || idx == 5 {
			idle += value
		}
	}
	if total == 0 {
		return 0, 0, fmt.Errorf("invalid cpu totals from /proc/stat")
	}
	return total, idle, nil
}

func readMemoryUsedPercent() (float64, error) {
	blob, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}

	var totalKB float64
	var availableKB float64
	for _, line := range strings.Split(string(blob), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			totalKB, _ = strconv.ParseFloat(fields[1], 64)
		case "MemAvailable:":
			availableKB, _ = strconv.ParseFloat(fields[1], 64)
		}
	}
	if totalKB <= 0 {
		return 0, fmt.Errorf("missing MemTotal in /proc/meminfo")
	}
	if availableKB < 0 {
		availableKB = 0
	}
	usedPct := ((totalKB - availableKB) / totalKB) * 100.0
	return clampFloat(usedPct, 0, 100), nil
}

func readTotalGPUUsage() (float64, float64, string, string, bool) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return -1, 0, "", "nvidia-smi not found", false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 900*time.Millisecond)
	defer cancel()

	cmd := exec.CommandContext(
		ctx,
		path,
		"--query-gpu=utilization.gpu,memory.used,memory.total,name",
		"--format=csv,noheader,nounits",
	)
	output, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(err.Error())
		if ctx.Err() == context.DeadlineExceeded {
			msg = "nvidia-smi timed out"
		}
		return -1, 0, "", msg, true
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	utilSum := 0.0
	memUsedSum := 0.0
	memTotalSum := 0.0
	count := 0
	firstName := ""

	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 4)
		if len(parts) < 4 {
			continue
		}

		util, utilErr := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		memUsed, usedErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		memTotal, totalErr := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
		name := strings.TrimSpace(parts[3])
		if utilErr != nil || usedErr != nil || totalErr != nil {
			continue
		}
		if firstName == "" {
			firstName = name
		}
		utilSum += util
		memUsedSum += memUsed
		memTotalSum += memTotal
		count++
	}

	if count == 0 {
		return -1, 0, "", "no NVIDIA GPUs detected", true
	}

	avgUtil := utilSum / float64(count)
	memPct := 0.0
	if memTotalSum > 0 {
		memPct = (memUsedSum / memTotalSum) * 100.0
	}

	label := firstName
	if count > 1 {
		label = fmt.Sprintf("%s +%d", firstName, count-1)
	}
	return clampFloat(avgUtil, 0, 100), clampFloat(memPct, 0, 100), label, fmt.Sprintf("%d GPU(s)", count), true
}

func pollTickCmd() tea.Cmd {
	return tea.Tick(metricsPollInterval, func(at time.Time) tea.Msg {
		return pollTickMsg{at: at}
	})
}

func waitForStreamEventCmd(streamID int64, ch <-chan service.StreamEvent) tea.Cmd {
	return func() tea.Msg {
		event, ok := <-ch
		if !ok {
			return streamEventMsg{streamID: streamID, ok: false}
		}

		events := make([]service.StreamEvent, 0, 64)
		events = append(events, event)
		for len(events) < 64 {
			select {
			case next, ok := <-ch:
				if !ok {
					return streamEventMsg{streamID: streamID, events: events, ok: true}
				}
				events = append(events, next)
			default:
				return streamEventMsg{streamID: streamID, events: events, ok: true}
			}
		}
		return streamEventMsg{streamID: streamID, events: events, ok: true}
	}
}

func waitForStreamErrCmd(streamID int64, ch <-chan error) tea.Cmd {
	return func() tea.Msg {
		err, ok := <-ch
		return streamErrMsg{streamID: streamID, err: err, ok: ok}
	}
}

func (m *Model) startStream(runID string, fromSeq int64) tea.Cmd {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return nil
	}
	if fromSeq < 0 {
		fromSeq = 0
	}

	if m.streamCancel != nil {
		m.streamCancel()
	}
	m.streamID++
	currentStreamID := m.streamID

	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel
	stream := make(chan service.StreamEvent, 128)
	streamErr := make(chan error, 1)
	m.streamChan = stream
	m.streamErrChan = streamErr

	go func() {
		err := m.service.StreamRun(ctx, runID, fromSeq, stream)
		streamErr <- err
		close(streamErr)
	}()

	return tea.Batch(
		waitForStreamEventCmd(currentStreamID, m.streamChan),
		waitForStreamErrCmd(currentStreamID, m.streamErrChan),
	)
}

func (m *Model) finalizeTerminalRun(status *service.RunStatus) tea.Cmd {
	if status == nil {
		m.running = false
		m.finalizingRun = false
		m.finalizingSince = time.Time{}
		m.pendingTerminal = nil
		if m.streamCancel != nil {
			m.streamCancel()
			m.streamCancel = nil
		}
		m.streamID++
		m.streamChan = nil
		m.streamErrChan = nil
		m.statusText = "Run finished."
		return nil
	}

	m.running = false
	m.finalizingRun = false
	m.finalizingSince = time.Time{}
	m.pendingTerminal = nil
	if m.streamCancel != nil {
		m.streamCancel()
		m.streamCancel = nil
	}
	m.streamID++
	m.streamChan = nil
	m.streamErrChan = nil

	m.appendTelemetryLine(fmt.Sprintf("%s | status | %s", statusTimestamp(status), status.Status))
	switch status.Status {
	case "completed":
		m.statusText = fmt.Sprintf("Run %s completed", shortRunID(status.RunID))
		if status.Result != nil {
			return tea.Batch(
				saveBundleCmd(m.store, status.RunID, m.currentConfig, status.Result, m.events),
				loadHistoryCmd(m.store),
			)
		}
	case "failed":
		m.errorText = fmt.Sprintf("Run %s failed: %s", shortRunID(status.RunID), status.Error)
		m.statusText = "Run failed"
	case "cancelled":
		m.statusText = fmt.Sprintf("Run %s cancelled", shortRunID(status.RunID))
	}
	return nil
}

func (m *Model) recoverTerminalStateForRestart(now time.Time) (tea.Cmd, bool) {
	if !(m.running || m.finalizingRun) {
		return nil, false
	}

	currentRunID := strings.TrimSpace(m.currentRunID)
	chooseTerminal := func(candidate *service.RunStatus) *service.RunStatus {
		if candidate == nil || !isTerminalRunStatus(candidate.Status) {
			return nil
		}
		candidateRunID := strings.TrimSpace(candidate.RunID)
		if currentRunID != "" && candidateRunID != "" && candidateRunID != currentRunID {
			return nil
		}
		return candidate
	}

	if status := chooseTerminal(m.pendingTerminal); status != nil {
		m.appendTelemetryLine(now.Format("15:04:05") + " | local | recovered stale finalizing state")
		return m.finalizeTerminalRun(status), true
	}
	if status := chooseTerminal(m.latestStatus); status != nil {
		m.appendTelemetryLine(now.Format("15:04:05") + " | local | recovered stale active state")
		return m.finalizeTerminalRun(status), true
	}
	return nil, false
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true
		m.resizePanels()
		m.applyFocusState()
		m.refreshHistoryView()
		if m.showConfigPathPrompt {
			m.ensureConfigPathChoiceVisible()
		}
		return m, nil

	case tea.BlurMsg:
		m.terminalFocused = false
		return m, nil

	case tea.FocusMsg:
		m.terminalFocused = true
		return m, sampleSystemMetricsCmd(m.gpuProbeEnabled)

	case spinner.TickMsg:
		if !m.running {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case pollTickMsg:
		cmds := []tea.Cmd{pollTickCmd(), sampleSystemMetricsCmd(m.gpuProbeEnabled)}
		if m.finalizingRun && m.pendingTerminal != nil {
			if m.streamChan == nil {
				cmds = append(cmds, m.finalizeTerminalRun(m.pendingTerminal))
				return m, tea.Batch(cmds...)
			}
			if !m.finalizingSince.IsZero() && msg.at.Sub(m.finalizingSince) > 4*time.Second {
				m.appendTelemetryLine(msg.at.Format("15:04:05") + " | local | finalize timeout; closing stream")
				cmds = append(cmds, m.finalizeTerminalRun(m.pendingTerminal))
				return m, tea.Batch(cmds...)
			}
		}

		if (m.running || m.finalizingRun) && strings.TrimSpace(m.currentRunID) != "" {
			cmds = append(cmds, fetchRunStatusCmd(m.service, m.currentRunID))
			if m.streamChan == nil && m.streamErrChan == nil {
				cmds = append(cmds, m.startStream(m.currentRunID, m.lastPolledEventSeq))
			}
		}
		return m, tea.Batch(cmds...)

	case systemMetricsMsg:
		if msg.err != nil {
			if strings.TrimSpace(m.gpuNote) == "" {
				m.gpuNote = "system metrics unavailable"
			}
			return m, nil
		}
		sampleRepeats := 1
		if !msg.sampledAt.IsZero() {
			sampleRepeats = estimateMetricsCatchupSamples(
				m.lastMetricsSampleAt,
				msg.sampledAt,
				metricsPollInterval,
				maxMetricsCatchupSamples,
			)
			m.lastMetricsSampleAt = msg.sampledAt
		}

		if m.cpuHasBaseline {
			deltaTotal := msg.snapshot.cpuTotalTicks - m.cpuLastTotal
			deltaIdle := msg.snapshot.cpuIdleTicks - m.cpuLastIdle
			if deltaTotal > 0 && deltaTotal >= deltaIdle {
				busyPct := (float64(deltaTotal-deltaIdle) / float64(deltaTotal)) * 100.0
				m.cpuUsagePct = clampFloat(busyPct, 0, 100)
				m.cpuHistory = appendUsageSampleN(m.cpuHistory, m.cpuUsagePct, 160, sampleRepeats)
			}
		} else {
			m.cpuHasBaseline = true
		}
		m.cpuLastTotal = msg.snapshot.cpuTotalTicks
		m.cpuLastIdle = msg.snapshot.cpuIdleTicks

		if msg.snapshot.memAvailable {
			m.memAvailable = true
			m.memUsagePct = clampFloat(msg.snapshot.memUsedPct, 0, 100)
			m.memHistory = appendUsageSampleN(m.memHistory, m.memUsagePct, 160, sampleRepeats)
		}

		m.gpuProbeEnabled = msg.snapshot.gpuProbeOK
		m.gpuAvailable = msg.snapshot.gpuAvailable
		m.gpuLabel = strings.TrimSpace(msg.snapshot.gpuLabel)
		m.gpuNote = strings.TrimSpace(msg.snapshot.gpuNote)
		if m.gpuAvailable {
			m.gpuUsagePct = clampFloat(msg.snapshot.gpuUtilPct, 0, 100)
			m.gpuMemUsagePct = clampFloat(msg.snapshot.gpuMemUsedPct, 0, 100)
			m.gpuHistory = appendUsageSampleN(m.gpuHistory, m.gpuUsagePct, 160, sampleRepeats)
		}
		return m, nil

	case defaultsLoadedMsg:
		if msg.err != nil {
			m.errorText = "Failed to load defaults: " + msg.err.Error()
			m.statusText = "Defaults unavailable. You can still paste custom JSON config."
			return m, nil
		}
		if m.configPinned {
			m.statusText = "Defaults loaded. Existing config preserved."
			return m, nil
		}
		blob, err := json.MarshalIndent(msg.defaults, "", "  ")
		if err != nil {
			m.errorText = "Could not render defaults JSON: " + err.Error()
			return m, nil
		}
		m.configEditor.SetValue(string(blob))
		m.statusText = "Defaults loaded. Tune config and press ctrl+r to run."
		return m, nil

	case historyLoadedMsg:
		if msg.err != nil {
			m.errorText = "Failed to load run history: " + msg.err.Error()
			return m, nil
		}
		m.historyItems = append([]storage.RunSummary(nil), msg.items...)
		sort.SliceStable(m.historyItems, func(i, j int) bool {
			return m.historyItems[i].SavedAt > m.historyItems[j].SavedAt
		})
		if m.historyCursor >= len(m.historyItems) {
			m.historyCursor = maxInt(0, len(m.historyItems)-1)
		}
		m.refreshHistoryView()
		return m, nil

	case configFileLoadedMsg:
		// Always close the prompt and restore focus when async load completes.
		m.showConfigPathPrompt = false
		m.configPathInput.Blur()
		m.applyFocusState()
		if msg.err != nil {
			m.errorText = "Config file load failed: " + msg.err.Error()
			return m, tea.ClearScreen
		}
		m.configEditor.SetValue(msg.text)
		m.configPinned = true
		m.lastConfigPath = strings.TrimSpace(msg.path)
		m.errorText = ""
		if m.lastConfigPath != "" {
			m.statusText = "Loaded config file: " + m.lastConfigPath
		} else {
			m.statusText = "Loaded config file."
		}
		return m, tea.ClearScreen

	case runStartedMsg:
		if msg.err != nil {
			m.errorText = "Run launch failed: " + msg.err.Error()
			m.statusText = "Run did not start."
			return m, nil
		}
		m.running = true
		m.currentRunID = msg.runID
		m.currentConfig = cloneMap(msg.config)
		m.latestStatus = nil
		m.lastPolledEventSeq = 0
		m.events = m.events[:0]
		m.finalizingRun = false
		m.finalizingSince = time.Time{}
		m.pendingTerminal = nil
		m.telemetryEntries = nil
		m.telemetryLines = nil
		m.telemetry.SetContent("")
		m.telemetryAutoFollow = true
		m.appendTelemetryLine(time.Now().Format("15:04:05") + " | local | stream connecting")
		m.results.SetContent("Waiting for first telemetry events...")
		m.statusText = fmt.Sprintf("Run %s started", shortRunID(msg.runID))

		return m, tea.Batch(
			m.spinner.Tick,
			m.startStream(msg.runID, 0),
			fetchRunStatusCmd(m.service, msg.runID),
		)

	case runCancelledMsg:
		if msg.err != nil {
			m.errorText = "Cancel failed: " + msg.err.Error()
			return m, nil
		}
		m.appendTelemetryLine(time.Now().Format("15:04:05") + " | local | cancel requested")
		m.statusText = "Cancel requested. Waiting for worker shutdown."
		return m, nil

	case streamEventMsg:
		if msg.streamID != m.streamID {
			return m, nil
		}
		if !msg.ok {
			m.streamChan = nil
			if m.finalizingRun && m.pendingTerminal != nil {
				return m, m.finalizeTerminalRun(m.pendingTerminal)
			}
			if m.running && strings.TrimSpace(m.currentRunID) != "" {
				m.appendTelemetryLine(time.Now().Format("15:04:05") + " | stream | channel closed; reconnecting")
				return m, tea.Batch(
					m.startStream(m.currentRunID, m.lastPolledEventSeq),
					fetchRunStatusCmd(m.service, m.currentRunID),
				)
			}
			return m, nil
		}
		m.errorText = ""
		terminalEvents := map[string]bool{
			"run_complete":          true,
			"run_failed":            true,
			"service_run_complete":  true,
			"service_run_failed":    true,
			"service_run_cancelled": true,
		}
		terminalSeen := false
		lastEvent := service.StreamEvent{}
		formatted := make([]string, 0, len(msg.events))
		for _, event := range msg.events {
			m.events = append(m.events, event)
			if event.Seq > m.lastPolledEventSeq {
				m.lastPolledEventSeq = event.Seq
			}
			formatted = append(formatted, formatStreamEvent(event))
			lastEvent = event
			if terminalEvents[event.Event] {
				terminalSeen = true
			}
		}
		m.appendTelemetryLines(formatted)
		if lastEvent.Event != "" {
			if percent, ok := extractProgress(lastEvent.Payload); ok {
				m.statusText = fmt.Sprintf("Run %s in progress: %.1f%%", shortRunID(m.currentRunID), percent*100)
			} else {
				m.statusText = fmt.Sprintf("Run %s event: %s", shortRunID(m.currentRunID), lastEvent.Event)
			}
		}

		cmds := []tea.Cmd{}
		if m.streamChan != nil {
			cmds = append(cmds, waitForStreamEventCmd(m.streamID, m.streamChan))
		}
		if terminalSeen && strings.TrimSpace(m.currentRunID) != "" {
			cmds = append(cmds, fetchRunStatusCmd(m.service, m.currentRunID))
		}
		if len(cmds) == 0 {
			return m, nil
		}
		return m, tea.Batch(cmds...)

	case streamErrMsg:
		if msg.streamID != m.streamID {
			return m, nil
		}
		if !msg.ok {
			m.streamErrChan = nil
			if m.finalizingRun && m.pendingTerminal != nil && m.streamChan == nil {
				return m, m.finalizeTerminalRun(m.pendingTerminal)
			}
			return m, nil
		}
		m.streamErrChan = nil
		if msg.err == nil || errors.Is(msg.err, context.Canceled) {
			if m.finalizingRun && m.pendingTerminal != nil && m.streamChan == nil {
				return m, m.finalizeTerminalRun(m.pendingTerminal)
			}
			return m, nil
		}
		m.errorText = "Telemetry stream error: " + msg.err.Error()
		m.appendTelemetryLine(time.Now().Format("15:04:05") + " | stream_error | " + msg.err.Error())
		if (m.running || m.finalizingRun) && strings.TrimSpace(m.currentRunID) != "" {
			m.appendTelemetryLine(time.Now().Format("15:04:05") + " | local | stream reconnecting")
			return m, tea.Batch(
				m.startStream(m.currentRunID, m.lastPolledEventSeq),
				fetchRunStatusCmd(m.service, m.currentRunID),
			)
		}
		return m, nil

	case runStatusMsg:
		if msg.err != nil {
			m.errorText = "Status poll failed: " + msg.err.Error()
			return m, nil
		}
		if msg.status == nil {
			m.errorText = "Status poll failed: empty response"
			return m, nil
		}
		currentRunID := strings.TrimSpace(m.currentRunID)
		statusRunID := strings.TrimSpace(msg.status.RunID)
		if currentRunID != "" && statusRunID != "" && statusRunID != currentRunID {
			return m, nil
		}
		m.latestStatus = msg.status
		m.results.SetContent(renderRunStatus(msg.status))

		if msg.status.LatestSeq > m.lastPolledEventSeq && strings.TrimSpace(msg.status.LatestEvent) != "" {
			m.lastPolledEventSeq = msg.status.LatestSeq
			line := fmt.Sprintf("%s | %s", trimTime(msg.status.LatestEventTimestamp), msg.status.LatestEvent)
			if pct, ok := extractProgress(msg.status.LatestEventPayload); ok {
				line += fmt.Sprintf(" | %.1f%%", pct*100.0)
			}
			m.appendTelemetryLine(line)
		}

		if isTerminalRunStatus(msg.status.Status) {
			m.pendingTerminal = msg.status
			if m.streamChan != nil {
				m.running = false
				m.finalizingRun = true
				if m.finalizingSince.IsZero() {
					m.finalizingSince = time.Now()
				}
				m.statusText = fmt.Sprintf("Run %s %s; draining telemetry...", shortRunID(msg.status.RunID), msg.status.Status)
				return m, nil
			}
			return m, m.finalizeTerminalRun(msg.status)
		}
		return m, nil

	case bundleSavedMsg:
		if msg.err != nil {
			m.errorText = "Could not save run bundle: " + msg.err.Error()
			return m, nil
		}
		m.statusText = fmt.Sprintf("Saved run bundle: %s", filepathBase(msg.summary.Directory))
		return m, loadHistoryCmd(m.store)

	case bundleLoadedMsg:
		if msg.err != nil {
			m.errorText = "Could not load bundle: " + msg.err.Error()
			return m, nil
		}
		m.results.SetContent(renderBundle(msg.bundle))
		m.statusText = fmt.Sprintf("Loaded bundle %s", filepathBase(msg.bundle.Summary.Directory))
		return m, nil

	case tea.KeyMsg:
		if m.showConfigPathPrompt {
			switch msg.String() {
			case "ctrl+c", "q":
				if m.streamCancel != nil {
					m.streamCancel()
				}
				return m, tea.Quit
			case "up":
				m.setConfigPathChoiceCursor(m.configPathChoiceCursor - 1)
				return m, nil
			case "down":
				m.setConfigPathChoiceCursor(m.configPathChoiceCursor + 1)
				return m, nil
			case "pgup":
				m.setConfigPathChoiceCursor(m.configPathChoiceCursor - m.configPathListVisibleRows())
				return m, nil
			case "pgdown":
				m.setConfigPathChoiceCursor(m.configPathChoiceCursor + m.configPathListVisibleRows())
				return m, nil
			case "esc":
				m.showConfigPathPrompt = false
				m.configPathInput.Blur()
				m.applyFocusState()
				m.statusText = "Config file load cancelled."
				return m, tea.ClearScreen
			case "enter":
				path := strings.TrimSpace(m.configPathInput.Value())
				if path == "" && len(m.configPathChoices) > 0 {
					path = m.configPathChoices[m.configPathChoiceCursor]
				}
				m.showConfigPathPrompt = false
				m.configPathInput.Blur()
				m.applyFocusState()
				if path == "" {
					m.errorText = "Config file path is required."
					return m, tea.ClearScreen
				}
				m.lastConfigPath = path
				m.errorText = ""
				m.statusText = "Loading config file..."
				return m, loadConfigFileCmd(path)
			}
			var cmd tea.Cmd
			before := m.configPathInput.Value()
			m.configPathInput, cmd = m.configPathInput.Update(msg)
			if m.configPathInput.Value() != before {
				m.syncConfigPathChoiceToInput()
			}
			return m, cmd
		}

		switch msg.String() {
		case "ctrl+c", "q":
			if m.streamCancel != nil {
				m.streamCancel()
			}
			return m, tea.Quit
		case "tab":
			m.focusPane = nextFocusPane(m.focusPane)
			m.applyFocusState()
			m.statusText = "Focus: " + focusPaneLabel(m.focusPane)
			return m, nil
		case "shift+tab", "backtab":
			m.focusPane = prevFocusPane(m.focusPane)
			m.applyFocusState()
			m.statusText = "Focus: " + focusPaneLabel(m.focusPane)
			return m, nil
		case "?":
			m.showHelp = !m.showHelp
			return m, nil
		case "ctrl+o":
			m.showConfigPathPrompt = true
			m.configPathInput.SetValue(m.lastConfigPath)
			m.configPathInput.CursorEnd()
			m.configPathInput.Focus()
			if err := m.refreshConfigPathChoices(); err != nil {
				m.errorText = "Could not list .json files in current directory: " + err.Error()
			} else {
				m.errorText = ""
			}
			m.statusText = "Choose a JSON file with up/down or type a path, then press Enter."
			m.applyFocusState()
			return m, nil
		case "ctrl+r":
			if m.running || m.finalizingRun {
				if recoverCmd, recovered := m.recoverTerminalStateForRestart(time.Now()); recovered {
					if m.running || m.finalizingRun {
						return m, recoverCmd
					}
					cfg, err := parseConfigJSON(m.configEditor.Value())
					if err != nil {
						m.errorText = "Config parse error: " + err.Error()
						return m, recoverCmd
					}
					m.errorText = ""
					return m, tea.Batch(recoverCmd, startRunCmd(m.service, cfg))
				}
				m.errorText = "A run is active or finalizing. Wait for completion before starting another."
				if strings.TrimSpace(m.currentRunID) != "" {
					return m, fetchRunStatusCmd(m.service, m.currentRunID)
				}
				return m, nil
			}
			cfg, err := parseConfigJSON(m.configEditor.Value())
			if err != nil {
				m.errorText = "Config parse error: " + err.Error()
				return m, nil
			}
			m.errorText = ""
			return m, startRunCmd(m.service, cfg)
		case "ctrl+x":
			if !m.running || strings.TrimSpace(m.currentRunID) == "" {
				m.errorText = "No active run to cancel."
				return m, nil
			}
			return m, cancelRunCmd(m.service, m.currentRunID)
		case "enter":
			if m.focusPane == paneHistory && len(m.historyItems) > 0 {
				sel := clampInt(m.historyCursor, 0, len(m.historyItems)-1)
				item := m.historyItems[sel]
				return m, loadBundleCmd(m.store, item.Directory)
			}
		case "up", "k":
			if m.focusPane == paneHistory && len(m.historyItems) > 0 {
				m.historyCursor = clampInt(m.historyCursor-1, 0, len(m.historyItems)-1)
				m.refreshHistoryView()
				return m, nil
			}
		case "down", "j":
			if m.focusPane == paneHistory && len(m.historyItems) > 0 {
				m.historyCursor = clampInt(m.historyCursor+1, 0, len(m.historyItems)-1)
				m.refreshHistoryView()
				return m, nil
			}
		case "ctrl+l":
			m.telemetryEntries = nil
			m.telemetryLines = nil
			m.telemetry.SetContent("")
			m.telemetryAutoFollow = true
			m.events = nil
			m.statusText = "Telemetry cleared"
			return m, nil
		}

		switch m.focusPane {
		case paneConfig:
			var cmd tea.Cmd
			before := m.configEditor.Value()
			m.configEditor, cmd = m.configEditor.Update(msg)
			if m.configEditor.Value() != before {
				m.configPinned = true
			}
			return m, cmd
		case paneTelemetry:
			var cmd tea.Cmd
			m.telemetry, cmd = m.telemetry.Update(msg)
			m.telemetryAutoFollow = m.telemetry.AtBottom()
			return m, cmd
		case paneResults:
			var cmd tea.Cmd
			m.results, cmd = m.results.Update(msg)
			return m, cmd
		case paneHistory:
			var cmd tea.Cmd
			m.history, cmd = m.history.Update(msg)
			return m, cmd
		}

	case tea.MouseMsg:
		switch m.focusPane {
		case paneTelemetry:
			var cmd tea.Cmd
			m.telemetry, cmd = m.telemetry.Update(msg)
			m.telemetryAutoFollow = m.telemetry.AtBottom()
			return m, cmd
		case paneResults:
			var cmd tea.Cmd
			m.results, cmd = m.results.Update(msg)
			return m, cmd
		case paneHistory:
			var cmd tea.Cmd
			m.history, cmd = m.history.Update(msg)
			return m, cmd
		}
	}

	return m, nil
}

func (m Model) View() string {
	if !m.ready {
		return "Booting blackswan-tui..."
	}

	innerWidth := maxInt(40, m.width-2)
	innerHeight := maxInt(12, m.height-2)

	header := headerStyle.Render("Blackswan Command Deck")

	statusPrefix := "*"
	if m.running {
		statusPrefix = m.spinner.View()
	}
	statusBody := strings.TrimSpace(m.statusText)
	if statusBody == "" {
		statusBody = "Ready"
	}
	statusLine := statusStyle.Render(statusPrefix + " " + statusBody)
	if strings.TrimSpace(m.errorText) != "" {
		statusLine = errorStyle.Render(m.errorText)
	}

	configPanel := renderPanel(
		"Config JSON",
		highlightJSONKeys(m.configEditor.View()),
		m.configPanelW,
		m.configPanelH,
		m.focusPane == paneConfig,
	)
	telemetryPanel := renderPanel(
		"Live Telemetry",
		m.telemetry.View(),
		m.telemetryW,
		m.telemetryH,
		m.focusPane == paneTelemetry,
	)
	rightColumn := telemetryPanel
	if m.resourceH >= 5 {
		rightColumn = lipgloss.JoinVertical(
			lipgloss.Left,
			renderPanel(
				"System Monitor",
				m.renderSystemMonitor(),
				m.resourceW,
				m.resourceH,
				false,
			),
			telemetryPanel,
		)
	}

	topRow := lipgloss.JoinHorizontal(lipgloss.Top, configPanel, rightColumn)

	bottomRow := lipgloss.JoinHorizontal(lipgloss.Top,
		renderPanel(
			"Results Explorer",
			m.results.View(),
			m.resultsW,
			m.resultsH,
			m.focusPane == paneResults,
		),
		renderPanel(
			"Run History",
			m.history.View(),
			m.historyW,
			m.historyH,
			m.focusPane == paneHistory,
		),
	)

	parts := []string{header, statusLine}
	if m.showConfigPathPrompt {
		promptWidth := clampInt(innerWidth-4, 42, 90)
		listRows := m.configPathListVisibleRows()
		promptHeight := clampInt(9+listRows, 12, maxInt(12, innerHeight-2))
		promptBody := strings.Join([]string{
			"Path to local JSON config file:",
			m.configPathInput.View(),
			"",
			".json files in current directory:",
			m.renderConfigPathChoices(listRows),
			"",
			"up/down select | enter load | esc cancel",
		}, "\n")
		parts = append(parts, renderPanel("Load Config File", promptBody, promptWidth, promptHeight, true))
	}
	parts = append(parts, topRow, bottomRow)
	if m.showHelp {
		parts = append(parts, helpStyle.Render("ctrl+r run | ctrl+o load file | ctrl+x cancel | tab/shift+tab cycle panes | up/down or wheel scroll focused pane | enter load history item | ctrl+l clear telemetry | q quit"))
	}

	body := strings.Join(parts, "\n")
	body = fitTextHeight(body, innerHeight)
	return lipgloss.NewStyle().
		Background(chromeBG).
		Foreground(lipgloss.Color("#E8F0F2")).
		Width(innerWidth).
		Height(innerHeight).
		Padding(0, 1).
		Render(body)
}

func (m *Model) applyFocusState() {
	if m.showConfigPathPrompt {
		m.configEditor.Blur()
		m.configPathInput.Focus()
		return
	}
	m.configPathInput.Blur()
	if m.focusPane == paneConfig {
		m.configEditor.Focus()
		return
	}
	m.configEditor.Blur()
}

func nextFocusPane(current focusPane) focusPane {
	switch current {
	case paneConfig:
		return paneTelemetry
	case paneTelemetry:
		return paneResults
	case paneResults:
		return paneHistory
	default:
		return paneConfig
	}
}

func prevFocusPane(current focusPane) focusPane {
	switch current {
	case paneConfig:
		return paneHistory
	case paneTelemetry:
		return paneConfig
	case paneResults:
		return paneTelemetry
	default:
		return paneResults
	}
}

func focusPaneLabel(pane focusPane) string {
	switch pane {
	case paneConfig:
		return "config"
	case paneTelemetry:
		return "telemetry"
	case paneResults:
		return "results"
	case paneHistory:
		return "history"
	default:
		return "unknown"
	}
}

func renderPanel(title, body string, width, height int, focused bool) string {
	borderColor := panelBorder
	if focused {
		borderColor = accentSecondary
	}
	style := panelStyle.Copy().
		BorderForeground(borderColor).
		Width(width).
		Height(height)

	titleLine := panelTitleStyle.Render(title)
	return style.Render(titleLine + "\n" + body)
}

func (m *Model) resizePanels() {
	if m.width <= 0 || m.height <= 0 {
		return
	}

	usableW := maxInt(40, m.width-6)
	usableH := maxInt(14, m.height-10)

	topH := int(math.Round(float64(usableH) * 0.58))
	topH = clampInt(topH, 8, usableH-5)
	bottomH := usableH - topH
	if bottomH < 5 {
		bottomH = 5
	}

	leftW := int(math.Round(float64(usableW) * 0.54))
	leftW = clampInt(leftW, 30, usableW-20)
	rightW := usableW - leftW

	resultsW := int(math.Round(float64(usableW) * 0.68))
	resultsW = clampInt(resultsW, 28, usableW-16)
	historyW := usableW - resultsW

	configInnerW := maxInt(20, leftW-6)
	configInnerH := maxInt(5, topH-4)
	configEditorH := maxInt(3, configInnerH-1)
	m.configEditor.SetWidth(configInnerW)
	m.configEditor.SetHeight(configEditorH)
	m.configPanelW = configInnerW + 4
	m.configPanelH = configEditorH + 3

	telemetryInnerW := maxInt(18, rightW-6)
	rightTotalH := maxInt(6, m.configPanelH)
	resourcePanelH := 0
	telemetryPanelH := rightTotalH
	if rightTotalH >= 12 {
		resourcePanelH = clampInt(int(math.Round(float64(rightTotalH)*0.38)), 5, rightTotalH-5)
		telemetryPanelH = rightTotalH - resourcePanelH
	}

	resourceViewH := 0
	if resourcePanelH > 0 {
		resourceViewH = maxInt(2, resourcePanelH-3)
	}
	telemetryViewH := maxInt(2, telemetryPanelH-3)

	m.resourceW = telemetryInnerW + 4
	if resourcePanelH > 0 {
		m.resourceH = resourceViewH + 3
	} else {
		m.resourceH = 0
	}

	m.telemetry.Width = telemetryInnerW
	m.telemetry.Height = telemetryViewH
	m.telemetryW = telemetryInnerW + 4
	m.telemetryH = telemetryViewH + 3

	resultsInnerW := maxInt(22, resultsW-6)
	resultsInnerH := maxInt(4, bottomH-4)
	resultsViewH := maxInt(3, resultsInnerH-1)
	m.results.Width = resultsInnerW
	m.results.Height = resultsViewH
	m.resultsW = resultsInnerW + 4
	m.resultsH = resultsViewH + 3

	historyInnerW := maxInt(16, historyW-6)
	historyInnerH := maxInt(4, bottomH-4)
	historyViewH := maxInt(3, historyInnerH-1)
	m.history.Width = historyInnerW
	m.history.Height = historyViewH
	m.historyW = historyInnerW + 4
	m.historyH = historyViewH + 3
	m.configPathInput.Width = clampInt(usableW-22, 20, 78)

	if len(m.telemetryEntries) > 0 {
		m.rebuildTelemetryContent(m.telemetryAutoFollow)
	}

	m.refreshHistoryView()
}

func (m *Model) refreshHistoryView() {
	if len(m.historyItems) == 0 {
		m.history.SetContent("No saved runs yet.\nCompleted runs are stored under ./runs")
		m.history.SetYOffset(0)
		m.historyCursorTopLine = 0
		m.historyCursorBottomLine = 0
		m.historyRenderedLines = 0
		return
	}

	if m.historyCursor >= len(m.historyItems) {
		m.historyCursor = len(m.historyItems) - 1
	}
	if m.historyCursor < 0 {
		m.historyCursor = 0
	}

	contentWidth := maxInt(1, m.history.Width)
	lines := make([]string, 0, len(m.historyItems))
	m.historyCursorTopLine = 0
	m.historyCursorBottomLine = 0
	for idx, item := range m.historyItems {
		cursor := " "
		if idx == m.historyCursor {
			cursor = "▶"
		}
		frac := item.RecommendedFraction * 100
		line := fmt.Sprintf("%s %s | %.2f%% | %s", cursor, trimTime(item.SavedAt), frac, safeMode(item.ObjectiveMode))
		wrapped := wrapLineToWidth(line, contentWidth)
		lineTop := len(lines)
		for _, segment := range wrapped {
			if idx == m.historyCursor {
				segment = historySelectedLineStyle.Render(segment)
			}
			lines = append(lines, segment)
		}
		lineBottom := len(lines) - 1
		if idx == m.historyCursor {
			m.historyCursorTopLine = lineTop
			m.historyCursorBottomLine = lineBottom
		}
	}
	m.history.SetContent(strings.Join(lines, "\n"))
	m.historyRenderedLines = len(lines)
	m.ensureHistoryCursorVisible()
}

func (m *Model) ensureHistoryCursorVisible() {
	if m.historyRenderedLines == 0 {
		m.history.SetYOffset(0)
		return
	}
	visibleRows := maxInt(1, m.history.Height)
	cursorTop := clampInt(m.historyCursorTopLine, 0, m.historyRenderedLines-1)
	cursorBottom := clampInt(m.historyCursorBottomLine, cursorTop, m.historyRenderedLines-1)
	top := clampInt(m.history.YOffset, 0, m.historyRenderedLines-1)
	bottom := top + visibleRows - 1
	scrollMargin := clampInt(visibleRows/4, 1, 2)
	upperBound := top + scrollMargin
	lowerBound := bottom - scrollMargin
	if cursorTop < upperBound {
		m.history.SetYOffset(cursorTop - scrollMargin)
		return
	}
	if cursorBottom > lowerBound {
		m.history.SetYOffset(cursorBottom - (visibleRows - 1 - scrollMargin))
		return
	}
	m.history.SetYOffset(top)
}

func wrapLineToWidth(line string, width int) []string {
	width = maxInt(1, width)
	runes := []rune(line)
	if len(runes) == 0 {
		return []string{""}
	}
	if len(runes) <= width {
		return []string{line}
	}
	segments := make([]string, 0, (len(runes)/width)+1)
	start := 0
	for start < len(runes) {
		end := start + width
		if end > len(runes) {
			end = len(runes)
		}
		segments = append(segments, string(runes[start:end]))
		start = end
	}
	return segments
}

func parseConfigJSON(raw string) (map[string]any, error) {
	clean := strings.TrimSpace(raw)
	if clean == "" {
		return map[string]any{}, nil
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(clean), &parsed); err != nil {
		return nil, err
	}
	return parsed, nil
}

func cloneMap(src map[string]any) map[string]any {
	if src == nil {
		return map[string]any{}
	}
	blob, err := json.Marshal(src)
	if err != nil {
		copyMap := make(map[string]any, len(src))
		for key, value := range src {
			copyMap[key] = value
		}
		return copyMap
	}
	var out map[string]any
	if err := json.Unmarshal(blob, &out); err != nil {
		copyMap := make(map[string]any, len(src))
		for key, value := range src {
			copyMap[key] = value
		}
		return copyMap
	}
	return out
}

func renderRunStatus(status *service.RunStatus) string {
	if status == nil {
		return "Waiting for run status..."
	}

	lines := []string{
		fmt.Sprintf("Run ID: %s", status.RunID),
		fmt.Sprintf("Status: %s", status.Status),
	}
	if status.StartedAt != "" {
		lines = append(lines, fmt.Sprintf("Started: %s", trimTime(status.StartedAt)))
	}
	if status.FinishedAt != "" {
		lines = append(lines, fmt.Sprintf("Finished: %s", trimTime(status.FinishedAt)))
	}
	if status.Error != "" {
		lines = append(lines, "", "Error:", status.Error)
	}

	if status.Result != nil {
		lines = append(lines, "", "Result snapshot:")
		if fraction, ok := asFloat(status.Result["recommended_fraction"]); ok {
			lines = append(lines, fmt.Sprintf("  Recommended fraction: %.2f%%", fraction*100.0))
		}
		if wealth, ok := asFloat(status.Result["expected_final_net_worth"]); ok {
			lines = append(lines, fmt.Sprintf("  Expected final net worth: $%.0f", wealth))
		}
		if cvar, ok := asFloat(status.Result["cvar_shortfall"]); ok {
			lines = append(lines, fmt.Sprintf("  CVaR shortfall: $%.0f", cvar))
		}
		if objective, ok := status.Result["primary_objective_mode"].(string); ok {
			lines = append(lines, fmt.Sprintf("  Objective mode: %s", objective))
		}
		lines = append(lines, "", "Result fields (condensed):")
		lines = append(lines, summarizeResultFields(status.Result, 14)...)
	}
	return strings.Join(lines, "\n")
}

func renderBundle(bundle *storage.RunBundle) string {
	if bundle == nil {
		return "No bundle selected"
	}
	lines := []string{
		fmt.Sprintf("Run ID: %s", bundle.Summary.RunID),
		fmt.Sprintf("Saved at: %s", trimTime(bundle.Summary.SavedAt)),
		fmt.Sprintf("Objective mode: %s", safeMode(bundle.Summary.ObjectiveMode)),
		fmt.Sprintf("Recommended fraction: %.2f%%", bundle.Summary.RecommendedFraction*100.0),
		fmt.Sprintf("Expected final net worth: $%.0f", bundle.Summary.ExpectedFinalNetWorth),
		"",
		fmt.Sprintf("Events captured: %d", len(bundle.Events)),
	}
	lines = append(lines, "", "Result fields (condensed):")
	lines = append(lines, summarizeResultFields(bundle.Result, 14)...)
	return strings.Join(lines, "\n")
}

func (m Model) renderSystemMonitor() string {
	innerW := maxInt(18, m.resourceW-6)
	meterW := clampInt(innerW-16, 8, 24)
	trendW := maxInt(10, innerW-6)
	maxBodyLines := maxInt(2, m.resourceH-3)

	cpuLine := fmt.Sprintf("CPU  %5.1f%% %s", m.cpuUsagePct, renderUsageMeter(m.cpuUsagePct, meterW))
	memLine := "MEM  n/a"
	if m.memAvailable {
		memLine = fmt.Sprintf("MEM  %5.1f%% %s", m.memUsagePct, renderUsageMeter(m.memUsagePct, meterW))
	}
	gpuLine := "GPU  n/a"
	if m.gpuAvailable {
		gpuLine = fmt.Sprintf("GPU  %5.1f%% %s", m.gpuUsagePct, renderUsageMeter(m.gpuUsagePct, meterW))
	} else if strings.TrimSpace(m.gpuNote) != "" {
		gpuLine = truncateText("GPU  n/a ("+m.gpuNote+")", innerW)
	}

	summary := []string{cpuLine, memLine, gpuLine}
	if maxBodyLines <= len(summary) {
		return strings.Join(summary[:maxBodyLines], "\n")
	}

	lines := append([]string{}, summary...)
	lines = append(lines, "cpu~ "+renderUsageTrend(m.cpuHistory, trendW))
	if m.gpuAvailable {
		lines = append(lines, "gpu~ "+renderUsageTrend(m.gpuHistory, trendW))
	} else if strings.TrimSpace(m.gpuNote) != "" {
		lines = append(lines, truncateText("gpu~ unavailable: "+m.gpuNote, innerW))
	}
	if m.memAvailable {
		lines = append(lines, "mem~ "+renderUsageTrend(m.memHistory, trendW))
	}
	if m.gpuAvailable {
		lines = append(lines, fmt.Sprintf("VRAM %5.1f%% %s", m.gpuMemUsagePct, renderUsageMeter(m.gpuMemUsagePct, meterW)))
	}
	if strings.TrimSpace(m.gpuLabel) != "" {
		lines = append(lines, truncateText("GPU: "+m.gpuLabel, innerW))
	}
	if len(lines) > maxBodyLines {
		lines = lines[:maxBodyLines]
	}
	return strings.Join(lines, "\n")
}

func renderUsageMeter(percent float64, width int) string {
	width = maxInt(4, width)
	p := clampFloat(percent, 0, 100)
	filled := int(math.Round((p / 100.0) * float64(width)))
	filled = clampInt(filled, 0, width)
	return "[" + strings.Repeat("#", filled) + strings.Repeat("-", width-filled) + "]"
}

func renderUsageTrend(samples []float64, width int) string {
	width = maxInt(4, width)
	if len(samples) == 0 {
		return strings.Repeat(".", width)
	}

	levels := []rune("._-:=+*#%@")
	out := make([]rune, width)
	for idx := range out {
		out[idx] = '.'
	}

	window := samples
	if len(window) > width {
		window = window[len(window)-width:]
	}
	start := width - len(window)
	for idx := 0; idx < len(window); idx++ {
		source := window[idx]
		p := clampFloat(source, 0, 100)
		levelIdx := int(math.Round((p / 100.0) * float64(len(levels)-1)))
		levelIdx = clampInt(levelIdx, 0, len(levels)-1)
		out[start+idx] = levels[levelIdx]
	}
	return string(out)
}

func appendUsageSample(history []float64, value float64, maxLen int) []float64 {
	if maxLen <= 0 {
		maxLen = 120
	}
	history = append(history, clampFloat(value, 0, 100))
	if len(history) > maxLen {
		history = history[len(history)-maxLen:]
	}
	return history
}

func appendUsageSampleN(history []float64, value float64, maxLen int, count int) []float64 {
	count = clampInt(count, 1, maxMetricsCatchupSamples)
	for idx := 0; idx < count; idx++ {
		history = appendUsageSample(history, value, maxLen)
	}
	return history
}

func estimateMetricsCatchupSamples(prev, current time.Time, interval time.Duration, maxSamples int) int {
	if prev.IsZero() || current.IsZero() || interval <= 0 {
		return 1
	}
	if !current.After(prev) {
		return 1
	}
	gap := current.Sub(prev)
	if gap <= interval {
		return 1
	}
	samples := int(gap / interval)
	samples = clampInt(samples, 1, maxSamples)
	return samples
}

func highlightJSONKeys(text string) string {
	if strings.IndexByte(text, '"') == -1 {
		return text
	}
	matches := jsonKeyPattern.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return text
	}

	var builder strings.Builder
	builder.Grow(len(text) + len(matches)*8)
	cursor := 0

	for _, bounds := range matches {
		start := bounds[0]
		end := bounds[1]
		if start < cursor {
			continue
		}
		builder.WriteString(text[cursor:start])
		match := text[start:end]
		colonIdx := strings.LastIndex(match, ":")
		if colonIdx <= 0 {
			builder.WriteString(match)
			cursor = end
			continue
		}
		keyPortion := match[:colonIdx]
		suffix := match[colonIdx:]
		keyStyle := jsonKeyStyle
		if jsonKeyHasObjectValue(text, end) {
			keyStyle = jsonObjectKeyStyle
		}
		builder.WriteString(keyStyle.Render(keyPortion))
		builder.WriteString(suffix)
		cursor = end
	}
	builder.WriteString(text[cursor:])
	return builder.String()
}

func jsonKeyHasObjectValue(text string, valueSearchStart int) bool {
	valueStart := nextNonWhitespaceIndex(text, valueSearchStart)
	return valueStart < len(text) && text[valueStart] == '{'
}

func nextNonWhitespaceIndex(text string, start int) int {
	idx := start
	for idx < len(text) {
		switch text[idx] {
		case ' ', '\t', '\n', '\r':
			idx++
		default:
			return idx
		}
	}
	return idx
}

func fitTextHeight(text string, height int) string {
	if height <= 0 {
		return ""
	}
	lines := strings.Split(text, "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func (m *Model) appendTelemetryLine(line string) {
	m.appendTelemetryLines([]string{line})
}

func (m *Model) appendTelemetryLines(lines []string) {
	if len(lines) == 0 {
		return
	}
	entries := make([]string, 0, len(lines))
	for _, raw := range lines {
		entry := normalizeTelemetryEntry(raw)
		if entry == "" {
			continue
		}
		entries = append(entries, entry)
	}
	if len(entries) == 0 {
		return
	}
	shouldFollow := m.focusPane != paneTelemetry || m.telemetryAutoFollow || m.telemetry.AtBottom()

	m.telemetryEntries = append(m.telemetryEntries, entries...)
	if len(m.telemetryEntries) > 1200 {
		m.telemetryEntries = m.telemetryEntries[len(m.telemetryEntries)-1200:]
	}
	m.rebuildTelemetryContent(shouldFollow)
}

func (m *Model) rebuildTelemetryContent(shouldFollow bool) {
	width := maxInt(8, m.telemetry.Width)
	rendered := make([]string, 0, len(m.telemetryEntries))
	for _, entry := range m.telemetryEntries {
		rendered = append(rendered, wrapTelemetryEntry(entry, width)...)
	}
	if len(rendered) > 4000 {
		rendered = rendered[len(rendered)-4000:]
	}
	m.telemetryLines = rendered
	m.telemetry.SetContent(strings.Join(rendered, "\n"))
	if shouldFollow {
		m.telemetry.GotoBottom()
		m.telemetryAutoFollow = true
	}
}

func normalizeTelemetryEntry(raw string) string {
	line := strings.ReplaceAll(raw, "\n", " ")
	line = strings.TrimSpace(line)
	if line == "" {
		return ""
	}
	return line
}

func wrapTelemetryEntry(entry string, width int) []string {
	entry = normalizeTelemetryEntry(entry)
	if entry == "" {
		return nil
	}
	width = maxInt(8, width)
	if len([]rune(entry)) <= width {
		return []string{entry}
	}

	lines := make([]string, 0, 4)
	remaining := entry
	continuationPrefix := "  "
	first := true

	for {
		limit := width
		if !first {
			limit = maxInt(4, width-len([]rune(continuationPrefix)))
		}
		head, tail := splitWrappedSegment(remaining, limit)
		if !first {
			head = continuationPrefix + head
		}
		lines = append(lines, head)
		if tail == "" {
			break
		}
		remaining = tail
		first = false
	}
	return lines
}

func splitWrappedSegment(raw string, width int) (string, string) {
	width = maxInt(1, width)
	runes := []rune(strings.TrimSpace(raw))
	if len(runes) <= width {
		return strings.TrimSpace(string(runes)), ""
	}

	cut := width
	for idx := width; idx > 0; idx-- {
		ch := runes[idx-1]
		if ch == ' ' || ch == '|' || ch == ',' || ch == ';' {
			cut = idx
			break
		}
	}

	head := strings.TrimSpace(string(runes[:cut]))
	tail := strings.TrimSpace(string(runes[cut:]))
	if head == "" {
		head = strings.TrimSpace(string(runes[:width]))
		tail = strings.TrimSpace(string(runes[width:]))
	}
	return head, tail
}

func summarizeResultFields(result map[string]any, maxFields int) []string {
	if len(result) == 0 {
		return []string{"  (empty)"}
	}
	keys := make([]string, 0, len(result))
	for key := range result {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	if maxFields <= 0 {
		maxFields = len(keys)
	}
	limit := minInt(maxFields, len(keys))

	lines := make([]string, 0, limit+1)
	for idx := 0; idx < limit; idx++ {
		key := keys[idx]
		lines = append(lines, fmt.Sprintf("  %s: %s", key, previewResultValue(result[key])))
	}
	if len(keys) > limit {
		lines = append(lines, fmt.Sprintf("  ... %d more fields", len(keys)-limit))
	}
	return lines
}

func previewResultValue(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case string:
		return strconv.Quote(truncateText(strings.TrimSpace(v), 72))
	case bool:
		if v {
			return "true"
		}
		return "false"
	case float64, float32, int, int64, int32, uint, uint64, uint32, json.Number:
		return fmt.Sprintf("%v", v)
	case map[string]any:
		return fmt.Sprintf("{%d keys}", len(v))
	case []any:
		return previewResultSlice(v)
	default:
		blob, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("<%T>", value)
		}
		return truncateText(string(blob), 72)
	}
}

func previewResultSlice(items []any) string {
	if len(items) == 0 {
		return "[0 items]"
	}
	sample := make([]string, 0, minInt(len(items), 2))
	for idx := 0; idx < len(items) && idx < 2; idx++ {
		sample = append(sample, previewInlineResultValue(items[idx]))
	}
	body := strings.Join(sample, ", ")
	if len(items) > 2 {
		body += ", ..."
	}
	return fmt.Sprintf("[%d items: %s]", len(items), body)
}

func previewInlineResultValue(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case string:
		return strconv.Quote(truncateText(strings.TrimSpace(v), 24))
	case map[string]any:
		return fmt.Sprintf("{%d keys}", len(v))
	case []any:
		return fmt.Sprintf("[%d items]", len(v))
	default:
		return truncateText(fmt.Sprintf("%v", v), 24)
	}
}

func truncateText(raw string, maxLen int) string {
	if maxLen < 4 {
		maxLen = 4
	}
	if len(raw) <= maxLen {
		return raw
	}
	return raw[:maxLen-3] + "..."
}

func formatStreamEvent(event service.StreamEvent) string {
	timeStamp := trimTime(event.Timestamp)
	if timeStamp == "" {
		timeStamp = time.Now().Format("15:04:05")
	}
	parts := []string{timeStamp, event.Event}
	if pct, ok := extractProgress(event.Payload); ok {
		parts = append(parts, fmt.Sprintf("%.1f%%", pct*100.0))
	}
	if phase, ok := event.Payload["cvar_mode"].(string); ok && phase != "" {
		parts = append(parts, phase)
	}
	return strings.Join(parts, " | ")
}

func extractProgress(payload map[string]any) (float64, bool) {
	if payload == nil {
		return 0, false
	}
	if value, ok := asFloat(payload["percent_complete"]); ok {
		return value, true
	}
	if completed, ok := asFloat(payload["completed"]); ok {
		if total, ok := asFloat(payload["total"]); ok && total > 0 {
			return completed / total, true
		}
	}
	return 0, false
}

func asFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint64:
		return float64(v), true
	case uint32:
		return float64(v), true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		return f, err == nil
	default:
		return 0, false
	}
}

func shortRunID(runID string) string {
	id := strings.TrimSpace(runID)
	if len(id) <= 8 {
		return id
	}
	return id[:8]
}

func trimTime(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	parsed, err := time.Parse(time.RFC3339, raw)
	if err == nil {
		return parsed.Local().Format("2006-01-02 15:04:05")
	}
	return raw
}

func statusTimestamp(status *service.RunStatus) string {
	if status == nil {
		return time.Now().Format("15:04:05")
	}
	if ts := trimTime(status.UpdatedAt); strings.TrimSpace(ts) != "" {
		return ts
	}
	if ts := trimTime(status.LatestEventTimestamp); strings.TrimSpace(ts) != "" {
		return ts
	}
	return time.Now().Format("15:04:05")
}

func isTerminalRunStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

func safeMode(raw string) string {
	if strings.TrimSpace(raw) == "" {
		return "unknown"
	}
	return raw
}

func filepathBase(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	parts := strings.Split(strings.ReplaceAll(path, "\\", "/"), "/")
	return parts[len(parts)-1]
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clampFloat(v, low, high float64) float64 {
	if v < low {
		return low
	}
	if v > high {
		return high
	}
	return v
}

func clampInt(v, low, high int) int {
	if v < low {
		return low
	}
	if v > high {
		return high
	}
	return v
}
