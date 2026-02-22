package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"blackswan-tui/internal/service"
	"blackswan-tui/internal/storage"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
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
)

type defaultsLoadedMsg struct {
	defaults map[string]any
	err      error
}

type historyLoadedMsg struct {
	items []storage.RunSummary
	err   error
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
	snapshot systemMetricsSnapshot
	err      error
}

type focusPane int

const (
	paneConfig focusPane = iota
	paneTelemetry
	paneResults
	paneHistory
)

type Model struct {
	service *service.Manager
	store   *storage.Store

	ready bool
	width int
	height int

	configEditor textarea.Model
	telemetry    viewport.Model
	results      viewport.Model
	history      viewport.Model
	spinner      spinner.Model

	focusPane   focusPane
	showHelp    bool

	statusText string
	errorText  string

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

	historyItems  []storage.RunSummary
	historyCursor int

	telemetryEntries []string
	telemetryLines   []string
	telemetryAutoFollow bool

	cpuUsagePct    float64
	memUsagePct    float64
	memAvailable   bool
	gpuUsagePct    float64
	gpuMemUsagePct float64
	gpuAvailable   bool
	gpuLabel       string
	gpuNote        string
	gpuProbeEnabled bool
	cpuLastTotal   uint64
	cpuLastIdle    uint64
	cpuHasBaseline bool
	cpuHistory     []float64
	memHistory     []float64
	gpuHistory     []float64

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

	return Model{
		service:      svc,
		store:        store,
		configEditor: cfgEditor,
		telemetry:    telemetry,
		results:      results,
		history:      history,
		spinner:      spin,
		focusPane:    paneConfig,
		showHelp:     true,
		statusText:   "Service connected. Loading defaults...",
		gpuProbeEnabled: true,
		gpuNote:      "probing...",
		telemetryAutoFollow: true,
		configPanelW: 74,
		configPanelH: 22,
		resourceW:    54,
		resourceH:    8,
		telemetryW:   54,
		telemetryH:   22,
		resultsW:     54,
		resultsH:     16,
		historyW:     44,
		historyH:     16,
	}
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
		snapshot, err := collectSystemMetrics(gpuProbeEnabled)
		return systemMetricsMsg{snapshot: snapshot, err: err}
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
	return tea.Tick(1200*time.Millisecond, func(at time.Time) tea.Msg {
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
		return m, nil

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

		if m.cpuHasBaseline {
			deltaTotal := msg.snapshot.cpuTotalTicks - m.cpuLastTotal
			deltaIdle := msg.snapshot.cpuIdleTicks - m.cpuLastIdle
			if deltaTotal > 0 && deltaTotal >= deltaIdle {
				busyPct := (float64(deltaTotal-deltaIdle) / float64(deltaTotal)) * 100.0
				m.cpuUsagePct = clampFloat(busyPct, 0, 100)
				m.cpuHistory = appendUsageSample(m.cpuHistory, m.cpuUsagePct, 160)
			}
		} else {
			m.cpuHasBaseline = true
		}
		m.cpuLastTotal = msg.snapshot.cpuTotalTicks
		m.cpuLastIdle = msg.snapshot.cpuIdleTicks

		if msg.snapshot.memAvailable {
			m.memAvailable = true
			m.memUsagePct = clampFloat(msg.snapshot.memUsedPct, 0, 100)
			m.memHistory = appendUsageSample(m.memHistory, m.memUsagePct, 160)
		}

		m.gpuProbeEnabled = msg.snapshot.gpuProbeOK
		m.gpuAvailable = msg.snapshot.gpuAvailable
		m.gpuLabel = strings.TrimSpace(msg.snapshot.gpuLabel)
		m.gpuNote = strings.TrimSpace(msg.snapshot.gpuNote)
		if m.gpuAvailable {
			m.gpuUsagePct = clampFloat(msg.snapshot.gpuUtilPct, 0, 100)
			m.gpuMemUsagePct = clampFloat(msg.snapshot.gpuMemUsedPct, 0, 100)
			m.gpuHistory = appendUsageSample(m.gpuHistory, m.gpuUsagePct, 160)
		}
		return m, nil

	case defaultsLoadedMsg:
		if msg.err != nil {
			m.errorText = "Failed to load defaults: " + msg.err.Error()
			m.statusText = "Defaults unavailable. You can still paste custom JSON config."
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
			m.configEditor, cmd = m.configEditor.Update(msg)
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

	header := headerStyle.Render("BlackSwan Command Deck")
	subtitle := subHeaderStyle.Render("Interactive Monte Carlo cockpit powered by Charm + local Python bridge")

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
		m.configEditor.View(),
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

	parts := []string{header, subtitle, statusLine, topRow, bottomRow}
	if m.showHelp {
		parts = append(parts, helpStyle.Render("ctrl+r run | ctrl+x cancel | tab/shift+tab cycle panes | up/down or wheel scroll focused pane | enter load history item | ctrl+l clear telemetry | q quit"))
	}

	body := strings.Join(parts, "\n")
	return lipgloss.NewStyle().
		Background(chromeBG).
		Foreground(lipgloss.Color("#E8F0F2")).
		Width(innerWidth).
		Height(innerHeight).
		Padding(0, 1).
		Render(body)
}

func (m *Model) applyFocusState() {
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

	if len(m.telemetryEntries) > 0 {
		m.rebuildTelemetryContent(m.telemetryAutoFollow)
	}

	m.refreshHistoryView()
}

func (m *Model) refreshHistoryView() {
	if len(m.historyItems) == 0 {
		m.history.SetContent("No saved runs yet.\nCompleted runs are stored under ./runs")
		return
	}

	if m.historyCursor >= len(m.historyItems) {
		m.historyCursor = len(m.historyItems) - 1
	}
	if m.historyCursor < 0 {
		m.historyCursor = 0
	}

	lines := make([]string, 0, len(m.historyItems))
	for idx, item := range m.historyItems {
		cursor := " "
		if idx == m.historyCursor {
			cursor = "▶"
		}
		frac := item.RecommendedFraction * 100
		line := fmt.Sprintf("%s %s | %.2f%% | %s", cursor, trimTime(item.SavedAt), frac, safeMode(item.ObjectiveMode))
		lines = append(lines, line)
	}
	m.history.SetContent(strings.Join(lines, "\n"))
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
	step := float64(len(samples)) / float64(width)
	if step <= 0 {
		step = 1
	}

	out := make([]rune, width)
	for idx := 0; idx < width; idx++ {
		source := int(math.Floor(float64(idx) * step))
		if source >= len(samples) {
			source = len(samples) - 1
		}
		p := clampFloat(samples[source], 0, 100)
		levelIdx := int(math.Round((p / 100.0) * float64(len(levels)-1)))
		levelIdx = clampInt(levelIdx, 0, len(levels)-1)
		out[idx] = levels[levelIdx]
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
