package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type handshakePayload struct {
	Event string `json:"event"`
	Host  string `json:"host"`
	Port  int    `json:"port"`
	Token string `json:"token"`
	PID   int    `json:"pid"`
}

type runCreateResponse struct {
	RunID  string `json:"run_id"`
	Status string `json:"status"`
}

type cancelResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type defaultsResponse struct {
	Defaults map[string]any `json:"defaults"`
}

type runsListResponse struct {
	Runs []RunStatus `json:"runs"`
}

type apiError struct {
	Error string `json:"error"`
}

type StreamEvent struct {
	Seq       int64          `json:"seq"`
	Timestamp string         `json:"timestamp"`
	Event     string         `json:"event"`
	Payload   map[string]any `json:"payload"`
}

type RunStatus struct {
	RunID                string         `json:"run_id"`
	Status               string         `json:"status"`
	CreatedAt            string         `json:"created_at"`
	UpdatedAt            string         `json:"updated_at"`
	StartedAt            string         `json:"started_at"`
	FinishedAt           string         `json:"finished_at"`
	CancelRequested      bool           `json:"cancel_requested"`
	Error                string         `json:"error"`
	Traceback            string         `json:"traceback"`
	Result               map[string]any `json:"result"`
	LatestSeq            int64          `json:"latest_seq"`
	LatestEvent          string         `json:"latest_event"`
	LatestEventTimestamp string         `json:"latest_event_timestamp"`
	LatestEventPayload   map[string]any `json:"latest_event_payload"`
}

type Manager struct {
	rootDir string

	mu        sync.RWMutex
	started   bool
	cmd       *exec.Cmd
	host      string
	port      int
	token     string
	processID int

	logsMu sync.Mutex
	logs   []string

	httpClient *http.Client
}

func NewManager(rootDir string) *Manager {
	return &Manager{
		rootDir: rootDir,
		httpClient: &http.Client{
			Timeout: 45 * time.Second,
		},
	}
}

func (m *Manager) appendLog(line string) {
	m.logsMu.Lock()
	defer m.logsMu.Unlock()
	m.logs = append(m.logs, line)
	if len(m.logs) > 600 {
		m.logs = m.logs[len(m.logs)-600:]
	}
}

func (m *Manager) Logs() string {
	m.logsMu.Lock()
	defer m.logsMu.Unlock()
	return strings.Join(m.logs, "\n")
}

func (m *Manager) resolvePython() string {
	venvPython := filepath.Join(m.rootDir, ".venv", "bin", "python")
	if info, err := os.Stat(venvPython); err == nil && !info.IsDir() {
		return venvPython
	}
	if custom := strings.TrimSpace(os.Getenv("BLACKSWAN_PYTHON")); custom != "" {
		return custom
	}
	return "python3"
}

func (m *Manager) Start(ctx context.Context) error {
	m.mu.Lock()
	if m.started {
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()

	pythonBin := m.resolvePython()
	scriptPath := filepath.Join(m.rootDir, "python_service", "blackswan_service.py")

	cmd := exec.Command(pythonBin, "-u", scriptPath)
	cmd.Dir = m.rootDir

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("create stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start python service: %w", err)
	}

	handshakeLine := make(chan string, 1)
	stdoutScannerDone := make(chan struct{})
	go func() {
		defer close(stdoutScannerDone)
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
		first := true
		for scanner.Scan() {
			line := scanner.Text()
			if first {
				first = false
				handshakeLine <- line
				continue
			}
			m.appendLog("service stdout: " + line)
		}
		if scanErr := scanner.Err(); scanErr != nil {
			m.appendLog("service stdout scan error: " + scanErr.Error())
		}
	}()

	go func() {
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
		for scanner.Scan() {
			m.appendLog("service stderr: " + scanner.Text())
		}
		if scanErr := scanner.Err(); scanErr != nil {
			m.appendLog("service stderr scan error: " + scanErr.Error())
		}
	}()

	waitHandshake := 12 * time.Second
	var line string
	select {
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return ctx.Err()
	case line = <-handshakeLine:
	case <-time.After(waitHandshake):
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return fmt.Errorf("service handshake timed out after %s", waitHandshake)
	}

	var handshake handshakePayload
	if err := json.Unmarshal([]byte(line), &handshake); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return fmt.Errorf("parse handshake: %w", err)
	}
	if handshake.Host == "" || handshake.Port <= 0 || handshake.Token == "" {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return fmt.Errorf("invalid handshake payload: %s", line)
	}

	m.mu.Lock()
	m.started = true
	m.cmd = cmd
	m.host = handshake.Host
	m.port = handshake.Port
	m.token = handshake.Token
	m.processID = handshake.PID
	m.mu.Unlock()

	go func() {
		if err := cmd.Wait(); err != nil {
			m.appendLog("service process exited with error: " + err.Error())
		} else {
			m.appendLog("service process exited")
		}
		<-stdoutScannerDone
		m.mu.Lock()
		m.started = false
		m.cmd = nil
		m.mu.Unlock()
	}()

	healthCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := m.Health(healthCtx); err != nil {
		_ = m.Stop()
		return fmt.Errorf("service health check failed: %w", err)
	}
	return nil
}

func (m *Manager) endpoint() (string, string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if !m.started || m.host == "" || m.port <= 0 || m.token == "" {
		return "", "", fmt.Errorf("service is not running")
	}
	return fmt.Sprintf("http://%s:%d", m.host, m.port), m.token, nil
}

func (m *Manager) Stop() error {
	m.mu.RLock()
	cmd := m.cmd
	started := m.started
	m.mu.RUnlock()

	if !started || cmd == nil || cmd.Process == nil {
		m.mu.Lock()
		m.started = false
		m.cmd = nil
		m.host = ""
		m.port = 0
		m.token = ""
		m.processID = 0
		m.mu.Unlock()
		return nil
	}

	_ = cmd.Process.Signal(os.Interrupt)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		m.mu.RLock()
		running := m.started
		m.mu.RUnlock()
		if !running {
			return nil
		}
		time.Sleep(75 * time.Millisecond)
	}

	_ = cmd.Process.Kill()
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		m.mu.RLock()
		running := m.started
		m.mu.RUnlock()
		if !running {
			return nil
		}
		time.Sleep(75 * time.Millisecond)
	}

	m.mu.Lock()
	m.started = false
	m.cmd = nil
	m.host = ""
	m.port = 0
	m.token = ""
	m.processID = 0
	m.mu.Unlock()
	return nil
}

func (m *Manager) doJSON(ctx context.Context, method, path string, payload any, out any) error {
	endpoint, token, err := m.endpoint()
	if err != nil {
		return err
	}

	var body io.Reader
	if payload != nil {
		blob, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal request payload: %w", err)
		}
		body = bytes.NewReader(blob)
	}

	req, err := http.NewRequestWithContext(ctx, method, endpoint+path, body)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("X-Blackswan-Token", token)
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		blob, _ := io.ReadAll(resp.Body)
		var apiErr apiError
		if json.Unmarshal(blob, &apiErr) == nil && strings.TrimSpace(apiErr.Error) != "" {
			return fmt.Errorf("api %s %s: %s", method, path, apiErr.Error)
		}
		return fmt.Errorf("api %s %s failed with status %d", method, path, resp.StatusCode)
	}

	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func (m *Manager) Health(ctx context.Context) error {
	endpoint, _, err := m.endpoint()
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("health endpoint returned %d", resp.StatusCode)
	}
	return nil
}

func (m *Manager) Defaults(ctx context.Context) (map[string]any, error) {
	var response defaultsResponse
	if err := m.doJSON(ctx, http.MethodGet, "/defaults", nil, &response); err != nil {
		return nil, err
	}
	if response.Defaults == nil {
		response.Defaults = map[string]any{}
	}
	return response.Defaults, nil
}

func (m *Manager) ListRuns(ctx context.Context) ([]RunStatus, error) {
	var response runsListResponse
	if err := m.doJSON(ctx, http.MethodGet, "/runs", nil, &response); err != nil {
		return nil, err
	}
	return response.Runs, nil
}

func (m *Manager) StartRun(ctx context.Context, config map[string]any) (string, error) {
	if config == nil {
		config = map[string]any{}
	}
	var response runCreateResponse
	if err := m.doJSON(ctx, http.MethodPost, "/runs", map[string]any{"config": config}, &response); err != nil {
		return "", err
	}
	if strings.TrimSpace(response.RunID) == "" {
		return "", fmt.Errorf("service did not return a run id")
	}
	return response.RunID, nil
}

func (m *Manager) CancelRun(ctx context.Context, runID string) error {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return fmt.Errorf("run id is required")
	}
	var response cancelResponse
	path := fmt.Sprintf("/runs/%s/cancel", url.PathEscape(runID))
	if err := m.doJSON(ctx, http.MethodPost, path, map[string]any{}, &response); err != nil {
		return err
	}
	if strings.TrimSpace(response.Status) == "" {
		return fmt.Errorf("unexpected cancel response")
	}
	return nil
}

func (m *Manager) GetRun(ctx context.Context, runID string) (*RunStatus, error) {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return nil, fmt.Errorf("run id is required")
	}
	var status RunStatus
	path := fmt.Sprintf("/runs/%s", url.PathEscape(runID))
	if err := m.doJSON(ctx, http.MethodGet, path, nil, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *Manager) StreamRun(ctx context.Context, runID string, fromSeq int64, sink chan<- StreamEvent) error {
	defer close(sink)

	endpoint, token, err := m.endpoint()
	if err != nil {
		return err
	}
	query := url.Values{}
	query.Set("from", strconv.FormatInt(fromSeq, 10))
	streamURL := fmt.Sprintf("%s/runs/%s/stream?%s", endpoint, url.PathEscape(runID), query.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, streamURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("X-Blackswan-Token", token)
	req.Header.Set("Accept", "text/event-stream")

	streamClient := &http.Client{}
	resp, err := streamClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		blob, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("stream request failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(blob)))
	}

	const (
		initialScanBuffer = 128 * 1024
		maxScanBuffer     = 16 * 1024 * 1024
	)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, initialScanBuffer), maxScanBuffer)

	currentEvent := ""
	dataLines := make([]string, 0, 4)
	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		raw := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]

		var event StreamEvent
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			return fmt.Errorf("decode stream event: %w", err)
		}
		if event.Event == "" {
			event.Event = currentEvent
		}
		if event.Payload == nil {
			event.Payload = map[string]any{}
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case sink <- event:
			return nil
		}
	}

	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if strings.TrimSpace(line) == "" {
			if err := flush(); err != nil {
				return err
			}
			currentEvent = ""
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			dataLines = append(dataLines, payload)
			continue
		}
	}
	if err := flush(); err != nil {
		return err
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return fmt.Errorf("stream event exceeded max size (%d bytes)", maxScanBuffer)
		}
		return err
	}
	return nil
}
