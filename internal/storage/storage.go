package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"blackswan-tui/internal/service"
)

type Store struct {
	rootDir string
	runsDir string
}

type RunSummary struct {
	RunID                string  `json:"run_id"`
	SavedAt              string  `json:"saved_at"`
	Status               string  `json:"status"`
	ObjectiveMode        string  `json:"objective_mode"`
	RecommendedFraction  float64 `json:"recommended_fraction"`
	ExpectedFinalNetWorth float64 `json:"expected_final_net_worth"`
	Directory            string  `json:"directory"`
}

type RunBundle struct {
	Summary RunSummary         `json:"summary"`
	Config  map[string]any     `json:"config"`
	Result  map[string]any     `json:"result"`
	Events  []service.StreamEvent `json:"events"`
}

func NewStore(rootDir string) (*Store, error) {
	runsDir := filepath.Join(rootDir, "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return nil, fmt.Errorf("create runs dir: %w", err)
	}
	return &Store{rootDir: rootDir, runsDir: runsDir}, nil
}

func (s *Store) RunsDir() string {
	return s.runsDir
}

func (s *Store) SaveRun(runID string, config map[string]any, result map[string]any, events []service.StreamEvent) (RunSummary, error) {
	if strings.TrimSpace(runID) == "" {
		runID = "unknown"
	}
	if config == nil {
		config = map[string]any{}
	}
	if result == nil {
		result = map[string]any{}
	}

	now := time.Now().UTC()
	stamp := now.Format("20060102-150405")
	shortID := runID
	if len(shortID) > 8 {
		shortID = shortID[:8]
	}
	dirName := fmt.Sprintf("%s-%s", stamp, shortID)
	dirPath := filepath.Join(s.runsDir, dirName)
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return RunSummary{}, fmt.Errorf("create run bundle dir: %w", err)
	}

	summary := RunSummary{
		RunID:                 runID,
		SavedAt:               now.Format(time.RFC3339),
		Status:                "completed",
		ObjectiveMode:         asString(result["primary_objective_mode"]),
		RecommendedFraction:   asFloat(result["recommended_fraction"]),
		ExpectedFinalNetWorth: asFloat(result["expected_final_net_worth"]),
		Directory:             dirPath,
	}

	if execution, ok := result["execution"].(map[string]any); ok {
		if mode := asString(execution["mode"]); mode != "" && summary.Status == "completed" {
			summary.Status = mode
		}
	}

	if err := writeJSON(filepath.Join(dirPath, "summary.json"), summary); err != nil {
		return RunSummary{}, err
	}
	if err := writeJSON(filepath.Join(dirPath, "config.json"), config); err != nil {
		return RunSummary{}, err
	}
	if err := writeJSON(filepath.Join(dirPath, "result.json"), result); err != nil {
		return RunSummary{}, err
	}
	if err := writeJSON(filepath.Join(dirPath, "events.json"), events); err != nil {
		return RunSummary{}, err
	}

	bundle := RunBundle{
		Summary: summary,
		Config:  config,
		Result:  result,
		Events:  events,
	}
	if err := writeJSON(filepath.Join(dirPath, "bundle.json"), bundle); err != nil {
		return RunSummary{}, err
	}
	return summary, nil
}

func (s *Store) List(limit int) ([]RunSummary, error) {
	entries, err := os.ReadDir(s.runsDir)
	if err != nil {
		return nil, fmt.Errorf("read runs dir: %w", err)
	}

	summaries := make([]RunSummary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		summaryPath := filepath.Join(s.runsDir, entry.Name(), "summary.json")
		blob, err := os.ReadFile(summaryPath)
		if err != nil {
			continue
		}
		var summary RunSummary
		if err := json.Unmarshal(blob, &summary); err != nil {
			continue
		}
		if summary.Directory == "" {
			summary.Directory = filepath.Join(s.runsDir, entry.Name())
		}
		summaries = append(summaries, summary)
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].SavedAt > summaries[j].SavedAt
	})

	if limit > 0 && len(summaries) > limit {
		summaries = summaries[:limit]
	}
	return summaries, nil
}

func (s *Store) LoadBundle(directory string) (*RunBundle, error) {
	dir := strings.TrimSpace(directory)
	if dir == "" {
		return nil, fmt.Errorf("directory is required")
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(s.runsDir, dir)
	}

	blob, err := os.ReadFile(filepath.Join(dir, "bundle.json"))
	if err == nil {
		var bundle RunBundle
		if json.Unmarshal(blob, &bundle) == nil {
			if bundle.Summary.Directory == "" {
				bundle.Summary.Directory = dir
			}
			return &bundle, nil
		}
	}

	var summary RunSummary
	if err := readJSON(filepath.Join(dir, "summary.json"), &summary); err != nil {
		return nil, err
	}
	var config map[string]any
	if err := readJSON(filepath.Join(dir, "config.json"), &config); err != nil {
		return nil, err
	}
	var result map[string]any
	if err := readJSON(filepath.Join(dir, "result.json"), &result); err != nil {
		return nil, err
	}
	var events []service.StreamEvent
	_ = readJSON(filepath.Join(dir, "events.json"), &events)

	summary.Directory = dir
	bundle := &RunBundle{
		Summary: summary,
		Config:  config,
		Result:  result,
		Events:  events,
	}
	return bundle, nil
}

func writeJSON(path string, value any) error {
	blob, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal json for %s: %w", path, err)
	}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

func readJSON(path string, out any) error {
	blob, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	if err := json.Unmarshal(blob, out); err != nil {
		return fmt.Errorf("decode %s: %w", path, err)
	}
	return nil
}

func asFloat(value any) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case uint:
		return float64(v)
	case uint32:
		return float64(v)
	case uint64:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(strings.TrimSpace(v), 64)
		return f
	default:
		return 0.0
	}
}

func asString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", v)
	}
}
