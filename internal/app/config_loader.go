package app

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LoadConfigFile reads a local JSON config file and requires a top-level object.
func LoadConfigFile(path string) (map[string]any, string, error) {
	rawPath := strings.TrimSpace(path)
	if rawPath == "" {
		return nil, "", fmt.Errorf("config file path is required")
	}
	if strings.Contains(rawPath, "://") {
		return nil, "", fmt.Errorf("only local filesystem paths are supported")
	}

	resolvedPath, err := filepath.Abs(rawPath)
	if err != nil {
		return nil, "", fmt.Errorf("resolve config path %q: %w", rawPath, err)
	}

	blob, err := os.ReadFile(resolvedPath)
	if err != nil {
		return nil, resolvedPath, fmt.Errorf("read config file %q: %w", resolvedPath, err)
	}

	var parsed any
	if err := json.Unmarshal(blob, &parsed); err != nil {
		return nil, resolvedPath, fmt.Errorf("parse config JSON %q: %w", resolvedPath, err)
	}

	cfg, ok := parsed.(map[string]any)
	if !ok {
		return nil, resolvedPath, fmt.Errorf("config JSON must be a top-level object")
	}
	return cfg, resolvedPath, nil
}

// FormatConfigJSON renders config as pretty-printed JSON for the editor.
func FormatConfigJSON(cfg map[string]any) (string, error) {
	if cfg == nil {
		cfg = map[string]any{}
	}
	blob, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", fmt.Errorf("render config JSON: %w", err)
	}
	return string(blob), nil
}
