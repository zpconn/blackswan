package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadConfigFileSuccess(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"risk":{"alpha":0.05}}`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg, resolved, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile returned error: %v", err)
	}

	wantResolved, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("filepath.Abs: %v", err)
	}
	if resolved != wantResolved {
		t.Fatalf("resolved path mismatch: got %q want %q", resolved, wantResolved)
	}
	if _, ok := cfg["risk"].(map[string]any); !ok {
		t.Fatalf("expected risk object, got: %#v", cfg["risk"])
	}
}

func TestLoadConfigFileRejectsNonObjectJSON(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`["not","an","object"]`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	_, _, err := LoadConfigFile(path)
	if err == nil {
		t.Fatalf("expected error for non-object JSON")
	}
	if !strings.Contains(err.Error(), "top-level object") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadConfigFileRejectsURL(t *testing.T) {
	t.Parallel()

	_, _, err := LoadConfigFile("https://example.com/config.json")
	if err == nil {
		t.Fatalf("expected URL rejection error")
	}
	if !strings.Contains(err.Error(), "local filesystem paths") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFormatConfigJSONNilMap(t *testing.T) {
	t.Parallel()

	text, err := FormatConfigJSON(nil)
	if err != nil {
		t.Fatalf("FormatConfigJSON returned error: %v", err)
	}
	if strings.TrimSpace(text) != "{}" {
		t.Fatalf("unexpected formatted text: %q", text)
	}
}
