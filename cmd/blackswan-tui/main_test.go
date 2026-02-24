package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveStartupConfigEmptyPath(t *testing.T) {
	t.Parallel()

	text, source, err := resolveStartupConfig("")
	if err != nil {
		t.Fatalf("resolveStartupConfig returned error: %v", err)
	}
	if text != "" || source != "" {
		t.Fatalf("expected empty startup config for empty path, got text=%q source=%q", text, source)
	}
}

func TestResolveStartupConfigSuccess(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"simulation":{"n_sims":123}}`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	text, source, err := resolveStartupConfig(path)
	if err != nil {
		t.Fatalf("resolveStartupConfig returned error: %v", err)
	}
	if !strings.Contains(text, `"n_sims": 123`) {
		t.Fatalf("unexpected config JSON text: %q", text)
	}
	wantSource, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("filepath.Abs: %v", err)
	}
	if source != wantSource {
		t.Fatalf("resolved source mismatch: got %q want %q", source, wantSource)
	}
}

func TestResolveStartupConfigFailsForInvalidShape(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`["invalid"]`), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	_, _, err := resolveStartupConfig(path)
	if err == nil {
		t.Fatalf("expected resolveStartupConfig to fail for non-object JSON")
	}
}
