package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"blackswan-tui/internal/app"
	"blackswan-tui/internal/service"
	"blackswan-tui/internal/storage"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	configPath := flag.String("config", "", "Path to a JSON config file to preload in the TUI editor.")
	flag.Parse()

	startupConfigJSON, startupConfigSource, err := resolveStartupConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load startup config: %v\n", err)
		os.Exit(1)
	}

	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to determine working directory: %v\n", err)
		os.Exit(1)
	}
	rootDir, _ = filepath.Abs(rootDir)

	svc := service.NewManager(rootDir)
	startupCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := svc.Start(startupCtx); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start blackswan local service: %v\n", err)
		logDump := svc.Logs()
		if logDump != "" {
			fmt.Fprintln(os.Stderr, "service logs:")
			fmt.Fprintln(os.Stderr, logDump)
		}
		os.Exit(1)
	}
	defer func() {
		_ = svc.Stop()
	}()

	store, err := storage.NewStore(rootDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize run storage: %v\n", err)
		os.Exit(1)
	}

	model := app.NewModelWithOptions(svc, store, app.ModelOptions{
		InitialConfigJSON: startupConfigJSON,
		InitialConfigPath: startupConfigSource,
	})
	program := tea.NewProgram(
		model,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
		tea.WithReportFocus(),
	)
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "tui exited with error: %v\n", err)
		os.Exit(1)
	}
}

func resolveStartupConfig(configPath string) (string, string, error) {
	if configPath == "" {
		return "", "", nil
	}

	cfg, resolvedPath, err := app.LoadConfigFile(configPath)
	if err != nil {
		return "", "", err
	}
	text, err := app.FormatConfigJSON(cfg)
	if err != nil {
		return "", "", err
	}
	return text, resolvedPath, nil
}
