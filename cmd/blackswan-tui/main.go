package main

import (
	"context"
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

	model := app.NewModel(svc, store)
	program := tea.NewProgram(model, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "tui exited with error: %v\n", err)
		os.Exit(1)
	}
}
