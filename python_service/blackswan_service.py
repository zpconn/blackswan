#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import multiprocessing as mp
import os
import queue as queue_mod
import secrets
import signal
import threading
import traceback
import uuid
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blackswan import DEFAULT_CONFIG, run_monte_carlo_optimization


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _run_worker(config: dict[str, Any], out_queue: mp.Queue) -> None:
    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        out_queue.put(
            {
                "kind": "progress",
                "event": str(event),
                "payload": _to_jsonable(payload),
                "timestamp": _now_iso(),
            }
        )

    try:
        result = run_monte_carlo_optimization(
            config=config,
            verbose=False,
            progress_callback=progress_callback,
        )
        out_queue.put({"kind": "result", "result": _to_jsonable(result), "timestamp": _now_iso()})
    except Exception as exc:
        out_queue.put(
            {
                "kind": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": _now_iso(),
            }
        )


@dataclass
class RunRecord:
    run_id: str
    config: dict[str, Any]
    created_at: str
    status: str = "queued"
    updated_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_requested: bool = False
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_seq: int = 1
    process: Optional[mp.Process] = None
    queue: Optional[mp.Queue] = None
    monitor_thread: Optional[threading.Thread] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.updated_at = self.created_at
        self.condition = threading.Condition(self.lock)

    def append_event(self, event: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self.condition:
            item = {
                "seq": int(self.next_seq),
                "timestamp": _now_iso(),
                "event": str(event),
                "payload": _to_jsonable(payload or {}),
            }
            self.next_seq += 1
            self.events.append(item)
            if len(self.events) > 8_000:
                self.events = self.events[-8_000:]
            self.updated_at = item["timestamp"]
            self.condition.notify_all()
            return item

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        with self.lock:
            return [event for event in self.events if int(event["seq"]) > seq]

    def wait_for_event(self, seq: int, timeout: float) -> bool:
        with self.condition:
            if self.events and int(self.events[-1]["seq"]) > seq:
                return True
            self.condition.wait(timeout=timeout)
            return bool(self.events and int(self.events[-1]["seq"]) > seq)

    def is_terminal(self) -> bool:
        with self.lock:
            return self.status in TERMINAL_STATUSES

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            latest_seq = 0 if not self.events else int(self.events[-1]["seq"])
            latest_event = "" if not self.events else str(self.events[-1].get("event", ""))
            latest_event_timestamp = "" if not self.events else str(self.events[-1].get("timestamp", ""))
            latest_event_payload = {} if not self.events else _to_jsonable(self.events[-1].get("payload") or {})
            return {
                "run_id": self.run_id,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "cancel_requested": self.cancel_requested,
                "error": self.error,
                "traceback": self.traceback,
                "result": self.result,
                "latest_seq": latest_seq,
                "latest_event": latest_event,
                "latest_event_timestamp": latest_event_timestamp,
                "latest_event_payload": latest_event_payload,
            }


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._active_run_id: Optional[str] = None

    def _has_active_run(self) -> bool:
        if self._active_run_id is None:
            return False
        record = self._runs.get(self._active_run_id)
        if record is None:
            self._active_run_id = None
            return False
        return not record.is_terminal()

    def create_run(self, config: dict[str, Any]) -> tuple[Optional[RunRecord], Optional[str]]:
        with self._lock:
            if self._has_active_run():
                return None, "A run is already active. Cancel or wait for completion before starting another."

            run_id = uuid.uuid4().hex
            record = RunRecord(
                run_id=run_id,
                config=_to_jsonable(config),
                created_at=_now_iso(),
            )
            self._runs[run_id] = record
            self._active_run_id = run_id

        out_queue: mp.Queue = mp.Queue()
        process = mp.Process(target=_run_worker, args=(config, out_queue), daemon=True)
        record.queue = out_queue
        record.process = process

        process.start()
        record.started_at = _now_iso()
        record.status = "running"
        record.append_event("service_run_started", {"pid": process.pid})

        monitor = threading.Thread(target=self._monitor_run, args=(record,), daemon=True)
        record.monitor_thread = monitor
        monitor.start()
        return record, None

    def _mark_terminal(self, record: RunRecord) -> None:
        with record.lock:
            if record.finished_at is None:
                record.finished_at = _now_iso()
            record.updated_at = _now_iso()
            with record.condition:
                record.condition.notify_all()
        with self._lock:
            if self._active_run_id == record.run_id:
                self._active_run_id = None

    def _monitor_run(self, record: RunRecord) -> None:
        assert record.process is not None
        assert record.queue is not None

        process = record.process
        out_queue = record.queue

        while True:
            message = None
            try:
                message = out_queue.get(timeout=0.25)
            except queue_mod.Empty:
                message = None

            if message is not None:
                kind = message.get("kind")
                if kind == "progress":
                    record.append_event(
                        str(message.get("event", "progress")),
                        _to_jsonable(message.get("payload") or {}),
                    )
                elif kind == "result":
                    with record.lock:
                        record.result = _to_jsonable(message.get("result"))
                        record.status = "completed"
                        record.error = None
                        record.traceback = None
                    record.append_event("service_run_complete", {"status": "completed"})
                    self._mark_terminal(record)
                    break
                elif kind == "error":
                    with record.lock:
                        record.status = "failed"
                        record.error = str(message.get("error", "Unknown worker error"))
                        record.traceback = message.get("traceback")
                    record.append_event("service_run_failed", {"error": record.error})
                    self._mark_terminal(record)
                    break

            if not process.is_alive():
                exit_code = process.exitcode
                with record.lock:
                    if record.status not in TERMINAL_STATUSES:
                        if record.cancel_requested:
                            record.status = "cancelled"
                            record.error = None
                            record.traceback = None
                            record.append_event("service_run_cancelled", {"exit_code": exit_code})
                        else:
                            record.status = "failed"
                            record.error = f"Worker exited unexpectedly with code {exit_code}."
                            record.traceback = None
                            record.append_event("service_run_failed", {"error": record.error})
                self._mark_terminal(record)
                break

        try:
            out_queue.close()
        except Exception:
            pass

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._runs.values())
        snapshots = [record.snapshot() for record in records]
        snapshots.sort(key=lambda item: item["created_at"], reverse=True)
        return snapshots

    def cancel_run(self, run_id: str) -> tuple[bool, str]:
        record = self.get_run(run_id)
        if record is None:
            return False, "Run not found."

        with record.lock:
            if record.status in TERMINAL_STATUSES:
                return False, f"Run is already {record.status}."
            record.cancel_requested = True
            if record.status != "cancel_requested":
                record.status = "cancel_requested"
            record.append_event("service_cancel_requested", {})

        process = record.process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        return True, "Cancellation signal sent."

    def shutdown(self) -> None:
        with self._lock:
            records = list(self._runs.values())
        for record in records:
            process = record.process
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        content_length = 0

    if content_length <= 0:
        return {}

    raw = handler.rfile.read(content_length)
    if not raw:
        return {}

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Request JSON body must be an object.")
    return parsed


def _build_handler(manager: RunManager, token: str):
    class ServiceHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            blob = json.dumps(_to_jsonable(payload), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _is_authorized(self) -> bool:
            header_token = self.headers.get("X-Blackswan-Token")
            if header_token == token:
                return True
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer ") and auth[7:] == token:
                return True
            return False

        def _require_auth(self) -> bool:
            if self._is_authorized():
                return True
            self._send_json({"error": "Unauthorized."}, status=HTTPStatus.UNAUTHORIZED)
            return False

        def _send_sse(self, record: RunRecord, from_seq: int) -> None:
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            seq = max(0, int(from_seq))
            try:
                while True:
                    events = record.events_after(seq)
                    if events:
                        for event in events:
                            frame = (
                                f"id: {int(event['seq'])}\\n"
                                f"event: {event['event']}\\n"
                                f"data: {json.dumps(_to_jsonable(event), separators=(',', ':'), ensure_ascii=False)}\\n\\n"
                            )
                            self.wfile.write(frame.encode("utf-8"))
                            seq = int(event["seq"])
                        self.wfile.flush()

                    if record.is_terminal() and not record.events_after(seq):
                        break

                    if not events:
                        got_new = record.wait_for_event(seq, timeout=10.0)
                        if not got_new:
                            self.wfile.write(b": keepalive\\n\\n")
                            self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            segments = [part for part in path.split("/") if part]

            if path == "/health":
                self._send_json({"status": "ok", "active_run": manager._active_run_id})
                return

            if path == "/defaults":
                if not self._require_auth():
                    return
                self._send_json({"defaults": _to_jsonable(DEFAULT_CONFIG)})
                return

            if path == "/runs":
                if not self._require_auth():
                    return
                self._send_json({"runs": manager.list_runs()})
                return

            if len(segments) == 2 and segments[0] == "runs":
                if not self._require_auth():
                    return
                run_id = segments[1]
                record = manager.get_run(run_id)
                if record is None:
                    self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(record.snapshot())
                return

            if len(segments) == 3 and segments[0] == "runs" and segments[2] == "stream":
                if not self._require_auth():
                    return
                run_id = segments[1]
                record = manager.get_run(run_id)
                if record is None:
                    self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                query = parse_qs(parsed.query)
                from_seq = 0
                try:
                    from_seq = int(query.get("from", ["0"])[0])
                except ValueError:
                    from_seq = 0
                self._send_sse(record, from_seq)
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            segments = [part for part in path.split("/") if part]

            if path == "/runs":
                if not self._require_auth():
                    return
                try:
                    payload = _read_json_body(self)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                raw_config = payload.get("config", {})
                if raw_config is None:
                    raw_config = {}
                if not isinstance(raw_config, dict):
                    self._send_json(
                        {"error": "config must be a JSON object."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                record, error = manager.create_run(raw_config)
                if record is None:
                    self._send_json({"error": error}, status=HTTPStatus.CONFLICT)
                    return
                self._send_json(
                    {"run_id": record.run_id, "status": record.status},
                    status=HTTPStatus.CREATED,
                )
                return

            if len(segments) == 3 and segments[0] == "runs" and segments[2] == "cancel":
                if not self._require_auth():
                    return
                run_id = segments[1]
                ok, message = manager.cancel_run(run_id)
                if not ok:
                    status = HTTPStatus.NOT_FOUND if message == "Run not found." else HTTPStatus.CONFLICT
                    self._send_json({"error": message}, status=status)
                    return
                self._send_json({"status": "cancel_requested", "message": message})
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    return ServiceHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Blackswan local HTTP service bridge for TUI clients")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    token = args.token.strip() or secrets.token_urlsafe(24)
    manager = RunManager()

    handler = _build_handler(manager, token)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True

    host, port = server.server_address
    handshake = {
        "event": "service_ready",
        "host": host,
        "port": int(port),
        "token": token,
        "pid": os.getpid(),
    }
    print(json.dumps(handshake, separators=(",", ":"), ensure_ascii=False), flush=True)

    def _request_stop(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        manager.shutdown()
        try:
            server.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
