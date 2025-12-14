"""
Tracing utilities for profiling async and sync functions.
Outputs Chrome Trace/Perfetto format.
"""

import asyncio
import functools
import inspect
import json
import queue
import threading
import time
import atexit
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from io import TextIOWrapper
from typing import Any, Callable


class EventType(str, Enum):
    BEGIN = "B"
    END = "E"
    METADATA = "M"


@dataclass
class TraceEvent:
    name: str
    ph: EventType
    pid: int
    tid: int
    ts: float
    args: dict[str, Any] = field(default_factory=dict)
    cat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "ph": self.ph.value,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
            "args": self.args,
        }
        if self.cat is not None:
            result["cat"] = self.cat
        return result


@dataclass
class ScopeContext:
    attributes: dict[str, Any] = field(default_factory=dict)


trace_context: ContextVar[ScopeContext | None] = ContextVar("trace_context", default=None)


class TraceCollector:
    def __init__(self, flush_interval_sec: float = 1.0, output_file: str = "trace_events.jsonl"):
        self.event_queue: queue.Queue[TraceEvent] = queue.Queue()
        self.flush_interval_sec = flush_interval_sec
        self.output_file = output_file
        self.shutdown_event = threading.Event()
        self.flusher_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flusher_thread.start()
        self.metadata_events: dict[tuple[int, int], TraceEvent] = {}
        self.next_fake_pid = 0
        self.thread_id_to_fake_pid: dict[int, int] = {}

    def add_event(self, event: TraceEvent):
        self.event_queue.put(event)

    def get_timestamp(self) -> float:
        return time.perf_counter() * 1e6

    def get_all_events_immediately_available(self) -> list[TraceEvent]:
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _write_events(self, events: list[TraceEvent], f: TextIOWrapper) -> None:
        for event in events:
            if event.pid not in self.thread_id_to_fake_pid:
                self.thread_id_to_fake_pid[event.pid] = self.next_fake_pid
                self.next_fake_pid += 1
            event.pid = self.thread_id_to_fake_pid[event.pid]

            if event.ph == EventType.METADATA:
                if (event.pid, event.tid) in self.metadata_events:
                    continue
                self.metadata_events[(event.pid, event.tid)] = event

            json.dump(event.to_dict(), f)
            f.write("\n")
        f.flush()

    def _flush_worker(self):
        with open(self.output_file, "a") as f:
            while not self.shutdown_event.is_set():
                events_to_write = self.get_all_events_immediately_available()
                try:
                    event = self.event_queue.get(timeout=self.flush_interval_sec)
                    events_to_write.append(event)
                    events_to_write.extend(self.get_all_events_immediately_available())
                except queue.Empty:
                    continue
                self._write_events(events_to_write, f)
            self._write_events(self.get_all_events_immediately_available(), f)

    def shutdown(self):
        self.shutdown_event.set()
        self.flusher_thread.join(timeout=5.0)


_trace_collector: TraceCollector | None = None


def _atexit_trace_shutdown():
    global _trace_collector
    if _trace_collector is not None:
        _trace_collector.shutdown()
        _trace_collector = None


atexit.register(_atexit_trace_shutdown)


def trace_init(flush_interval_sec: float = 1.0, output_file: str = "trace_events.jsonl") -> None:
    global _trace_collector
    _trace_collector = TraceCollector(flush_interval_sec, output_file)


def trace_shutdown() -> None:
    global _trace_collector
    if _trace_collector is None:
        return
    _trace_collector.shutdown()
    _trace_collector = None


def scope(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for tracing async and sync functions."""

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            if _trace_collector is None:
                return await func(*args, **kwargs)

            thread_id = threading.current_thread().ident or 0
            thread_name = threading.current_thread().name
            try:
                task = asyncio.current_task()
                coroutine_name = task.get_name() if task else f"sync:{thread_name}"
            except RuntimeError:
                coroutine_name = f"sync:{thread_name}"

            begin_event = TraceEvent(
                name=func.__name__,
                ph=EventType.BEGIN,
                pid=thread_id,
                tid=hash(coroutine_name) % 1000000,
                ts=_trace_collector.get_timestamp(),
                args={"track": coroutine_name, "thread": thread_name},
                cat="async",
            )
            _trace_collector.add_event(begin_event)

            scope_ctx = ScopeContext()
            token = trace_context.set(scope_ctx)

            try:
                return await func(*args, **kwargs)
            finally:
                end_event = TraceEvent(
                    name=func.__name__,
                    ph=EventType.END,
                    pid=thread_id,
                    tid=hash(coroutine_name) % 1000000,
                    ts=_trace_collector.get_timestamp(),
                    args={**scope_ctx.attributes},
                    cat="async",
                )
                _trace_collector.add_event(end_event)
                trace_context.reset(token)

        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            if _trace_collector is None:
                return func(*args, **kwargs)

            thread_id = threading.current_thread().ident or 0
            thread_name = threading.current_thread().name

            begin_event = TraceEvent(
                name=func.__name__,
                ph=EventType.BEGIN,
                pid=thread_id,
                tid=0,
                ts=_trace_collector.get_timestamp(),
                args={"thread": thread_name},
                cat="sync",
            )
            _trace_collector.add_event(begin_event)

            scope_ctx = ScopeContext()
            token = trace_context.set(scope_ctx)

            try:
                return func(*args, **kwargs)
            finally:
                end_event = TraceEvent(
                    name=func.__name__,
                    ph=EventType.END,
                    pid=thread_id,
                    tid=0,
                    ts=_trace_collector.get_timestamp(),
                    args={**scope_ctx.attributes},
                    cat="sync",
                )
                _trace_collector.add_event(end_event)
                trace_context.reset(token)

        return sync_wrapper


def get_scope_context() -> ScopeContext:
    result = trace_context.get(ScopeContext())
    assert result is not None
    return result


def update_scope_context(values: dict[str, Any]) -> None:
    result = trace_context.get(ScopeContext())
    assert result is not None
    result.attributes.update(values)
