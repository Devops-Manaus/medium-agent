import json
import logging
import os
from datetime import datetime

from crewai.events import crewai_event_bus
from crewai.events.types.agent_events import AgentExecutionCompletedEvent
from crewai.events.types.llm_events import LLMCallFailedEvent
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

_logger: logging.Logger | None = None


def setup_run_logger(run_id: str) -> logging.Logger:
    """Cria e configura o logger para uma execução específica.
    Salva em logs/run_<run_id>.log e também emite no console.
    """
    global _logger

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/run_{run_id}.log"

    logger = logging.getLogger(f"crewai_run.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _logger = logger
    _register_event_handlers(logger)

    logger.info("=" * 60)
    logger.info(f"NOVA EXECUÇÃO — run_id={run_id} | log={log_path}")
    logger.info("=" * 60)
    return logger


def get_logger() -> logging.Logger:
    """Retorna o logger da sessão atual. Cria um fallback se não foi inicializado."""
    global _logger
    if _logger is None:
        _logger = setup_run_logger(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return _logger


def _register_event_handlers(logger: logging.Logger) -> None:
    """Subscreve os eventos do crewAI event bus para captura no log."""

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def on_tool_start(source, event: ToolUsageStartedEvent):
        args_str = json.dumps(event.tool_args, ensure_ascii=False) if isinstance(event.tool_args, dict) else str(event.tool_args)
        logger.info(f"TOOL_START  [{event.agent_role or 'agent'}] → {event.tool_name} | args: {args_str[:300]}")

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def on_tool_end(source, event: ToolUsageFinishedEvent):
        duration = (event.finished_at - event.started_at).total_seconds()
        output_preview = str(event.output)[:300]
        logger.info(f"TOOL_END    [{event.agent_role or 'agent'}] ← {event.tool_name} | {duration:.1f}s | output: {output_preview}")

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def on_tool_error(source, event: ToolUsageErrorEvent):
        logger.error(f"TOOL_ERROR  [{event.agent_role or 'agent'}] {event.tool_name} | {event.error}")

    @crewai_event_bus.on(TaskStartedEvent)
    def on_task_start(source, event: TaskStartedEvent):
        task_name = getattr(event.task, "name", None) or (str(event.task)[:60] if event.task else "unknown")
        logger.info(f"TASK_START  {task_name}")

    @crewai_event_bus.on(TaskCompletedEvent)
    def on_task_complete(source, event: TaskCompletedEvent):
        task_name = getattr(event.task, "name", None) or (str(event.task)[:60] if event.task else "unknown")
        output_preview = str(event.output.raw)[:500] if event.output else ""
        logger.info(f"TASK_DONE   {task_name} | output: {output_preview}")

    @crewai_event_bus.on(TaskFailedEvent)
    def on_task_fail(source, event: TaskFailedEvent):
        task_name = getattr(event.task, "name", None) or (str(event.task)[:60] if event.task else "unknown")
        logger.error(f"TASK_FAIL   {task_name} | {getattr(event, 'error', '')}")

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def on_agent_done(source, event: AgentExecutionCompletedEvent):
        role = getattr(event.agent, "role", "agent")
        logger.info(f"AGENT_DONE  [{role}] | output: {str(event.output)[:300]}")

    @crewai_event_bus.on(LLMCallFailedEvent)
    def on_llm_fail(source, event: LLMCallFailedEvent):
        logger.error(f"LLM_FAILED  {str(event.error)[:500]}")
