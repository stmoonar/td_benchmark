from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable


_LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class _RankFormatter(logging.Formatter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        ms = int(record.msecs)
        return f"[{ts}.{ms:03d}][rank{self.rank}][{record.levelname}] {record.getMessage()}"


def get_logger(name: str = "moe_bench", run_dir: str | Path | None = None, rank: int = 0, level: str = "info") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_LEVELS[level])
    logger.propagate = False
    if getattr(logger, "_moe_bench_config", None) == (str(run_dir), rank, level):
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = _RankFormatter(rank)
    if run_dir is not None:
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_dir / f"rank{rank}.log", encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(_LEVELS[level])
    logger.addHandler(handler)
    if rank == 0 and run_dir is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(_LEVELS[level])
        logger.addHandler(console)
    logger._moe_bench_config = (str(run_dir), rank, level)  # type: ignore[attr-defined]
    return logger


def log_diagnostics(logger: logging.Logger, enabled: bool, supplier: Callable[[], str]) -> None:
    if not enabled:
        return
    logger.info(supplier())
