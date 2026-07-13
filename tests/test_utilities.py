import logging
import time

from moe_bench.logging_util import get_logger, log_diagnostics
from moe_bench.timing import StageTimer


def test_get_logger_writes_rank_log_without_duplicate_handlers(tmp_path):
    logger = get_logger("moe_bench.test", run_dir=tmp_path, rank=1, level="debug")
    logger_again = get_logger("moe_bench.test", run_dir=tmp_path, rank=1, level="debug")

    assert logger is logger_again
    assert len(logger.handlers) == 1

    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()

    log_text = (tmp_path / "logs" / "rank1.log").read_text(encoding="utf-8")
    assert "[rank1][INFO] hello" in log_text


def test_log_diagnostics_does_not_call_supplier_when_disabled(tmp_path):
    logger = get_logger("moe_bench.diag", run_dir=tmp_path, rank=0, level="info")

    called = False

    def supplier():
        nonlocal called
        called = True
        return "expensive"

    log_diagnostics(logger, enabled=False, supplier=supplier)

    assert called is False


def test_stage_timer_records_elapsed_milliseconds():
    timer = StageTimer()

    with timer.stage("sleep"):
        time.sleep(0.001)

    assert timer.elapsed_ms["sleep"] > 0
    assert isinstance(timer.elapsed_ms["sleep"], float)
