import time

from automl_engine.utils.console import CONSOLE_WIDTH

STAGE_W = 10
NAME_W = 15
CONTENT_W = CONSOLE_WIDTH - STAGE_W - NAME_W


def log_model_score(name, score, stage="", log=False):
    if not log:
        return

    stage_str = f"[{stage}]"

    # =====================================================
    # TEXT / SYSTEM MESSAGE
    # =====================================================
    if not isinstance(score, (int, float)):

        msg = str(score)

        if len(msg) > CONTENT_W:
            msg = msg[:CONTENT_W - 3] + "..."

        line = (
            f"{stage_str:<{STAGE_W}}"
            f"{name:<{NAME_W}}"
            f"{msg:>{CONTENT_W}}"
        )

        print(line)
        return

    # =====================================================
    # NUMERIC SCORE ROW
    # =====================================================
    if abs(score) >= 1e6:
        score_str = f"{score:.4e}"
    else:
        score_str = f"{score:.4f}"

    line = (
        f"{stage_str:<{STAGE_W}}"
        f"{name:<{NAME_W}}"
        f"{score_str:>{CONTENT_W}}"
    )

    print(line)

def log_start(msg):
    text = f"[START] {msg}"

    if len(text) > CONSOLE_WIDTH:
        text = text[:CONSOLE_WIDTH - 3] + "..."

    print(text.ljust(CONSOLE_WIDTH))


def log_end(msg, start_time):
    elapsed = time.perf_counter() - start_time
    text = f"[END] {msg} in {elapsed:.2f}s"

    if len(text) > CONSOLE_WIDTH:
        text = text[:CONSOLE_WIDTH - 3] + "..."

    print(text.ljust(CONSOLE_WIDTH))