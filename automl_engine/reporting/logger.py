# automl_engine/reporting/logging.py

import time
from typing import Final, Union

from .console import CONSOLE_WIDTH


STAGE_W: Final[int] = 12
NAME_W: Final[int] = 20
CONTENT_W: Final[int] = CONSOLE_WIDTH - STAGE_W - NAME_W


def log_model_score(
    name: str,
    score: Union[int, float, str],
    stage: str = "",
    log: bool = False,
) -> None:
    """
    Log a model score or system message in aligned tabular format.

    Parameters
    ----------
    name : str
        Model or component name.
    score : int | float | str
        Numeric score or textual message.
    stage : str, optional
        Pipeline stage label (e.g., TRAIN, VALID, SEARCH).
    log : bool, optional
        If False, logging is skipped.

    Returns
    -------
    None
        Prints formatted log output to stdout.
    """
    if not log:
        return

    stage_str: str = f"[{stage}]"

    # =====================================================
    # TEXT / SYSTEM MESSAGE
    # =====================================================
    if not isinstance(score, (int, float)):

        msg: str = str(score)

        if len(msg) > CONTENT_W:
            msg = msg[: CONTENT_W - 3] + "..."

        line: str = (
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
        score_str: str = f"{score:.4e}"
    else:
        score_str = f"{score:.4f}"

    line: str = (
        f"{stage_str:<{STAGE_W}}"
        f"{name:<{NAME_W}}"
        f"{score_str:>{CONTENT_W}}"
    )

    print(line)


def log_start(msg: str) -> None:
    """
    Log the start of an execution stage.

    Parameters
    ----------
    msg : str
        Description of the process starting.

    Returns
    -------
    None
        Prints formatted start message.
    """
    text: str = f"[START] {msg}"

    if len(text) > CONSOLE_WIDTH:
        text = text[: CONSOLE_WIDTH - 3] + "..."

    print(text.ljust(CONSOLE_WIDTH))


def log_end(msg: str, start_time: float) -> None:
    """
    Log completion of an execution stage with elapsed runtime.

    Parameters
    ----------
    msg : str
        Description of the completed process.
    start_time : float
        Start timestamp obtained from time.perf_counter().

    Returns
    -------
    None
        Prints formatted end message with execution time.
    """
    elapsed: float = time.perf_counter() - start_time
    text: str = f"[END] {msg} in {elapsed:.2f}s"

    if len(text) > CONSOLE_WIDTH:
        text = text[: CONSOLE_WIDTH - 3] + "..."

    print(text.ljust(CONSOLE_WIDTH))
