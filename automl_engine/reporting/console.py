# automl_engine/reporting/console.py

from typing import Final

CONSOLE_WIDTH: Final[int] = 50


def print_section(title: str, *, timestamp: bool = False) -> None:
    """
    Print a formatted console section header.

    Parameters
    ----------
    title : str
        Title of the section.
    timestamp : bool, optional
        If True, prepends the current time (HH:MM:SS) to the title.

    Returns
    -------
    None
        Prints formatted output to stdout.
    """
    line: str = "=" * CONSOLE_WIDTH

    if timestamp:
        from datetime import datetime
        now: str = datetime.now().strftime("%H:%M:%S")
        title = f"[{now}] {title}"

    print("\n" + line)
    print(title.upper().center(CONSOLE_WIDTH))
    print(line)


def print_subsection(title: str) -> None:
    """
    Print a formatted subsection title.

    Parameters
    ----------
    title : str
        Subsection name.

    Returns
    -------
    None
        Prints formatted output to stdout.
    """
    text: str = f"--- {title} ---"
    print(f"\n{text.center(CONSOLE_WIDTH)}")


def print_result_block(
    *,
    model: str,
    metric: str,
    mean: float,
    std: float,
    runtime: float,
) -> None:
    """
    Print the final AutoML result summary block.

    Parameters
    ----------
    model : str
        Name of the best-performing model.
    metric : str
        Evaluation metric used.
    mean : float
        Mean cross-validation score.
    std : float
        Standard deviation of the score.
    runtime : float
        Total runtime in seconds.

    Returns
    -------
    None
        Prints formatted result summary to stdout.
    """
    line: str = "=" * CONSOLE_WIDTH

    print("\n" + line)
    print("RESULT".center(CONSOLE_WIDTH))
    print(line)

    print(f"{'Best Model':<15}: {model}")
    print(f"{'Performance':<15}: {metric} = {mean:.4f} ± {std:.4f}")
    print(f"{'Runtime':<15}: {runtime:.2f}s")

    print(line)


def print_row(
    left: str,
    right: str,
    lw: int = 30,
    rw: int = 12,
) -> None:
    """
    Print a two-column formatted row.

    Parameters
    ----------
    left : str
        Left column content.
    right : str
        Right column content.
    lw : int, optional
        Width of the left column.
    rw : int, optional
        Width of the right column.

    Returns
    -------
    None
        Prints formatted row to stdout.
    """
    print(f"{left:<{lw}}{right:>{rw}}")
