CONSOLE_WIDTH = 50


def print_section(title, *, timestamp=False):
    line = "=" * CONSOLE_WIDTH

    if timestamp:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        title = f"[{now}] {title}"

    print("\n" + line)
    print(title.upper().center(CONSOLE_WIDTH))
    print(line)


def print_subsection(title):
    text = f"--- {title} ---"
    print(f"\n{text.center(CONSOLE_WIDTH)}")


def print_result_block(
    *,
    model,
    metric,
    mean,
    std,
    runtime,
):
    line = "=" * CONSOLE_WIDTH

    print("\n" + line)
    print("RESULT".center(CONSOLE_WIDTH))
    print(line)

    print(f"{'Best Model':<15}: {model}")
    print(f"{'Performance':<15}: {metric} = {mean:.4f} ± {std:.4f}")
    print(f"{'Runtime':<15}: {runtime:.2f}s")

    print(line)