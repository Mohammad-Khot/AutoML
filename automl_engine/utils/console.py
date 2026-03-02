def print_section(title, *, timestamp=False):
    line = "=" * 12

    if timestamp:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] {line} {title.upper()} {line}")
    else:
        print(f"\n{line} {title.upper()} {line}")


def print_subsection(title: str):
    print(f"\n  ▶ {title}")