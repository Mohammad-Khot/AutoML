# reporting/preprocessing.py

from .console import print_section, CONSOLE_WIDTH


def _format_steps(details: dict) -> str:
    """
    Convert recursive estimator description into a readable string.
    """
    if not details:
        return "None"

    name = details.get("name", "Unknown")

    # Leaf node
    if not details.get("steps"):
        return name

    # Flatten pipeline steps
    steps = []
    for step in details["steps"]:
        for _, sub in step.items():
            steps.append(_format_steps(sub))

    return " → ".join(steps)


def print_preprocessing(summary: dict) -> None:
    print_section("PREPROCESSING PIPELINE")

    transformers = summary.get("transformers", {})

    if not transformers:
        print("No preprocessing applied.")
        return

    for name, info in transformers.items():
        cols = info.get("columns", [])
        details = info.get("details", {})

        step_str = _format_steps(details)

        print(f"{name.upper():<15}: {step_str}")
        print(f"{'Columns':<15}: {cols}")
        print("-" * CONSOLE_WIDTH)

    # Feature selector
    selector = summary.get("selector")

    if selector:
        print(f"{'Feature Sel':<15}: {selector['name']}")

    if summary.get("feature_engineering"):
        print("FE  :", summary["feature_engineering"])
