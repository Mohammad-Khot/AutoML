import time


def log_model_score(name: object, score: object, stage="", log: bool = False):
    if log:
        prefix = f"[{stage}]" if stage else "[MODEL]"

        if isinstance(score, (int, float)):
            print(f"{prefix} {name:<15} score={score:.4f}")
        else:
            print(f"{prefix} {name:<15} {score}")


def log_start(msg):
    print(f"[START] {msg}")


def log_end(msg, start_time):
    print(f"[END] {msg} in {time.perf_counter() - start_time:.2f}s")
