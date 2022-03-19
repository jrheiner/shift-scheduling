import copy
import datetime
import functools

import shift_scheduling as s
import timeit
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ["BENCHMARK_MODE"] = "ON"


def benchmark_run(input_data, weeks):
    nodes, edges = s.create_schedule(input_data, output_file="run.csv")
    r = []
    for _ in range(2):
        r.append(_single_run(input_data))
    return {
        "mean": np.mean(r),
        "std": np.std(r),
        "weeks": int(weeks),
        "nodes": int(nodes),
        "edges": int(edges)
    }


def _single_run(input_data):
    t = timeit.Timer(functools.partial(s.create_schedule, input_data, output_file="run.csv"))
    return min(t.repeat(1, 1))


def plot_benchmark(runs_list: list, plot_title):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x = np.array([run["weeks"] for run in runs_list[0]])
    labels = [f"{run['weeks']} ({run['nodes']})" for run in runs_list[0]]
    for runs in runs_list:
        y = np.array([run["mean"] for run in runs])
        y_std = np.array([run["std"] for run in runs])
        ax.plot(x, y, "-")
        ax.plot(x, y, "x")
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
    plt.suptitle(plot_title)
    plt.ylabel("Execution time in seconds")
    plt.xlabel("Shift scheduling time frame in weeks (nodes)")
    plt.xticks(x, labels, rotation=45)
    plt.show()


def benchmark_case(benchmark_config, number_of_weeks):
    benchmark_config = copy.deepcopy(benchmark_config)
    runs = []
    for idx in range(number_of_weeks):
        weeks = 4 * (idx + 1)
        benchmark_config["end_date"] = benchmark_config["start_date"] + datetime.timedelta(
            weeks=weeks) - datetime.timedelta(days=1)
        runs.append(benchmark_run(benchmark_config, weeks=weeks))
        print(f"Finished configuration {idx + 1}/{number_of_weeks}")
    return runs


if __name__ == "__main__":
    benchmark_config = {
        "shifts": 2,
        "staff_per_shift": 3,
        "total_staff": 15,
        "work_days": [
            "Mo",
            "Tu",
            "We",
            "Th",
            "Fr",
            "Sa",
            "Su"
        ],
        "start_date": datetime.datetime.strptime("2022-01-01", "%Y-%m-%d"),
        "end_date": datetime.datetime.strptime("2022-01-01", "%Y-%m-%d"),
        "soft_constraints": {
            "balanced_weekends": False
        }
    }
    results = []
    results.append(benchmark_case(benchmark_config=benchmark_config, number_of_weeks=13))
    print(results[0])

    benchmark_config["soft_constraints"]["balanced_weekends"] = True
    results.append(benchmark_case(benchmark_config=benchmark_config, number_of_weeks=13))
    print(results[1])
    plot_benchmark(results, plot_title="Time complexity (balanced_weekends = true)")

    os.remove("run.csv")
