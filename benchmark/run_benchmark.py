import functools

import shift_scheduling as s
import timeit
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

os.environ["BENCHMARK_MODE"] = "ON"


def benchmark_run(input_file):
    weeks, nodes, edges, _ = os.path.basename(input_file).split(".")
    r = []
    for _ in range(2):
        r.append(_single_run(input_file))
    return {
        "mean": np.mean(r),
        "std": np.std(r),
        "weeks": int(weeks),
        "nodes": int(nodes),
        "edges": int(edges)
    }


def _single_run(input_file):
    t = timeit.Timer(functools.partial(s.create_schedule, input_file, output_file="run.csv"))
    return min(t.repeat(5, 1))


input_files = [f for f in glob.glob("./input/*.json")]
print(f"Collected {len(input_files)} benchmark configuration{'' if len(input_files) == 1 else 's'}")

runs = []
for idx, input_file in enumerate(input_files):
    runs.append(benchmark_run(input_file))
    print(f"Finished configuration {idx + 1} [{os.path.basename(input_file)}]")

print(runs)
print(len(runs))
os.remove("run.csv")

y = np.array([run["mean"] for run in runs])
y_std = np.array([run["std"] for run in runs])
x = np.array([run["weeks"] for run in runs])
labels = [f"{run['weeks']} ({run['nodes']})" for run in runs]

fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(x, y, "-")
ax.plot(x, y, "x")
ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
plt.suptitle("Time complexity (balanced_weekends = false)")
plt.ylabel("Execution time in seconds")
plt.xlabel("Shift scheduling time frame in weeks (nodes)")
plt.xticks(x, labels, rotation=45)
plt.show()
