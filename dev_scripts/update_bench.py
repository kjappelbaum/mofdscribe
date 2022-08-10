# -*- coding: utf-8 -*-
"""
This script does the following

(1) It loops over all the taks it knows. Each tasks corresponds to a subfolder in the top level
`bench` folder.
(2) In each subfolder, it globs for all `.json` file.
(3) For each `.json` file, we look for a corresponding `.rst` file. If there is None, we create it.
(4) We read all `.json` files, validate them, and then compile the results into one long dataframe. We append some metadata for sphinx-needs to the `.rst` file. For each entry, generate a unique id
by taking `R-{first_letter_of_symbol}-{first_letter_of_feat}{year}{month}{day}{hour}{minute}{second}`
(5) This dataframe is used to generate a boxplot which we save as `html`.
(5) We copy the `.rst` and `.html` file to the corresponding `docs/source/leaderboars/` folder.
The `.rst` files will end up in the `docs/source/leaderboards/task_models` folder.
"""

import os
import shutil
from glob import glob
from pathlib import Path

import holoviews as hv
import pandas as pd
from loguru import logger

from mofdscribe.bench.mofbench import BenchResult, BenchTaskEnum

hv.extension("bokeh")


class NoTaskException(Exception):
    pass


METRIC_CARD_TEMPLATE = """Metric card
~~~~~~~~~~~~~~~~~~~

.. regressionmetrics:: {name}
   :id: {id}
   :start_time: {start_time}
   :end_time: {end_time}
   :version: {version}
   :features: {features}
   :name: {name}
   :task: {task}
   :model_type: {model_type}
   :reference: {reference}
   :implementation: {implementation}
   :mofdscribe_version: {mofdscribe_version}
   :mean_squared_error: {mean_squared_error}
   :mean_absolute_error: {mean_absolute_error}
   :r2_score: {r2_score}
   :max_error: {max_error}
   :mean_absolute_percentage_error: {mean_absolute_percentage_error}
   :top_5_in_top_5: {top_5_in_top_5}
   :top_10_in_top_10: {top_10_in_top_10}
   :top_50_in_top_50: {top_50_in_top_50}
   :top_100_in_top_100: {top_100_in_top_100}
   :top_500_in_top_500: {top_500_in_top_500}
   :session_info: {session_info}
"""

DOC_DIR = "../docs"


def make_plot(df, outname):
    cols = [
        "mean squared error",
        "mean absolute error",
        "r2 score",
        "max error",
        "mean absolute percentage error",
        "top 5 in top 5",
        "top 10 in top 10",
        "top 50 in top 50",
        "top 100 in top 100",
        "top 500 in top 500",
    ]

    f = hv.HoloMap(
        {column: hv.BoxWhisker(df, kdims="name", vdims=column) for column in cols}, kdims="metric"
    ).opts(framewise=True, width=450, invert_axes=True)
    hv.util.output(
        widget_location="right",
    )
    hv.save(f, outname, fmt="html")


def update_rst(file, bench_result):
    mean_metrics = bench_result.metrics.average_metrics()
    id = Path(file).stem
    top_keys = [
        "name",
        "start_time",
        "end_time",
        "version",
        "features",
        "name",
        "task",
        "model_type",
        "reference",
        "mofdscribe_version",
        "implementation",
        "session_info",
    ]
    card_dict = {}
    for key in top_keys:
        card_dict[key] = getattr(bench_result, key)

    for k, v in dict(mean_metrics).items():
        try:
            card_dict[k] = v.round(2)
        except AttributeError:
            # integers
            card_dict[k] = v
    card_dict["id"] = id

    with open(file, "r") as handle:
        content = handle.read()

    content += "\n\n" + METRIC_CARD_TEMPLATE.format(**card_dict)

    stem = Path(file).stem
    file_new = file.replace(stem, "R-" + stem)
    with open(file_new, "w") as handle:
        handle.write(content)

    return file_new


def compile_task(task):
    """
    Compile the results of a task into a dataframe.
    """
    # glob for all json files
    task_dir = os.path.join("..", "bench_results", str(task))
    logger.info("Compiling task {}".format(task))

    versions = os.listdir(task_dir)
    latest_version = max(versions)
    for version in versions:
        versionname = "latest" if version == latest_version else version
        json_files = glob(os.path.join(task_dir, version, "*.json"))
        if len(json_files) == 0:
            raise NoTaskException()
        logger.info(f"Found {len(json_files)} json files")
        stems = [Path(f).stem for f in json_files]
        # find all matching rst file
        rst_files = [os.path.join(task_dir, version, f + ".rst") for f in stems]
        # check if the .rst files exist, otherwise create
        for rst_file in rst_files:
            if not os.path.exists(rst_file):
                with open(rst_file, "w") as f:
                    f.write("")

        # read all json files
        dfs = []

        for json in json_files:
            res = BenchResult.parse_file(json)

            df = pd.DataFrame(res.metrics.concatenated_metrics().dict())
            df["name"] = res.name

            dfs.append(df)

        df_all = pd.concat(dfs)

        df_all.columns = [c.replace("_", " ") for c in df_all.columns]
        df_all.to_csv(os.path.join(task_dir, version, "metrics.csv"))
        html_path = os.path.join(task_dir, version, f"{task}_plot_{versionname}.html")
        make_plot(df_all, html_path)

        for rst_file, json in zip(rst_files, json_files):
            new_file = update_rst(rst_file, BenchResult.parse_file(json))
            shutil.copy(
                new_file,
                os.path.join(DOC_DIR, "source", "leaderboards", f"{task}_models", versionname),
            )

        shutil.copy(html_path, os.path.join(DOC_DIR, "source", "leaderboards"))


if __name__ == "__main__":
    for task in BenchTaskEnum._member_names_:
        try:
            compile_task(task)
        except NoTaskException:
            pass
