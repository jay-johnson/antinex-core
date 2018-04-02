#!/usr/bin/env python

import os
import sys
import json
import argparse
import pandas as pd
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_utils.utils import ppj


name = "convert-to-json"
log = build_colorized_logger(name=name)


def convert_bottom_rows_to_json():
    """convert_bottom_rows_to_json

    Convert the last few rows in a dataset to JSON
    """

    parser = argparse.ArgumentParser(
                description=("Convert the last few "
                             "rows in a dataset to JSON"))
    parser.add_argument(
        "-f",
        help=("dataset to use default /opt/antinex-datasets/v1/webapps/"
              "django/training-ready/v1_django_cleaned.csv"),
        required=False,
        dest="dataset")
    parser.add_argument(
        "-b",
        help=("rows from bottom of dataset"),
        required=False,
        dest="rows_from_bottom")
    parser.add_argument(
        "-d",
        help="enable debug",
        required=False,
        dest="debug",
        action="store_true")
    args = parser.parse_args()

    debug = False
    dataset = ("/opt/antinex-datasets/v1/webapps/django/training-ready/"
               "v1_django_cleaned.csv")
    bottom_row_idx = -5
    if args.debug:
        debug = True
    if args.dataset:
        dataset = args.dataset
    if args.rows_from_bottom:
        bottom_row_idx = int(args.rows_from_bottom)
        if bottom_row_idx > 0:
            bottom_row_idx = -1 * bottom_row_idx

    if not os.path.exists(dataset):
        log.error(("failed to find file={}")
                  .format(
                    dataset))
        sys.exit(1)

    if debug:
        log.info(("reading dataset={}")
                 .format(
                    dataset))
    df = pd.read_csv(
        dataset)
    output_predict_rows = []

    if debug:
        log.info(("building predict_rows={}")
                 .format(
                    -1 * bottom_row_idx))
    for ridx, row in df.iloc[bottom_row_idx:].iterrows():
        new_row = json.loads(row.to_json())
        new_row["_dataset_index"] = ridx
        output_predict_rows.append(new_row)
    # end of building rows from dataset

    print(ppj(output_predict_rows))

# end of convert_bottom_rows_to_json


if __name__ == "__main__":
    convert_bottom_rows_to_json()
