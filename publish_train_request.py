#!/usr/bin/env python

import os
import sys
import datetime
import json
import argparse
import pandas as pd
from antinex_core.log.setup_logging import build_colorized_logger
from celery_connectors.publisher import Publisher

parser = argparse.ArgumentParser(description=("Launch a Train "
                                              "Request into the AntiNex "
                                              "core"))
parser.add_argument(
    "-f",
    help=("request json file to use default: "
          "./training/django-antinex-simple.json"),
    required=False,
    dest="request_file")
parser.add_argument(
    "-d",
    help="debug",
    required=False,
    dest="debug",
    action="store_true")
args = parser.parse_args()

name = "train-publisher"
log = build_colorized_logger(name=name)

log.info("{} - start".format(name))

request_file = "./training/django-antinex-simple.json"
if args.request_file:
    request_file = args.request_file

exchange_name = "webapp.train.requests"
routing_key = "webapp.train.requests"
queue_name = "webapp.train.requests"
auth_url = "redis://localhost:6379/6"
serializer = "json"

if not os.path.exists(request_file):
    log.error(("Missing request file={}")
              .format(
                request_file))
    sys.exit(1)

req_data = None
with open(request_file, "r") as cur_file:
    req_data = json.loads(cur_file.read())

if not os.path.exists(request_file):
    log.error(("Did not find request data in file={}")
              .format(
                request_file))
    sys.exit(1)

# import ssl
# Connection("amqp://", login_method='EXTERNAL', ssl={
#            "ca_certs": '/etc/pki/tls/certs/something.crt',
#            "keyfile": '/etc/something/system.key',
#            "certfile": '/etc/something/system.cert',
#            "cert_reqs": ssl.CERT_REQUIRED,
#          })
#
ssl_options = {}
app = Publisher(
    name,
    auth_url,
    ssl_options)

if not app:
    log.error(("Failed to connect to broker={}")
              .format(
                  auth_url))
else:

    # Now send:
    now = datetime.datetime.now().isoformat()
    body = req_data
    body["created"] = now
    log.info("loading predict_rows")
    predict_rows_df = pd.read_csv(req_data["dataset"])
    predict_rows = []
    for idx, org_row in predict_rows_df.iterrows():
        new_row = json.loads(org_row.to_json())
        new_row["idx"] = len(predict_rows) + 1
        predict_rows.append(new_row)
    body["predict_rows"] = pd.DataFrame(predict_rows).to_json()

    log.info(("using predict_rows={}")
             .format(
                len(predict_rows)))

    log.info(("Sending msg={} "
              "ex={} rk={}")
             .format(
                str(body)[0:10],
                exchange_name,
                routing_key))

    # Publish the message:
    msg_sent = app.publish(
        body=body,
        exchange=exchange_name,
        routing_key=routing_key,
        queue=queue_name,
        serializer=serializer,
        retry=True)

    log.info(("End - {} sent={}")
             .format(
                name,
                msg_sent))
# end of valid publisher or not
