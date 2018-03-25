import json
import pandas as pd
from kombu import Connection
from kombu import Producer
from kombu import Exchange
from kombu import Queue
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERROR
from antinex_utils.consts import FAILED


name = "antinex-prc-send-res"
log = build_colorized_logger(name=name)


def send_results_to_broker(
        loc,
        final_results):
    """send_results_to_broker

    :param loc: api-generated dictionary for where to send the results
    :param final_results: prediction results from the worker
    """

    status = ERROR
    org_model = None
    org_rounded = None
    org_train_scaler = None
    org_test_scaler = None
    org_train_scaler_dataset = None
    org_test_scaler_dataset = None

    if len(final_results) > 0 and len(final_results) > 0:

        if final_results["data"]["sample_predictions"]:
            final_results["data"]["sample_predictions"] = json.loads(
                pd.Series(
                    final_results["data"]["sample_predictions"]).to_json(
                        orient="records"))
        if final_results["data"]["rounded"]:
            final_results["data"]["rounded"] = json.loads(
                pd.Series(
                    final_results["data"]["rounded"]).to_json(
                        orient="records"))

        final_results["data"].pop("predictions", None)
        final_results["data"]["model_json"] = \
            final_results["data"]["model"].to_json()

        # remove values that cannot be serialized to json (for now)
        org_model = final_results["data"].pop("model", None)
        org_rounded = final_results["data"].pop("rounded_predictions", None)
        org_train_scaler = final_results["data"].pop("scaler_train", None)
        org_test_scaler = final_results["data"].pop("scaler_test", None)
        org_train_scaler_dataset = final_results["data"].pop(
            "scaled_train_dataset", None)
        org_test_scaler_dataset = final_results["data"].pop(
            "scaled_test_dataset", None)

        source = loc["source"]
        auth_url = loc["auth_url"]
        ssl_options = loc["ssl_options"]
        exchange = loc["exchange"]
        exchange_type = loc["exchange_type"]
        routing_key = loc["routing_key"]
        queue = loc["queue"]
        delivery_mode = loc["delivery_mode"]
        manifest = loc["manifest"]

        log.debug(("CORERES - sending response back to source={} "
                   "ssl={} exchange={} routing_key={}")
                  .format(
                    source,
                    ssl_options,
                    exchange,
                    routing_key))

        try:
            conn = None
            if len(ssl_options) > 0:
                log.debug("connecting with ssl")
                conn = Connection(
                    auth_url,
                    login_method="EXTERNAL",
                    ssl=ssl_options)
            else:
                log.debug("connecting without ssl")
                conn = Connection(
                    auth_url)
            # end of connecting

            conn.connect()

            log.debug("getting channel")
            channel = conn.channel()

            core_exchange = Exchange(
                exchange,
                type=exchange_type,
                durable=True)

            log.debug("creating producer")
            producer = Producer(
                channel=channel,
                auto_declare=True,
                serializer="json")

            try:
                log.debug("declaring exchange")
                producer.declare()
            except Exception as k:
                log.error(("declare exchange failed with ex={}")
                          .format(
                            k))
            # end of try to declare exchange which can fail if it exists

            core_queue = Queue(
                queue,
                core_exchange,
                routing_key=routing_key,
                durable=True)

            try:
                log.debug("declaring queue")
                core_queue.maybe_bind(conn)
                core_queue.declare()
            except Exception as k:
                log.error(("declare queue={} routing_key={} failed with ex={}")
                          .format(
                            queue,
                            routing_key,
                            k))
            # end of try to declare queue which can fail if it exists

            log.info(("publishing exchange={} routing_key={} persist={}")
                     .format(
                        core_exchange.name,
                        routing_key,
                        delivery_mode))

            send_data_to_rest_api = {
                "results": final_results,
                "manifest": manifest
            }

            producer.publish(
                body=send_data_to_rest_api,
                exchange=core_exchange.name,
                routing_key=routing_key,
                auto_declare=True,
                serializer="json",
                delivery_mode=delivery_mode)

        except Exception as e:
            log.info(("Failed to publish to core req={} with ex={}")
                     .format(
                        str(final_results)[0:32],
                        e))
        # try/ex

        status = SUCCESS

        log.info(("send_results_to_broker - done"))
    else:
        log.info(("CORERES - nothing to send back final_results={} ")
                 .format(
                    final_results))
        status = FAILED
    # publish to the core if enabled

    # put this back into the results
    if org_model:
        final_results["data"]["model"] = org_model
    if org_rounded:
        final_results["data"]["rounded_predictions"] = org_rounded

    # could be improved by checking assignment with a list
    final_results["data"]["scaler_train"] = org_train_scaler
    final_results["data"]["scaler_test"] = org_test_scaler
    final_results["data"]["scaled_train_dataset"] = org_train_scaler_dataset
    final_results["data"]["scaled_test_dataset"] = org_test_scaler_dataset

    return status
# end of send_results_to_broker
