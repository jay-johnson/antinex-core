#!/usr/bin/env python

import os
import sys
import json
import numpy as np
import pandas as pd
from antinex_core.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.build_scaler_train_and_test_datasets import \
    build_scaler_train_and_test_datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


name = "antinex-scaler-django"
log = build_colorized_logger(name=name)


def build_model(
        num_features,
        loss,
        optimizer,
        metrics):
    """build_model

    Build the Keras Deep Neural Network Model

    :param num_features: number of features
    :param loss: loss function to apply
    :param optimizer: optimizer to use
    :param metrics: list of metrics
    """
    log.info("building model")
    model = Sequential()
    model.add(
        Dense(
            250,
            input_dim=num_features,
            kernel_initializer="uniform",
            activation="relu"))
    model.add(
        Dense(
            1,
            kernel_initializer="uniform",
            activation="sigmoid"))

    log.info(("compiling loss={} optimizer={} metrics={}")
             .format(
                loss,
                optimizer,
                metrics))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)

    return model
# end of build_model


def run_antinex_scaler_normalization_on_django_dataset():
    """run_antinex_scaler_normalization_on_django_dataset

    Process the Django AntiNex dataset using scaler normalization

    """

    # noqa git clone https://github.com/jay-johnson/antinex-datasets.git /opt/antinex-datasets
    dataset = os.getenv(
        "DATASET",
        ("/opt/antinex-datasets/v1/webapps/"
         "django/training-ready/v1_django_cleaned.csv"))
    model_backup_file = os.getenv(
        "MODEL_BACKUP_FILE",
        "/tmp/{}-full-model.h5".format(
            name))
    model_weights_file = os.getenv(
        "MODEL_WEIGHTS_FILE",
        "/tmp/{}-weights.h5".format(
            name))
    model_json_file = os.getenv(
        "MODEL_JSON_FILE",
        "/tmp/{}-model.json".format(
            name))
    model_image_file = os.getenv(
        "MODEL_IMAGE_FILE",
        "/tmp/{}-predictions-vs-correct.png".format(
            name))
    footnote_text = os.getenv(
        "PLOT_FOOTNOTE",
        "AntiNex v1")
    image_title = "{} - Predictions and Correct Predictions".format(
        name)
    show_predictions = bool(os.getenv(
        "SHOW_PREDICTIONS",
        "1") == "1")

    seed = 42
    np.random.seed(seed)

    features_to_process = [
        "idx",
        "arp_hwlen",
        "arp_hwtype",
        "arp_id",
        "arp_op",
        "arp_plen",
        "arp_ptype",
        "dns_default_aa",
        "dns_default_ad",
        "dns_default_an",
        "dns_default_ancount",
        "dns_default_ar",
        "dns_default_arcount",
        "dns_default_cd",
        "dns_default_id",
        "dns_default_length",
        "dns_default_ns",
        "dns_default_nscount",
        "dns_default_opcode",
        "dns_default_qd",
        "dns_default_qdcount",
        "dns_default_qr",
        "dns_default_ra",
        "dns_default_rcode",
        "dns_default_rd",
        "dns_default_tc",
        "dns_default_z",
        "dns_id",
        "eth_id",
        "eth_type",
        "icmp_addr_mask",
        "icmp_code",
        "icmp_gw",
        "icmp_id",
        "icmp_ptr",
        "icmp_seq",
        "icmp_ts_ori",
        "icmp_ts_rx",
        "icmp_ts_tx",
        "icmp_type",
        "icmp_unused",
        "ip_id",
        "ip_ihl",
        "ip_len",
        "ip_tos",
        "ip_version",
        "ipv6_fl",
        "ipv6_hlim",
        "ipv6_nh",
        "ipv6_plen",
        "ipv6_tc",
        "ipv6_version",
        "ipvsix_id",
        "pad_id",
        "tcp_dport",
        "tcp_fields_options.MSS",
        "tcp_fields_options.NOP",
        "tcp_fields_options.SAckOK",
        "tcp_fields_options.Timestamp",
        "tcp_fields_options.WScale",
        "tcp_id",
        "tcp_seq",
        "tcp_sport",
        "udp_dport",
        "udp_id",
        "udp_len",
        "udp_sport"
    ]

    if not os.path.exists(dataset):
        log.error(("Failed to find dataset={}")
                  .format(
                        dataset))

    log.info(("loading dataset={}")
             .format(
                dataset))
    predict_feature = "label_value"
    predict_rows_df = pd.read_csv(dataset)
    log.info("converting to json")

    found_columns = list(predict_rows_df.columns.values)
    num_features = len(found_columns) - 1
    min_scaler_range = -1
    max_scaler_range = 1
    max_records = 100000
    verbose = 1
    test_size = 0.2
    batch_size = 32
    epochs = 15
    num_splits = 2
    loss = "binary_crossentropy"
    optimizer = "adam"
    metrics = [
        "accuracy"
    ]
    label_rules = {
        "labels": [
            "not_attack",
            "not_attack",
            "attack"
        ],
        "label_values": [
            -1,
            0,
            1
        ]
    }

    should_set_labels = False
    labels_dict = {}
    if "labels" in label_rules and "label_values" in label_rules:
        label_rows = label_rules["label_values"]
        for idx, lidx in enumerate(label_rows):
            if len(label_rules["labels"]) >= idx:
                should_set_labels = True
                labels_dict[str(lidx)] = \
                    label_rules["labels"][idx]
    # end of compiling labels dictionary

    # define a function with no arguments for Keras Scikit-Learn API to work
    # https://keras.io/scikit-learn-api/
    def set_model():
        """set_model"""
        return build_model(
            num_features=num_features,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)

    scaler_res = build_scaler_train_and_test_datasets(
        label=name,
        train_features=features_to_process,
        test_feature=predict_feature,
        df=predict_rows_df,
        test_size=test_size,
        seed=seed,
        scaler_cast_to_type="float32",
        min_feature_range=min_scaler_range,
        max_feature_range=max_scaler_range)

    if scaler_res["status"] != SUCCESS:
        log.error(("Failed to scaler train and test datasets for csv={}")
                  .format(
                    dataset))
        sys.exit(1)

    scaler_res_data = scaler_res["data"]
    datanode = {
        "X_train": scaler_res_data["x_train"],
        "Y_train": scaler_res_data["y_train"],
        "X_test": scaler_res_data["x_test"],
        "Y_test": scaler_res_data["y_test"]
    }

    predict_rows_df[predict_feature] = scaler_res_data["scaled_test_dataset"]
    train_scaler_df = pd.DataFrame(
        scaler_res_data["scaled_train_dataset"],
        columns=features_to_process)
    test_scaler_df = pd.DataFrame(
        scaler_res_data["scaled_test_dataset"],
        columns=[predict_feature])

    sample_rows = train_scaler_df[features_to_process]
    target_rows = test_scaler_df
    num_samples = len(sample_rows.index)
    num_target_rows = len(target_rows.index)

    log.info(("scaled datasets ready "
              "x_train={} y_train={} "
              "x_test={} y_test={}")
             .format(
                len(datanode["X_train"]),
                len(datanode["Y_train"]),
                len(datanode["X_test"]),
                len(datanode["Y_test"])))

    log.info("building KerasClassifier")

    model = KerasClassifier(
        build_fn=set_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose)

    log.info(("fitting model epochs={} batch={} "
              "rows={} features={}")
             .format(
                epochs,
                batch_size,
                len(datanode["X_train"]),
                len(found_columns)))

    model.fit(
        datanode["X_train"],
        datanode["Y_train"],
        validation_data=(
            datanode["X_test"],
            datanode["Y_test"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose)

    log.info("building estimators")

    estimators = []
    estimators.append(
        ("standardize",
            StandardScaler()))
    estimators.append(
        ("mlp",
            model))

    log.info("building pipeline")
    pipeline = Pipeline(estimators)

    # noqa https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    log.info(("starting classification "
              "StratifiedKFold "
              "splits={} seed={}")
             .format(
                num_splits,
                seed))

    kfold = StratifiedKFold(
        n_splits=num_splits,
        random_state=seed)

    log.info(("classification cross_val_score"))
    results = cross_val_score(
        pipeline,
        datanode["X_train"],
        datanode["Y_train"],
        cv=kfold)
    scores = [
        results.std(),
        results.mean()
    ]
    accuracy = {
        "accuracy": results.mean() * 100
    }

    log.info(("classification accuracy={} samples={}")
             .format(
                accuracy["accuracy"],
                num_samples))

    # make predictions
    predictions = model.predict(
        sample_rows.values,
        verbose=verbose)

    log.info(("classification confusion_matrix samples={} "
              "predictions={} target_rows={}")
             .format(
                num_samples,
                len(predictions),
                num_target_rows))

    cm = confusion_matrix(
        target_rows.values,
        predictions)

    log.info(("classification has confusion_matrix={} "
              "predictions={} target_rows={}")
             .format(
                cm,
                len(predictions),
                num_target_rows))

    rounded = [round(x[0]) for x in predictions]
    sample_predictions = []

    log.info(("scores={} accuracy={} "
              "merging samples={} with predictions={} "
              "labels={}")
             .format(
                scores,
                accuracy.get("accuracy", None),
                len(sample_rows.index),
                len(rounded),
                labels_dict))

    ridx = 0
    for idx, row in predict_rows_df.iterrows():
        if len(sample_predictions) > max_records:
            log.info(("hit max={} predictions")
                     .format(
                        max_records))
            break
        new_row = json.loads(row.to_json())
        cur_value = rounded[ridx]
        if predict_feature in row:
            new_row["_original_{}".format(
                    predict_feature)] = \
                int(row[predict_feature])
        else:
            new_row["_original_{}".format(
                    predict_feature)] = \
                "missing-from-dataset"
        if cur_value != int(row[predict_feature]):
            new_row["prediction_status"] = 1
        else:
            new_row["prediction_status"] = 0

        new_row[predict_feature] = int(cur_value)

        if should_set_labels:
            new_row["label_name"] = \
                labels_dict[str(int(cur_value))]
        new_row["_row_idx"] = ridx
        new_row["_count"] = idx
        sample_predictions.append(new_row)
        ridx += 1
    # end of merging samples with predictions

    log.info(("creating merged_predictions_df from sample_predictions={}")
             .format(
                len(sample_predictions)))

    merged_predictions_df = pd.DataFrame(
        sample_predictions)

    if show_predictions:
        for row_num, row in merged_predictions_df.iterrows():
            log.info(("row={} original_{}={} predicted={}")
                     .format(
                        row_num,
                        predict_feature,
                        row["_original_{}".format(
                            predict_feature)],
                        row[predict_feature]))
        # end seeing original vs predicted
    # if showing all predictions to log

    log.debug("saving weights")

    # there are some known issues saving weights:
    # https://github.com/keras-team/keras/issues/4875
    model.model.save_weights(model_weights_file)

    log.debug("saving full keras model to file")

    model.model.save(model_backup_file)

    log.debug("saving keras model as json to file")

    with open(
            model_json_file,
            "w") as json_file:
        json_file.write(model.model.to_json())

    log.info(("neural network created "
              "merged_predictions_df.index={} columns={} "
              "with cross_val_score={} and accuracy={}")
             .format(
                len(merged_predictions_df.index),
                merged_predictions_df.columns.values,
                scores,
                accuracy.get("accuracy", None)))

    log.info(("saved model_backup_file={} "
              "model_weights_file={} model_json_file={} "
              "image_title={}")
             .format(
                model_backup_file,
                model_weights_file,
                model_json_file,
                image_title))

    sns.set(font="serif")
    sns.set_context(
        "paper",
        rc={
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 10})
    fig, ax = plt.subplots(
        figsize=(15.0, 10.0))

    ax = sns.distplot(
        merged_predictions_df["prediction_status"])

    ax.get_figure().text(
        0.90,
        0.01,
        footnote_text,
        va="bottom",
        fontsize=8,
        color="#888888")

    # More seaborn examples:

    # ax = sns.regplot(
    #     x=merged_predictions_df[predict_feature],
    #     y=merged_predictions_df["_original_{}".format(
    #         predict_feature)],
    #     marker="+")
    # ax.get_figure().text(
    #     0.90,
    #     0.01,
    #     footnote_text,
    #     va="bottom",
    #     fontsize=8,
    #     color="#888888")

    # g = sns.FacetGrid(
    #     data=merged_predictions_df[[
    #         predict_feature,
    #         "_original_{}".format(
    #             predict_feature)]],
    #     size=15.0,
    #     palette="Set2",
    #     col=predict_feature,
    #     hue=predict_feature)
    # g.map(
    #     sns.regplot,
    #     predict_feature,
    #     "_original_{}".format(
    #         predict_feature))
    # g.fig.suptitle(
    #     image_title)
    # g.fig.get_children()[-1].set_bbox_to_anchor(
    #     (1.1, 0.5, 0, 0))
    # g.fig.text(
    #    0.90,
    #    0.01,
    #    footnote_text,
    #    va="bottom",
    #    fontsize=8,
    #    color="#888888")

    # g = sns.pairplot(
    #    data=merged_predictions_df[[
    #        predict_feature,
    #        "_original_{}".format(
    #            predict_feature)]],
    #    diag_kind="kde")
    # g.fig.suptitle(
    #    image_title)
    # g.fig.text(
    #    0.90,
    #    0.01,
    #    footnote_text,
    #    va="bottom",
    #    fontsize=8,
    #    color="#888888")

    # sns.lmplot(
    #    x=predict_feature,
    #    y="_original_{}".format(
    #        predict_feature),
    #    hue=predict_feature,
    #    data=merged_predictions_df,
    #    markers=["o", "x"],
    #    palette="Set1")

    # g = sns.jointplot(
    #     x=predict_feature,
    #     y="_original_{}".format(
    #         predict_feature),
    #     data=merged_predictions_df,
    #     kind="reg")

    # g = sns.heatmap(
    #     merged_predictions_df[[
    #         predict_feature,
    #         "_original_{}".format(
    #             predict_feature)]].values.astype("int"),
    #     annot=True,
    #     annot_kws={
    #        "size": 16})

    # plt.show()
    ax.get_figure().savefig(
        model_image_file)
    # fig.savefig(
    #    model_image_file)
    # g.fig.savefig(
    #    model_image_file)

    log.info(("saved model_backup_file={} model_weights_file={} "
              "model_json_file={} image_file={}")
             .format(
                model_backup_file,
                model_weights_file,
                model_json_file,
                model_image_file))

# end of run_antinex_scaler_normalization_on_django_dataset


if __name__ == "__main__":
    run_antinex_scaler_normalization_on_django_dataset()
