from enum import Enum

import pandas as pd
import tensorflow as tf


class Kind(Enum):
    IsFeature = 0
    IsLabel = 1
    Other = 2


class DataConfig(object):
    def __init__(self, config):
        self.config = config
        self.columns = []
        if "columns" in config:
            self.columns = config["columns"]
        else:
            raise ValueError("data_config must specify columns")

        def sort_by_index(elem):
            return elem["index"]

        self.columns.sort(key=sort_by_index)

        if "cross_column" in config:
            self.cross_column = config["cross_column"]
        else:
            self.cross_column = None

        if "hash_bucket" in config and config["hash_bucket"]:
            self.biggest_hash_bucket = max(config["hash_bucket"])
            self.hash_bucket = pd.Series(config["hash_bucket"])
        self.type_dict = {"DT_STRING": tf.string, "DT_INT64": tf.int64, "DT_FLOAT": tf.float32}
        self.default_value = {"DT_STRING": [""], "DT_INT64": [0], "DT_FLOAT": [0.0]}
        self.feature_columns = {}
        for column in self.columns:
            d_type = column["type"]
            idx = column["index"]
            if d_type in self.type_dict:
                self.feature_columns["key%d" % idx] = tf.FixedLenFeature([], self.type_dict[d_type])
            else:
                raise TypeError("cannot convert type [%s] to tf type." % d_type)

    def column_exists(self, name):
        for col in self.columns:
            if name == col["name"]:
                return True
        return False

    def get_column_by_name(self, name):
        for col in self.columns:
            if name == col["name"]:
                return col
        return None

    def get_column_by_index(self, index):
        for col in self.columns:
            if index == col["index"]:
                return col
        return None

    def have_one_hot(self):
        for col in self.columns:
            if col["n_class"]:
                return True
        return False

    def get_linear_feature_columns(self):
        """
        get crossed columns.
        if specify cross_column in data config, will only cross the specified column,
        else cross all the one hot columns
        cross_column format in data config:
        ----------------------------------
        "cross_column": [
            {
            "columns":["key1","key2"],
            "boundaries":{
                "key1":[0,10,100],
                "key2":null
                }
            },
            {
            "columns":["key1","key2","key3"],
            "boundaries":{
                "key1":[0,10,100],
                "key2":[5,15,25],
                "key3":null
                }
            }
        ]
        -----------------------------------
        if column type is string boundary must be null,
        if column type is number boundaries must be list
        """
        linear_feature_columns = []
        if self.cross_column:
            crossed_columns = []
            for cross_column in self.cross_column:
                columns = cross_column["columns"]
                boundaries = cross_column["boundaries"]
                hash_bucket_size = []
                tmp_cross = []
                for c in columns:
                    if self.column_exists(c):
                        col = self.get_column_by_name(c)
                        d_type = col["type"]
                        if d_type == "DT_STRING":
                            if col["n_class"]:
                                hash_bucket_size.append(col["n_class"])
                                tmp_cross.append(c)
                            else:
                                raise ValueError("column %s`s type is DT_STRING, but it`s not in onehot_column" % c)
                        else:
                            if not boundaries:
                                raise ValueError("column %s must specify boundaries")
                            if col["kind"] == Kind.IsFeature.value:
                                col_tensor = tf.feature_column.numeric_column(c)
                                tmp_cross.append(tf.feature_column.bucketized_column(col_tensor, boundaries[c]))
                            else:
                                raise ValueError("column %s must be in features" % c)
                    else:
                        raise ValueError("column %s must be in columns" % c)
                crossed_columns.append(tf.feature_column.crossed_column(tmp_cross, max(hash_bucket_size)))
            linear_feature_columns = crossed_columns
        elif self.have_one_hot():
            crossed_columns = []
            crossed_features = []
            idx = 0
            for col_outer in self.columns:
                if not col_outer["n_class"]:
                    continue
                for col_inner in self.columns[idx + 1:]:
                    if not col_inner["n_class"]:
                        continue
                    crossed_features.append(
                        ([col_outer["name"], col_inner["name"]], max(col_outer["n_class"], col_inner["n_class"])))
                idx += 1
            for cols, categorical_size in crossed_features:
                try:
                    hash_bucket_size = self.hash_bucket[self.hash_bucket > categorical_size].real[0]
                except IndexError:
                    hash_bucket_size = self.biggest_hash_bucket
                crossed_columns.append(tf.feature_column.crossed_column(cols, hash_bucket_size))

            linear_feature_columns = crossed_columns  # + base_columns
        return linear_feature_columns

    def get_one_hot_feature_columns(self):
        feature_columns = []
        for col in self.columns:
            if col["kind"] != Kind.IsFeature.value:
                continue
            if col["n_class"]:
                try:
                    # get hash number just greater than one hot class number
                    hash_bucket_size = self.hash_bucket[self.hash_bucket > col["n_class"]].real[0]
                except IndexError:
                    hash_bucket_size = self.biggest_hash_bucket
                categorical = tf.feature_column.categorical_column_with_hash_bucket(col["name"], hash_bucket_size)
                feature_columns.append(tf.feature_column.indicator_column(categorical))
            elif col["mean"]:
                def normalizer(m, s):
                    def func(x):
                        return (tf.cast(x, tf.float32) - m) / s

                    return func

                mean = col["mean"]
                std = col["std_dev"]
                if std == 0:
                    continue
                feature_columns.append(
                    tf.feature_column.numeric_column(col["name"], normalizer_fn=normalizer(mean, std)))
            else:
                feature_columns.append(tf.feature_column.numeric_column(col["name"]))
        return feature_columns

    def get_features_labels_from_orc(self, serialized):
        columns = tf.parse_example(serialized, features=self.feature_columns)
        features = dict()
        labels = []
        for col in self.columns:
            if col["kind"] == Kind.IsFeature.value:
                features[col["name"]] = columns["key%d" % col["index"]]
            elif col["kind"] == Kind.IsLabel.value:
                labels.append(columns["key%d" % col["index"]])
        labels = tf.stack(labels, axis=1)
        return features, labels

    def get_features_from_orc(self, serialized):
        columns = tf.parse_example(serialized, features=self.feature_columns)
        features = dict()
        for col in self.columns:
            if col["kind"] == Kind.IsFeature.value:
                features[col["name"]] = columns["key%d" % col["index"]]
        return features

    def get_features_labels_from_csv(self, records):
        default_value = self.get_default_value()
        columns = tf.decode_csv(records, default_value)
        features = dict()
        labels = []
        for c in self.columns:
            kind = c["kind"]
            if kind == Kind.IsFeature.value:
                features[c["name"]] = columns[c["index"]]
            elif kind == Kind.IsLabel.value:
                labels.append(columns[c["index"]])
        labels = tf.stack(labels, axis=1)
        return features, labels

    def get_features_from_csv(self, records):
        default_value = self.get_default_value()
        columns = tf.decode_csv(records, default_value)
        features = dict()
        for c in self.columns:
            kind = c["kind"]
            if kind == Kind.IsFeature.value:
                features[c["name"]] = columns[c["index"]]
        return features

    def get_default_value(self):
        default_value = []
        for col in self.columns:
            kind = col["index"]
            if kind == Kind.IsFeature.value:
                if col["type"] == "DT_STRING":
                    default_value.append([""])
                else:
                    if col["mean"]:
                        default_value.append(col["mean"])
                    else:
                        default_value.append(self.default_value[col["type"]])
            else:
                default_value.append(self.default_value[col["type"]])
        return default_value
