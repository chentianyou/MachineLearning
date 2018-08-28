import json
import os
import sys

import tensorflow as tf
from tensorflow.contrib import factorization

tf.app.flags.DEFINE_string("config", "", "the config of kmeans")
FLAGS = tf.app.flags.FLAGS


class KMeansConfig(object):
    def __init__(self, file):
        with open(file) as f:
            config = json.load(f)
        if "model_dir" in config:
            self.model_dir = config["model_dir"]
        else:
            raise ValueError("config must specify model_dir")

        if "n_cluster" in config:
            self.n_cluster = config["n_cluster"]
        else:
            raise ValueError("config must specify n_cluster")

        if "data_dir" in config:
            self.data_dir = config["data_dir"]
        else:
            raise ValueError("config must specify data_dir")

        if "epoch" in config:
            self.epoch = config["epoch"]
        else:
            self.epoch = 1

        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        else:
            self.batch_size = 50

        if "data_type" in config:
            self.data_type = config["data_type"]
        else:
            ValueError("config must specify data_type")

        if "have_label" in config:
            self.have_label = config["have_label"]
        else:
            self.have_label = False

        if "num_input" in config:
            self.num_input = config["num_input"]
        else:
            self.num_input = None

        if not isinstance(self.data_type, list):
            ValueError("data_type must be list")
        self.default_value = []
        self.default_dic = {"DT_STRING": "", "DT_FLOAT": 0.0, "DT_INT64": 0}
        for dtype in self.data_type:
            if dtype in self.default_dic:
                self.default_value.append([self.default_dic[dtype]])
            else:
                ValueError("data_type %s is error, must be DT_STRING,DT_FLOAT or DT_INT64" % dtype)


def main(_):
    config = KMeansConfig(FLAGS.config)
    files = []
    if tf.gfile.Exists(config.data_dir):
        if tf.gfile.IsDirectory(config.data_dir):
            file_list = tf.gfile.ListDirectory(config.data_dir)
            files.extend([os.path.join(config.data_dir, i) for i in file_list])
        else:
            files.append(config.data_dir)
    else:
        raise ValueError("path %s is not exists." % config.data_dir)

    # dfs = []
    # for filename in files:
    #     dfs.append(pd.read_csv(filename, header=None))
    # credits_data = pd.concat(dfs, ignore_index=True)
    # columns = credits_data.columns
    # train_data = credits_data[columns[:-1]].values
    # label_data = credits_data[columns[-1]].values
    # train_data_scale = preprocessing.scale(train_data)
    #
    # def input_feature_fn():
    #     return tf.train.limit_epochs(
    #         tf.convert_to_tensor(train_data_scale, dtype=tf.float32), num_epochs=1)
    def input_feature_fn(is_feature=True):
        def data_decode(records):
            columns = tf.decode_csv(records=records, record_defaults=config.default_value)
            features = dict([(str(k), columns[k]) for k in range(len(config.data_type) - 1)])
            labels = columns[-1]
            if is_feature:
                return features
            else:
                return labels

        dataset = tf.data.TextLineDataset(files)
        if is_feature:
            dataset = dataset.batch(config.batch_size)
            dataset = dataset.repeat(config.epoch)
        dataset = dataset.map(data_decode)
        return dataset

    feature_columns = []
    for i in range(len(config.data_type) - 1):
        feature_columns.append(tf.feature_column.numeric_column(str(i)))
    iter_step = int(config.num_input / config.batch_size)
    kmeans = factorization.KMeansClustering(
        config.n_cluster,
        model_dir=config.model_dir,
        use_mini_batch=True,
        mini_batch_steps_per_iteration=iter_step,
        feature_columns=feature_columns,
    )
    # train
    print("start train")
    kmeans.train(input_feature_fn)
    cluster_centers = kmeans.cluster_centers()
    print('score:', kmeans.score(input_feature_fn))
    print('cluster centers:', cluster_centers)

    # map the input points to their clusters
    cluster_indices = list(kmeans.predict_cluster_index(input_feature_fn))
    result_file = os.path.join(config.model_dir, "result.txt")
    if tf.gfile.Exists(result_file):
        tf.gfile.Remove(result_file)

    dataset = input_feature_fn(is_feature=False)
    iterator = dataset.make_one_shot_iterator()
    label_tensor = iterator.get_next()
    with tf.Session() as sess:
        with open(result_file, "w") as f:
            f.write("label,cluster_index\n")
            idx = 0
            while True:
                try:
                    label_val = sess.run(label_tensor)
                    cluster_index = cluster_indices[idx]
                    f.write("%s,%s\n" % (label_val, cluster_index))
                    idx += 1
                except tf.errors.OutOfRangeError:
                    break
    print("result write to %s" % result_file)


if __name__ == '__main__':
    if FLAGS.config == "":
        print("must specify config.")
        sys.exit(0)
    if not tf.gfile.Exists(FLAGS.config):
        print("config file not exists.")
        sys.exit(0)
    tf.app.run()
