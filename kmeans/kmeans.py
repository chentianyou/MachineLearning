import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.contrib import factorization

n_cluster = 2
model_dir = "./model_dir"


def main():
    credits_data = pd.read_csv("../testdata/credit_train.csv", header=None)
    columns = credits_data.columns
    train_data = credits_data[columns[:-1]].copy().values
    label_data = credits_data[columns[-1]].copy().values
    train_data_scale = preprocessing.scale(train_data)

    def input_func():
        features = tf.convert_to_tensor(train_data_scale, dtype=tf.float32)
        features = tf.train.limit_epochs(features, num_epochs=5)
        return features

    k_means = factorization.KMeansClustering(
        num_clusters=n_cluster,
        model_dir=model_dir,
        use_mini_batch=False)

    # train
    num_iterations = 10
    previous_centers = None
    print("start train")
    for _ in range(num_iterations):
        k_means.train(input_func)
        cluster_centers = k_means.cluster_centers()
        if previous_centers is not None:
            print('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
        print('score:', k_means.score(input_func))
    print('cluster centers:', cluster_centers)

    # map the input points to their clusters
    cluster_indices = list(k_means.predict_cluster_index(input_func))
    # with open("result.txt", "w") as f:
    #     for i, point in enumerate(label_data):
    #         cluster_index = cluster_indices[i]
    #         f.write("%s:%d\n" % (point, cluster_index))
    #     f.close()
    rate = dict()
    rate["11"] = [0, 0]
    rate["00"] = [0, 0]
    rate["01"] = [0, 0]
    rate["10"] = [0, 0]
    for i, point in enumerate(label_data):
        cluster_index = cluster_indices[i]
        point = str(point)
        rate[point+"1"][0] += 1
        rate[point+"0"][0] += 1
        if point == "1" and cluster_index == 0:
            rate[point+"1"][1] += 1
        if point == "0" and cluster_index == 1:
            rate[point+"0"][1] += 1
        if point == "1" and cluster_index == 1:
            rate[point+"0"][1] += 1
        if point == "0" and cluster_index == 0:
            rate[point+"1"][1] += 1

    print(rate)
    print("11", rate["11"][1] / rate["11"][0])
    print("00", rate["00"][1] / rate["00"][0])
    print("10", rate["10"][1] / rate["10"][0])
    print("01", rate["01"][1] / rate["01"][0])


if __name__ == '__main__':
    main()
