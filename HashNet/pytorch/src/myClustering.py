# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import os


def kmeans_cluster(K=20):
    cluster = KMeans(K, random_state=0)
    return cluster


def dbscan_cluster(eps=0.5):
    cluster = DBSCAN(eps=eps)
    return cluster


def kmeans_center(hamming_code, K):
    cluster = kmeans_cluster(K)
    cluster.fit(hamming_code)
    return cluster.cluster_centers_


def calculate_cluster_circles(cluster, hamming_code, K, circles_min=0, circles_step=0.01, circles_max=10):
    # predict the cluster id
    cluster_id = cluster.fit_predict(hamming_code)
    cluster_centers = cluster.cluster_centers_
    # initial the circles_index (simply makes each row the same)
    circle_index = np.arange(circles_min, circles_max, circles_step)
    circles_index = np.zeros([K, len(circle_index)])
    circles_number = np.zeros([K, len(circle_index)])
    for i in range(K):
        circles_index[i] = circle_index

    # calculate the number in the circle by different clusters
    for i in range(K):
        cluster_distance_i = np.sum(np.abs(hamming_code[cluster_id == i] - cluster_centers[i]), axis=-1) / 2
        for j in range(len(circles_index[i])):
            circles_number[i, j] = np.sum(cluster_distance_i >= circles_index[i, j])

    return [circles_index, circles_number, cluster_centers]


def calculate_cluster_circles_dbscan(cluster, hamming_code, circles_min=0, circles_step=0.01, circles_max=10):
    cluster_id = cluster.fit_predict(hamming_code)
    K = cluster_id.max()
    cluster_centers = np.zeros([K, 48])

    circle_index = np.arange(circles_min, circles_max, circles_step)
    circles_index = np.zeros([K, len(circle_index)])
    circles_number = np.zeros([K, len(circle_index)])
    for i in range(K):
        circles_index[i] = circle_index
    for i in range(K):
        cluster_centers[i] = hamming_code[cluster_id == i].mean(axis=0)
    for i in range(K):
        cluster_distance_i = np.sum(np.abs(hamming_code[cluster_id == i] - cluster_centers[i]), axis=-1) / 2
        for j in range(len(circles_index[i])):
            circles_number[i, j] = np.sum(cluster_distance_i >= circles_index[i, j])
    return [circles_index, circles_number, cluster_centers]


if __name__ == "__main__":
    # Parameters initialization
    print "Not for this use"


class SavedCluster:
    def __init__(self, job_dataset, K, code_test, net='ResNet152', circles_max=30):
        self.datapath = 'save_for_load/saved_cluster/%s/%s' % (job_dataset, net)
        self.datapath_cluster = '%s/cluster_%s.npy' % (self.datapath, str(K))
        self.datapath_circle = '%s/circle_%s.npz' % (self.datapath, str(K))
        self.K_value = K
        if not os.path.exists(self.datapath):
            os.mkdir(self.datapath)
        if not os.path.exists(self.datapath_cluster):
            # create the cluster and save them
            print '%s not exists. A new clustering file is being creating' % (self.datapath_cluster)
            cluster = kmeans_cluster(K)
            [circles_index, clusterCircleNums, clusterCenters] = calculate_cluster_circles(cluster, code_test, K=K,
                                                                                           circles_max=circles_max)
            saved_cluster = cluster
            saved_circle = [circles_index, clusterCircleNums, clusterCenters]
            self.cluster = saved_cluster
            self.circle = saved_circle
            np.save(self.datapath_cluster, saved_cluster)
            np.savez(self.datapath_circle, circles_index, clusterCircleNums, clusterCenters)
        else:
            # load the cluster and unpack them
            self.cluster = np.load(self.datapath_cluster)
            self.circle_vars = np.load(self.datapath_circle)
            circles_index, clusterCircleNums, clusterCenters = self.circle_vars['arr_0'], self.circle_vars['arr_1'], \
                                                               self.circle_vars['arr_2']
            self.circle = [circles_index, clusterCircleNums, clusterCenters]

    def poly_fit(self, deg=2):
        K = self.K_value
        self.datapath_polyparams = '%s/polyparams_K_%s.npy' % (self.datapath, str(K))
        circles_index, clusterCircleNums = self.circle[0], self.circle[1]
        if not os.path.exists(self.datapath_polyparams):
            # calculate polyparams and save them
            poly_params = np.zeros([K, deg + 1])
            X_index = circles_index[0]
            for i in range(K):
                # print K
                Y_dps = clusterCircleNums[i]
                poly_params[i] = np.polyfit(X_index, Y_dps, deg)
            np.save(self.datapath_polyparams, poly_params)
            self.poly_params = poly_params
        else:
            poly_params = np.load(self.datapath_polyparams)
            self.poly_params = poly_params
        return poly_params

    def plot_functions(self, deg=2):
        K = self.K_value
        poly_params = self.poly_fit(deg)
        circles_index, clusterCircleNums, clusterCenters = self.get_circle_vars()
        fit_tmp = np.array([circles_index ** i for i in range(poly_params.shape[1] - 1, -1, -1)])
        fit_out = np.array([np.matmul(poly_params[i], fit_tmp[:, i, :]) for i in range(K)])
        import matplotlib.pyplot as plt
        for i in range(K):
            plt.subplot(K / 10, 10, i + 1)
            plt.plot(circles_index[i], clusterCircleNums[i])
            plt.plot(circles_index[i], fit_out[i])

    def get_cluster(self):
        return self.cluster

    def get_circle(self):
        return self.circle

    def get_circle_vars(self):
        return self.circle[0], self.circle[1], self.circle[2]

    def get_poly_params(self):
        return self.poly_params


class SavedDBscanCluster:
    def __init__(self, job_dataset, eps, code_test, net='ResNet152'):
        self.datapath = 'save_for_load/saved_cluster/%s/%s' % (job_dataset, net)
        self.datapath_cluster = '%s/cluster_eps_%s.npy' % (self.datapath, str(eps))
        self.datapath_circle = '%s/circle_eps_%s.npz' % (self.datapath, str(eps))
        self.eps = eps
        if not os.path.exists(self.datapath):
            os.mkdir(self.datapath)
        if not os.path.exists(self.datapath_cluster):
            # create the cluster and save them
            cluster = dbscan_cluster(eps)
            [circles_index, clusterCircleNums, clusterCenters] = calculate_cluster_circles_dbscan(cluster, code_test,
                                                                                                  circles_max=30)
            saved_cluster = cluster
            saved_circle = [circles_index, clusterCircleNums, clusterCenters]
            self.cluster = saved_cluster
            self.circle = saved_circle
            np.save(self.datapath_cluster, saved_cluster)
            np.savez(self.datapath_circle, circles_index, clusterCircleNums, clusterCenters)
        else:
            # load the cluster and unpack them
            self.cluster = np.load(self.datapath_cluster)
            self.circle_vars = np.load(self.datapath_circle)
            circles_index, clusterCircleNums, clusterCenters = self.circle_vars['arr_0'], self.circle_vars['arr_1'], \
                                                               self.circle_vars['arr_2']
            self.circle = [circles_index, clusterCircleNums, clusterCenters]

    def poly_fit(self, deg=2):
        self.datapath_polyparams = '%s/polyparams_eps_%s.npy' % (self.datapath, str(self.eps))
        circles_index, clusterCircleNums = self.circle[0], self.circle[1]
        if not os.path.exists(self.datapath_polyparams):
            # calculate polyparams and save them
            poly_params = np.zeros([K, deg + 1])
            X_index = circles_index[0]
            for i in range(K):
                # print K
                Y_dps = clusterCircleNums[i]
                poly_params[i] = np.polyfit(X_index, Y_dps, deg)
            np.save(self.datapath_polyparams, poly_params)
            self.poly_params = poly_params
        else:
            poly_params = np.load(self.datapath_polyparams)
            self.poly_params = poly_params
        return poly_params

    def get_cluster(self):
        return self.cluster

    def get_circle(self):
        return self.circle

    def get_circle_vars(self):
        return self.circle[0], self.circle[1], self.circle[2]

    def get_poly_params(self):
        return self.poly_params
