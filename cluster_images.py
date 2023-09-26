import os
import pandas as pd
import numpy as np
import random

from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class KerasFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=32):
        model = VGG16()
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        self.batch_size = batch_size

    def fit(self, X):
        return self

    def transform(self, X):
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        features = []

        for idx in range(n_batches):
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            batch_images = X[start:end]
            batch_features = self.process_batch(batch_images)
            features.extend(batch_features)

        return list(zip(X, np.vstack(features)))

    def process_batch(self, batch_images):
        batch_data = [preprocess_input(np.array(load_img(img, target_size=(224, 224)))) for img in batch_images]
        batch_data = np.array(batch_data)
        return self.model.predict(batch_data)


class DimReducer(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=100, random_state=1):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        _, features = zip(*X)
        self.pca.fit(features)
        return self

    def transform(self, X):
        filenames, features = zip(*X)
        reduced_features = self.pca.transform(features)
        return list(zip(filenames, reduced_features))


class ClusterKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, X, y=None):
        filenames, features = zip(*X)
        self.kmeans.fit(features)
        self.labels_ = self.kmeans.labels_

        self.groups = {}
        for filename, label in zip(filenames, self.labels_):
            if label not in self.groups:
                self.groups[label] = []
            self.groups[label].append(filename)

        return self


def main(img_path: str, n_sample: int = None):

    list_imgs = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    if n_sample is not None:
        list_imgs = random.sample(list_imgs, k=n_sample)

    pipeline = Pipeline([
      ("feature_extractor", KerasFeatureExtractor(batch_size=32)),
      ("reducer", DimReducer()),
      ("clustering", ClusterKMeans())
    ])
    pipeline.fit(list_imgs)

    return pipeline

"""Example usage:

    import requests

    for img in df["path"].dropna():

        img_path = os.path.join(img_base_dir, img.split("/")[-1])
        if os.path.exists(img_path):
            continue

        img_data = requests.get(img).content
        with open(img_path, 'wb') as handler:
            handler.write(img_data)
        
    pipe_obj = main(img_base_dir)
    generated_groups = pipe_obj.named_steps["clustering"].groups

"""