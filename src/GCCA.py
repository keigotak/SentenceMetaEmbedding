import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig #eigh
import pickle
from pathlib import Path


class GCCA:
    def __init__(self, tag=None):
        self.tag = tag
        if self.tag is None:
            self.eta_path = Path('../models/gcca_eta.pkl')
            self.mean_vector_path = Path('../models/gcca_mean_vector.pkl')
        else:
            self.eta_path = Path(f'../models/gcca_{self.tag}_eta.pkl')
            self.mean_vector_path = Path(f'../models/gcca_{self.tag}_mean_vector.pkl')

        self.eta = None
        self.mean_vector = None
        self.std_vector = None
        self.dim = None
        self.tau = 0.1

    def centering(self, x):
        '''
        :param x: axis0 is number of sentences and axis1 is dimension of each embedding
        :return: centerized embedding vectors
        '''
        avg_by_columns = np.mean(x, axis=0)
        centered_x = x - avg_by_columns
        return centered_x

    def get_variance(self, x):
        std_for_columns = np.std(x, axis=0)
        return std_for_columns

    def get_covariance_matrics(self, x1, x2):
        '''
        :param x1: source embedding vector 1
        :param x2: source embedding vector 2
        :return: covariance matrix
        '''
        return np.outer(x1, x2)

    def fit(self, vectors):
        '''
        :param vectors: axis0 is embedding source, axis1 is batch size, and axis2 is dimention of each source
        :return: None
        '''
        centerized_vectors = [self.centering(vector) for vector in vectors]  # centerized by average for each column

        ## getting covariance matrix for each src embedding pair of
        covariance_matrices = [] # dict にする
        for i in range(len(centerized_vectors)):  # for src1 to calculate covariance matrix
            for j in range(len(centerized_vectors)):  # for src2 to calculate covariance matrix
                cm_sum = np.zeros((len(centerized_vectors[i][0]), len(centerized_vectors[j][0])))
                for k in range(len(centerized_vectors[0])):  # for batch sentences
                    cm_sum += self.get_covariance_matrics(centerized_vectors[i][k], centerized_vectors[j][k])
                cm_sum /= len(centerized_vectors[0]) # 求めるのは期待値なので数で割る必要がある
                if i == j:
                    cm_sum = cm_sum + self.tau * self.std_vector[j] * np.eye(cm_sum.shape[0])
                covariance_matrices.append(cm_sum.tolist())

        ## unpacking covariance_matrices to allocate diagonal matrix and other components, separately.
        ## ここの実装はスピードが遅ければ見直す
        cross_vectors_a = [] # diagonal
        cross_vectors_b = [] # not diagonal
        flg_new_row = False
        flg_diagonal = False
        col, previous_dim = 0, 0
        zero = 1e-10
        for i, cm in enumerate(covariance_matrices):  ## for each covariance matrix
            if i % len(vectors) == 0:
                flg_new_row = True
            if i % len(vectors) == 0 and i != 0:
                col += 1
                previous_dim = len(cm[0])
            if i % len(vectors) == col:
                flg_diagonal = True

            for j, c in enumerate(cm):
                if flg_new_row:
                    if flg_diagonal:
                        cross_vectors_a.append(c.copy())
                        cross_vectors_b.append([zero] * len(c))
                    else:
                        cross_vectors_a.append([zero] * len(c))
                        cross_vectors_b.append(c.copy())
                else:
                    if flg_diagonal:
                        cross_vectors_a[col * previous_dim + j].extend(c)
                        cross_vectors_b[col * previous_dim + j].extend([zero] * len(c))
                    else:
                        try:
                            cross_vectors_a[col * previous_dim + j].extend([zero] * len(c))
                            cross_vectors_b[col * previous_dim + j].extend(c)
                        except IndexError:
                            print(len(cross_vectors_a))
                            print(len(cross_vectors_a[0]))
                            print(col*previous_dim + j)
            flg_new_row = False
            flg_diagonal = False

        ## solving generalized eigen equation
        eigen_values, eigen_vectors = eig(a=np.array(cross_vectors_a), b=np.array(cross_vectors_b)) # in ascending order
        eigen_values_real_part, eigen_vectors_real_part = eigen_values.real, eigen_vectors.real # complex part check

        ## 固有値の大きいものから取る
        c = zip(eigen_values_real_part, eigen_vectors_real_part)
        cc = sorted(c, reverse=True, key=lambda x: x[0])
        sorted_eigen_values_real_part, sorted_eigen_vectors_real_part = zip(*cc)

        ## stacking all eigen vectors
        if self.dim is not None:
            self.eta = eigen_vectors_real_part[:self.dim]
        else:
            self.eta = eigen_vectors_real_part  ## dim: d1 + d2, d1 + d2
        # self.eta = [self.eta[:, :len(vectors[0][0])], self.eta[:, len(vectors[0][0]):]]


    def prepare(self, all_vectors):
        '''
        :param vectors: axis0 is embedding source, axis1 is batch size, and axis2 is dimention of each source
        :return: transformed vectors
        '''
        mean_vectors = []
        std_vectors = []

        for vectors in all_vectors:
            m = np.mean(vectors, axis=0)
            mean_vectors.append(m.copy())
            s = np.std(vectors, axis=0)
            std_vectors.append(np.mean(s).copy())
        self.mean_vector = mean_vectors
        self.std_vector = std_vectors

    def transform(self, vectors):
        '''
        :param vectors: axis0 is embedding source, axis1 is batch size, and axis2 is dimention of each source
        :return: transformed vectors axis0: batch size, axis1: meta embedding dim
        '''
        rets = []

        batch_size = len(vectors[0])
        for b in range(batch_size):
            tmp = []
            for i in range(len(vectors)):
                tmp.extend(vectors[i][b] - self.mean_vector[i]) # same as Poerner et al 2019 EMNLP https://www.aclweb.org/anthology/D19-1173.pdf
            rets.append(np.dot(self.eta, tmp).tolist())
        return np.array(rets)

    def save_model(self):
        with self.eta_path.open('wb') as f:
            pickle.dump(self.eta, f)
        with self.mean_vector_path.open('wb') as f:
            pickle.dump(self.mean_vector, f)

    def load_model(self):
        with self.eta_path.open('rb') as f:
            self.eta = pickle.load(f)
        with self.mean_vector_path.open('rb') as f:
            self.mean_vector = pickle.load(f)


if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.random.randn(5, 10)
    x2 = np.random.randn(5, 12)
    # x1 = [[40, 80], [80, 90], [90, 100]]
    # x2 = [[40, 80], [80, 90], [90, 100]]

    gcca = GCCA()
    print(gcca.get_covariance_matrics(x1, x2))
    gcca.prepare([x1, x2])

    gcca.fit([x1, x2])

    print(gcca.eta)
    print(gcca.eta[0].shape)
    print(gcca.eta[1].shape)

    gcca.transform([x1, x2])
