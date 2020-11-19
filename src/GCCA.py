import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig



class GCCA:
    def __init__(self):
        self.eta = None

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
        div_vectors = [self.get_variance(vector) for vector in vectors]

        ## getting covariance matrix for each src embedding pair of
        covariance_matrices = []
        for i in range(len(centerized_vectors)):  # for src1 to calculate covariance matrix
            for j in range(len(centerized_vectors)):  # for src2 to calculate covariance matrix
                cm_sum = np.zeros((len(centerized_vectors[i][0]), len(centerized_vectors[j][0])))
                for k in range(len(centerized_vectors[0])):  # for batch sentences
                    cm_sum += self.get_covariance_matrics(centerized_vectors[i][k], centerized_vectors[j][k])
                covariance_matrices.append(cm_sum.tolist())

        ## unpacking covariance_matrices to allocate diagonal matrix and other components, separately.
        cross_vectors_a = []
        cross_vectors_b = []
        flg_new_row = False
        flg_diagonal = False
        col, previous_dim = 0, 0
        for i, cm in enumerate(covariance_matrices):  ## for each covariance matrix
            if i % len(vectors) == 0:
                flg_new_row = True
            if i % len(vectors) == 0 and i != 0:
                col += 1
                previous_dim += len(cm[0])
            if i % len(vectors) == col:
                flg_diagonal = True

            for j, c in enumerate(cm):
                if flg_new_row:
                    if flg_diagonal:
                        cross_vectors_a.append(c.copy())
                        cross_vectors_b.append([0.0] * len(c))
                    else:
                        cross_vectors_a.append([0.0] * len(c))
                        cross_vectors_b.append(c.copy())
                else:
                    if flg_diagonal:
                        cross_vectors_a[col * previous_dim + j].extend(c)
                        cross_vectors_b[col * previous_dim + j].extend([0.0] * len(c))
                    else:
                        cross_vectors_a[col * previous_dim + j].extend([0.0] * len(c))
                        cross_vectors_b[col * previous_dim + j].extend(c)
            flg_new_row = False
            flg_diagonal = False

        ## solving generalized eigen equation
        eigen_values, eigen_vectors = eig(a=np.array(cross_vectors_a), b=np.array(cross_vectors_b))
        eigen_values_real_part, eigen_vectors_real_part = eigen_values.real, eigen_vectors.real

        ## stacking all eigen vectors
        self.eta = eigen_vectors_real_part  ## d1 + d2 * d1 + d2

    def transform(self, vectors):
        '''
        :param vectors: axis0 is embedding source, axis1 is batch size, and axis2 is dimention of each source
        :return: transformed vectors
        '''
        return self.eta @ vectors


if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.random.randn(5, 10)
    x2 = np.random.randn(5, 12)
    x1 = [[40, 80], [80, 90], [90, 100]]
    x2 = [[40, 80], [80, 90], [90, 100]]

    gcca = GCCA()
    print(gcca.get_covariance_matrics(x1, x2))
    eigen_vectors = gcca.fit([x1, x2])

    print(eigen_vectors)
    print(eigen_vectors.shape)
