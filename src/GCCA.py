import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig



class GCCA:
    def __init__(self):
        pass

    def fit(self, vectors):
        avg_vectors = [np.mean(vector) for vector in vectors]
        div_vectors = [np.std(vector) for vector in vectors]

        cross_vectors_a = [[] for _ in range(vectors.shape[0] * vectors.shape[1])]
        cross_vectors_b = [[] for _ in range(vectors.shape[0] * vectors.shape[1])]
        for i in range(vectors.shape[0]):
            for j in range(vectors.shape[0]):
                ## calculating cross vectors for each input vector
                centered_vector_i = vectors[i] - avg_vectors[i]  ## minus average from each input vector
                centered_vector_j = vectors[j] - avg_vectors[j]  ## minus average from each input vector
                cross_vectors = np.outer(centered_vector_i, centered_vector_j)  ## calculate cross vector

                ## for stability, adding slight value based on input variance
                tau = 0.1  ## hyper paramter
                sigma = div_vectors[i]
                stability_matrix = tau * sigma * np.eye(cross_vectors.shape[0])
                cross_vectors += stability_matrix

                ## unpacking covariance matrices, and packing each element of covariance matrices to global 2d matrix
                ## A is the left and B is the right of eigen equation
                for k in range(cross_vectors.shape[0]):
                    if i == j:
                        cross_vectors_a[cross_vectors.shape[0] * i + k].extend(cross_vectors[k].tolist())
                        cross_vectors_b[cross_vectors.shape[0] * i + k].extend([0.0] * cross_vectors.shape[1])
                    else:
                        cross_vectors_a[cross_vectors.shape[0] * i + k].extend([0.0] * cross_vectors.shape[1])
                        cross_vectors_b[cross_vectors.shape[0] * i + k].extend(cross_vectors[k].tolist())

        ## solving generalized eigen equation
        # top_k = 2
        # eigen_values, eigen_vectors = eigs(A=np.array(cross_vectors_a), k=top_k, M=np.array(cross_vectors_b))
        eigen_values, eigen_vectors = eig(a=np.array(cross_vectors_a), b=np.array(cross_vectors_b))
        eigen_values_real_part, eigen_vectors_real_part = eigen_values.real, eigen_vectors.real

        ## stacking all eigen vectors
        eigen_vector_matrix = eigen_vectors_real_part

        return eigen_vector_matrix

    def transform(self, vectors):
        pass


if __name__ == '__main__':
    np.random.seed(0)
    # arr = np.random.randn(2, 10)

    arr = np.array([[1., 0.], [0., 1.]])
    '''
    [[ 0.5  0.5  0.5  0.5]
     [-0.5  0.5 -0.5  0.5]
     [-0.5  0.5  0.5 -0.5]
     [ 0.5  0.5 -0.5 -0.5]]
    '''

    gcca = GCCA()
    eigen_vectors = gcca.fit(arr)
    print(eigen_vectors)
    print(eigen_vectors.shape)
