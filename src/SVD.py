from pathlib import Path
import numpy as np
import pickle

class SVD:
    def __init__(self, tag=None):
        self.tag = tag
        if self.tag is None:
            self.u_path = Path(f'../models/svd_u.txt')
            self.s_path = Path(f'../models/svd_s.txt')
            self.vt_path = Path(f'../models/svd_vt.txt')
            self.mean_vector_path = Path(f'../models/svd_{self.tag}_mean_vector.txt')
        else:
            self.u_path = Path(f'../models/svd_{self.tag}_u.txt')
            self.s_path = Path(f'../models/svd_{self.tag}_s.txt')
            self.vt_path = Path(f'../models/svd_{self.tag}_vt.txt')
            self.mean_vector_path = Path(f'../models/svd_{self.tag}_mean_vector.txt')

        self.U = None
        self.s = None
        self.VT = None
        self.mean_vector = None

    def prepare(self, all_vectors):
        mean_vectors = []

        for vectors in all_vectors:
            m = np.mean(vectors, axis=0)
            mean_vectors.append(m.copy())
        self.mean_vector = mean_vectors

    def fit(self, vectors):
        centerized_vectors = [vectors[i] - self.mean_vector[i] for i in range(len(vectors))]
        concat_vectors = np.concatenate(centerized_vectors, axis=1)
        self.U, self.s, self.VT = np.linalg.svd(concat_vectors, full_matrices=True)

    def transform(self, vectors):
        centerized_vectors = [vectors[i] - self.mean_vector[i] for i in range(len(vectors))]
        ret = np.einsum('pq, rs->rq', self.VT, np.concatenate(centerized_vectors, axis=1)) # 22 * 22, 5 * 22
        return ret

    def save_model(self):
        with self.mean_vector_path.open('w') as f:
            f.write('\n'.join([' '.join(map(str, v)) for v in self.mean_vector]))
        with self.u_path.open('w') as f:
            f.write('\n'.join([' '.join(map(str, v)) for v in self.U]))
        with self.s_path.open('w') as f:
            f.write('\n'.join([str(v) for v in self.s]))
        with self.vt_path.open('w') as f:
            f.write('\n'.join([' '.join(map(str, v)) for v in self.VT]))

    def load_model(self):
        with self.mean_vector_path.open('r') as f:
            self.mean_vector = [np.array(list(map(float, l.strip().split(' ')))) for l in f.readlines()]
        with self.u_path.open('r') as f:
            self.U = np.array([list(map(float, l.strip().split(' '))) for l in f.readlines()])
        with self.s_path.open('r') as f:
            self.s = np.array([float(l.strip()) for l in f.readlines()])
        with self.vt_path.open('r') as f:
            self.VT = np.array([list(map(float, l.strip().split(' '))) for l in f.readlines()])


if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.random.randn(5, 10)
    x2 = np.random.randn(5, 12)
    # x1 = [[40, 80], [80, 90], [90, 100]]
    # x2 = [[40, 80], [80, 90], [90, 100]]
    svd = SVD()
    svd.prepare([x1, x2])
    svd.fit([x1, x2])
    svd.save_model()
    svd.load_model()

    print(svd.U.shape)
    print(svd.s.shape)
    print(svd.VT.shape)

    svd.transform([x1, x2])
