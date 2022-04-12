from pathlib import Path
import numpy as np
import pickle
import torch
import dask
import dask.array as da

class SVD:
    def __init__(self, tag=None):
        self.tag = tag
        if self.tag is None:
            self.u_path = Path(f'../models/svd_u.txt')
            self.s_path = Path(f'../models/svd_s.txt')
            self.vt_path = Path(f'../models/svd_vt.txt')
            self.mean_vector_path = Path(f'../models/svd_{self.tag}_mean_vector.txt')
        else:
            self.u_path = Path(f'../models/svd_{self.tag}_u.pt')
            self.s_path = Path(f'../models/svd_{self.tag}_s.pt')
            self.vt_path = Path(f'../models/svd_{self.tag}_vt.pt')
            self.mean_vector_path = Path(f'../models/svd_{self.tag}_mean_vector.pt')

        self.U = None
        self.s = None
        self.VT = None
        self.mean_vector = None

    def prepare(self, all_vectors):
        mean_vectors = [torch.mean(torch.as_tensor(v), dim=0) for v in all_vectors]

        # for vectors in all_vectors:
        #     m = np.mean(vectors, axis=0)
        #     mean_vectors.append(m.copy())
        self.mean_vector = mean_vectors

    def fit(self, vectors):
        centerized_vectors = [torch.as_tensor(vectors[i]) - self.mean_vector[i] for i in range(len(vectors))]
        backend = 'dask'
        if backend == 'torch':
            concat_vectors = torch.cat(centerized_vectors, dim=1)
            self.U, self.s, self.VT = torch.linalg.svd(concat_vectors, full_matrices=True)
        elif backend == 'dask':
            concat_vectors = da.from_array(torch.cat(centerized_vectors, dim=1).numpy())
            rets = dask.compute(da.linalg.svd(concat_vectors))
            self.U, self.s, self.VT = rets[0]
        else:
            concat_vectors = torch.cat(centerized_vectors, dim=1).numpy()
            self.U, self.s, self.VT = np.linalg.svd(concat_vectors, full_matrices=True)


    def transform(self, vectors):
        centerized_vectors = [torch.as_tensor(vectors[i]) - self.mean_vector[i] for i in range(len(vectors))]
        # ret = np.einsum('pq, rs->rq', self.VT, np.concatenate(centerized_vectors, axis=1)) # 22 * 22, 5 * 22
        # ret = da.compute(da.einsum('pq,rs->rq', self.VT.numpy(), np.concatenate(centerized_vectors, axis=1)))
        ret = da.compute(da.matmul(np.concatenate(centerized_vectors, axis=1), self.VT.numpy()))

        return ret[0]

    def save_model(self):
        with self.mean_vector_path.open('wb') as f:
            torch.save([m for m in self.mean_vector], f)
            # f.write('\n'.join([' '.join(map(str, v.tolist())) for v in self.mean_vector]))
        with self.u_path.open('wb') as f:
            torch.save(torch.as_tensor(self.U, dtype=torch.double), f)
            # f.write('\n'.join([' '.join(map(str, v)) for v in self.U.tolist()]))
        with self.s_path.open('wb') as f:
            torch.save(torch.as_tensor(self.s, dtype=torch.double), f)
            # f.write('\n'.join([str(v) for v in self.s.tolist()]))
        with self.vt_path.open('wb') as f:
            torch.save(torch.as_tensor(self.VT, dtype=torch.double), f)
            # f.write('\n'.join([' '.join(map(str, v)) for v in self.VT.tolist()]))

    def load_model(self):
        with self.mean_vector_path.open('rb') as f:
            self.mean_vector = torch.load(f)
            # self.mean_vector = [np.array(list(map(float, l.strip().split(' ')))) for l in f.readlines()]
        with self.u_path.open('rb') as f:
            self.U = torch.load(f)
            # self.U = np.array([list(map(float, l.strip().split(' '))) for l in f.readlines()])
        with self.s_path.open('rb') as f:
            self.s = torch.load(f)
            # self.s = np.array([float(l.strip()) for l in f.readlines()])
        with self.vt_path.open('rb') as f:
            self.VT = torch.load(f)
            # self.VT = np.array([list(map(float, l.strip().split(' '))) for l in f.readlines()])


if __name__ == '__main__':
    np.random.seed(0)
    x1 = torch.as_tensor(np.random.randn(5, 10))
    x2 = torch.as_tensor(np.random.randn(5, 12))
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
