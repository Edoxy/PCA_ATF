import numpy as np
'''
Implementaizone della PCA tramite la decomposizione SVD 
Created on April 2023

author: Edoardo Vay
'''

class PCA:
    def __init__(self, n_components: int = None) -> None:
        '''
        Class implementation of PCA:
        _n_components: int, default None; Number of dimension of the new subspace
        '''
        self._n_components = n_components
        # will contain the mean
        self.mean = None
        self._eig_vectors = None
        self._eig_value = None
        self.S = None
        self.U = None
        self.V = None
        self._fitted = False
        return
    
    def fit(self, data: np.array) -> None:
        '''
        Function that finds the principal direction
        data: np.array
        '''
        n = data.shape[0]
        p = data.shape[1]
        self.mean = np.mean(data, axis=0)
        centered_data = data
        self.U, self.S, Vt = np.linalg.svd(centered_data)
        self.V = Vt.T
        if self._n_components == None:
            self._n_components = np.min(n, p)
        else:
            self._n_components = min(self._n_components, min(n, p))

        self._eig_value = self.S**2 / self._n_components
        self._eig_vectors = self.V[:, 0:self._n_components]

        self._fitted = True
        return
    
    def transform(self, x: np.array) -> np.array:
        '''
        Function that transform the data in the principal components
        '''
        if not self._fitted:
            print(NotImplementedError, 'Before tranforming you need to fit the model')
            return None
        centered_x = x - self.mean
        pca_x = centered_x.dot(self._eig_vectors)
        return pca_x
    
    def inverse_transform(self, pca_x: np.array) -> np.array:
        '''
        Function that reconstruct the data in the original space
        '''
        if not self._fitted:
            print(NotImplementedError, 'Before tranforming you need to fit the model')
            return None
        centered_x = pca_x.dot(self._eig_vectors.T)
        x = centered_x + self.mean
        return x

if __name__ == '__main__':
    print('Test Class ...')
    pca = PCA(n_components=10)
    x = np.zeros((10, 10))
    pca.transform(x)
