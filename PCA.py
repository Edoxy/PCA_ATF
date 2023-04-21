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
    
    def fit(self, data: np.array, svd_decomposition = True) -> None:
        '''
        Function that finds the principal direction
        data: np.array
        svd_decomposition: bool, If true uses the svd decomposition to find the eighenvectors
        '''
        n = data.shape[0]
        p = data.shape[1]

        # calcoliamo il vettore medio e centriamo i dati
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean

        # Imposta il numero massimo di componenti che potremo ottenere
        # anche nel caso non fosse stato scelto il paramentro 
        if self._n_components == None:
            # Imposta il numero massimo di componenti che potremo ottenere
            # nel caso non fosse stato scelto il paramentro
            self._n_components = np.min(n, p)
        else:
            self._n_components = min(self._n_components, min(n, p))

        # calcolo delle componenti principali
        if svd_decomposition:
            ## SVD
            self.U, self.S, Vt = np.linalg.svd(centered_data)
            self.V = Vt.T

            self._eig_value = self.S**2 / self._n_components
            self._eig_vectors = self.V[:, 0:self._n_components]

        else:
            ## CLASSICO
            cov = np.matmul(centered_data, centered_data.T)
            v, w = np.linalg.eigh(cov)
            v = v[::-1]
            self._eig_value = (v + 1e-13)/p
            # print(np.min(v))
            w = w[:, ::-1]
            # Calcoliamo gli autovettori della Matrice di covarianza originale
            w = np.matmul(centered_data.T, w)
            self._eig_vectors = w[:, 0:self._n_components]

            for i in range(self._n_components):
                self._eig_vectors[:, i] = -1/(np.sqrt(v[i])) * self._eig_vectors[:, i]
        
        # Abilitiamo l'uso delle funzioni di trasformazione
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
