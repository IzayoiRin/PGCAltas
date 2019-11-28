import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class NonFitError(Exception):
    pass


class LDAexFeatures(object):
    """
    search for the projection of a dataset which maximizes the between class scatter
    to within class scatter ratio Sw/Sb of this projected dataset
    target:
        transform a dataset A through transformation matrix W, the ratio of between class scatter
    to within class scatter of the transformed dataset Y is maximized.
        Y = wTA
        adjusting w, LDA aims to find the projection of maximum separability

    For original dataset A m*n:
    Scatter Within(Sw):
        c = Rm*n xj = Rn*1 uc = Rn*1
        (xj-uc) dot (xj-uc)T = Rn*n Sw = Rn*n
        Sw = (ci-ucT)T dot (ci-ucT)
           = sigma((xj-uc) dot (xj-uc)T)
    Scatter Between(Sb):
        u = Rn*1, Sb = R n*n
        Sb = sigma(m(uc-u)(uc-u)T)
    Scatter Total(St):
        St = Sw + Sb
    Ratio = Sw / Sb

    For transformed dateset Y
    Sw* = wT dot Sw dot w
    Sb* = wT dot Sb dot w
    ratio = Sw* / Sb*
    w calculated by the Eigenvectors of Sw-1Sb
    argMax(w) Sw* / Sb* same as constrained optimization:
        argMax(w) wT dot Sb dot w
        s.t. wT dot Sw dot w
    Lagrangian form:
        L = wT dot Sb dot w - lambda(wT dot Sw dot w - K)
        dL / dw = Sb dot w - lambda Sw dot w = 0
        Sw-1 Sb w = lambda w
        * [Sw-1Sb-diag(lambda)] = 0
        * this equation accomplished by numpy.linalg.eig(Sw-1Sb)
    Steps: Raschka, S. (2015)
        Standardize Normalized(mean=0 dev=1)
        calculate miu, miu c of each class
        calculate Sb and Sw scatter matrix
        calculate eigenvectors of Sw-1Sb
        select corresponding k largest eigenvectors to create w =Rn*k
        transform dataset with w
    Transform:
        Y = Xw
        original dataset X = R m*n
        transform matrix w = R n*k
        transformed dataset Y = R m*k
    """

    def __init__(self, max_egi=None):
        """
        m: samples n: factors
        :param dataset: Rm*n
        :param labels: Rm*1
        """
        self.dataset = None
        self.labels = None
        self.max_egi = max_egi

        self.scatter_w = None
        self.scatter_b = None

    def fit(self, dataset: np.ndarray, labels: np.ndarray, standard=False):
        """
        fit the Linear Discriminant Machine from input dataset according to it's labels
        :param standard: should standardize dataset
        :param dataset: training dataset
        :param labels: training labels
        """
        self.dataset = dataset
        self.labels = labels
        if standard:
            self.__standard_scale()
        self._init_params()
        self._cal_eigen()
        return self

    def transform(self, dataset: np.ndarray):
        """
        transform input dataset with fitted params
        :param dataset: testing_dataset
        :return: transformed_dataset
        """
        if self.transform_matrix is None:
            raise NonFitError('Fit dataset first')
        return dataset.dot(self.transform_matrix)

    def fit_transform(self, dataset: np.ndarray, labels: np.ndarray):
        """
        fit form input and transformed it
        :param dataset: training dataset
        :param labels: training label
        :return: transformed dataset
        """
        return self.fit(dataset, labels).transform(dataset)

    def __standard_scale(self):
        self.dataset = StandardScaler().fit_transform(self.dataset)

    def _init_params(self):
        """
        initial params from dataset and labels
        """
        m, n = self.dataset.shape
        # calculate the mean vector of whole dataset
        self.miu = np.mean(self.dataset, axis=0).reshape(n, 1)
        # group dataset by class labels
        classes_idxs = [np.argwhere(self.labels[:, 0] == label).reshape(-1) for label in np.unique(self.labels)]
        miu_c, data_c, n_c = list(), list(), list()
        for idx in classes_idxs:
            # get per class dataset
            c = self.dataset[idx, :]
            data_c.append(c)
            # get per class mean vector
            miu_c.append(np.mean(c, axis=0))
            # calculate the rows of per class
            n_c .append(len(c))
        miu_c = np.array(miu_c).T
        n_c = np.array(n_c)
        # sum the sw of per class up
        self.scatter_w = self._cal_sw(data_c, miu_c, len(np.unique(self.labels)))
        # sum the sb of per class up
        self.scatter_b = self._cal_sb(n_c, miu_c)

    @staticmethod
    def _cal_sw(c_dat, c_miu, total_c):
        """
        Calculate the scatter matrices for the SW (Scatter within)
        :param c_dat: data of per classes R c*n
        :param c_miu: mean vectors for per classes R n*1
        :return: scatter matrix within R n*n
        """
        ret = [(c_dat[i] - c_miu[:, i]).T.dot((c_dat[i] - c_miu[:, i]))
               for i in range(total_c)]
        return sum(ret)

    def _cal_sb(self, c_rows, c_miu):
        """
        Calculate the scatter matrices for the SB (Scatter Between)
        :param c_rows: the rows of per class Nc [scale]
        :param c_miu: mean vectors for per classes R n*1
        :return: scatter matrix between R n*n
        """
        return c_rows * (c_miu - self.miu).dot((c_miu - self.miu).T)

    def _cal_eigen(self):
        """
        Compute the Eigenvalues and Eigenvectors of SW^-1 SB
        """
        assert self.scatter_w is not None and self.scatter_b is not None, "Initial params first"
        t = np.linalg.inv(self.scatter_w).dot(self.scatter_b)
        self.eigval, self.eigvec = np.linalg.eig(t)
        eigval_dict = dict(sorted([(i, np.abs(v)) for i, v in enumerate(self.eigval)], key=lambda x: x[0]))
        self.selected = list(eigval_dict.keys())[:self.max_egi]

    @property
    def eigenpairs(self):
        if not hasattr(self, 'selected'):
            return
        return list(zip(self.eigval[self.selected, ], self.eigvec[:, self.selected]))

    @property
    def transform_matrix(self):
        if not hasattr(self, 'selected'):
            return
        return self.eigvec[:, self.selected]


if __name__ == '__main__':
    rectangles = np.array([[1, 1.5, 1.7, 1.45, 1.1, 1.6, 1.8], [1.8, 1.55, 1.45, 1.6, 1.65, 1.7, 1.75], [1.8, 1.55, 1.45, 1.6, 1.65, 1.7, 1.75]])
    triangles = np.array([[0.1, 0.5, 0.25, 0.4, 0.3, 0.6, 0.35, 0.15, 0.4, 0.5, 0.48],
                          [1.1, 1.5, 1.3, 1.2, 1.11, 1.0, 1.4, 1.2, 1.368, 1.5, 1.0], [1.1, 1.5, 1.3, 1.2, 1.15, 1.0, 1.4, 1.2, 1.3, 1.5, 1.0]])
    circles = np.array([[1.5, 1.55, 1.52, 1.4, 1.3, 1.6, 1.35, 1.45, 1.4, 1.5, 1.48, 1.51, 1.52, 1.49, 1.41, 1.39, 1.6,
                         1.35, 1.55, 1.47, 1.57, 1.48,
                         1.55, 1.555, 1.525, 1.45, 1.35, 1.65, 1.355, 1.455, 1.45, 1.55, 1.485, 1.515, 1.525, 1.495,
                         1.415, 1.395, 1.65, 1.355, 1.555, 1.475, 1.575, 1.485]
                           , [1.3, 1.35, 1.33, 1.32, 1.315, 1.30, 1.34, 1.32, 1.33, 1.35, 1.30, 1.31, 1.35, 1.33, 1.32,
                              1.315, 1.55, 1.34, 1.28, 1.23, 1.25, 1.29,
                              1.35, 1.355, 1.335, 1.325, 1.66, 1.305, 1.345, 1.325, 1.335, 1.355, 1.305, 1.315, 1.355,
                              1.335, 1.325, 1.3155, 1.385, 1.345, 1.285, 1.235, 1.255, 1.295],
                        [1.3, 1.35, 1.33, 1.32, 1.315, 1.30, 1.34, 1.32, 1.33, 1.35, 1.30, 1.31, 1.35, 1.33, 1.32,
                         1.315, 1.38, 1.34, 1.28, 1.23, 1.25, 1.29,
                         1.35, 1.355, 1.335, 1.325, 1.3155, 1.305, 1.345, 1.325, 1.335, 1.355, 1.305, 1.315, 1.355,
                         1.335, 1.325, 2.1, 1.385, 1.345, 1.285, 1.28, 1.255, 1.295]
                        ])

    rectangles, triangles, circles = rectangles.T, triangles.T, circles.T
    classes = [rectangles, triangles, circles]
    new_dataset = np.concatenate(classes, 0)
    print(new_dataset.shape)
    rows = []
    for idx, i in enumerate(classes):
        row = len(i)
        rows.append(np.array([str(idx + 1) for _ in range(row)]).reshape(-1, 1))
    labels = np.concatenate(rows, 0)
    lda0 = LDAexFeatures(max_egi=2)
    y = lda0.fit_transform(new_dataset, labels)
    a,b,c = range(0,7), range(7, 18), range(18,62)
    a, b, c = y[list(a), :],y[list(b), :],y[list(c), :]
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.set_xlim(-3, 3)
    ax0.set_ylim(-4, 3)
    for l, c, m in zip([a,b,c], ['r', 'g', 'b'], ['s', 'x', 'o']):
        ax0.scatter(l[:,0], l[:,1],
                    c=c, marker=m, label=l, edgecolors='black')
    ax0.legend(loc='upper right')
    plt.show()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    dp = LDA(n_components=2).fit_transform(new_dataset, labels.reshape(-1))
    print(dp[1])