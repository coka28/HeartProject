import numpy as np
from numpy.random import random, normal, permutation
from scipy.special import gammaln, multigammaln
import pickle

import matplotlib.pyplot as plt

np.warnings.filterwarnings("ignore")


# import dataset and create feature vectors
with open("heart.csv", "r") as csvfile:
    lines = csvfile.read().splitlines()
    hdr = lines[0].split(",")
    fields = [ln.split(",") for ln in lines[1:]]


# feature                                                               # scale

age = [int(f[0]) for f in fields]                                       # cardinal
sex = [0 if f[1]=="F" else 1 for f in fields]                           # nominal
pain = [{"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}[f[2]] for f in fields]  # nominal
rbp = [int(f[3]) for f in fields]                                       # cardinal
cho = [int(f[4]) for f in fields]                                       # cardinal
fbs = [0 if f[5]=="0" else 1 for f in fields]                           # nominal
ecg = [{"LVH": 0, "Normal": 1, "ST": 2}[f[6]] for f in fields]          # nominal
mhr = [int(f[7]) for f in fields]                                       # cardinal
ean = [1 if f[8]=="Y" else 0 for f in fields]                           # nominal
olp = [float(f[9]) for f in fields]                                     # cardinal
sts = [{"Down": 0, "Flat": 1, "Up":2}[f[10]] for f in fields]           # nominal

dis = [int(f[11]) for f in fields]                                      # nominal

# nominal = 0; cardinal = 1
cat = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]


X = np.array([
        age, sex, pain, rbp, cho, fbs, ecg, mhr, ean, olp, sts
    ]).T

y = np.array([dis]).T

# impute missing values with mean
X[X[:, 3]==0, 3] = X[X[:, 3] != 0, 3].mean()
X[X[:, 4]==0, 4] = X[X[:, 4] != 0, 4].mean()

# continuous features get standardized & transformed by sigmoid function into range [0, 1]
# k-nary features get encoded by ceil(ld(k)) binary features
X2 = np.concatenate((
    [1 / (1 + np.exp(-(X[:, i] - X[:, i].mean()) / X[:, i].std())) for i in [0, 3, 4, 7, 9]],
    np.array([(X[:, 6] // 2 == 1).astype("float64"), (X[:, 6] % 2 == 1).astype("float64")]),
    np.array([(X[:, 10] // 2 == 1).astype("float64"), (X[:, 10] % 2 == 1).astype("float64")]),
    np.array([(X[:, 2] // 2 == 1).astype("float64"), (X[:, 2] % 2 == 1).astype("float64")]),
    np.array([X[:, 1], X[:, 5], X[:, 8]])), axis=0).T


def test_train_split(X, ratio=0.8):
    N = np.arange(len(X))
    test = np.random.permutation(N)[int(len(X)*ratio):]
    train = np.delete(N, test)

    return train, test


class RBM:
    """
    class for a restricted boltzmann machine.
    output nodes may have sigmoidal or linear activation functions.
    input vectors are assumed to be scaled between 0 and 1.
    """

    _f = {
            "sigmoid": lambda x, wt, b : 1 / (1 + np.exp(-x@wt-b)),
            "linear": lambda x, wt, b : x@wt+b,
        }

    _p = {
            "sigmoid": lambda x : random(x.shape) < x,
            "linear": lambda x : x + normal(size=x.shape)
        }

    def __init__(self, nvis, ncont, nhid, fout="sigmoid"):
        self.nvis = nvis
        self.cont = ncont
        self.nhid = nhid

        self.wt = normal(scale=1e-2, size=(nvis, nhid))
        self.vb = np.zeros((1, nvis))
        self.hb = np.zeros((1, nhid))

        self.hf = RBM._f[fout]
        self.hp = RBM._p[fout]
        

    def train(self, X, k=1, eta=1e-3, eta_decay=1000, nr_epoch=1000, mb_size=50, L2_regul=1e-3):
        n, m = X.shape
        X = X.copy()
        f_eta = 1 - np.log(eta_decay) / nr_epoch

        print("| 0 %" + " "*90 + "100 % |\n|", end="")
        
        for i in range(nr_epoch):
            ind = permutation(np.arange(n))
            mini_batches = [ind[g: g+mb_size] for g in range(0, n-mb_size-n%mb_size, mb_size)]

            for mb in mini_batches:
                v = v0 = X[mb]
                p = p0 = self.hf(v0, self.wt, self.hb)

                for j in range(k):
                    v = 1 / (1 + np.exp(-p @ self.wt.T - self.vb))
                    p = self.hf(v, self.wt, self.hb)
                    h = self.hp(p)
                
                v = 1 / (1 + np.exp(-p @ self.wt.T - self.vb))
                v[:, :self.cont] = np.log((1e-3 + v[:, :self.cont]) / (1 + 1e-3 - v[:, :self.cont]))
                v0[:, :self.cont] = np.log((1e-3 + v0[:, :self.cont]) / (1 + 1e-3 - v0[:, :self.cont]))

                self.wt += eta * ((v0.T @ p0 - v.T @ p) / mb_size - L2_regul * np.mean(self.wt**2))
                self.vb += eta * (v0 - v).mean(axis=0)
                self.hb += eta * (p0 - p).mean(axis=0)

            eta *= f_eta

            if not i%(nr_epoch//100):
                print("-", end="")

        print("|")
        
    def hidp(self, X):
        return self.hf(X, self.wt, self.hb)

    def infer(self, X, indices=(), k=100):
        v = X.copy()
        ind = np.array(indices)
        v[:, ind] = 0.5
        p = self.hf(v, self.wt, self.hb)

        for i in range(k):
            v[:, ind] = 1 / (1 + np.exp(-p @ self.wt.T - self.vb))[:, ind]
            p = self.hf(v, self.wt, self.hb)
            h = self.hp(p)
        
        v[:, ind] = 1 / (1 + np.exp(-p @ self.wt.T - self.vb))[:, ind]

        return v[:, ind]

# function references for pickling autoencoder object
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x

def dsigmoid(x):
    return np.exp(-x - 2*np.log1p(np.exp(-x)))

def dlinear(x):
    return 1

# having a function at index 0 (input layer) makes an if statement in line 233 unnecessary
def _dummy(x):
    return 0

class AutoEncoder:
    """
    class for an autoencoder, pretraining weights layer-by-layer using RBMs, with subsequent
    fine-tuning using stochastic gradient descent on least-residual squares objective.
    coding layer has linear activation function, all other nodes have sigmoidal activation
    functions. 
    """

    def __init__(self):
        self.wt = None
        self.f = None
        self.df = None
        
    def pre_train(self, Xtrain, ninp, layout, L2_regul=1e-3, k=1, nr_epoch=1000, eta=1e-1):
        """
        pre-train RBMs with set nr of hidden nodes.
        layout must be a tuple with the nr of hidden layers for each layer.
        ninp is the number of visible input nodes to the RBM (could also be inferred from
        shape of Xtrain matrix).
        """
        rbms = [RBM(ninp, 5, layout[0])] + \
               [RBM(L1, 0, L2) for L1, L2 in zip(layout[:-2], layout[1:-1])] + \
               [RBM(layout[-2], 0, layout[-1], "linear")]
        x = Xtrain
        for rbm in rbms:
            rbm.train(x, k=k, eta=eta, nr_epoch=nr_epoch, L2_regul=L2_regul)
            x = rbm.hidp(x)

        self.wt = [np.vstack((rbm.wt, rbm.hb)) for rbm in rbms]
        self.f = [_dummy] + [sigmoid for _ in rbms[:-1]] + [linear]
        self.df = [_dummy] + [dsigmoid for _ in rbms[:-1]] + [dlinear]
        for i in range(len(self.wt)-1, -1, -1):
            self.wt.append(np.vstack((self.wt[i][:-1].T, rbms[i].vb)))
            self.f.append(sigmoid)
            self.df.append(dsigmoid)
            

    def fine_tune(self, Xtrain, eta=1e-2, nr_epoch=1000, mb_size=50, eta_decay=100, L2_regul=0, L2_regul_rise=0):
        """
        fine-tuning using stochastic gradient descent on minibatches of size mb_size.
        eta-decay is an exponential decrease, after nr_epoch epochs, the learning rate eta will
        exponentially drop from eta0 to eta0 / eta_decay.
        L2-regularization can be linearly ramped up/down using start value L2_regul and end value
        L2_regul_rise.
        """
        
        n, m = Xtrain.shape
        f_eta = 1 - np.log(eta_decay) / nr_epoch
        d_L2 = (L2_regul_rise - L2_regul) / nr_epoch

        x = [np.zeros((mb_size, m))] + [np.zeros((mb_size, wt.shape[1])) for wt in self.wt]
        y = [np.ones((mb_size, m+1))] + [np.ones((mb_size, wt.shape[1]+1)) for wt in self.wt]
        e = [np.zeros((mb_size, m))] + [np.zeros((mb_size, wt.shape[1])) for wt in self.wt]

        for i in range(nr_epoch):
            ind = permutation(np.arange(n))
            mini_batches = [ind[g: g+mb_size] for g in range(0, n-mb_size-n%mb_size, mb_size)]

            for mb in mini_batches:
                y[0][:, :-1] = Xtrain[mb]

                for j, wt in enumerate(self.wt):
                    x[j+1][:] = y[j] @ wt
                    y[j+1][:, :-1] = self.f[j+1](x[j+1])

                e[-1][:] = self.df[-1](x[-1]) * (y[-1][:, :-1] - y[0][:, :-1])
                for j in range(len(self.wt)-1, -1, -1):
                    e[j][:] = self.df[j](x[j]) * (e[j+1][:, None, :] * self.wt[j][None, :-1]).sum(axis=2)
                    self.wt[j] -= eta * (np.mean(e[j+1][:, None, :] * y[j][:, :, None], axis=0) + L2_regul * self.wt[j])
            
            eta *= f_eta
            L2_regul += d_L2

            if not i%(nr_epoch//100):
                print("%d / 100  \t" % (i*100 // nr_epoch), np.sum(abs(self.reconstruct(Xtrain) - Xtrain)))

    def reconstruct(self, X):
        # reconstruct input vector in output layer. useful for measuring the ability
        # of the encoder to pass the data through the code-layer bottleneck.
        x = X

        for f, wt in zip(self.f[1:], self.wt):
            x = f(np.hstack((x, np.ones((X.shape[0], 1)))) @ wt)

        return x

    def encode(self, X):
        # get encoded input vectors
        x = X

        for f, wt in zip(self.f[1:len(self.wt)//2+1], self.wt[:len(self.wt)//2+1]):
            x = f(np.hstack((x, np.ones((X.shape[0], 1)))) @ wt)

        return x

    def save_to_file(self, filename):
        with open(filename, "wb") as pfile:
            pickle.dump((self.wt, self.f, self.df), pfile)

    @staticmethod
    def create_from_file(filename):
        with open(filename, "rb") as pfile:
            wt, f, df = pickle.load(pfile)
        ae = AutoEncoder()
        ae.wt = wt
        ae.f = f
        ae.df = df

        return ae
        

class BayesianHierarchicalClustering:

    """
    BHC was proposed in 2005 by Katherine A. Heller and Zoubin Ghahramani.
    It splits the data vectors into an unspecified nr of clusters, using bayesian
    probability as a metric to greedily and hierarchically merge sub-trees, starting
    from individual points.
    
    template for implementation comes from
    https://github.com/caponetto/bhc

    resources:
        [BHC] http://mlg.eng.cam.ac.uk/zoubin/papers/icml05heller.pdf
        [GCP] https://thaines.com/content/misc/gaussian_conjugate_prior_cheat_sheet.pdf
    """

    def __init__(self):
        pass

    @staticmethod
    def _log_d(alpha, n_pts, log_dk):
        # [BHC] fig. 3
        if n_pts == 1 and log_dk is None:
            return np.log(alpha)
        else:
            dk_t1 = np.log(alpha) + gammaln(n_pts)
            dk_t2 = log_dk
            a = np.maximum(dk_t1, dk_t2)
            b = np.minimum(dk_t1, dk_t2)
            return a + np.log(1 + np.exp(b - a))    # logsumexp to avoid overflow

    @staticmethod
    def _mlh(data, smat0, scale0, deg0, mu0, log_prior0):
        # maximum likelihood estimate of mu, cov; using inverse wishart prior [GCP] eqns 7-11
        data = np.atleast_2d(data)
        n, d = data.shape
        mean = np.mean(data, axis=0)
        scale = scale0 + n
        deg = deg0 + n
        smat = np.nan_to_num(np.cov((data-mean[None]).T), nan=0) * (n-1)
        dt = (mean - mu0)[None]
        smat += smat0 + scale0 * deg0 / scale * np.dot(dt.T, dt)
        log_prior = np.log(2) * deg * d / 2 + np.log(2 * np.pi / scale) * d / 2 + \
                    multigammaln(deg / 2, d) - np.log(np.linalg.det(smat)) * deg / 2
        mu = (scale0 * mu0 + n * mean) / (scale0 + n)
        return log_prior - log_prior0 - np.log(2*np.pi) * n * d / 2, mu, smat

    def create(self, points, alpha, scale, g, label):
        N, m = points.shape
        mu0 = np.mean(points, axis=0)
        smat0 = np.cov(points.T) / g
        deg0 = m + 1
        log_prior0 = np.log(2) * deg0 * m / 2 + m / 2 * np.log(2 * np.pi / scale) + \
                     multigammaln(deg0 / 2, m) - deg0 / 2 * np.log(np.linalg.det(smat0))

        n = np.ones(N)      # nr per cluster
        c = np.arange(N)    # cluster membership
        pt = np.arange(N)   # list of indices of clusters/points not merged yet

        log_d = np.log(alpha)[None].repeat(N)
        log_p = np.array([self._mlh(pt, smat0, scale, deg0, mu0, log_prior0)[0] for pt in points])

        merge_info = np.zeros((0, 5))
        merge_score = np.zeros(N)
        n_label_pos = np.array(label).flatten()
        n_label_neg = np.array(1 - label).flatten()
        merge_n = np.ones(N, dtype="int64")
        merge_cov = (smat0 * g)[None].repeat(N, axis=0)
        merge_mu = points.copy()
        merge_children = [() for _ in range(N)]


        for i in range(N-1):
            for j in range(i+1, N):
                nk = n[i] + n[j]
                log_dij = log_d[i] + log_d[j]
                log_dk = self._log_d(alpha, nk, log_dij)
                log_pik = np.log(alpha) + gammaln(nk) - log_dk
                merge = np.vstack((points[i], points[j]))
                log_pk, _, _ = self._mlh(merge, smat0, scale, deg0, mu0, log_prior0)
                log_r1 = log_pik + log_pk                           # [BHC] eqn 3
                log_r2 = - log_dk + log_dij + log_p[i] + log_p[j]   # 
                log_r = log_r1 - log_r2                             #
                merge_score[i] = log_d[i] + log_p[i]
                merge_score[j] = log_d[j] + log_p[j]
                merge_info = np.vstack((merge_info, [i, j, log_r, log_r1, log_r2]))

        while pt.size > 1:
            max_log_rk = np.max(merge_info[:, 2])
            pos = np.min(np.argwhere(merge_info[:, 2] == max_log_rk))
            i, j, log_rk, log_r1, log_r2 = merge_info[pos]
            i, j = int(i), int(j)

            pos = (merge_info[:, 0] == i) | (merge_info[:, 1] == i) | \
                  (merge_info[:, 0] == j) | (merge_info[:, 1] == j)
            merge_info[pos, 2] = -np.inf

            k = n.size
            nk = n[i] + n[j]
            n = np.append(n, nk)

            merge_n = np.append(merge_n, nk)
            n_label_pos = np.append(n_label_pos, n_label_pos[i]+n_label_pos[j])
            n_label_neg = np.append(n_label_neg, n_label_neg[i]+n_label_neg[j])
            merge_score = np.append(merge_score, log_rk)
            merge_children.append((i, j))
            
            log_dij = log_d[i] + log_d[j]
            log_dk = self._log_d(alpha, nk, log_dij)
            log_d = np.append(log_d, log_dk)
            c[(c==i)|(c==j)] = k

            pt = pt[~((pt==i)|(pt==j))]
            pt = np.append(pt, k)

            t1 = np.maximum(log_r1, log_r2)
            t2 = np.minimum(log_r1, log_r2)
            log_pk = t1 + np.log(1 + np.exp(t2 - t1))       # logsumexp
            log_p = np.append(log_p, log_pk)

            pts_k = points[c==k]

            # recalculate mle for current merge to store params
            _, mu, cov = self._mlh(pts_k, smat0, scale, deg0, mu0, log_prior0)
            merge_mu = np.vstack((merge_mu, mu))
            merge_cov = np.append(merge_cov, cov[None], axis=0)
            
            for i in range(pt.size - 1):
                nk = n[i] + n[k]
                log_dik = log_d[i] + log_d[k]
                log_dh = self._log_d(alpha, nk, log_dik)
                log_pih = np.log(alpha) + gammaln(nk) - log_dh
                merge = np.vstack((pts_k, points[c==pt[i]]))
                log_ph, _, _ = self._mlh(merge, smat0, scale, deg0, mu0, log_prior0)
                log_p_ik = log_p[k] + log_p[pt[i]]
                log_r1 = log_pih + log_ph
                log_r2 = log_dik - log_dh + log_p_ik
                log_r = log_r1 - log_r2
                merge_info = np.vstack((merge_info, [k, pt[i], log_r, log_r1, log_r2]))

        self.merge_score = merge_score - merge_score.max()
        self.mu = merge_mu
        self.cov = merge_cov
        self.merge_n = merge_n
        self.label_pos = n_label_pos
        self.label_neg = n_label_neg
        self.merge_children = merge_children
        self.log_prior = np.zeros(self.mu.shape[0])
        self.log_y1_freq = np.log((self.label_pos + 1) / (self.label_pos + self.label_neg + 2))
        self.log_y0_freq = np.log((self.label_neg + 1) / (self.label_pos + self.label_neg + 2))

        def rec_prior(k, children):
            if len(children) == 0: return 1
            l, r = children
            lcount = rec_prior(l, self.merge_children[l])
            rcount = rec_prior(r, self.merge_children[r])
            self.log_prior[l] = self.merge_score[l] - np.log(lcount/N)
            self.log_prior[r] = self.merge_score[r] - np.log(rcount/N)
            return lcount + rcount

        self.log_prior[-1] = rec_prior(len(self.mu)-1, self.merge_children[-1])
            

    def predict(self, data):
        # calculate probability p(y=0) and p(y=1) using prior p(c=i), pdf P(x | c=i, theta[i]),
        # and rate of diseased patients within each cluster:
        # p(y=1) = Sum(p(y=1|c=i) * p(x|c=i, theta[i]) * p(c=i), i in clusters)
        # and vice versa for p(y=0).
        # probs are unnormalized, but knowing both, we can normalize by p(y=1) / (p(y=1) + p(y=0))
        n, m = data.shape
        k = len(self.mu)
        
        # calc pdf for each point for each gaussian
        dµ = data[:, None] - self.mu[None, :]
        mahal = ((dµ[..., None, :] @ self.cov) * dµ[..., None, :]).sum(axis=-1)[..., 0]
        log_pdf = -np.log(2*np.pi) * (m/2) - 0.5 * np.log(np.linalg.det(self.cov)[None]) - 0.5 * mahal
        py1 = self.log_y1_freq[None, :] + log_pdf + self.log_prior[None, :]
        py1 = np.exp(np.max(py1, axis=1)) * np.sum(np.exp(py1-np.max(py1, axis=1)[:, None]), axis=1)
        py0 = self.log_y0_freq[None, :] + log_pdf + self.log_prior[None, :]
        py0 = np.exp(np.max(py0, axis=1)) * np.sum(np.exp(py0-np.max(py0, axis=1)[:, None]), axis=1)
        return py1 / (py0 + py1)


class BayesianHierarchicalClustering2:

    """
    BHC2 creates two set of clusters; one for each class. Not sure whether this works better
    in any circumstances.

    BHC was proposed in 2005 by Katherine A. Heller and Zoubin Ghahramani.
    It splits the data vectors into an unspecified nr of clusters, using bayesian
    probability as a metric to greedily and hierarchically merge sub-trees, starting
    from individual points.
    
    template for implementation comes from
    https://github.com/caponetto/bhc

    resources:
        [BHC] http://mlg.eng.cam.ac.uk/zoubin/papers/icml05heller.pdf
        [GCP] https://thaines.com/content/misc/gaussian_conjugate_prior_cheat_sheet.pdf
    """

    def __init__(self):
        pass

    @staticmethod
    def _log_d(alpha, n_pts, log_dk):
        # [BHC] fig. 3
        if n_pts == 1 and log_dk is None:
            return np.log(alpha)
        else:
            dk_t1 = np.log(alpha) + gammaln(n_pts)
            dk_t2 = log_dk
            a = np.maximum(dk_t1, dk_t2)
            b = np.minimum(dk_t1, dk_t2)
            return a + np.log(1 + np.exp(b - a))    # logsumexp to avoid overflow

    @staticmethod
    def _mlh(data, smat0, scale0, deg0, mu0, log_prior0):
        # maximum likelihood estimate of mu, cov; using inverse wishart prior [GCP] eqns 7-11
        data = np.atleast_2d(data)
        n, d = data.shape
        mean = np.mean(data, axis=0)
        scale = scale0 + n
        deg = deg0 + n
        smat = np.nan_to_num(np.cov((data).T), nan=0) * (n-1)
        dt = (mean - mu0)[None]
        smat += smat0 + scale0 * deg0 / scale * np.dot(dt.T, dt)
        log_prior = np.log(2) * deg * d / 2 + np.log(2 * np.pi / scale) * d / 2 + \
                    multigammaln(deg / 2, d) - np.log(np.linalg.det(smat)) * deg / 2
        mu = (scale0 * mu0 + n * mean) / (scale0 + n)
        return log_prior - log_prior0 - np.log(2*np.pi) * n * d / 2, mu, smat

    def _create(self, points, alpha, scale, g):
        N, m = points.shape
        mu0 = np.mean(points, axis=0)
        smat0 = np.cov(points.T) / g
        deg0 = m + 1
        log_prior0 = np.log(2) * deg0 * m / 2 + m / 2 * np.log(2 * np.pi / scale) + \
                     multigammaln(deg0 / 2, m) - deg0 / 2 * np.log(np.linalg.det(smat0))

        n = np.ones(N)      # nr per cluster
        c = np.arange(N)    # cluster membership
        pt = np.arange(N)   # list of indices of clusters/points not merged yet

        log_d = np.log(alpha)[None].repeat(N)
        log_p = np.array([self._mlh(pt, smat0, scale, deg0, mu0, log_prior0)[0] for pt in points])

        merge_info = np.zeros((0, 5))
        merge_score = np.zeros(N)
        merge_n = np.ones(N)
        merge_cov = (smat0 * g)[None].repeat(N, axis=0)
        merge_mu = points.copy()
        merge_children = [() for _ in range(N)]


        for i in range(N-1):
            for j in range(i+1, N):
                nk = n[i] + n[j]
                log_dij = log_d[i] + log_d[j]
                log_dk = self._log_d(alpha, nk, log_dij)
                log_pik = np.log(alpha) + gammaln(nk) - log_dk
                merge = np.vstack((points[i], points[j]))
                log_pk, _, _ = self._mlh(merge, smat0, scale, deg0, mu0, log_prior0)
                log_r1 = log_pik + log_pk                           # [BHC] eqn 3
                log_r2 = - log_dk + log_dij + log_p[i] + log_p[j]   # 
                log_r = log_r1 - log_r2                             #
                merge_score[i] = log_d[i] + log_p[i]
                merge_score[j] = log_d[j] + log_p[j]
                merge_info = np.vstack((merge_info, [i, j, log_r, log_r1, log_r2]))

        while pt.size > 1:
            max_log_rk = np.max(merge_info[:, 2])
            pos = np.min(np.argwhere(merge_info[:, 2] == max_log_rk))
            i, j, log_rk, log_r1, log_r2 = merge_info[pos]
            i, j = int(i), int(j)

            pos = (merge_info[:, 0] == i) | (merge_info[:, 1] == i) | \
                  (merge_info[:, 0] == j) | (merge_info[:, 1] == j)
            merge_info[pos, 2] = -np.inf

            k = n.size
            nk = n[i] + n[j]
            n = np.append(n, nk)

            merge_n = np.append(merge_n, nk)
            merge_score = np.append(merge_score, log_rk)
            merge_children.append((i, j))
            
            log_dij = log_d[i] + log_d[j]
            log_dk = self._log_d(alpha, nk, log_dij)
            log_d = np.append(log_d, log_dk)
            c[(c==i)|(c==j)] = k

            pt = pt[~((pt==i)|(pt==j))]
            pt = np.append(pt, k)

            t1 = np.maximum(log_r1, log_r2)
            t2 = np.minimum(log_r1, log_r2)
            log_pk = t1 + np.log(1 + np.exp(t2 - t1))       # logsumexp
            log_p = np.append(log_p, log_pk)

            pts_k = points[c==k]

            # recalculate mle for current merge to store params
            _, mu, cov = self._mlh(pts_k, smat0, scale, deg0, mu0, log_prior0)
            merge_mu = np.vstack((merge_mu, mu))
            merge_cov = np.append(merge_cov, cov[None], axis=0)
            
            for i in range(pt.size - 1):
                nk = n[i] + n[k]
                log_dik = log_d[i] + log_d[k]
                log_dh = self._log_d(alpha, nk, log_dik)
                log_pih = np.log(alpha) + gammaln(nk) - log_dh
                merge = np.vstack((pts_k, points[c==pt[i]]))
                log_ph, _, _ = self._mlh(merge, smat0, scale, deg0, mu0, log_prior0)
                log_p_ik = log_p[k] + log_p[pt[i]]
                log_r1 = log_pih + log_ph
                log_r2 = log_dik - log_dh + log_p_ik
                log_r = log_r1 - log_r2
                merge_info = np.vstack((merge_info, [k, pt[i], log_r, log_r1, log_r2]))

        merge_score -= merge_score.max()
        log_prior = np.zeros(merge_mu.shape[0])

        def rec_prior(k, children):
            if len(children) == 0: return 1
            l, r = children
            lcount = rec_prior(l, merge_children[l])
            rcount = rec_prior(r, merge_children[r])
            log_prior[l] = merge_score[l] - np.log(lcount/N)
            log_prior[r] = merge_score[r] - np.log(rcount/N)
            return lcount + rcount

        log_prior[-1] = rec_prior(len(merge_mu)-1, merge_children[-1])
        return merge_n, merge_mu, merge_cov, log_prior

    def create(self, points, alpha, scale, g, labels):
        xpos = points[labels.flatten() == 1]
        xneg = points[labels.flatten() == 0]

        pos_n, self.pos_mu, self.pos_cov, self.pos_log_prior = self._create(xpos, alpha, scale, g)
        neg_n, self.neg_mu, self.neg_cov, self.neg_log_prior = self._create(xneg, alpha, scale, g)

    def predict(self, data):
        # calculate probability p(y=0) and p(y=1) using prior p(c=i), pdf P(x | c=i, theta[i]),
        # and rate of diseased patients within each cluster:
        # p(y=1) = Sum(p(y=1|c=i) * p(x|c=i, theta[i]) * p(c=i), i in clusters)
        # and vice versa for p(y=0).
        # probs are unnormalized, but knowing both, we can normalize by p(y=1) / (p(y=1) + p(y=0))
        n, m = data.shape
        
        # calc pdf for each point for each gaussian
        dµ_pos = data[:, None] - self.pos_mu[None, :]
        mahal_pos = ((dµ_pos[..., None, :] @ self.pos_cov) * dµ_pos[..., None, :]).sum(axis=-1)[..., 0]
        log_pdf_pos = -np.log(2*np.pi) * (m/2) - 0.5 * np.log(np.linalg.det(self.pos_cov)[None]) - 0.5 * mahal_pos
        dµ_neg = data[:, None] - self.neg_mu[None, :]
        mahal_neg = ((dµ_neg[..., None, :] @ self.neg_cov) * dµ_neg[..., None, :]).sum(axis=-1)[..., 0]
        log_pdf_neg = -np.log(2*np.pi) * (m/2) - 0.5 * np.log(np.linalg.det(self.neg_cov)[None]) - 0.5 * mahal_neg
        
        py1 = log_pdf_pos + self.pos_log_prior[None, :]
        py1 = np.exp(np.max(py1, axis=1)) * np.sum(np.exp(py1-np.max(py1, axis=1)[:, None]), axis=1)
        py0 = log_pdf_neg + self.neg_log_prior[None, :]
        py0 = np.exp(np.max(py0, axis=1)) * np.sum(np.exp(py0-np.max(py0, axis=1)[:, None]), axis=1)
        return py1 / (py0 + py1)


"""
example for using the classes and functions in this script:

if True:
	train, test = test_train_split(X2, 0.75)
	Xtrain = X2[train]
	Xtest = X2[test]
	ae = AutoEncoder()
	ae.pre_train(Xtrain, 14, (80, 200, 200, 6), L2_regul=1e-2, k=1, nr_epoch=1500, eta=2e-2)
	ae.fine_tune(Xtrain, eta=1e-1, nr_epoch=5000, mb_size=50, eta_decay=10, L2_regul=0, L2_regul_rise=1e-3)

	Ytrain = y[train]
	Ztrain = ae.encode(Xtrain)
	Ytest = y[test]
	Ztest = ae.encode(Xtest)

	bhc = BayesianHierarchicalClustering()
	bhc.create(Ztrain, 1, 2, 4, Ytrain)

	# ^p(y = 1)
	print("accuracy on training set: \t%.3f" % (100*bhc.predict(Ztrain)))
	print("accuracy on test set:     \t%.3f" % (100*bhc.predict(Ztest)))

"""
