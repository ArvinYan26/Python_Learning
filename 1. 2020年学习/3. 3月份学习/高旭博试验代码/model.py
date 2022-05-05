import numpy as np


class SilvaModel:

    def __init__(self,
                 A: np.ndarray, K: int,
                 lambd: float = 0.6,
                 Delta: float = 0.07,
                 epsilon: float = 0.05,
                 omega: (float, float) = (0, 1)):

        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        assert K > 1
        assert lambd >= 0
        assert Delta >= 0
        assert epsilon >= 0
        assert len(omega) == 2 and omega[0] < omega[1]

        self._A = A
        self._V = A.shape[0]  # number of vertices
        self._K = K
        self._lambd = lambd
        self._Delta = Delta
        self._epsilon = epsilon
        self._omega = omega

        # Initialize the position of each particle.
        self._p = np.random.randint(self._V, size=self._K)
        # print(self._p)

        self._Prand = A / A.sum(axis=1, keepdims=True)
        # print(self._Prand)

        self._N = np.ones((self._V, self._K))
        for k in range(self._K):
            self._N[self._p[k], k] = 2
        # print(self._N)

        # Calculate the dominance value.
        self._Nbar = self._N / self._N.sum(axis=1, keepdims=True)
        # print(self._Nbar)

        self._Nbar_diff_norm = np.inf

        self._E = np.full(self._K, self._omega[0] +
                          (self._omega[1] - self._omega[0]) / self._K)
        # print(self._E)

    def iterate(self):

        next_position = np.zeros(self._K, dtype=int)
        # print(self._E)

        for k in range(self._K):
            Ppref = 1.0 * self._A
            for j in range(self._V):
                Ppref[:, j] = Ppref[:, j] * self._Nbar[j, k]
            Ppref = np.copy(Ppref) / Ppref.sum(axis=1, keepdims=True)

            Prean = 1.0 * np.zeros((self._V, self._V))
            # XXX: this is an implementation detail. It is not described in the
            # paper.
            dominated = np.where(
                self._Nbar[:, k] == np.max(self._Nbar, axis=1))
            for u in dominated:
                Prean[:, u] = 1.0
            Prean = np.copy(Prean) / Prean.sum(axis=1, keepdims=True)

            S = 0 if self._E[k] > self._omega[0] else 1

            P = (1 - S) * (self._lambd * Ppref + (1 - self._lambd) *
                           self._Prand) + S * Prean

            next_position[k] = np.random.choice(self._V, p=P[self._p[k], :])
            # print("particle {} went from {} to {}.".format(k, self._p[k],
            #                                                next_position[k]))

        for k in range(self._K):
            self._N[next_position[k], k] = self._N[next_position[k], k] + 1
            self._p[k] = next_position[k]

            is_dominated = self._Nbar[next_position[k], k] == np.max(
                self._Nbar[next_position[k], :])

            self._E[k] = np.clip(self._E[k] +
                                 (1.0 if is_dominated else -
                                  1.0) *
                                 self._Delta, self._omega[0], self._omega[1])

        next_Nbar = self._N / self._N.sum(axis=1, keepdims=True)
        self._Nbar_diff_norm = np.linalg.norm(next_Nbar - self._Nbar,
                                              ord=np.inf)
        self._Nbar = next_Nbar

        # print(self._Nbar_diff_norm)

    def has_converged(self) -> bool:
        return self._Nbar_diff_norm < self._epsilon

    def result(self) -> np.ndarray:
        return np.copy(self._Nbar)


if __name__ == "__main__":

    A = np.array(
        [[0, 1, 1, 0, 0, 0],
         [1, 0, 1, 0, 0, 0],
         [1, 1, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 1],
         [0, 0, 0, 1, 0, 1],
         [0, 0, 0, 1, 1, 0]])

    model = SilvaModel(A, 2)

    for t in range(100):
        model.iterate()

    while not model.has_converged():
        model.iterate()

    print(model.result())
    print(np.argmax(model.result(), axis=1))