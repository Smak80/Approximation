import numpy as np

class MLP:
    __a = 1
    __b = 1
    __eta = 0.005

    def __init__(self,
                 inp: list,
                 out: list,
                 neuronNum: tuple = (3, 2),
                 epoches = 1000,
                 epsilon = 0.002):
        self.__max_epoches = epoches
        self.__epsilon = epsilon
        self.__layers = len(neuronNum)+2
        iln = len(inp[0])
        oln = len(out[0])
        nN = [iln, oln]
        self.__nN = np.insert(nN, 1, neuronNum)
        self.__inp = np.array(inp)
        self.__out = np.array(out)
        self.__w = [None for i in range(self.__layers - 1)]
        for i in range(self.__layers - 1):
            self.__w[i] = np.random.rand(
                self.__nN[i] + 1,
                self.__nN[i+1] +
                (0 if i == self.__layers - 2 else 1)
            )
        self.learn()

    def nonLinAct(self, x):
        return np.array(self.__a * np.tanh(self.__b * x))

    def nonLinActDer(self, x):
        return np.array(self.__b / self.__a * \
                        (self.__a - self.nonLinAct(x)) *\
                        (self.__a + self.nonLinAct(x)))

    def linAct(self,x):
        return np.array(x)

    def linActDer(self, x):
        return np.array(1)

    def learn(self):
        l = np.array(
            [None for i in range(self.__layers)]
        )
        l_err = np.array(
            [None for i in range(1, self.__layers)]
        )
        l_delta = np.array(
            [None for i in range(1, self.__layers)]
        )
        inp = self.__inp
        out = self.__out
        k = 0
        err_n = self.__epsilon+100
        while k<self.__max_epoches and err_n>self.__epsilon:
            err_n = 0
            k+=1
            for i in range(len(inp)):
                l[0] = np.insert(inp[i], 0, 1)
                l[0] = np.array([l[0]])
                for j in range(1, self.__layers - 1):
                    l[j] = self.nonLinAct(
                        np.dot(l[j-1], self.__w[j-1])
                    )
                l[self.__layers-1] = self.linAct(
                    np.dot(l[self.__layers-2],
                           self.__w[self.__layers-2])
                )
                l_err[self.__layers-2] = out[i] - l[self.__layers-1]
                err_n +=  0.5*(out[i] - l[self.__layers-1])**2
                l_delta[self.__layers-2] = \
                l_err[self.__layers-2]*(
                    self.linActDer(l[self.__layers - 1])
                )
                for j in range(self.__layers - 2, 0, - 1):
                    l_err[j-1] = np.dot(l_delta[j], self.__w[j].T)
                    l_delta[j-1] = l_err[j-1] * self.nonLinActDer(l[j])
                deltaW = [None for i in range(self.__layers-1)]
                for j in range(self.__layers - 2, -1, -1):
                    deltaW[j] = self.__eta*np.dot(l_delta[j].T, l[j])
                    self.__w[j] += deltaW[j].T
            err_n /= len(inp)
            print("Epoche", k, "Error=", err_n)

    def calc(self, inps: list):
        ys = np.array([])
        for i in range(len(inps)):
            inp  = np.insert(inps[i], 0, 1)
            inp = np.array([inp])
            hl = len(self.__nN)-2
            for lr in range(hl):
                inp = self.nonLinAct(
                    np.dot(inp, self.__w[lr])
                )
            ys = np.append(
                ys,
                self.linAct(np.dot(inp, self.__w[hl]))
            )
        return ys