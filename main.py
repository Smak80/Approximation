from mpl_toolkits.mplot3d import Axes3D
import dataloader as dl
from MLPerceptron import MLP
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
pi = 3.141592653589793
def mesh_data(x, y, z):
    cnt = 1
    while (cnt < len(x) and x[cnt]==x[0]):
        cnt+=1
    x = np.array([x[i] for i in range(0, len(x), cnt)])
    y = np.array([y[i] for i in range(cnt)])
    nx, ny = np.meshgrid(x, y)
    nz = np.reshape(z, (len(x), len(y)))
    return nx, ny, nz

def plot2d():
    ld = dl.loader(dimensions=2)
    tsi = ld.getTestInp()
    tso = ld.getTestOut()
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    mlp = MLP(tri, tro, (20,25), epoches=2500)
    plt.plot(tri, tro, "r+")
    out = mlp.calc(tri)
    plt.plot(tri, out, "b-")
    out = mlp.calc(tsi)
    plt.plot(tsi, out, "go")
    plt.show()

def plot3d():
    ld = dl.loader(dimensions=3)
    ai = ld.getAllInp()
    ao = ld.getAllOut()
    x, y, z = mesh_data(ai.T[0], ai.T[1], ao.T[0])
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    axes.scatter(x, y, z, color="#bbff8016")
    mlp = MLP(ai, ao, (25,), epoches=1000)
    out = mlp.calc(ai)
    x, y, z = mesh_data(ai.T[0], ai.T[1], out)
    axes.plot_surface(x, y, z, cmap=cm.get_cmap("coolwarm"),
                      linewidth=0, antialiased=True)
    plt.show()


if __name__=="__main__":
    plot2d()




