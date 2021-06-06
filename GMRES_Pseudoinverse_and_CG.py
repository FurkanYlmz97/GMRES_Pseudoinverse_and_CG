import numpy as np
# import cupy as np
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
# from cupy import linalg as LA


def conjugate_grad(A, b, N=None):

    if N is None:
        N = A.shape[1]

    r = b
    d = r
    x = []
    x.append(np.zeros(len(b)))
    for k in range(N):

        alfa = (np.dot(r, r)) / (np.dot(np.dot(d, A), d))
        x.append(x[len(x)-1] + alfa*d)
        rnew = r - alfa * np.dot(A, d)
        beta = (np.dot(rnew, rnew)) / (np.dot(r, r))
        r = rnew
        d = r + beta*d
    return np.array(x)


def conjugate_grad_90(A, b, N=None):

    if N is None:
        N = A.shape[1]

    r = b
    d = r
    x = []
    q = []
    x.append(np.zeros(len(b)))
    for k in range(N):

        alfa = (np.dot(r, r)) / (np.dot(np.dot(d, A), d))
        x.append(x[len(x)-1] + alfa*d)
        rnew = r - alfa * np.dot(A, d)
        beta = (np.dot(rnew, rnew)) / (np.dot(r, r))
        r = rnew
        d = r + beta*d
        q.append(r/LA.norm(r))
    return np.array(x), np.array(q).T


def cgAlgo(A, b, N=None):
    if N is None:
        N = A.shape[1]
    X = []
    for bb in b:
        x = []
        for index in range(bb.shape[1]):
            x.append(conjugate_grad(A, bb[:, index], N))
        X.append(x)

    X = np.array(X)
    X = np.transpose(X, (2, 0, 3, 1))
    return X


def gmresAlgo(A, b, N=None):
    if N is None:
        N = A.shape[1]
    X = []
    for bb in b:
        x = []
        for index in range(bb.shape[1]):
            x.append(gmres(A, bb[:, index], N))
        X.append(x)

    X = np.array(X)
    X = np.transpose(X, (2, 0, 3, 1))
    return X


def createMatrix(m, tao):

    A = np.identity(m)
    for i in range(m):
        A[i] = np.random.rand(m) * 2 - np.ones(m)
        A[i, i] = 1
        if i-1 >= 0:
            A[i, 0:i] = 0
    A = A + A.T - np.identity(m)
    A[np.abs(A) > tao] = 0
    A += np.identity(m)
    return A


def Pinverse(A, b):

    pinverse = np.linalg.pinv(A)
    x = []
    for bs in b:
        x.append(pinverse@bs)
    return np.array(x)


def findErrors(A, x, x0, b):

    es = []
    eo = []

    for xx in x:
        dum = - xx + x0
        dum = np.linalg.norm(dum, ord=2, axis=0) ** 2
        es.append(np.sqrt(np.mean(dum)))

    for xx, bb in zip(x, b):
        dum = bb - A @ xx
        dum = np.linalg.norm(dum, ord=2, axis=0) ** 2
        eo.append(np.sqrt(np.mean(dum)))
    return es, eo


def gmres(A, b, iter):

    q =[]
    a_iter = []
    a_iter.append(0)
    b_iter = []
    b_iter.append(0)
    T = []

    Tt= []


    q.append(np.zeros(len(b))) #  q0
    q.append(b/LA.norm(b)) #  q1
    x = []
    # x.append(np.zeros(len(b)))
    for k in range(1, iter+1):

        e1 = np.zeros(k+1)
        e1[0] = 1
        v = np.dot(A, q[k])
        a_iter.append(np.dot(q[k].T, v))
        v = v - b_iter[k-1] * q[k-1] - a_iter[k] * q[k]
        b_iter.append(LA.norm(v))
        q.append(v / b_iter[k])

        if k > 1:
            T = np.zeros((k, k))
            T[np.arange(k - 1) + 1, np.arange(k - 1)] = np.array(b_iter[1:-1])
            T[np.arange(k - 1), np.arange(k - 1) + 1] = np.array(b_iter[1:-1])
            T[np.arange(k), np.arange(k)] = np.array(a_iter[1:])

        if k > 2:
            dum = np.zeros((1, len(T[len(T)-1])))
            dum[0][dum.shape[1]-1] = b_iter[len(b_iter)-1]
            Tt = np.concatenate((T, dum))
            Q, R = LA.qr(Tt)
            dm = np.dot(Q.T, e1)
            Rm = R[0:(R.shape[1]), :]
            y = LA.norm(b) * np.dot(LA.inv(Rm), dm)
            x.append(np.dot(np.array(q[1:len(q)-1]).T, y))
    return x


def gmres_90(A, b, iter):

    q =[]
    a_iter = []
    a_iter.append(0)
    b_iter = []
    b_iter.append(0)
    T = []
    Tt= []

    q.append(np.zeros(len(b))) #  q0
    q.append(b/LA.norm(b)) #  q1
    x = []
    # x.append(np.zeros(len(b)))
    for k in range(1, iter+1):

        e1 = np.zeros(k+1)
        e1[0] = 1
        v = np.dot(A, q[k])
        a_iter.append(np.dot(q[k].T, v))
        v = v - b_iter[k-1] * q[k-1] - a_iter[k] * q[k]
        b_iter.append(LA.norm(v))
        q.append(v / b_iter[k])

        if k > 1:
            T = np.zeros((k, k))
            T[np.arange(k - 1) + 1, np.arange(k - 1)] = np.array(b_iter[1:-1])
            T[np.arange(k - 1), np.arange(k - 1) + 1] = np.array(b_iter[1:-1])
            T[np.arange(k), np.arange(k)] = np.array(a_iter[1:])

        if k > 2:
            dum = np.zeros((1, len(T[len(T)-1])))
            dum[0][dum.shape[1]-1] = b_iter[len(b_iter)-1]
            Tt = np.concatenate((T, dum))
            Q, R = LA.qr(Tt)
            dm = np.dot(Q.T, e1)
            Rm = R[0:(R.shape[1]), :]
            y = LA.norm(b) * np.dot(LA.inv(Rm), dm)
            x.append(np.dot(np.array(q[1:len(q)-1]).T, y))
    return x, np.array(q[1:len(q)]).T


if __name__ == '__main__':

    #  Todo: Initialize
    start_time = time.time()
    Ax100t01 = createMatrix(m=100, tao=0.1)
    Ax500t01 = createMatrix(m=500, tao=0.1)
    Ax10000t01 = createMatrix(m=10000, tao=0.1)

    Ax100t001 = createMatrix(m=100, tao=0.01)
    Ax500t001 = createMatrix(m=500, tao=0.01)
    Ax10000t001 = createMatrix(m=10000, tao=0.01)

    X0x100 = np.random.normal(0, 1, (100, 10))
    X0x500 = np.random.normal(0, 1, (500, 10))
    X0x10000 = np.random.normal(0, 1, (10000, 10))
    # print("--- %s seconds ---" % (time.time() - start_time))

    b0x100t01 = Ax100t01@X0x100
    b0x500t01 = Ax500t01@X0x500
    b0x10000t01 = Ax10000t01@X0x10000

    b0x100t001 = Ax100t001@X0x100
    b0x500t001 = Ax500t001@X0x500
    b0x10000t001 = Ax10000t001@X0x10000
    # print("--- %s seconds ---" % (time.time() - start_time))

    bx100t01 = []
    bx100t01.append(b0x100t01 + np.random.normal(0, 0.0001, (b0x100t01.shape[0], b0x100t01.shape[1])))
    bx100t01.append(b0x100t01 + np.random.normal(0, 0.01, (b0x100t01.shape[0], b0x100t01.shape[1])))
    bx100t01.append(b0x100t01 + np.random.normal(0, 1, (b0x100t01.shape[0], b0x100t01.shape[1])))

    bx500t01 = []
    bx500t01.append(b0x500t01 + np.random.normal(0, 0.0001, (b0x500t01.shape[0], b0x500t01.shape[1])))
    bx500t01.append(b0x500t01 + np.random.normal(0, 0.01, (b0x500t01.shape[0], b0x500t01.shape[1])))
    bx500t01.append(b0x500t01 + np.random.normal(0, 1, (b0x500t01.shape[0], b0x500t01.shape[1])))

    bx10000t01 = []
    bx10000t01.append(b0x10000t01 + np.random.normal(0, 0.0001, (b0x10000t01.shape[0], b0x10000t01.shape[1])))
    bx10000t01.append(b0x10000t01 + np.random.normal(0, 0.01, (b0x10000t01.shape[0], b0x10000t01.shape[1])))
    bx10000t01.append(b0x10000t01 + np.random.normal(0, 1, (b0x10000t01.shape[0], b0x10000t01.shape[1])))

    bx100t001 = []
    bx100t001.append(b0x100t001 + np.random.normal(0, 0.0001, (b0x100t001.shape[0], b0x100t001.shape[1])))
    bx100t001.append(b0x100t001 + np.random.normal(0, 0.01, (b0x100t001.shape[0], b0x100t001.shape[1])))
    bx100t001.append(b0x100t001 + np.random.normal(0, 1, (b0x100t001.shape[0], b0x100t001.shape[1])))

    bx500t001 = []
    bx500t001.append(b0x500t001 + np.random.normal(0, 0.0001, (b0x500t001.shape[0], b0x500t001.shape[1])))
    bx500t001.append(b0x500t001 + np.random.normal(0, 0.01, (b0x500t001.shape[0], b0x500t001.shape[1])))
    bx500t001.append(b0x500t001 + np.random.normal(0, 1, (b0x500t001.shape[0], b0x500t001.shape[1])))

    bx10000t001 = []
    bx10000t001.append(b0x10000t001 + np.random.normal(0, 0.0001, (b0x10000t001.shape[0], b0x10000t001.shape[1])))
    bx10000t001.append(b0x10000t001 + np.random.normal(0, 0.01, (b0x10000t001.shape[0], b0x10000t001.shape[1])))
    bx10000t001.append(b0x10000t001 + np.random.normal(0, 1, (b0x10000t001.shape[0], b0x10000t001.shape[1])))
    # print("--- %s seconds ---" % (time.time() - start_time))
    #  Todo: Initialize

    # Todo: Test Purpose only
    # test_x1 = Pinverse(Ax100t01, bx100t01)
    # es, eo = findErrors(Ax100t01, test_x1, X0x100, bx100t01)
    # print("Es: ", es[0], "Eo: ", eo[0])

    # test_x = cgAlgo(Ax100t01, bx100t01, N=50)
    # error_es = []
    # error_eo = []
    # for time in test_x:
    #     es, eo = findErrors(Ax100t01, time, X0x100, bx100t01)
    #     error_es.append(es[0])
    #     error_eo.append(eo[0])
    # plt.plot(error_es, label='ES Error')
    # plt.plot(error_eo, label='EO Error')
    # plt.legend()
    # plt.yscale("log")
    # plt.show()
    # print("Es: ", error_es[len(error_es)-1], "Eo: ", error_eo[len(error_eo)-1])

    # test_x = gmres(Ax100t01, bx100t01[0][:, 0], 100)
    #
    # errors = []
    # for xxxxx in test_x:
    #     errors.append(np.linalg.norm(np.dot(Ax100t01, xxxxx) - bx100t01[0][:, 0], ord=2, axis=0) ** 2)
    #
    # plt.plot(errors)
    # plt.yscale("log")
    # plt.show()

    # test_x = gmresAlgo(Ax100t01, bx100t01, N=7)
    # error_es = []
    # error_eo = []
    # for time in test_x:
    #     es, eo = findErrors(Ax100t01, time, X0x100, bx100t01)
    #     error_es.append(es[0])
    #     error_eo.append(eo[0])
    # plt.plot(error_es, label='ES Error')
    # plt.plot(error_eo, label='EO Error')
    # plt.legend()
    # plt.yscale("log")
    # plt.show()
    # print("Es: ", error_es[len(error_es)-1], "Eo: ", error_eo[len(error_eo)-1])
    # Todo: Test Purpose only

    # Todo: Plotting 16
    As = [Ax100t01, Ax100t001, Ax500t01, Ax500t001, Ax10000t01, Ax10000t001]
    bs = [bx100t01, bx100t001, bx500t01, bx500t001, bx10000t01, bx10000t001]
    xs = [X0x100, X0x100, X0x500, X0x500, X0x10000, X0x10000]
    plot_titles = ["A(100x100) tao=0.1 Pseudoinverse",
                   "A(100x100) tao=0.1 GMRES",
                   "A(100x100) tao=0.1 CG",
                   "A(100x100) tao=0.01 Pseudoinverse",
                   "A(100x100) tao=0.01 GMRES",
                   "A(100x100) tao=0.01 CG",
                   "A(500x500) tao=0.1 Pseudoinverse",
                   "A(500x500) tao=0.1 GMRES",
                   "A(500x500) tao=0.1 CG",
                   "A(500x500) tao=0.01 Pseudoinverse",
                   "A(500x500) tao=0.01 GMRES",
                   "A(500x500) tao=0.01 CG",
                   "A(10000x10000) tao=0.1 Pseudoinverse",
                   "A(10000x10000) tao=0.1 GMRES",
                   "A(10000x10000) tao=0.1 CG",
                   "A(10000x10000) tao=0.01 Pseudoinverse",
                   "A(10000x10000) tao=0.01 GMRES",
                   "A(10000x10000) tao=0.01 CG"
                   ]
    Ns = [ 0,
           23,
           23,
           0,
           8,
           8,
           0,
           55,
           55,
           0,
           9,
           9,
           0,
           4,
           4,
           0,
           14,
           14
           ]
    noises = [0.0001, 0.01, 1]
    count = 0

    for A, b, x in zip(As, bs, xs):

        x_pred = Pinverse(A, b)
        es, _ = findErrors(A, x_pred, x, b)
        plt.plot(noises, es, label='ES Error Pseudoinverse')
        plt.scatter(noises, es, color='orange')
        plt.legend()
        plt.title(plot_titles[count])
        count += 1
        plt.ylabel("Log(Error)")
        plt.xlabel("Log(Noise std)")
        for i_y, i_x in zip(es, noises):
            plt.annotate("("+ str(i_x) + " ," + str(round(i_y, 3)) + ")", (i_x, i_y))
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(plot_titles[count-1] + ".png")
        # plt.show()
        plt.close()

        x_pred = gmresAlgo(A, b, N=Ns[count])
        x_pred = x_pred[len(x_pred)-1]
        es, _ = findErrors(A, x_pred, x, b)
        plt.plot(noises, es, label='ES Error GMRES')
        plt.scatter(noises, es, color='orange')
        plt.legend()
        plt.title(plot_titles[count])
        count += 1
        plt.yscale("log")
        plt.ylabel("Log(Error)")
        plt.xscale("log")
        plt.xlabel("Log(Noise std)")
        for i_y, i_x in zip(es, noises):
            plt.annotate("("+ str(i_x) + " ," + str(round(i_y, 3)) + ")", (i_x, i_y))
        plt.savefig(plot_titles[count-1] + ".png")
        # plt.show()
        plt.close()

        x_pred = cgAlgo(A, b, N=Ns[count])
        x_pred = x_pred[len(x_pred) - 1]
        es, _ = findErrors(A, x_pred, x, b)
        plt.plot(noises, es, label='ES Error CG')
        plt.scatter(noises, es, color='orange')
        plt.legend()
        plt.title(plot_titles[count])
        count += 1
        plt.yscale("log")
        plt.ylabel("Log(Error)")
        plt.xscale("log")
        plt.xlabel("Log(Noise std)")
        for i_y, i_x in zip(es, noises):
            plt.annotate("("+ str(i_x) + " ," + str(round(i_y, 3)) + ")", (i_x, i_y))
        plt.savefig(plot_titles[count-1] + ".png")
        # plt.show()
        plt.close()
    # Todo: Plotting 16

    # Todo: Plotting CG
    As = [Ax100t01, Ax100t001, Ax500t01, Ax500t001, Ax10000t01, Ax10000t001]
    bs = [bx100t01, bx100t001, bx500t01, bx500t001, bx10000t01, bx10000t001]
    xs = [X0x100, X0x100, X0x500, X0x500, X0x10000, X0x10000]
    noises = [0.0001, 0.01, 1]
    count = 0

    labels = ["Eo-> A(100x100) Tao: 0.1",
              "Eo-> A(100x100) Tao: 0.01",
              "Eo-> A(500x500) Tao: 0.1",
              "Eo-> A(500x500) Tao: 0.01",
              "Eo-> A(10000x10000) Tao: 0.1",
              "Eo-> A(10000x10000) Tao: 0.01",
              ]

    labelses = ["Es-> A(100x100) Tao: 0.1",
              "Es-> A(100x100) Tao: 0.01",
              "Es-> A(500x500) Tao: 0.1",
              "Es-> A(500x500) Tao: 0.01",
              "Es-> A(10000x10000) Tao: 0.1",
              "Es-> A(10000x10000) Tao: 0.01",
              ]
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    N = 30
    for A, b, x, label, label2 in zip(As, bs, xs, labels, labelses):

        x_pred = cgAlgo(A, b, N=N)
        error_es_s = []
        error_eo_s = []
        error_es_m = []
        error_eo_m = []
        error_es_b = []
        error_eo_b = []
        for time in x_pred:
            es, eo = findErrors(A, time, x, b)
            error_es_s.append(es[0])
            error_eo_s.append(eo[0])
            error_es_m.append(es[1])
            error_eo_m.append(eo[1])
            error_es_b.append(es[2])
            error_eo_b.append(eo[2])

        ax0.plot(error_eo_s, label=label)
        ax3.plot(error_es_s, label=label2)
        # ax0.scatter(error_eo_s, np.arange(N+1), color='orange')
        ax1.plot(error_eo_m, label=label)
        ax4.plot(error_es_m, label=label2)
        # ax1.scatter(error_eo_m, np.arange(N+1), color='orange')
        ax2.plot(error_eo_b, label=label)
        ax5.plot(error_es_b, label=label2)
        # ax2.scatter(error_eo_b, np.arange(N+1), color='orange')

    ax0.set_title("Eo Error with Gaussian Noise std: 0.0001")
    ax0.set_yscale("log")
    ax0.set_ylabel("Log(Error)")
    ax0.set_xlabel("Iteration")
    ax0.legend()
    fig0.show()

    ax1.set_title("Eo Error with Gaussian Noise std: 0.01")
    ax1.set_yscale("log")
    ax1.set_ylabel("Log(Error)")
    ax1.set_xlabel("Iteration")
    ax1.legend()
    fig1.show()

    ax2.set_title("Eo Error with Gaussian Noise std: 1")
    ax2.set_yscale("log")
    ax2.set_ylabel("Log(Error)")
    ax2.set_xlabel("Iteration")
    ax2.legend()
    fig2.show()

    ax3.set_title("Es Error with Gaussian Noise std: 0.0001")
    ax3.set_yscale("log")
    ax3.set_ylabel("Log(Error)")
    ax3.set_xlabel("Iteration")
    ax3.legend()
    fig3.show()

    ax4.set_title("Es Error with Gaussian Noise std: 0.01")
    ax4.set_yscale("log")
    ax4.set_ylabel("Log(Error)")
    ax4.set_xlabel("Iteration")
    ax4.legend()
    fig4.show()

    ax5.set_title("Es Error with Gaussian Noise std: 1")
    ax5.set_yscale("log")
    ax5.set_ylabel("Log(Error)")
    ax5.set_xlabel("Iteration")
    ax5.legend()
    fig5.show()
    # Todo: Plotting CG

    # Todo: Plotting GMRES
    As = [Ax100t01, Ax100t001, Ax500t01, Ax500t001, Ax10000t01, Ax10000t001]
    bs = [bx100t01, bx100t001, bx500t01, bx500t001, bx10000t01, bx10000t001]
    xs = [X0x100, X0x100, X0x500, X0x500, X0x10000, X0x10000]
    noises = [0.0001, 0.01, 1]
    count = 0

    labels = ["A(100x100) Tao: 0.1",
              "A(100x100) Tao: 0.01",
              "A(500x500) Tao: 0.1",
              "A(500x500) Tao: 0.01",
              "A(10000x10000) Tao: 0.1",
              "A(10000x10000) Tao: 0.01",
              ]

    labelses = ["Es-> A(100x100) Tao: 0.1",
              "Es-> A(100x100) Tao: 0.01",
              "Es-> A(500x500) Tao: 0.1",
              "Es-> A(500x500) Tao: 0.01",
              "Es-> A(10000x10000) Tao: 0.1",
              "Es-> A(10000x10000) Tao: 0.01",
              ]

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    N = 30
    for A, b, x, label, label2 in zip(As, bs, xs, labels, labelses):

        x_pred = gmresAlgo(A, b, N=N)
        error_es_s = []
        error_eo_s = []
        error_es_m = []
        error_eo_m = []
        error_es_b = []
        error_eo_b = []
        for time in x_pred:
            es, eo = findErrors(A, time, x, b)
            error_es_s.append(es[0])
            error_eo_s.append(eo[0])
            error_es_m.append(es[1])
            error_eo_m.append(eo[1])
            error_es_b.append(es[2])
            error_eo_b.append(eo[2])

        ax0.plot(np.arange(start=3, stop=N+1), error_eo_s, label=label)
        ax1.plot(np.arange(start=3, stop=N+1), error_eo_m, label=label)
        ax2.plot(np.arange(start=3, stop=N+1), error_eo_b, label=label)

        ax3.plot(np.arange(start=3, stop=N+1), error_es_s, label=label2)
        ax4.plot(np.arange(start=3, stop=N+1), error_es_m, label=label2)
        ax5.plot(np.arange(start=3, stop=N+1), error_es_b, label=label2)

    ax0.set_title("Eo Error with Gaussian Noise std: 0.0001")
    ax0.set_yscale("log")
    ax0.set_ylabel("Log(Error)")
    ax0.set_xlabel("Dimension K")
    ax0.legend()
    fig0.show()

    ax1.set_title("Eo Error with Gaussian Noise std: 0.01")
    ax1.set_yscale("log")
    ax1.set_ylabel("Log(Error)")
    ax1.set_xlabel("Dimension K")
    ax1.legend()
    fig1.show()

    ax2.set_title("Eo Error with Gaussian Noise std: 1")
    ax2.set_yscale("log")
    ax2.set_ylabel("Log(Error)")
    ax2.set_xlabel("Dimension K")
    ax2.legend()
    fig2.show()

    ax3.set_title("Es Error with Gaussian Noise std: 0.0001")
    ax3.set_yscale("log")
    ax3.set_ylabel("Log(Error)")
    ax3.set_xlabel("Dimension K")
    ax3.legend()
    fig3.show()

    ax4.set_title("Es Error with Gaussian Noise std: 0.01")
    ax4.set_yscale("log")
    ax4.set_ylabel("Log(Error)")
    ax4.set_xlabel("Dimension K")
    ax4.legend()
    fig4.show()

    ax5.set_title("Es Error with Gaussian Noise std: 1")
    ax5.set_yscale("log")
    ax5.set_ylabel("Log(Error)")
    ax5.set_xlabel("Dimension K")
    ax5.legend()
    fig5.show()
    # Todo: Plotting GMRES

    # Todo: Show Orthogonality
    test_x, test_q = conjugate_grad_90(Ax100t01, bx100t01[0][:, 0], 10)
    iden = test_q.T @ test_q
    iden = iden - np.identity(len(iden))
    print("For A(100x100) tao: 0.1 & b(100) with lowest std gaussian")
    print("Max: ", np.max(iden), "Min: ", np.min(iden), "Mean: ", np.mean(iden))

    test_x, test_q = conjugate_grad_90(Ax500t01, bx500t01[0][:, 0], 10)
    iden = test_q.T @ test_q
    iden = iden - np.identity(len(iden))
    print("For A(500x500) tao: 0.1 & b(500) with lowest std gaussian")
    print("Max: ", np.max(iden), "Min: ", np.min(iden), "Mean: ", np.mean(iden))

    test_x, test_q = conjugate_grad_90(Ax10000t01, bx10000t01[0][:, 0], 10)
    iden = test_q.T @ test_q
    iden = iden - np.identity(len(iden))
    print("For A(10000x10000) tao: 0.1 & b(10000) with lowest std gaussian")
    print("Max: ", np.max(iden), "Min: ", np.min(iden), "Mean: ", np.mean(iden))
    # Todo: Show Orthogonality
