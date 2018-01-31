SAVE_PARAMS_EVERY = 5000

import glob
import random
import numpy as np
import os.path as op
import _pickle as pickle

def load_saved_params():
    """
    帮助函数，用于读取保存的权重并重置循环
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        print(f)
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        print(iter)
        if (iter > st):
            st = iter
    print(f)
    print(st)

    if st > 0:
        with open("saved_params_%d.npy" % st, "rb") as f:
            print(f)
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ SGD(Stochastic Gradient Descent)

    参数:
    f -- 优化目标函数，该函数应该有一个输入和两个输出，返回输入的代价和梯度
    x0 -- SGD初始样本数量
    step -- 步长
    iterations -- 循环次数
    postprocessing -- 后处理函数，用于当我们想要归一化词向量
    PRINT_EVERY -- 确定经过多少次循环输出代价

    返回值:
    x -- SGD结束后返回权重
    """

    # 用于迭代学习率
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in range(start_iter + 1, iterations + 1):
        # Don't forget to apply the postprocessing after every iteration!
        # You might want to print the progress every few iterations.

        cost = None
        cost, grad = f(x)

        x = x - step * grad

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d: %f" % (iter, expcost))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("")


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q3_sgd.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    #load_saved_params()
    sanity_check()
    #your_sanity_checks()
