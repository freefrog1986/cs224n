import numpy as np
import random

def gradcheck_naive(f, x):
    """
    函数f的梯度检查器
   
    参数:
    f -- 函数f，输入参数x，输出代价和梯度
    x -- 梯度检查的点(numpy array)

    """

    rndstate = random.getstate() # 获取当前随机数的环境状态
    random.setstate(rndstate) # 设置当前随机数的环境状态

    fx, grad = f(x) # 计算函数f对当前点x的输出和梯度
    h = 1e-4        # 设置一个极小的值，用于计算梯度

    # 对x进行迭代
    # 设置flags=['multi_index']能够得到所有维度的索引 
    # 设置op_flags=['readwrite']使我们能够对x进行读写操作
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        random.setstate(rndstate) # 设置当前随机数的环境状态
        tmp1 = np.copy(x) 
        tmp1[ix] = tmp1[ix] + h # 计算(x + h)
        f1, _ = f(tmp1) # 计算f(x + h)
        
        random.setstate(rndstate)
        tmp2 = np.copy(x) 
        tmp2[ix] = tmp2[ix] - h # 计算(x - h)
        f2, _ = f(tmp2) # 计算f(x - h)

        numgrad = (f1 - f2) / (2 * h) # 近似计算梯度(f(x+h) - f(x-h))/2h

        # 对比近似梯度和f计算得到的梯度是否一致，分母使用max(1,...)是考虑到了两者为小数的情况
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("梯度检查失败.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # x的下一个元素。

    print("通过梯度检查!")


def sanity_check():
    """
    一些基本的测试例子。
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("")


def your_sanity_checks():
    """
    可以添加自己的测试例子
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
