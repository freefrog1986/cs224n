import numpy as np

def softmax(x):
    """
    对输入x的每一行计算softmax。

    为了提高运算速度，代码使用numpy的np.exp, np.sum, np.reshape, np.max,以及广播运算。

    关于广播运算，点击下面链接:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    该函数应对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。
    
    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x,axis=1) # 得到每行的最大值，用于缩放每行的元素，避免溢出
        x-=tmp.reshape((x.shape[0],1)) # 使每行减去所在行的最大值（广播运算）

        x = np.exp(x) # 第一步，计算所有值以e为底的x次幂
        tmp = np.sum(x, axis = 1) # 将每行求和并保存
        x /= tmp.reshape((x.shape[0], 1)) # 所有元素除以所在行的元素和（广播运算）

    else:
        # 向量
        tmp = np.max(x) # 得到最大值
        x -= tmp # 利用最大值缩放数据
        x = np.exp(x) # 对所有元素求以e为底的x次幂
        tmp = np.sum(x) # 求元素和
        x /= tmp # 求somftmax
    return x


def test_softmax_basic():
    """
    一些基本的测试softmax的例子
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def test_softmax():
    """
    该函数用于添加自己的测试例子
    """
    print("Running your tests...")
    ### YOUR CODE HERE

    ### END YOUR CODE


if __name__ == "__main__":
    #test_softmax_basic()
    test1 = np.array([1,  2, 3, 4])
    print('原始向量',test1)
    print('经过softmax后',softmax(test1))

