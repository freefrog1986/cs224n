import numpy as np

def sigmoid(x):
    """
    计算输入x的sigmoid。

    参数:
    x -- 常量或者numpy array.

    返回:
    s -- sigmoid(x)
    """
    s = np.true_divide(1, 1 + np.exp(-x)) # 使用np.true_divide进行加法运算
    return s


def sigmoid_grad(s):
    """
    计算sigmoid的梯度，这里的参数s应该是x作为输入的sigmoid的返回值。

    参数:
    s -- 常数或者numpy array。

    返回:
    ds -- 梯度。
    """
    ds = s * (1 - s) # 可以证明：sigmoid函数关于输入x的导数等于`sigmoid(x)(1-sigmoid(x))`
    return ds


def test_sigmoid_basic():
    """
    用于测试sigmoid及其导数例子
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print(g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print("You should verify these results by hand!\n")


def test_sigmoid():
    """
    可以在下面添加自己的测试例子。
    """
    print("Running your tests...")


if __name__ == "__main__":
    test_sigmoid_basic();
    
