import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    激活函数为sigmoid的两层神经网络的前向和后向传播

    前向传播，代价函数为交叉熵函数，后向传播计算所有参数的梯度.

    参数:
    data -- 维度为（M x Dx）的矩阵, 每行代表一个样本.
    labels -- 维度为（M x Dy）的矩阵, 每行是一个one-hot向量.
    params -- 模型的权重
    dimensions -- 元组数据包括，输入维度, 隐藏层神经元的数量，输出维度
    """

    ### 设置网络权重
    ofs = 0 # 用于提取权重，初始化为0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2]) #输入维度, 隐藏层神经元的数量，输出维度

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H)) # 输入层权重W1，维度（Dx, H）
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H)) # 输入层权重b1，维度（1, H）
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy)) # 隐藏层权重W2，维度（H, Dy）
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy)) # 隐藏层权重b2，维度（1, Dy）

    ### 前向传播
    M = np.shape(data)[0]

    z1 = np.zeros([M,H]) # 初始化z1，维度（M, H）
    z1 = np.dot(data, W1) + b1 # 计算z1， 维度(M, H)
    g1 = sigmoid(z1) # 计算g1, 维度(M, H)

    z2 = np.zeros([M, Dy]) # 初始化z2, 维度(M, Dy)
    z2 = np.dot(g1, W2) + b2 # 计算z2, 维度(M, Dy)
    g2 = softmax(z2) # 计算g2也就是输出， 维度(M, Dy)

    cost = - np.sum(labels * np.log(g2)) # 计算代价函数，交叉熵

    ###后向传播
    dW1 = data.T # z1对于W1的梯度
    db1 = np.ones([1, M]) # z1对于b1的梯度
    dz1 = sigmoid_grad(g1) # g1对于z1的梯度
    dg1 = W2.T # z2对于g1的梯度
    dz2 = g2 - labels # 代价函数对于z2的导数

    dW2 = g1.T # z2对于W2的导数
    db2 = np.ones([1, M]) # z2对于b2的导数

    gradW1 = np.dot(dW1, np.dot(dz2, dg1) * dz1) # 利用链式法则计算代价对于W1的导数
    gradb1 = np.dot(db1, np.dot(dz2, dg1) * dz1) # 利用链式法则计算代价对于b1的导数

    gradW2 = np.dot(dW2, dz2) # 利用链式法则计算代价对于W2的导数
    gradb2 = np.dot(db2, dz2) # 利用链式法则计算代价对于b2的导数

    ### 保存梯度
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    设置神经网络的数据集和权重，并使用gradcheck进行测试
    """
    print("运行神经网络单元测试...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0]) # 创建数据集
    
    ### 创建one-hot标签向量 
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1
    
    ### 创建权重矩阵
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    ### 利用梯度检查器进行测试
    gradcheck_naive(lambda params:
       forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    可以在这里打造自己的例子
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
