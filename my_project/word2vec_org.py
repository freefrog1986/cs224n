import numpy as np
import random

from softmax import softmax
from gradcheck import gradcheck_naive
from sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ 
    行归一化函数

    用于归一化矩阵的每一行，使每行的均为单位长度
    """
    x /= np.sqrt((x * x).sum(axis=1)).reshape(-1,1)
    return x


def test_normalize_rows():
    print("测试归一化函数...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("测试成功！")


def softmaxCostAndGradient(predicted, target, outputVectors):
    """ 
    word2vec模型的softmax代价函数
    
    此函数处理对一个目标词向量进行预测，并以此为word2vec模型的单元模块。使用交叉熵作为代价函数并计算梯度。（可以理解为神经网络输出层的过程，输出层的输入为predicted,权重矩阵为outputVectors，矩阵相乘得到y_hat, 真实值是y_true）

    参数:
    predicted -- 数据类型numpy ndarray, 文本中的预测词向量(\hat{v})
    target -- 数据类型integer, 目标单词的索引值
    outputVectors -- 所有tokens的输出向量（行）

    返回值:
    cost -- softmax的交叉熵代价（cross entropy cost）
    gradPred -- 预测词向量的梯度
    grad -- 输出词向量矩阵的梯度

    """
     ### YOUR CODE HERE
    ## Gradient for $\hat{\bm{v}}$:

    #  Calculate the predictions:
    vhat = predicted
    z = np.dot(outputVectors, vhat)
    preds = softmax(z)

    #  Calculate the cost:
    cost = -np.log(preds[target])

    #  Gradients
    z = preds.copy()
    z[target] -= 1.0

    grad = np.outer(z, vhat)
    gradPred = np.dot(outputVectors.T, z)
    ### END YOUR CODE

    return cost, gradPred, grad

def getNegativeSamples(target, dataset, K):
    """ 对K个非目标词向量进行采样"""

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ word2vec模型的负采样代价函数

    本函数计算一个预测词向量和一个目标词向量情况下的代价和梯度，并使用负采样技巧，作为word2vec的单元模块。K是负采样数量。

    注意: dataset的初始化方式见test_word2vec函数.

    参数和返回值与softmaxCostAndGradient一致。
    """

    # 负采样. 
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    grad = np.zeros((outputVectors.shape[0],outputVectors.shape[1]))
    indices = indices[indices != target]

    # 1. 前向传播
    outvec = outputVectors[target].reshape(1,-1) # 目标词向量
    negvec = outputVectors[indices].reshape(-1,outputVectors.shape[1]) #负采样词向量
    y_score = np.dot(outvec,predicted.T)
    ns_score = np.dot(negvec,predicted.T)
    y_hat = sigmoid(y_score) # 目标预测值
    ns_hat = sigmoid(-ns_score) # 负采样预测值，注意这里取负了

    # 2. 计算代价
    cost_o2c = -np.log(y_hat) # 预测值与真实值的代价
    cost_n2c = -np.sum(np.log(ns_hat),0) # 负采样与真实值的代价
    cost = cost_o2c + cost_n2c 
    
    # 3. 后向传播
    gradPred = (y_hat-1) * outvec - np.dot((ns_hat-1).T,negvec) # 代价对于predictied梯度
    grad[target] = (y_hat-1) * predicted # 代价对于目标词向量的梯度
    grad[indices] = -np.dot((ns_hat-1),predicted) # 代价对于负采样词向量的梯度

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ word2vec的Skip-gram模型

    参数:
    currrentWord -- 中心单词，string
    C -- 窗口大小，integer
    contextWords -- 上下文单词列表，不大于2*C，list of strings
    tokens -- 词典，用于将单词映射为词向量中的索引
    inputVectors -- 所有单词的“输入”词向量矩阵（每行是一个词向量）
    outputVectors -- 所有单词的“输出”词向量矩阵（每行是一个词向量）
    word2vecCostAndGradient -- 单元模块函数，输入中心词向量、“输出”词向量矩阵和目标单词，得到代价和梯度。根据是否使用负采样，二选一。

    返回值:
    cost -- skip-gram模型的代价值
    grad -- 词向量的梯度
    """

    ### 初始化权重和代价
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]

    for j in contextWords:
        u_idx = tokens[j]
        c_cost, c_grad_in, c_grad_out = \
            word2vecCostAndGradient(vhat, u_idx, outputVectors, dataset)
        cost += c_cost
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """word2vec的连续袋模型（CBOW model）
    
    参数和返回值: 与skip-gram模型一样
    """

### 初始化权重和代价
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    target = tokens[currentWord] # 得到目标单词的索引

    ### 对窗口内的所有单词进行前后向传播
    for context in contextWords:
        idx = tokens[context]
        predicted = inputVectors[idx].reshape(1,-1) # 相当于用单词的one-hot向量与输入词向量矩阵相乘

        if word2vecCostAndGradient == negSamplingCostAndGradient:
            cost, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset, K=10)
        elif word2vecCostAndGradient == softmaxCostAndGradient:
            cost, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors)

    ### 保存中心词向量和输出词向量矩阵的梯度
    gradIn[idx] = gradPred
    gradOut = grad

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()