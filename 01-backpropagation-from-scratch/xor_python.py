
import random
import math

class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data # 真实值
        self.grad = 0.0 # 上游传下来的梯度
        self._backward = lambda: None # 该值的反向梯度
        self._parents = set(_parents) # 哪些参数计算出该值
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0 
        
        out._backward = _backward
        return out 

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data 
            other.grad += out.grad * self.data 
        
        out._backward = _backward
        return out
    
    __rmul__ = __mul__

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data,  (self, other), '-')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        
        out._backward = _backward
        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += out.grad * (out.data > 0) 
        
        out._backward = _backward
        return out 
    
    def __pow__(self, k):
        out = Value(self.data ** k, (self, ), f'**{k}')

        def _backward():
            self.grad += out.grad * k * (self.data ** (k - 1))
        
        out._backward = _backward
        return out 

    def tanh(self):
        # y = tanh(x) → dy/dx = 1 - y^2
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)
        out._backward = _backward
        return out

    def sigmoid(self):
        # σ(x) = 1 / (1 + e^{-x}) → dσ/dx = σ(x) * (1 - σ(x))
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out

    def backward(self):
        # 1. 先拓扑排序：从self出发，沿父母方向向上
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build(parent)
                topo.append(v)
        build(self)
        # print("topo: ", [])

        # 2. 输出自己的梯度是 1 
        self.grad = 1.0 

        # 3. 倒着走拓扑序，调用每个节点的 _backward
        for v in reversed(topo):
            v._backward()


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)] # 当前这一个神经元接收几个输入，则有几个权重
        self.b = Value(0.0)
    
    def __call__(self, x):
        """
        x: list;
        n = Neuron()
        h = n(x)

        权重 + 偏置 + 激活函数
        x, self.w, self.b 均是 Value 类型; 
        """
        act = sum([xi * wi for xi, wi in zip(x, self.w)], self.b) # sum([1,2], start=Value) 
        # return act.relu() 
        return act.tanh()

    def parameters(self):
        # 保存参数的顺序: w -> b ; Value类型; self.w 本身就是list ； list相加
        return self.w + [self.b]

class Layer:
    """
    叫法上有输入层，但只构建隐藏层和输出层；
    功能：返回当前层的n个神经元；
    输入：上一层的结果，Value类型

    l = Layer()
    h = l(x)
    """

    def __init__(self, n_in, n_out): # 输入和输出神经元个数
        # 构建当前层的神经元（有n_out个）
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons] # 每个神经元是单独计算的
    
    def parameters(self):
        # 神经元参数放一起

        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """ 将层Layer连接到一起
    for layer in self.layers
    """
    def __init__(self, n_in, n_list=[3,1]):
        
        self.layers = [Layer(n_in, n_list[0])] + [Layer(n_list[i], n_list[i+1]) for i in range(len(n_list)-1)]

    def __call__(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return x[0].sigmoid() # layer() 返回的是列表，取第一项
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":

    # a = Value(2.0)
    # b = Value(-3.0)
    # c = a + b # -1.0
    # d = a * b # -6.0
    # e = c * d # 6.0
    # e.backward()

    # print("a的梯度:", a.grad)
    # print("b的梯度:", b.grad)

    seed = random.random()

    print("seed: ", seed)
    random.seed(seed)

    # 1. 创建训练数据
    x_data = [[0,0], [0,1], [1,0], [1,1]]
    y_data = [[0], [1], [1], [0]]

    # 2. 构建模型：输入层 + 隐藏层（3个神经元） + 输出层（1个神经元）
    model = MLP(2, [3,1])

    # 3. 模型训练
    for epoch in range(2000):
        # 前向计算 + 计算损失
        loss = Value(0.0) # 要参与到计算图中，需要变成标量类
        for i in range(len(x_data)):
            x, y = x_data[i], y_data[i][0]
            pred = model(x)
            loss = loss + (pred - y) ** 2
        
       
        # 反向传播 
        loss.backward() # 可以给计算图各个中间节点、输入节点的grad都进行赋值，不只是w/b可训练参数；
        
        # 参数更新
        for p in model.parameters():
            p.data = p.data - 0.05 * p.grad 

        # 梯度清零
        for p in model.parameters():
            p.grad = 0.0        

        if epoch % 20 == 0:
            print(f"epoch: {epoch}, loss: {loss.data:.4f}")

    # 推理
    def predict(x):
        # x 可能是单个样本(如 [0,1])，也可能是批量网格矩阵(grid, N×2)
        # MLP 前向只吃单个样本 → 逐样本前向，统一返回 float 数组
        import numpy as np
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.array([model(row).data for row in x])
    from utils import xor_solution_vis
    xor_solution_vis.plot_xor_solution(x_data,y_data,predict,save_path="xor_solution_py.png")