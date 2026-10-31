import torch
import torch.nn as nn
from torch import autograd

class MyReLU(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        ctx: 上下文对象，用来保存反向需要的信息
        y = max(x, 0)
        """
        # 保存mask, 反向时用来判断哪些位置>0
        mask = (x > 0)
        ctx.save_for_backward(mask)
        y = x.masked_fill(~mask, 0.0)
        return y 
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: 上游传来的梯度 dL/dy
        返回 dL/dx = dL/dy * dy/dx
        dy/dx = 1 if x > 0 else 0
        """
        mask, = ctx.saved_tensors
        grad_x = grad_output * mask.float()
        return grad_x


class MyReLUModule(nn.Module):
    def forward(self, x):
        return MyReLU.apply(x)

if __name__ == "__main__":
    relu = MyReLUModule()

    x = torch.tensor([-2.0, -0.0, 0.0, 1.5, 3.0], requires_grad=True)
    # 自己实现
    y = relu(x)
    loss = y.sum()
    loss.backward()

    # 官方实现
    # y = nn.ReLU()(x)
    # loss = y.sum()
    # loss.backward()

    print("x            : ", x.data)
    print("ReLU output y: ", y.data)
    print("x.grad       : ", x.grad)

    # 输出结果：
    # x            :  tensor([-2.0000, -0.0000,  0.0000,  1.5000,  3.0000])
    # ReLU output y:  tensor([0.0000, 0.0000, 0.0000, 1.5000, 3.0000])
    # x.grad       :  tensor([0., 0., 0., 1., 1.])