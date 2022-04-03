import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
# 输入为 [N, C, H, W]，需要两个参数，in_planes为输特征通道数，K 为专家个数
class Attention(nn.Module):
    def __init__(self,in_planes, C, r):#ratio = 1 // r = 1 // 16（默认）
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, C // r)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        # 将输入特征全局池化为 [N, C, 1, 1]
        att=self.avgpool(x)
        # 将特征转化为二维 [N, C]
        att=att.view(att.shape[0],-1) 
        # 使用 sigmoid 函数输出归一化到 [0,1] 区间
        return F.relu(self.fc(att))
class ODConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,
                 groups=1,K=4,batchsize = 128):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes, C = in_planes, r = 16)
        self.weight = nn.Parameter(torch.randn(batchsize, K, self.out_planes, self.in_planes//self.groups,
             self.kernel_size, self.kernel_size),requires_grad=True)
        self.mtx = nn.Parameter(torch.randn(K, 1),requires_grad=True)
        self.fc1 = nn.Linear(in_planes // 16, kernel_size * kernel_size)
        self.fc2 = nn.Linear(in_planes // 16, in_planes * 1 // self.groups)
        self.fc3 = nn.Linear(in_planes // 16, out_planes * 1)
        self.fc4 = nn.Linear(in_planes // 16, K * 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_planes)
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        # 调用 attention 函数得到归一化的权重 [N, K]
        N,in_planels, H, W = x.shape
        att=self.attention(x)#[N, Cin // 16]
        att1 = self.sigmoid(self.fc1(att))#[N, kernel_size * kernel_size]
        att1 = att1.reshape(att1.shape[0], self.kernel_size, self.kernel_size)#[N, kernel_size, kernel_size]
        att2 = self.sigmoid(self.fc2(att))#[N, in_planes]
        att3 = self.sigmoid(self.fc3(att))#[N, out_planes]
        att4 = self.softmax(self.fc4(att))#[N, K]
        Weight = None
        for i in range(self.weight.shape[0]):
            if i == 0:
                Weight = torch.unsqueeze(self.weight[i, :, :, :, :, :] * att1[i, :, :], 0)
            else:
                Weight = torch.cat([Weight, torch.unsqueeze(self.weight[i, :, :, :, :, :] * att1[i, :, :], 0)], 0)        
        Weight = self.dropout(Weight)
        Weight2 = None
        for i in range(self.weight.shape[0]):
            if i == 0:
                Weight2 = torch.unsqueeze(Weight[i, :, :, :, :, :] * att2[i, None, :, None, None], 0)
            else:
                Weight2 = torch.cat([Weight2, torch.unsqueeze(Weight[i, :, :, :, :, :] * att2[i, None, :, None, None], 0)], 0)
        Weight2 = self.dropout(Weight2)
        Weight3 = None  
        for i in range(self.weight.shape[0]):
            if i == 0:
                Weight3 = torch.unsqueeze(Weight2[i, :, :, :, :, :] * att3[i, None, :, None, None,  None], 0)
            else:
                Weight3 = torch.cat([Weight3, torch.unsqueeze(Weight2[i, :, :, :, :, :] * att3[i, None, :, None, None, None], 0)], 0)

        Weight3 = self.dropout(Weight3)
        Weight4 = None
        for i in range(self.weight.shape[0]):
            if i == 0:
                Weight4 = torch.unsqueeze(Weight3[i, :, :, :, :, :] * att4[i, :, None, None, None, None], 0)
            else:
                Weight4 = torch.cat([Weight4, torch.unsqueeze(Weight3[i, :, :, :, :, :] * att4[i, :, None, None, None, None], 0)], 0)
        Weight4 = self.dropout(Weight4)
        Weight4 = torch.unsqueeze(Weight4, 6)
        Weight4 = Weight4.permute(0, 6, 2, 3, 4, 5, 1)
        Weight4 = torch.matmul(Weight4, self.mtx)
        x=x.view(1, -1, H, W)
        Weight4 = Weight4.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size, self.kernel_size)
        output=F.conv2d(x,weight=Weight4,
                  stride=self.stride, padding=self.padding,
                  groups=self.groups*N)
        # 形状恢复为 [N, C_out, H, W]        
        _, _, H, W = output.shape
        output=output.view(N, self.out_planes, H, W)
        output = self.relu(self.bn(output))
        return output
          