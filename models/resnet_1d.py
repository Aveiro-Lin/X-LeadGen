import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2','simple_conv_net',
            'simple_conv_net_tongleiduibi','simple_gru','simple_fc']


class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SimpleFC(nn.Module):  
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(SimpleFC, self).__init__()  
        self.DG_method=DG_method
        self.distill=distill
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=4, padding=14,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
#         self.input_size=
#         self.hidden_size=
#         self.num_layers=1
#         self.distill=distill
#         self.DG_method=DG_method

        self.fc_mid = nn.Linear(32*128,512)
    
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.fc = nn.Linear(512, num_classes)  
  
    def forward(self, x):  
        # Convolutional layer  
        x = self.conv1(x.transpose(-1,-2))  # 1000, N -->32,256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 32,128
        x= self.fc_mid(x.reshape(x.shape[0],32*128)) 
#         x = x.reshape(x.shape[0],512)
        x=self.relu(x)
        
        if self.DG_method=='DG_GR':

            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3


            xo=self.fc(x)
            if self.distill==True:
                return xo,y,x
            else:
                return xo,y
        else:

            xo = self.fc(x)

            if self.distill==True:
                return xo,x
            else:
                return xo

class SimpleGRU(nn.Module):  
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(SimpleGRU, self).__init__()  
        self.DG_method=DG_method
        self.distill=distill
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=4, padding=14,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
#         self.input_size=
#         self.hidden_size=
#         self.num_layers=1
#         self.distill=distill
#         self.DG_method=DG_method

        self.gru = nn.GRU(input_size=128, hidden_size=2, num_layers=1,  \
                          batch_first=True,bidirectional=True)  
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.fc = nn.Linear(512, num_classes)  
  
    def forward(self, x):  
        # Convolutional layer  
        x = self.conv1(x.transpose(-1,-2))  # 1000, N -->128,256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x,cell = self.gru(x.transpose(-1,-2)) 
        x = x.reshape(x.shape[0],512)
        x=self.relu(x)
        
        if self.DG_method=='DG_GR':

            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3


            xo=self.fc(x)
            if self.distill==True:
                return xo,y,x
            else:
                return xo,y
        else:

            xo = self.fc(x)

            if self.distill==True:
                return xo,x
            else:
                return xo    
    
class simple_Conv(nn.Module):
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(simple_Conv,self).__init__()
        self.distill=distill
        self.DG_method=DG_method
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=15, stride=8, padding=7,
                               bias=False) # change according to the leads of the input ECG data
        self.bn2 = nn.BatchNorm1d(16)
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)

    def forward(self,x):
        x=self.conv1(x.transpose(-1,-2))
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=x.view(x.shape[0],512)
        x=self.relu(x)
        
        if self.DG_method=='DG_GR':
            
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            
            
            xo=self.fc(x)
            if self.distill==True:
                return xo,y,x
            else:
                return xo,y
        else:
        
            xo = self.fc(x)
            
            if self.distill==True:
                return xo,x
            else:
                return xo
        
#         return x
class simple_conv_net_teacher_self_distill_leadI_net(nn.Module):
    
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(simple_conv_net_teacher_self_distill_leadI_net,self).__init__()
        self.DG_method=DG_method
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        
        
        
        
        self.conv2 = nn.Conv1d(32, 16, kernel_size=15, stride=8, padding=7, bias=False) 
        self.bn2 = nn.BatchNorm1d(16)
        
        
        
        
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        
        
        

        self.conv_lead1_F1_L1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_F2_L2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1L2_P1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1P1_T1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L2T1_T2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_T1=nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3, bias=False) 
        self.w_lead1_L1_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_P1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1=torch.tensor([1.]).requires_grad_()
        self.bn_lead1_L1=nn.BatchNorm1d(32)
        self.bn_lead1_L2=nn.BatchNorm1d(16)
        self.bn_lead1_P1=nn.BatchNorm1d(32)
        self.bn_lead1_T1=nn.BatchNorm1d(32)
        self.bn_lead1_T2=nn.BatchNorm1d(16)
        self.relu_lead1=nn.ReLU(inplace=True)
        self.fc_lead1=nn.Linear(512,num_classes)
        self.grl_lead1=GRL()
        self.fc_1_domain_lead1 = nn.Linear(512 , 64)
        self.fc_2_domain_lead1 = nn.Linear(64, domain_classes)
        self.relu_domain_lead1 = nn.LeakyReLU(inplace=True)
        self.dp_domain_lead1 = nn.Dropout (p=0.5)
        
    def forward(self,x_12lead):
        x=self.conv1(x_12lead.transpose(-1,-2))
        x_center_F1=self.bn1(x)    # 
        
        x=self.relu(x_center_F1)
        x=self.maxpool(x)
        
        
        x=self.conv2(x)
        x_center_F2=self.bn2(x)
        x=x_center_F2.view(x_center_F2.shape[0],512)
        x=self.relu(x)
        
        if self.DG_method=='teacher_self_distill_leadI_GRL':
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            xo=self.fc(x)
            
            
            lead1_L1=self.bn_lead1_L1(self.conv_lead1_F1_L1(x_center_F1))
            lead1_L2=self.bn_lead1_L2(self.conv_lead1_F2_L2(x_center_F2))
            lead1_P1=self.bn_lead1_P1(self.conv_lead1_L1L2_P1(F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[0]*lead1_L1+F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[1]*(F.interpolate(lead1_L2.unsqueeze(0),size=(32,500), mode='bilinear', align_corners=False).squeeze(0))))
            lead1_T1=self.bn_lead1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[0]*self.conv_lead1_T1(x_12lead.transpose(-1,-2))+self.conv_lead1_L1P1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[1]*lead1_P1+F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[2]*lead1_L1))                     # 各导联只有这个数字取得不同。
            lead1_T2=self.bn_lead1_T2(self.conv_lead1_L2T1_T2(F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[0]*(F.interpolate(lead1_T1.unsqueeze(0),size=(16,32) , mode='bilinear', align_corners=False).squeeze(0))+F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[1]*lead1_L2 ))
            x_lead1=lead1_T2.view(lead1_T2.shape[0],512)
            x_lead1=self.relu(x_lead1)
            y_lead1=self.grl_lead1(x_lead1)
            y_lead1=self.fc_1_domain_lead1(y_lead1)  # 64
            y_lead1=self.dp_domain_lead1(y_lead1)
            y_lead1=self.relu_domain_lead1(y_lead1)
            y_lead1=self.fc_2_domain_lead1(y_lead1)  # 3
            xo_lead1=self.fc_lead1(x_lead1)
            
            return xo,y,x_center_F1,x_center_F2,xo_lead1,y_lead1,lead1_T1,lead1_T2
        else:
            print(self.DG_method)
            print('forward error,check DG_method!')
            return 0
    
        
class SimpleGRU_teacher_self_distill_leadI_net(nn.Module):  
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(SimpleGRU_teacher_self_distill_leadI_net, self).__init__()  
        self.DG_method=DG_method
        self.distill=distill
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=4, padding=14,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        
        self.gru = nn.GRU(input_size=128, hidden_size=2, num_layers=1,  \
                          batch_first=True,bidirectional=True)  
        
        
        
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
  


        self.conv_lead1_F1_L1=nn.Conv1d(128,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_F2_L2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1L2_P1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1P1_T1=nn.Conv1d(32,128,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L2T1_T2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_T1=nn.Conv1d(12, 128, kernel_size=7, stride=4, padding=14,bias=False)
        self.w_lead1_L1_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_P1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1=torch.tensor([1.]).requires_grad_()
        self.bn_lead1_L1=nn.BatchNorm1d(32)
        self.bn_lead1_L2=nn.BatchNorm1d(16)
        self.bn_lead1_P1=nn.BatchNorm1d(32)
        self.bn_lead1_T1=nn.BatchNorm1d(128)
        self.bn_lead1_T2=nn.BatchNorm1d(16)
        self.relu_lead1=nn.ReLU(inplace=True)
        self.fc_lead1=nn.Linear(512,num_classes)
        self.grl_lead1=GRL()
        self.fc_1_domain_lead1 = nn.Linear(512 , 64)
        self.fc_2_domain_lead1 = nn.Linear(64, domain_classes)
        self.relu_domain_lead1 = nn.LeakyReLU(inplace=True)
        self.dp_domain_lead1 = nn.Dropout (p=0.5)
        
        
    def forward(self, x_12lead):  
        # Convolutional layer  
        x = self.conv1(x_12lead.transpose(-1,-2))  # 1000, N -->128,256
        x_center_F1 = self.bn1(x)
#         print(1,x_center_F1.shape)
        x = self.relu(x_center_F1)
        x = self.maxpool(x)
        
        x_center_F2,cell = self.gru(x.transpose(-1,-2)) 
        x = x_center_F2.reshape(x_center_F2.shape[0],512)
        x=self.relu(x)
        
        
        if self.DG_method=='teacher_self_distill_leadI_GRL':

            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            xo=self.fc(x)
            
            
            lead1_L1=self.bn_lead1_L1(self.conv_lead1_F1_L1(x_center_F1))
            lead1_L2=self.bn_lead1_L2(self.conv_lead1_F2_L2(x_center_F2.reshape(x_center_F2.shape[0],16,32)))
            lead1_P1=self.bn_lead1_P1(self.conv_lead1_L1L2_P1(F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[0]*lead1_L1+F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[1]*(F.interpolate(lead1_L2.unsqueeze(0),size=(32,256), mode='bilinear', align_corners=False).squeeze(0))))
            lead1_T1=self.bn_lead1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[0]*self.conv_lead1_T1(x_12lead.transpose(-1,-2))+self.conv_lead1_L1P1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[1]*lead1_P1+F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[2]*lead1_L1))                     # 各导联只有这个数字取得不同。
#             print(2,lead1_T1.shape)
            lead1_T2=self.bn_lead1_T2(self.conv_lead1_L2T1_T2(F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[0]*(F.interpolate(lead1_T1.unsqueeze(0),size=(16,32) , mode='bilinear', align_corners=False).squeeze(0))+F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[1]*lead1_L2 )).reshape(x_center_F2.shape[0],128,4)
            x_lead1=lead1_T2.view(lead1_T2.shape[0],512)
            x_lead1=self.relu(x_lead1)
            y_lead1=self.grl_lead1(x_lead1)
            y_lead1=self.fc_1_domain_lead1(y_lead1)  # 64
            y_lead1=self.dp_domain_lead1(y_lead1)
            y_lead1=self.relu_domain_lead1(y_lead1)
            y_lead1=self.fc_2_domain_lead1(y_lead1)  # 3
            xo_lead1=self.fc_lead1(x_lead1)
#             print(x_center_F2.shape,lead1_T2.shape)
            return xo,y,x_center_F1,x_center_F2,xo_lead1,y_lead1,lead1_T1,lead1_T2
        else:
            print(self.DG_method)
            print('forward error,check DG_method!')
            return 0
    
class simple_fc_teacher_self_distill_leadI_net(nn.Module):  
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(simple_fc_teacher_self_distill_leadI_net, self).__init__()  
        self.DG_method=DG_method
        self.distill=distill
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=4, padding=14,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        
        self.fc_mid = nn.Linear(32*128,512)
    
    
        self.grl = GRL()
        self.fc = nn.Linear(512 , num_classes)
        self.fc_1_domain = nn.Linear(512 , 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
  


        self.conv_lead1_F1_L1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_F2_L2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1L2_P1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1P1_T1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L2T1_T2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_T1=nn.Conv1d(12, 32, kernel_size=7, stride=4, padding=14,bias=False)
        self.w_lead1_L1_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_P1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1=torch.tensor([1.]).requires_grad_()
        self.bn_lead1_L1=nn.BatchNorm1d(32)
        self.bn_lead1_L2=nn.BatchNorm1d(16)
        self.bn_lead1_P1=nn.BatchNorm1d(32)
        self.bn_lead1_T1=nn.BatchNorm1d(32)
        self.bn_lead1_T2=nn.BatchNorm1d(16)
        self.relu_lead1=nn.ReLU(inplace=True)
        self.fc_lead1=nn.Linear(512,num_classes)
        self.grl_lead1=GRL()
        self.fc_1_domain_lead1 = nn.Linear(512 , 64)
        self.fc_2_domain_lead1 = nn.Linear(64, domain_classes)
        self.relu_domain_lead1 = nn.LeakyReLU(inplace=True)
        self.dp_domain_lead1 = nn.Dropout (p=0.5)

    def forward(self, x_12lead):  
        # Convolutional layer  
        x = self.conv1(x_12lead.transpose(-1,-2))  # 1000, N -->32,256
        x_center_F1 = self.bn1(x)
        
        x = self.relu(x_center_F1)
        x = self.maxpool(x) # 32,128
        
        
        x_center_F2= self.fc_mid(x.reshape(x.shape[0],32*128)) 
#         x = x.reshape(x.shape[0],512)
        x=self.relu(x_center_F2)
        
        if self.DG_method=='teacher_self_distill_leadI_GRL':

            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            xo=self.fc(x)
            
            
            lead1_L1=self.bn_lead1_L1(self.conv_lead1_F1_L1(x_center_F1))
            lead1_L2=self.bn_lead1_L2(self.conv_lead1_F2_L2(x_center_F2.reshape(x_center_F2.shape[0],16,32)))
            lead1_P1=self.bn_lead1_P1(self.conv_lead1_L1L2_P1(F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[0]*lead1_L1+F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[1]*(F.interpolate(lead1_L2.unsqueeze(0),size=(32,256), mode='bilinear', align_corners=False).squeeze(0))))
            lead1_T1=self.bn_lead1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[0]*self.conv_lead1_T1(x_12lead.transpose(-1,-2))+self.conv_lead1_L1P1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[1]*lead1_P1+F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[2]*lead1_L1))                     # 各导联只有这个数字取得不同。
            lead1_T2=self.bn_lead1_T2(self.conv_lead1_L2T1_T2(F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[0]*(F.interpolate(lead1_T1.unsqueeze(0),size=(16,32) , mode='bilinear', align_corners=False).squeeze(0))+F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[1]*lead1_L2 )).reshape(x_center_F2.shape[0],512)
            x_lead1=lead1_T2.view(lead1_T2.shape[0],512)
            x_lead1=self.relu(x_lead1)
            y_lead1=self.grl_lead1(x_lead1)
            y_lead1=self.fc_1_domain_lead1(y_lead1)  # 64
            y_lead1=self.dp_domain_lead1(y_lead1)
            y_lead1=self.relu_domain_lead1(y_lead1)
            y_lead1=self.fc_2_domain_lead1(y_lead1)  # 3
            xo_lead1=self.fc_lead1(x_lead1)
#             print(x_center_F2.shape,lead1_T2.shape)

            return xo,y,x_center_F1,x_center_F2,xo_lead1,y_lead1,lead1_T1,lead1_T2
        else:
            print(self.DG_method)
            print('forward error,check DG_method!')
            return 0
    
    
    
class ResNet_self(nn.Module):

    def __init__(self, block, layers, input_channels=1, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,DG_method=None,domain_classes=1,distill=False):
        super(ResNet_self, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.distill=distill
        self.DG_method=DG_method
        
#         self.domain_classes=domain_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_1_domain = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.fc_1_domain_n = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_n = nn.Linear(64, domain_classes)
        self.relu_domain_n=nn.LeakyReLU(inplace=True)
        self.dp_domain_n=nn.Dropout (p=0.5)
        self.fc_1_domain_a = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_a = nn.Linear(64, domain_classes)
        self.relu_domain_a=nn.LeakyReLU(inplace=True)
        self.dp_domain_a=nn.Dropout (p=0.5)
        self.fc_remover_1=nn.Linear(512*block.expansion,64)
        self.fc_remover_2=nn.Linear(64,num_classes)
#         self.relu_remover=nn.LeakyReLU(inplace=True)
        self.fc_1_domain_o = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_o = nn.Linear(64, domain_classes)
        self.relu_domain_o=nn.LeakyReLU(inplace=True)
        self.dp_domain_o=nn.Dropout (p=0.5)

        self.grl = GRL()
        self.grl_class = GRL()
        
        


        self.conv_lead1_F1_L1=nn.Conv1d(64,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_F2_L2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1L2_P1=nn.Conv1d(32,32,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L1P1_T1=nn.Conv1d(32,64,3,stride=1,padding=1,bias=False)
        self.conv_lead1_L2T1_T2=nn.Conv1d(16,16,3,stride=1,padding=1,bias=False)
        self.conv_lead1_T1=nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.w_lead1_L1_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_P1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_P1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L1_T1=torch.tensor([1.]).requires_grad_()
        self.w_lead1_L2_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1_T2=torch.tensor([1.]).requires_grad_()
        self.w_lead1_T1=torch.tensor([1.]).requires_grad_()
        self.bn_lead1_L1=nn.BatchNorm1d(32)
        self.bn_lead1_L2=nn.BatchNorm1d(16)
        self.bn_lead1_P1=nn.BatchNorm1d(32)
        self.bn_lead1_T1=nn.BatchNorm1d(64)
        self.bn_lead1_T2=nn.BatchNorm1d(16)
        self.relu_lead1=nn.ReLU(inplace=True)
        self.fc_lead1=nn.Linear(512,num_classes)
        self.grl_lead1=GRL()
        self.fc_1_domain_lead1 = nn.Linear(512 , 64)
        self.fc_2_domain_lead1 = nn.Linear(64, domain_classes)
        self.relu_domain_lead1 = nn.LeakyReLU(inplace=True)
        self.dp_domain_lead1 = nn.Dropout (p=0.5)

        
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
            

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x_12lead):
        # See note [TorchScript super()]
        
        x=x_12lead.transpose(1,2)
        
        x = self.conv1(x)
        x_center_F1 = self.bn1(x)
        x = self.relu(x_center_F1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_center_F2 = self.avgpool(x)
#         print(x_center_F2.shape)
        # 应该为16，32
        x = torch.flatten(x_center_F2, 1)
        
#         print(x.shape)
        
        #print("feature", x)
        
        
        
        
        
        
        
        
        if self.DG_method=='teacher_self_distill_leadI_GRL':
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            xo=self.fc(x)
            
            
            lead1_L1=self.bn_lead1_L1(self.conv_lead1_F1_L1(x_center_F1))
            lead1_L2=self.bn_lead1_L2(self.conv_lead1_F2_L2(x_center_F2.reshape(x_center_F2.shape[0],16,32)))
            lead1_P1=self.bn_lead1_P1(self.conv_lead1_L1L2_P1(F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[0]*lead1_L1+F.softmax(torch.cat([self.w_lead1_L1_P1,self.w_lead1_L2_P1]),dim=0)[1]*(F.interpolate(lead1_L2.unsqueeze(0),size=(32,500), mode='bilinear', align_corners=False).squeeze(0))))
            lead1_T1=self.bn_lead1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[0]*self.conv_lead1_T1(x_12lead.transpose(-1,-2))+self.conv_lead1_L1P1_T1(F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[1]*lead1_P1+F.softmax(torch.cat([self.w_lead1_T1,self.w_lead1_P1_T1,self.w_lead1_L1_T1]),dim=0)[2]*lead1_L1))                     # 各导联只有这个数字取得不同。
            lead1_T2=self.bn_lead1_T2(self.conv_lead1_L2T1_T2(F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[0]*(F.interpolate(lead1_T1.unsqueeze(0),size=(16,32) , mode='bilinear', align_corners=False).squeeze(0))+F.softmax(torch.cat([self.w_lead1_T1_T2,self.w_lead1_L2_T2]),dim=0)[1]*lead1_L2 )).reshape(x_center_F2.shape[0],512,1)
            x_lead1=lead1_T2.view(lead1_T2.shape[0],512)
            x_lead1=self.relu(x_lead1)
            y_lead1=self.grl_lead1(x_lead1)
            y_lead1=self.fc_1_domain_lead1(y_lead1)  # 64
            y_lead1=self.dp_domain_lead1(y_lead1)
            y_lead1=self.relu_domain_lead1(y_lead1)
            y_lead1=self.fc_2_domain_lead1(y_lead1)  # 3
            xo_lead1=self.fc_lead1(x_lead1)
            
            return xo,y,x_center_F1,x_center_F2,xo_lead1,y_lead1,lead1_T1,lead1_T2
        else:
            print(self.DG_method)
            print('forward error,check DG_method!')
            return 0
        
        

    def forward(self, x):
        return self._forward_impl(x)
    
    
class simple_Conv_tongleiduibi(nn.Module):
    def __init__(self,input_channels=1,num_classes=2,DG_method=None,domain_classes=1,distill=False):
        super(simple_Conv_tongleiduibi,self).__init__()
        self.distill=distill
        self.DG_method=DG_method
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=15, stride=8, padding=7,
                               bias=False) # change according to the leads of the input ECG data
        self.bn2 = nn.BatchNorm1d(16)
        self.grl = GRL()
        self.fc_A = nn.Linear(16512 , num_classes)
        self.fc_B1 = nn.Linear(512 , 64)
        self.fc_B2 = nn.Linear(64 , num_classes)
        
        self.fc = nn.Linear(512 , num_classes)
        
        
        self.fc_C1_1 = nn.Linear(512 , 100)
        self.fc_C2_1 = nn.Linear(100 , 1)
        
        self.fc_C1_2 = nn.Linear(512 , 100)
        self.fc_C2_2 = nn.Linear(100 , 1)
        
        self.fc_C1_3 = nn.Linear(512 , 100)
        self.fc_C2_3 = nn.Linear(100 , 1)
        
        self.fc_C1_4 = nn.Linear(512 , 100)
        self.fc_C2_4 = nn.Linear(100 , 1)
        
        self.fc_C1_5 = nn.Linear(512 , 100)
        self.fc_C2_5 = nn.Linear(100 , 1)
        
        self.fc_C1_6 = nn.Linear(512 , 100)
        self.fc_C2_6 = nn.Linear(100 , 1)
        
        self.fc_C1_7 = nn.Linear(512 , 100)
        self.fc_C2_7 = nn.Linear(100 , 1)
        
        self.fc_C1_8 = nn.Linear(512 , 100)
        self.fc_C2_8 = nn.Linear(100 , 1)
        
        self.fc_C1_9 = nn.Linear(512 , 100)
        self.fc_C2_9 = nn.Linear(100 , 1)
        
        self.fc_C1_10 = nn.Linear(512 , 100)
        self.fc_C2_10 = nn.Linear(100 , 1)
        
        self.fc_C1_11 = nn.Linear(512 , 100)
        self.fc_C2_11 = nn.Linear(100 , 1)
        
        self.fc_C1_12 = nn.Linear(512 , 100)
        self.fc_C2_12 = nn.Linear(100 , 1)
        
        self.fc_C1_13 = nn.Linear(512 , 100)
        self.fc_C2_13 = nn.Linear(100 , 1)
        
        self.fc_C1_14 = nn.Linear(512 , 100)
        self.fc_C2_14 = nn.Linear(100 , 1)
        
        self.fc_C1_15 = nn.Linear(512 , 100)
        self.fc_C2_15 = nn.Linear(100 , 1)
        
        self.fc_C1_16 = nn.Linear(512 , 100)
        self.fc_C2_16 = nn.Linear(100 , 1)
        
        self.fc_C1_17 = nn.Linear(512 , 100)
        self.fc_C2_17 = nn.Linear(100 , 1)
        
        self.fc_C1_18 = nn.Linear(512 , 100)
        self.fc_C2_18 = nn.Linear(100 , 1)
        
        self.fc_C1_19 = nn.Linear(512 , 100)
        self.fc_C2_19 = nn.Linear(100 , 1)
        
        self.fc_C1_20 = nn.Linear(512 , 100)
        self.fc_C2_20 = nn.Linear(100 , 1)
        
        self.fc_C1_21 = nn.Linear(512 , 100)
        self.fc_C2_21 = nn.Linear(100 , 1)
        
        self.fc_C1_22 = nn.Linear(512 , 100)
        self.fc_C2_22 = nn.Linear(100 , 1)
        
        self.fc_C1_23 = nn.Linear(512 , 100)
        self.fc_C2_23 = nn.Linear(100 , 1)
        
        self.fc_C1_24 = nn.Linear(512 , 100)
        self.fc_C2_24 = nn.Linear(100 , 1)
        
        self.fc_C1_25 = nn.Linear(512 , 100)
        self.fc_C2_25 = nn.Linear(100 , 1)
        
        self.fc_C1_26 = nn.Linear(512 , 100)
        self.fc_C2_26 = nn.Linear(100 , 1)
        
        self.fc_C1_27 = nn.Linear(512 , 100)
        self.fc_C2_27 = nn.Linear(100 , 1)
        
        self.fc_C3 = nn.Linear(27 , num_classes)
        self.fc_B1_domain = nn.Linear(512 , 128)
        self.fc_C1_domain = nn.Linear(512 , 64)
        self.fc_B2_domain = nn.Linear(128 , 32)
        self.fc_B3_domain = nn.Linear(32, domain_classes)
        self.fc_C2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.dp=nn.Dropout (p=0.5)
        self.softmax_B_layer = nn.Softmax(dim=1) 
        self.softmax_C_layer = nn.Softmax(dim=1) 
        self.sigmoid_B_layer = nn.Sigmoid()
        self.sigmoid_C_layer = nn.Sigmoid()
        
        
    def forward(self,x):
        x=self.conv1(x.transpose(-1,-2))
        x=self.bn1(x)
        x_A_1=self.relu(x)
        x=self.maxpool(x_A_1)
        x=self.conv2(x)
        x=self.bn2(x)
        x=x.view(x.shape[0],512)
        x=self.relu(x)
        
        if self.DG_method==None:
            xo=self.fc(x)
            return xo
        
        elif self.DG_method=='DG_GR_method_A':
            x_A_1=x_A_1.view(x_A_1.shape[0],-1)
            x_A_2=torch.cat((x_A_1,x),dim=1)
#             print(x_A_2.shape)  # 16512
            x=self.fc_A(x_A_2)
            return x
            
            
        elif self.DG_method=='DG_GR_method_B':
            
            y=self.grl(x)
            y=self.fc_B1_domain(y)  # 128
            y=self.relu_domain(y)
            y=self.dp_domain(y)
            y=self.fc_B2_domain(y)  # 32
            y=self.relu_domain(y)
            y=self.dp_domain(y)
            y=self.fc_B3_domain(y)  # domain_classes
            y=self.softmax_B_layer(y)
            
            xo=self.fc_B1(x)
            xo=self.fc_B2(xo)
            xo=self.sigmoid_B_layer(xo)
            
            return xo,y
        elif self.DG_method=='DG_GR_method_C':
            
            y=self.grl(x)
            y=self.fc_C1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_C2_domain(y)  # 4
            y=self.softmax_C_layer(y)
            
            
            xo=self.fc_C1_1(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_1=self.fc_C2_1(xo)
            
            xo=self.fc_C1_2(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_2=self.fc_C2_2(xo)
            
            xo=self.fc_C1_3(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_3=self.fc_C2_3(xo)
            
            xo=self.fc_C1_4(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_4=self.fc_C2_4(xo)
            
            xo=self.fc_C1_5(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_5=self.fc_C2_5(xo)
            
            xo=self.fc_C1_6(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_6=self.fc_C2_6(xo)
            
            xo=self.fc_C1_7(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_7=self.fc_C2_7(xo)
            
            xo=self.fc_C1_8(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_8=self.fc_C2_8(xo)
            
            xo=self.fc_C1_9(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_9=self.fc_C2_9(xo)
            
            xo=self.fc_C1_10(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_10=self.fc_C2_10(xo)
            
            xo=self.fc_C1_11(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_11=self.fc_C2_11(xo)
            
            xo=self.fc_C1_12(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_12=self.fc_C2_12(xo)
            
            xo=self.fc_C1_13(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_13=self.fc_C2_13(xo)
            
            xo=self.fc_C1_14(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_14=self.fc_C2_14(xo)
            
            xo=self.fc_C1_15(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_15=self.fc_C2_15(xo)
            
            xo=self.fc_C1_16(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_16=self.fc_C2_16(xo)
            
            xo=self.fc_C1_17(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_17=self.fc_C2_17(xo)
            
            xo=self.fc_C1_18(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_18=self.fc_C2_18(xo)
            
            xo=self.fc_C1_19(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_19=self.fc_C2_19(xo)
            
            xo=self.fc_C1_20(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_20=self.fc_C2_20(xo)
            
            xo=self.fc_C1_21(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_21=self.fc_C2_21(xo)
            
            xo=self.fc_C1_22(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_22=self.fc_C2_22(xo)
            
            xo=self.fc_C1_23(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_23=self.fc_C2_23(xo)
            
            xo=self.fc_C1_24(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_24=self.fc_C2_24(xo)
            
            xo=self.fc_C1_25(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_25=self.fc_C2_25(xo)
            
            xo=self.fc_C1_26(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_26=self.fc_C2_26(xo)
            
            xo=self.fc_C1_27(x)
            xo=self.dp(xo)
            xo=self.relu(xo)
            xo_27=self.fc_C2_27(xo)
            
            
            xo_C_concat= torch.cat((xo_1,xo_2,xo_3,xo_4,xo_5,xo_6,xo_7,xo_8,xo_9,xo_10, \
                                   xo_11,xo_12,xo_13,xo_14,xo_15,xo_16,xo_17,xo_18,xo_19,xo_20, \
                                   xo_21,xo_22,xo_23,xo_24,xo_25,xo_26,xo_27), dim=1)  
            
            xo_C_concat=self.sigmoid_C_layer(xo_C_concat)
            
            xo_C=self.fc_C3(xo_C_concat)
            
            
            return xo_C,y
            
            
        elif self.DG_method=='MMD':
            x_cls=self.fc(x)
            return x,x_cls
        
        else:
            print('forward error!')

class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels=1, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,DG_method=None,domain_classes=1,distill=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.distill=distill
        self.DG_method=DG_method
        
#         self.domain_classes=domain_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_1_domain = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.fc_1_domain_n = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_n = nn.Linear(64, domain_classes)
        self.relu_domain_n=nn.LeakyReLU(inplace=True)
        self.dp_domain_n=nn.Dropout (p=0.5)
        self.fc_1_domain_a = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_a = nn.Linear(64, domain_classes)
        self.relu_domain_a=nn.LeakyReLU(inplace=True)
        self.dp_domain_a=nn.Dropout (p=0.5)
        self.fc_remover_1=nn.Linear(512*block.expansion,64)
        self.fc_remover_2=nn.Linear(64,num_classes)
#         self.relu_remover=nn.LeakyReLU(inplace=True)
        self.fc_1_domain_o = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_o = nn.Linear(64, domain_classes)
        self.relu_domain_o=nn.LeakyReLU(inplace=True)
        self.dp_domain_o=nn.Dropout (p=0.5)

        self.grl = GRL()
        self.grl_class = GRL()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x=x.transpose(1,2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
#         print(x.shape)
        
        #print("feature", x)
        
        if self.DG_method=='DG_GR':
            
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            
            
            xo=self.fc(x)
            if self.distill==True:
                return xo,y,x
            else:
                return xo,y
        
        elif self.DG_method=='remover':
            
            y=self.fc_remover_1(x)  #64
            z=self.fc_remover_2(y)  # 3
            
            return y,z         # 64,3
        
        elif self.DG_method=='class_condition_GRL':
            
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            
            y_class=self.grl_class(x)
            
            y_n=self.fc_1_domain_n(y_class)
            y_n=self.dp_domain_n(y_n)
            y_n=self.relu_domain_n(y_n)
            y_n=self.fc_2_domain_n(y_n)
            
            y_a=self.fc_1_domain_a(y_class)
            y_a=self.dp_domain_a(y_a)
            y_a=self.relu_domain_a(y_a)
            y_a=self.fc_2_domain_a(y_a)
            
            y_o=self.fc_1_domain_o(y_class)
            y_o=self.dp_domain_o(y_o)
            y_o=self.relu_domain_o(y_o)
            y_o=self.fc_2_domain_o(y_o)
            
            x=self.fc(x)
            
            return x,y,y_n,y_a,y_o
        
        elif self.DG_method=='MMD':
            x_cls=self.fc(x)
            return x,x_cls
        
        else:
        
            xo = self.fc(x)
            
            if self.distill==True:
                return xo,x
            else:
                return xo

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_show_2d(nn.Module):

    def __init__(self, block, layers, input_channels=1, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,DG_method=None,domain_classes=1,distill=False):
        super(ResNet_show_2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.distill=distill
        self.DG_method=DG_method
        self.show_2d=False
        
#         self.domain_classes=domain_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) # change according to the leads of the input ECG data
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_1 = nn.Linear(512 * block.expansion, 2)
        self.fc_2 = nn.Linear(2, num_classes)
        self.fc_1_domain = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.LeakyReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.fc_1_domain_n = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_n = nn.Linear(64, domain_classes)
        self.relu_domain_n=nn.LeakyReLU(inplace=True)
        self.dp_domain_n=nn.Dropout (p=0.5)
        self.fc_1_domain_a = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_a = nn.Linear(64, domain_classes)
        self.relu_domain_a=nn.LeakyReLU(inplace=True)
        self.dp_domain_a=nn.Dropout (p=0.5)
        self.fc_remover_1=nn.Linear(512*block.expansion,64)
        self.fc_remover_2=nn.Linear(64,num_classes)
#         self.relu_remover=nn.LeakyReLU(inplace=True)
        self.fc_1_domain_o = nn.Linear(512 * block.expansion, 64)
        self.fc_2_domain_o = nn.Linear(64, domain_classes)
        self.relu_domain_o=nn.LeakyReLU(inplace=True)
        self.dp_domain_o=nn.Dropout (p=0.5)

        self.grl = GRL()
        self.grl_class = GRL()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x=x.transpose(1,2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
#         print(x.shape)
        
        #print("feature", x)
        
        xo1 = self.fc_1(x)
        xo = self.fc_2(xo1)

        if self.show_2d==True:
            return xo1
        else:
            return xo

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def _resnet_show_2d(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_show_2d(block, layers, **kwargs)
    return model

def _resnet_self(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_self(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet18_show_2d(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_show_2d('resnet18_show_2d', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def simple_gru(**kwargs):
    return SimpleGRU(**kwargs)

def simple_fc(**kwargs):
    return SimpleFC(**kwargs)

def simple_conv_net(**kwargs):
    return simple_Conv(**kwargs)

def simple_conv_net_tongleiduibi(**kwargs):
    return simple_Conv_tongleiduibi(**kwargs)

def simple_conv_net_teacher_self_distill_leadI (**kwargs):
    return simple_conv_net_teacher_self_distill_leadI_net(**kwargs)

def simple_fc_teacher_self_distill_leadI (**kwargs):
    return simple_fc_teacher_self_distill_leadI_net(**kwargs)

def simple_gru_teacher_self_distill_leadI (**kwargs):
    return SimpleGRU_teacher_self_distill_leadI_net(**kwargs)

def resnet18_teacher_self_distill_leadI (pretrained=False, progress=True,**kwargs):
    return _resnet_self('resnet18_teacher_self_distill_leadI_net', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
