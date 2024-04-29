import torch
import torch.nn as nn
import torch.nn.functional as F



# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
train_size = (1,3,256,256)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp: 
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        pools, convs, dynas = [],[],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(dynamic_filter(inchannels=k, kernel_size=3))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = AvgPool2d(base_size=120)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)
    def forward(self, x):
        
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.) 

        return out_low + out_high

