# -*- coding: utf-8 -*-

from torch import nn
from utils import *
import ipdb

vgg = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, 3, 1),
        nn.ReLU(),    # ReLU1_1    :4
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 128, 3, 1),
        nn.ReLU(),    # ReLU2_1    4:11
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 256, 3, 1),
        nn.ReLU(),    # ReLU3_1    11:18
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 512, 3, 1),
        nn.ReLU(),    # ReLU4_1  this is the last layer used    18:31
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),    # ReLU5_1
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3, 1),
        nn.ReLU(),
        )


decoder = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 256, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1),
        nn.ReLU(),

        )



class Net(nn.Module):    
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])     # input -> ReLU1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])   # ReLU1_1 -> ReLU2_1  
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # ReLU2_1 -> ReLU3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # ReLU3_1 -> ReLu4_1

        self.decoder = decoder
        self.criterion = nn.MSELoss(size_average=False, reduce=True)
          
    def encode_with_intermediate(self, input):
        results = [input]    # result[0] = input
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]    # ReLU1_1, ReLU2_1, ReLU3_1, ReLu4_1
        
    def calc_content_loss(self, input, target):  # f(g(t)) & t
        assert(input.size() == target.size())
        assert(target.requires_grad is False)
        L_c = self.criterion(input, target)
        return L_c
    
    def calc_style_loss(self, input, target):    # \phi_i(g(t)) & \phi_i(s)
        assert(input.size() == target.size())
        assert(target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        L_s = self.criterion(input_mean, target_mean) + self.criterion(input_std, target_std)
        return L_s
    
    def forward(self, content, style, args):
        assert( 0 <= args.alpha <= 1)
        content_f = self.encode_with_intermediate(content)[-1]
        style_fs = self.encode_with_intermediate(style)
        
        t = adaptive_instance_normalization(content_f, style_fs[-1])
        t = (1 - args.alpha) * content_f + args.alpha * t
        
        g_t = self.decoder(t)
        
        g_t_fs = self.encode_with_intermediate(g_t)
        
#        ipdb.set_trace()
        L_c = self.calc_content_loss(g_t_fs[-1], t)
        L_s = self.calc_style_loss(g_t_fs[0], style_fs[0])
        for i in range(3):
            L_s += self.calc_style_loss(g_t_fs[i+1], style_fs[i+1])
        
        return L_c, L_s
    
    
    
    
    
    





















