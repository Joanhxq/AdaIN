# -*- coding: utf-8 -*-


import argparse
import os
from dataset import FlatFolderDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import net
import torch
from torch import nn
from tqdm import tqdm
from sampler import InfiniteSamplerWrapper
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, required=True, help="Directory path to amount of content images")
    parser.add_argument('--style_dir', type=str, required=True, help="Directory path to amount of style images")
    parser.add_argument('--save_models', default='save_models', help="Path to save the trained model")
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch to train')
    parser.add_argument('--alpha', type=float, default=1.0, help='a smooth transition between content-similarity and style-similarity can be observed by changing α from 0 to 1.')
    parser.add_argument('--lambda_weight', type=float, default=10.0, help='The degree of style transfer can be controlled during training by adjusting the style weight λ')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of Adam')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='The decay rate of learning rate')
    parser.add_argument('--max_epoch', type=int, default=160000, help='Number of iterations')
    parser.add_argument('--save_model_interval', type=int, default=10000, help='The interval epoch to save trained model')

    return parser.parse_args()


def adjust_learning_rate(optimizer, iteration, args):
    lr = args.lr / (1.0 + args.lr_decay * iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    args = parse_args()
    
    os.makedirs(args.save_models, exist_ok=True)
    
    device = torch.device('cuda')
    
    vgg = net.vgg
    vgg.load_state_dict(torch.load("models/vgg_normalised.pth"))
    vgg_encoder = nn.Sequential(*list(vgg.children())[:31])
    for param in vgg_encoder.parameters():
        param.requires_grad = False
    
    decoder = net.decoder
    
    network = net.Net(vgg_encoder, decoder)
    network.train()
    network.to(device)
    
    train_transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
            ])
    
    content_dataset = FlatFolderDataset(args.content_dir, train_transform)
    style_dataset = FlatFolderDataset(args.style_dir, train_transform)
    
    content_iter = iter(DataLoader(content_dataset, batch_size = args.batch_size, sampler=InfiniteSamplerWrapper(content_dataset)))
    style_iter = iter(DataLoader(style_dataset, batch_size=args.batch_size, sampler=InfiniteSamplerWrapper(content_dataset)))
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    
    for i in tqdm(range(args.max_epoch)):
        adjust_learning_rate(optimizer, i, args)
        
        content_imgs = next(content_iter)['img'].to(device)
        style_imgs = next(style_iter)['img'].to(device)
        
        optimizer.zero_grad()
#        ipdb.set_trace()
        L_c, L_s = network(content_imgs, style_imgs, args)
        loss = L_c + args.lambda_weight * L_s
        loss.backward()
        
        optimizer.step()
        
        if (i+1) % args.save_model_interval == 0 or (i+1) == args.max_epoch:
            state_dict = decoder.state_dict()
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].to(torch.device('cpu'))
            torch.save(state_dict, '{:s}/decoder_iter_{:d}.pth'.format(args.save_models, i+1))
        
if __name__ == '__main__':
    train()