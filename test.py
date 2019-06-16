# -*- coding: utf-8 -*-


import argparse
from torchvision import transforms
import os
import net
from torch import nn
import torch
from PIL import Image
from utils import *
from torchvision.utils import save_image
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, help='File path to content image')
parser.add_argument('--style', type=str, help='File path to style image')
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
parser.add_argument('--img_size', type=int, default=512, help='Minimum size for images, keeping the original size if given 0')
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the stylized images')
parser.add_argument('--style_interpolation_weights', type=str, default='', help='The weight for blending the multiple style images')
parser.add_argument('--decoder', type=str, default='models/decoder.pth', help='Path for the arguments of decoder')
parser.add_argument('--alpha', type=float, default=1.0, help='a smooth transition between content-similarity and style-similarity can be observed by changing α from 0 to 1.0')
parser.add_argument('--perserve_color', action='store_true', help='If specified, preserve color of the content image')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def transform(size, crop=None):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, i) for i in os.listdir(args.content_dir)]
    
do_interpolation = False
if args.style:
    style_paths = args.style.split(',')  # list
    if len(style_paths) == 1:
        style_paths = [args.style]
    else:
        do_interpolation = True
        style_weights = [int(w) for w in args.style_interpolation_weights.split(',')]
        style_weights = [w/sum(style_weights) for w in style_weights]
else:
    style_paths = [os.path.join(args.style_dir, i) for i in os.listdir(args.style_dir)]
    

vgg_encoder = net.vgg
vgg_encoder.eval()
vgg_encoder.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])

decoder = net.decoder
decoder.eval()
decoder.load_state_dict(torch.load(args.decoder))

vgg_encoder.to(device)
decoder.to(device)

def style_transfer(vgg_encoder, decoder, content, style, alpha=1.0, style_weights=None):  # N x C x H x W    
    content_f = vgg_encoder(content)
    style_f = vgg_encoder(style)
    if style_weights:
        _, C, H, W = content_f.size()
        t = torch.FloatTensor(1, C, H, W).zero_().to(device)
        t_temp = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(style_weights):
            t = t + w * t_temp[i:i+1]
        content_f = content_f[0:1]
    else:
        t = adaptive_instance_normalization(content_f, style_f)
    T = (1 - alpha) * content_f + alpha * t
    return decoder(T)


#ipdb.set_trace()
for c in content_paths:
    if do_interpolation:
        style = torch.stack([transform(args.img_size, args.crop)(Image.open(p).convert('RGB')) for p in style_paths]).to(device)
        content = transform(args.img_size, args.crop)(Image.open(c).convert('RGB')).unsqueeze(0).expand_as(style).to(device)
        with torch.no_grad():
            output = style_transfer(vgg_encoder, decoder, content, style, args.alpha, style_weights)
        save_image(output.cpu(), '{:s}/{:s}_interpolation.png'.format(args.output_dir, os.path.basename(c).split('.')[0]))
    else:
        for s in style_paths:
            style = transform(args.img_size)(Image.open(s).convert('RGB'))
            content = transform(args.img_size)(Image.open(c).convert('RGB'))
            if args.perserve_color:
                style = coral(style, content)
            style = style.unsqueeze(0).to(device)
            content = content.unsqueeze(0).to(device)
            with torch.no_grad():
                output = style_transfer(vgg_encoder, decoder, content, style, args.alpha)
            save_image(output.cpu(), '{:s}/{:s}_stylized_by_{:s}.png'.format(args.output_dir, os.path.basename(c).split('.')[0], os.path.basename(s).split('.')[0]))

