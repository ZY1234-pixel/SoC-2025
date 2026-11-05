"""Runtime-only architecture and crop logic for the trained spine heatmap."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16


SIZE=384
LINE_HALF_LENGTH=144
ROWS=np.arange(64,320)
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]


def affine_for_line(points):
    points=np.asarray(points,dtype=np.float64)
    midpoint=points.mean(axis=0)
    tangent=points[1]-points[0]
    length=float(np.linalg.norm(tangent))
    if length<10:
        raise ValueError('Degenerate spine line')
    tangent/=length
    normal=np.array([-tangent[1],tangent[0]])
    scale=(2*LINE_HALF_LENGTH)/length
    return np.array([
        [scale*normal[0],scale*normal[1],SIZE/2-scale*np.dot(normal,midpoint)],
        [scale*tangent[0],scale*tangent[1],SIZE/2-scale*np.dot(tangent,midpoint)],
    ],dtype=np.float32)


def crop_with_line(image,coarse):
    matrix=affine_for_line(coarse)
    crop=cv2.warpAffine(image,matrix,(SIZE,SIZE),flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(180,180,180))
    return crop,matrix


def image_tensor(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
    return (tensor-MEAN)/STD


class ConvBlock(nn.Module):
    def __init__(self,channels_in,channels_out):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(channels_in,channels_out,3,padding=1),
            nn.BatchNorm2d(channels_out),nn.ReLU(inplace=True),
            nn.Conv2d(channels_out,channels_out,3,padding=1),
            nn.BatchNorm2d(channels_out),nn.ReLU(inplace=True))

    def forward(self,x):
        return self.net(x)


class SpineLineNet(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError('Runtime model loads all parameters from heatmap.pt')
        self.encoder=vgg16(weights=None).features[:23]
        self.dec3=ConvBlock(512+256,192)
        self.dec2=ConvBlock(192+128,96)
        self.dec1=ConvBlock(96+64,48)
        self.head=nn.Sequential(nn.Conv2d(49,32,3,padding=1),nn.ReLU(inplace=True),
                                nn.Conv2d(32,1,1))

    def forward(self,x):
        features=[]
        for i,layer in enumerate(self.encoder):
            x=layer(x)
            if i in (3,8,15,22):
                features.append(x)
        one,two,three,four=features
        x=self.dec3(torch.cat([F.interpolate(four,size=three.shape[-2:],mode='bilinear',align_corners=False),three],1))
        x=self.dec2(torch.cat([F.interpolate(x,size=two.shape[-2:],mode='bilinear',align_corners=False),two],1))
        x=self.dec1(torch.cat([F.interpolate(x,size=one.shape[-2:],mode='bilinear',align_corners=False),one],1))
        horizontal=torch.arange(SIZE,device=x.device,dtype=x.dtype)
        prior=torch.exp(-.5*((horizontal-SIZE/2)/55).square())[None,None,None,:]
        prior=prior.expand(x.shape[0],1,SIZE,SIZE)
        return self.head(torch.cat([x,prior],1)).squeeze(1)


@torch.no_grad()
def refine_endpoints(model,image,coarse,device='cuda'):
    crop,matrix=crop_with_line(image,coarse)
    logits=model(image_tensor(crop)[None].to(device))[0]
    probability=torch.softmax(logits[64:320],dim=-1)
    x=(probability*torch.arange(SIZE,device=probability.device)).sum(-1).cpu().numpy()
    confidence=probability.max(-1).values.cpu().numpy()
    valid=confidence>=np.quantile(confidence,.25)
    coefficients=np.polyfit(ROWS[valid],x[valid],1,w=np.sqrt(confidence[valid]))
    y=np.array([SIZE/2-LINE_HALF_LENGTH,SIZE/2+LINE_HALF_LENGTH])
    canonical=np.column_stack((np.polyval(coefficients,y),y)).astype(np.float32)
    inverse=cv2.invertAffineTransform(matrix)
    refined=cv2.transform(canonical[None],inverse)[0]
    return refined,{'line_confidence':float(np.mean(confidence)),
                    'slope':float(coefficients[0])}
