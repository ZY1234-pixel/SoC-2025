import os
from model import build_model
from engine.val import validate
from utils.converter import Converter
from data import build_dataset, build_dataloader

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

def main(cfg):
    val_dataset = build_dataset(cfg, 'val')
    val_dataloader = build_dataloader(val_dataset, 'val', cfg)
    converter = Converter(cfg['DATA']['DICT'])

    model = build_model(cfg)
    # model = model.cuda()
    model = model.cpu()
    model.eval()

    example = torch.rand(1,3,1760,1264)
    traced_script_module = torch.jit.trace(model,example)
    traced_script_module.save("mthv2.pt")

    # jit_model = torch.jit.load("mthv2.pt")
    # jit_model = torch.jit.script(jit_model)
    optimize_model = optimize_for_mobile(traced_script_module)
    optimize_model.save("mthv2.torchscript.pt")
    optimize_model._save_for_lite_interpreter("mthv2.torchscript.ptl")
    # check_model =
    # optimized_traced_model('mthv2.ptl')

if __name__ == '__main__':
    import yaml
    from utils.parser import default_parser

    parser = default_parser()
    args = parser.parse_args()
    args.config = 'configs/casia-hwdb.yaml'
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    #cfg = yaml.load(open(args.config, 'r'))
    main(cfg)