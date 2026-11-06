"""Run coarse YOLO spine detection followed by local heatmap refinement."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from infer_spine_auto_orientation import infer_spine
from spine_line_heatmap_runtime import SpineLineNet,refine_endpoints


ROOT=Path(__file__).resolve().parent


def weight_path(package_name,experiment_path):
    packaged=ROOT/'weights'/package_name
    return packaged if packaged.is_file() else ROOT/experiment_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir',type=Path)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--detector',type=Path,
                        default=weight_path('detector.pt','runs_6kpts/pose_1024_orientation_normalized_20261102/weights/best.pt'))
    parser.add_argument('--router',type=Path,
                        default=weight_path('direction.pt','runs_6kpts/pose_1024_orientft/weights/best.pt'))
    parser.add_argument('--heatmap',type=Path,
                        default=weight_path('heatmap.pt','runs_6kpts/spine_line_heatmap_20261103/best.pt'))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    yolo_device=0 if torch.cuda.is_available() else 'cpu'
    heatmap_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector=YOLO(str(args.detector))
    router=YOLO(str(args.router))
    heatmap=SpineLineNet(pretrained=False)
    checkpoint=torch.load(args.heatmap,map_location='cpu',weights_only=True)
    heatmap.load_state_dict(checkpoint['model'])
    heatmap.eval().to(heatmap_device)
    rows=[]
    images=sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    for path in images:
        image=cv2.imread(str(path))
        if image is None:
            continue
        prediction=infer_spine(detector,image,mode='auto',device=yolo_device,orientation_model=router)
        if prediction is None:
            rows.append([path.name,'miss',*(['']*9)])
            continue
        coarse=prediction['endpoints']
        refined,meta=refine_endpoints(heatmap,image,coarse,device=heatmap_device)
        preview=image.copy()
        for points,color in ((coarse,(0,0,255)),(refined,(255,0,0))):
            points=np.rint(points).astype(int)
            cv2.line(preview,tuple(points[0]),tuple(points[1]),color,
                     max(3,round(min(image.shape[:2])/300)))
        cv2.imwrite(str(args.output_dir/path.name),preview)
        rows.append([path.name,prediction['rotation'],meta['line_confidence'],
                     *[float(x) for x in coarse.flatten()],
                     *[float(x) for x in refined.flatten()]])
    with (args.output_dir/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle)
        writer.writerow(['image','rotation','line_confidence','coarse_x1','coarse_y1',
                         'coarse_x2','coarse_y2','refined_x1','refined_y1','refined_x2','refined_y2'])
        writer.writerows(rows)
    print(f'Processed {len(rows)} images using heatmap epoch {checkpoint["epoch"]}; '
          f'output: {args.output_dir}')


if __name__=='__main__':
    main()
