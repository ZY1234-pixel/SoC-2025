# UHDM Dataset

本目录不包含数据本体，只保留放置位置说明。

推荐结构：

- `dataset/input/`: 测试人员做推理时放输入图
- `dataset/UHDM/train/`: 训练集（可选）
- `dataset/UHDM/test/`: 验证集（可选）

UHDM 训练数据默认按以下命名匹配：

- `0001_moire.jpg`
- `0001_gt.jpg`


## 下载方式：Google网盘
https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=drive_link

## 下载方式：OpenDataLab

### CLI  下载
```bash
pip install openxlab #安装

pip install -U openxlab #版本升级

openxlab login #进行登录，输入对应的AK/SK

openxlab dataset info --dataset-repo OpenDataLab/UHDM #数据集信息及文件列表查看

openxlab dataset get --dataset-repo OpenDataLab/UHDM #数据集下载

openxlab dataset download --dataset-repo OpenDataLab/UHDM --source-path /README.md --target-path /path/to/local/folder #数据集文件下载
```

### SDK  下载
```bash
pip install openxlab #安装

pip install -U openxlab #版本升级

import openxlab
openxlab.login(ak=<Access Key>, sk=<Secret Key>) #进行登录，输入对应的AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/UHDM') #数据集信息及文件列表查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/UHDM', target_path='/path/to/local/folder/')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/UHDM',source_path='/README.md', target_path='/path/to/local/folder') #数据集文件下载
```