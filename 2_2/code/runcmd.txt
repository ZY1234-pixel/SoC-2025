## 环境配置
```sh
pip install -r doc\requirements.txt
```

## 数据预处理
```shell
python -m code.data.data_preparation
python -m code.data.data_preparation_all

python -m code.data.generate_meta_info
python -m code.data.generate_meta_info_all
```

## 训练第一阶段
```sh
python code\train.py -opt code\configs\NAFNet_phase1.yml
```

## 训练第二阶段
```sh
python code\train.py -opt code\configs\NAFNet_phase2.yml --resume output\model\last_model
```

## 全量数据微调
```sh
python code\train.py -opt code\configs\NAFNet_fintune_all.yml --resume output\model\best_model
```

## 测试脚本
```sh
python code\test\predict.py Data\Val_data myResults
```
