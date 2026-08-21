import torch
import paddle
import numpy as np


def convert_layernorm_weights(paddle_dict, pytorch_dict, old_prefix, new_prefix):

    # weight shape: [C] 转化为 shape: [1, C, 1, 1]
    if f"{old_prefix}.body.weight" in paddle_dict:
        weight = paddle_dict[f"{old_prefix}.body.weight"].numpy()
        pytorch_dict[f"{new_prefix}.weight"] = torch.from_numpy(weight).reshape(1, -1, 1, 1)

    if f"{old_prefix}.body.bias" in paddle_dict:
        bias = paddle_dict[f"{old_prefix}.body.bias"].numpy()
        pytorch_dict[f"{new_prefix}.bias"] = torch.from_numpy(bias).reshape(1, -1, 1, 1)


def convert_paddle_to_pytorch_v2(paddle_path, pytorch_path):

    paddle_state_dict = paddle.load(paddle_path)
    pytorch_state_dict = {}

    for key, value in paddle_state_dict.items():
        if isinstance(value, paddle.Tensor):
            value = value.numpy()

        new_key = key
        new_value = torch.from_numpy(value)

        # 处理 LayerNorm 权重的特殊转换
        if ".norm1.body.weight" in key or ".norm2.body.weight" in key:
            # 从 [C] 转换为 [1, C, 1, 1]
            new_value = new_value.reshape(1, -1, 1, 1)
            new_key = key.replace(".body.weight", ".weight")
        elif ".norm1.body.bias" in key or ".norm2.body.bias" in key:
            new_value = new_value.reshape(1, -1, 1, 1)
            new_key = key.replace(".body.bias", ".bias")

        # 处理 beta 和 gamma 参数
        if ".beta" in key or ".gamma" in key:
            # 从 [1, C, 1, 1] 保持不变
            pass

        pytorch_state_dict[new_key] = new_value

    torch.save(pytorch_state_dict, pytorch_path)
    print(f"成功转换权重: {paddle_path} -> {pytorch_path}")

    return pytorch_state_dict


if __name__ == "__main__":
    ## 修改到 paddle 权重路径
    paddle_path = "D:\\IMGDeblur\\file\\output\\model.pdparams"
    pytorch_path = "nafnet_ncnn.pth"

    convert_paddle_to_pytorch_v2(paddle_path, pytorch_path)