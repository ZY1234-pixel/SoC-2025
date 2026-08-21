import torch
from nafnet_pytorch_ncnn import NAFNet


def export_to_pnnx_v3(weight_path, output_path, height=512, width=512):

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 10],
        dec_blk_nums=[1, 1, 1, 1]
    )

    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 固定输入尺寸(必须是 16 的倍数)
    dummy_input = torch.randn(1, 3, height, width)

    # 导出为 TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path + ".pt")

    print(f"✓ 模型已导出为: {output_path}.pt")
    print(f"✓ 输入尺寸: [1, 3, {height}, {width}]")
    print("\n下一步:")
    print(f"pnnx {output_path}.pt inputshape=[1,3,{height},{width}]")


if __name__ == "__main__":
    export_to_pnnx_v3("D:\\IMGDeblur\\SoC-2025_2_Deblur\\2_2\\nafnet_ncnn.pth", "nafnet_model", height=1024, width=1024)