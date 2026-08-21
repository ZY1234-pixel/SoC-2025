import torch
from nafnet_pytorch_ncnn import NAFNet


def test_model():
    print("正在加载模型...")
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 10],
        dec_blk_nums=[1, 1, 1, 1]
    )

    state_dict = torch.load("nafnet_ncnn.pth")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("正在测试推理...")
    test_input = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = model(test_input)

    print(f"✓ 输入形状: {test_input.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ 模型测试成功!")


if __name__ == "__main__":
    test_model()