### 文件说明
####  目录中包含以下文件：  
<details>
<summary>checkpoint （文件夹，用于存放模型权重文件）</summary>
<ol>
    <li>best_decontamination_model_balanced.pth （模型权重文件）
    <li>model2.pth （训练断点恢复文件）</li>
</ol>
</details>
<details>
<summary>data （文件夹，用于存放数据集）</summary>
    <ol>
    <li>dataset （训练数据集。包含灰度图像和彩色图像。）</li>
    <li>test1（彩色图像测试数据集。）</li>
    <li>test2（灰度图像测试数据集。）</li>
    </ol>
<p>以上文件不在代码仓库中。从网盘下载数据集后可解压于此，供模型测试。数据集内有带污迹图像，以及对应的Ground Truth（无污迹图像）。带污迹图像位于input文件夹，无污迹图像位于target文件夹。
<br>彩色图像数据集来源于StainDoc数据集。训练集数据从其训练数据中采样与裁剪得到7025个样本。测试集来源于其测试集。测试集中的带污迹图像文本常因水迹而产生形变、纸面反光、字迹丢失。故PSNR能直接反映图像恢复的效果，但不能准确反映去污效果，仅供参考。
数据集链接https://www.selectdataset.com/dataset/e9ad004518c3c9b9da66703be4b3173f
<br>灰度图像数据集来源于上一期文档去污算法的成果。为其中的测试集图像制作了对应的Ground Truth图像，加入到训练集中。故test2中的灰度图像测试集来源于训练数据。

</details>
<li>dataLoader.py （代码文件，读取图片功能相关代码）
<li>evaluation.py （代码文件，评价模型的表现。读取评价数据集的图像，输入带污渍图像进行推理，与干净图像进行比较，计算PSNR。）
<li>example.py （代码文件，用于模型推理，运行图像去污功能）
<li>model2.py （代码文件，用于定义模型的结构）
<li>train_balancedV2.py （代码文件，用于训练模型）



### 使用说明

- <p>推理脚本example.py，用于测试算法功能。脚本中设置的参数可参考注释。如果需要修改脚本的参数，运行前注意检查以下几个参数是否正确：</p>

<ol>
    <li>--input_dir <br>str类型。用于推理的图像文件路径。需要进行去污处理的图像放在这个路径。</li>
    <li>--output_dir <br>str类型。输出处理结果文件的路径。去污处理的图像会保存到这个路径。</li>
    <li>--model_path <br>str类型。模型权重文件的路径，需要指向具体的文件名。</li>
    <li>--output_grayscale <br>bool类型。用于控制输出的图像是否为灰度图。True为输出灰度图像，False为输出彩色图像。进行灰度图像去污时建议设置为True（设置为False也能运行，但是图像色调会稍微偏蓝）。</li>
</ol>

<br>

- 
    <p>评价脚本evaluation.py，用于使用测试集测试模型的表现。脚本中设置的参数可参考注释。如果需要修改脚本的参数，运行前注意检查以下几个参数是否正确：
    </p>
<ol>
    <li>--input_dir <br>str类型。用于推理的图像文件路径。测试数据集中带污迹图像放在这个路径。</li>
    <li>-gt_dir <br>str类型。用于对照的干净图像文件路径。测试数据集中的Ground Truth图像放在这个路径。</li>
    <li>--output_dir <br>str类型。输出处理结果文件的路径。去污处理的图像会保存到这个路径。</li>
    <li>--model_path str类型。模型权重文件的路径，需要指向具体的文件名。</li>
    <li>--output_grayscale <br>bool类型。用于控制输出的图像是否为灰度图。True为输出灰度图像，False为输出彩色图像。进行灰度图像去污时建议设置为True（设置为False也能运行，但是图像色调会稍微偏蓝）。</li>
</ol>

<br>

- 
    <p>训练脚本train_balancedV2.py，用于训练图像去污模型。该脚本支持从检查点继续训练，并通过增加灰度图像在训练中的权重来改善模型对灰度图像的处理效果。脚本中设置的参数可参考注释。如果需要修改脚本的参数，运行前注意检查以下几个参数是否正确：
    </p>
<ol>
    <li>--epochs <br>int类型。训练轮数，默认值为50。</li>
    <li>--batch_size <br>int类型。批次大小，默认值为16。GPU内存不足时可减小此值。</li>
    <li>--gradient_accumulation_steps <br>int类型。梯度累积步数，默认值为4。实际batch_size = batch_size * gradient_accumulation_steps。</li>
    <li>--lr <br>float类型。初始学习率，默认值为5e-4。建议比初始训练时设置得小一些。</li>
    <li>--val_split <br>float类型。验证集比例，默认值为0.2（即20%的数据用于验证）。</li>
    <li>--patience <br>int类型。早停耐心值，默认值为15。当验证损失连续patience轮未改善时，训练将提前停止。</li>
    <li>--resume <br>flag类型。从检查点恢复训练的标志。添加此参数时，将从--checkpoint_path指定的检查点恢复训练。</li>
    <li>--grayscale_weight <br>float类型。灰度图像采样权重，默认值为20.0。相对于彩色图像，该值表示灰度图像在训练中的出现频率倍数。值越大，灰度图像在训练中的权重越高。</li>
    <li>--checkpoint_path <br>str类型。检查点路径，默认值为'./checkpoints/model2.pth'。用于保存训练断点和从断点恢复训练。</li>
    <li>--best_model_path <br>str类型。最佳模型路径，默认值为'./checkpoints/best_model2.pth'。用于加载初始权重。如果未设置--resume且该文件存在，将从此文件加载模型权重开始训练。</li>
</ol>
