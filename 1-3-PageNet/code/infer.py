import numpy as np

from model import build_model
from utils.converter import Converter
import yaml
from utils.parser import default_parser
import torch

from utils.decode import det_rec_nms, PageDecoder
from torchvision.transforms import Compose
from data.transforms_test import RandomResize, SizeAjust, ToTensor

import cv2
from PIL import Image, ImageDraw, ImageFont


class Detector:
    """
    手写/印刷文本检测+识别检测器
    核心功能：加载模型→图像预处理→模型推理→后处理（NMS+行解码）→字符识别→结果输出
    """

    def __init__(self):
        """初始化检测器：解析配置、加载模型、初始化转换器/解码器/图像变换"""
        parser = default_parser()
        args = parser.parse_args()
        args.config = 'configs/casia-hwdb.yaml'  # PageNet/
        self.cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        model = self.init_model()

    def init_model(self):
        """
        初始化模型相关组件：
        - 字符转换器：将模型输出的类别索引转为实际字符
        - 模型构建与加载：构建网络并设置为评估模式
        - 页面解码器：处理字符阅读顺序、行分割
        - 图像变换：测试阶段的图像预处理流水线
        """

        # dict
        self.converter = Converter(self.cfg['DATA']['DICT'])

        # build model
        self.model = build_model(self.cfg)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # os.makedirs(self.cfg['OUTPUT_FOLDER'], exist_ok=True)

        self.model.eval()

        # build processor
        layout = self.cfg['POST_PROCESS']['LAYOUT'] if 'LAYOUT' in self.cfg['POST_PROCESS'] else 'generic'
        self.page_decoder = PageDecoder(
            se_thres=self.cfg['POST_PROCESS']['SOL_EOL_CONF_THRES'],
            max_steps=self.cfg['POST_PROCESS']['READ_ORDER_MAX_STEP'],
            layout=layout
        )

        self.image_mode = self.cfg['DATA']['VAL']['IMAGE_MODE']

        tfm_cfgs = self.cfg['DATA']['VAL']

        transforms = []
        force_resize = tfm_cfgs['FORCE_RESIZE'] if 'FORCE_RESIZE' in tfm_cfgs else True
        transforms.append(RandomResize(tfm_cfgs['WIDTHS'], tfm_cfgs['MAX_HEIGHT'], force_resize))
        transforms.append(SizeAjust(tfm_cfgs['SIZE_STRIDE']))
        transforms.append(ToTensor())
        if len(transforms) == 0:
            return None
        self.transforms = Compose(transforms)

    def detect(self, image):
        """
        核心检测识别函数：
        输入：图像路径
        输出：识别的文本字符串（按从上到下的行顺序）
        流程：图像读取→预处理→模型推理→NMS后处理→行解码→字符识别→结果拼接
        """

        oriImage = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image = self.transforms(oriImage).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            pred_det_rec, pred_read_order, pred_sol, pred_eol = self.model(image)
        pred_det_rec = pred_det_rec[0].cpu().numpy()
        pred_read_order = pred_read_order[0].cpu().numpy()
        pred_sol = pred_sol[0].cpu().numpy()
        pred_eol = pred_eol[0].cpu().numpy()

        pred_det_rec = det_rec_nms(
            pred_det_rec=pred_det_rec,
            img_shape=image.shape[-2:],
            dis_weight=self.cfg['POST_PROCESS']['DIS_WEIGHT'],
            conf_thres=self.cfg['POST_PROCESS']['CONF_THRES'],
            nms_thres=self.cfg['POST_PROCESS']['NMS_THRES']
        )

        img = self.paint_det(oriImage, pred_det_rec)
        cv2.imwrite('res.jpg', img)
        line_results, _ = self.page_decoder.decode(
            output=pred_det_rec,
            pred_read=pred_read_order,
            pred_start=pred_sol,
            pred_end=pred_eol,
            img_shape=image.shape[-2:],
        )

        word_res = ''
        for line_result in line_results:
            line_word = [np.argmax(pred_det_rec[id][5:]) for id in line_result]
            line_word = self.converter.decode(line_word)
            word_res += line_word

        return word_res

    def paint_det(self, oriImage, det_rec):
        """
        绘制检测框和识别字符到图像上（可视化）
        :param oriImage: 原始灰度图像
        :param det_rec: NMS后的检测结果，格式：[x, y, w, h, conf, cls1, cls2, ...]
        :return: 绘制后的彩色图像
        """

        image = cv2.cvtColor(oriImage, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        det_size = det_rec.shape[0]
        font = ImageFont.truetype("SIMSUN.TTC", 40)  # PageNet/
        for i in range(det_size):
            box_xywh = det_rec[i][:4]
            word = self.converter.decode([np.argmax(det_rec[i][5:])])
            self.rectangle = draw.rectangle([(int)(box_xywh[0] - box_xywh[2] / 2), (int)(box_xywh[1] - box_xywh[3] / 2),
                                             (int)(box_xywh[0] + box_xywh[2] / 2),
                                             (int)(box_xywh[1] + box_xywh[3] / 2)], outline=(255, 0, 0), width=2)

            draw.text(((int)(box_xywh[0] + box_xywh[2] / 2), (int)(box_xywh[1] - box_xywh[3] / 2)), word, font=font,
                      fill=(0, 0, 0), stroke_width=1)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image


if __name__ == '__main__':
    detector = Detector()
    res = detector.detect('test.jpg')
    print("识别结果为：", res)
