from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import os

INPUT_IMG = "./hebrew/test/hebrew.jpg"
OUTPUT_IMG = "./output/hebrew_compare.jpg"
FONT_PATH = "./NotoSansHebrew-Regular.ttf"

os.makedirs("./output", exist_ok=True)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    rec_model_dir='./hebrew_rec_infer',
)

result = ocr.predict(input=INPUT_IMG)
font = ImageFont.truetype(FONT_PATH, 36) if os.path.exists(FONT_PATH) else ImageFont.load_default()

for rec in result:
    rec.print()
    rec.save_to_json(save_path="./output/")
    
    #从 .json 取数据（PaddleOCR 3.x 标准写法）
    data = rec.json['res']
    texts = data['rec_texts']
    boxes = data['rec_boxes']
    
    # 画布
    orig = Image.open(INPUT_IMG).convert("RGB")
    W, H = orig.size
    canvas = Image.new("RGB", (W*2+4, H+50), "white")
    draw = ImageDraw.Draw(canvas)
    
    # 左边：原图 + 红框
    left = orig.copy()
    ld = ImageDraw.Draw(left)
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        ld.rectangle([x1,y1,x2,y2], outline="#E74C3C", width=3)
    canvas.paste(left, (0, 50))
    
    # 右边：白底 + 绿框 + 文字居右
    right = Image.new("RGB", (W, H), (250,250,250))
    rd = ImageDraw.Draw(right)
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = map(int, b)
        rd.rectangle([x1,y1,x2,y2], outline="#27AE60", width=3)
        bbox = rd.textbbox((0,0), texts[i], font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        rd.text((x2-tw-8, y1+(y2-y1-th)//2), texts[i], fill="#1E8449", font=font)
    canvas.paste(right, (W+4, 50))   
    canvas.save(OUTPUT_IMG)
