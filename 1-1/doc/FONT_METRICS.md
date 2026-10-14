# 固定字体度量表

DOCX 排版规划使用固定的字符宽度数据，运行时不扫描系统字体。数据由支持字体离线生成，输出文件为 `Code/docflow_src/docflow/planning/font_metrics.json`。

## 支持字体

生成器使用项目根目录 `fonts/` 下的本地字体：

- `fonts/宋体.ttc` → 宋体 / SimSun
- `fonts/黑体.ttf` → 黑体 / SimHei
- `fonts/楷体.ttf` → 楷体 / KaiTi
- `fonts/仿宋.ttf` → 仿宋 / FangSong
- `fonts/times.ttf` → Times New Roman

字体二进制不提交到仓库，只提交生成后的度量表。

## 重新生成

```bash
python Code/scripts/generate_font_metrics.py
```

生成器需要 `fontTools`。输出只保存非默认宽度字符；CJK 默认使用全角宽度。

## 运行时规则

- 中文和全角字符使用对应中文字体的固定度量。
- 西文和半角字符使用 Times New Roman 的固定度量。
- 找不到字符时使用对应字体的默认宽度。
- 运行时不读取 `fonts/`，也不依赖系统字体安装情况。
