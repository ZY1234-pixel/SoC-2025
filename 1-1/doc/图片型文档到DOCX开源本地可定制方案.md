# 图片型文档还原为 DOCX 的开源 / 本地部署 / 可定制实现方案

> 版本：v1.0  
> 日期：2026-04-14  
> 适用场景：图片型 PDF、扫描图片、拍照文档、多样文档类型（合同、报告、票据、教材、论文、试卷、表单、宣传页等）  
> 目标：在完全本地部署、可替换组件、可持续迭代的前提下，将图片型文档解析为统一文档 IR，并渲染为尽可能可编辑、版式尽量保真的 `.docx` 文件。

---

## 1. 文档目标与设计原则

### 1.1 目标

本方案要解决的问题不是“做 OCR”，而是构建一条完整的 **文档理解 → 统一表示 → DOCX 渲染** 流水线。

最终目标包括：

1. **本地部署**：不依赖云 API，可在离线或内网环境运行。
2. **可定制**：可替换 OCR、布局分析、表格识别、公式识别、DOCX 渲染器。
3. **多文档类型**：兼容普通文本页、双栏页、表格页、论文页、票据页、表单页、图文混排页等。
4. **尽量保留文档语义与布局**：既不只提纯文本，也不把所有页面都退化成整页图片。
5. **面向工程可维护性**：便于调错、回归测试、人工修订和持续迭代。

### 1.2 设计原则

#### 原则 A：不要把“文字识别”当成“文档还原”

高质量 DOCX 还原依赖的不仅是 OCR 准确率，还包括：

- 布局区域检测
- 阅读顺序恢复
- 标题 / 正文 / 页眉 / 页脚 / 图注 / 脚注区分
- 表格结构恢复
- 公式恢复
- 页级设置恢复

Paddle 的 PP-StructureV3 明确把布局分析、表格识别、公式识别、阅读顺序恢复与 Markdown 导出放在同一条文档解析流水线上，说明“复杂文档解析”本身就是多模块系统，而不是单一 OCR 模型问题。  
参考：PaddleX PP-StructureV3、PaddleOCR PP-StructureV3 官方文档。

#### 原则 B：不要直接从 OCR 文本拼 DOCX

必须先构建 **统一文档 IR（Intermediate Representation）**。原因是：

- OCR / 布局 / 表格 / 公式结果需要统一抽象；
- 同一份 IR 应能支持 DOCX、HTML、Markdown、人工校对界面等多种输出；
- IR 要保留 provenance（来源信息）、bbox、reading order、block type 等高价值信息；
- 一旦直接把识别结果写入 Word，对模型替换、问题定位、后处理和多输出支持都会变差。

Docling 官方将 `DoclingDocument` 定义为统一文档表示，支持文本、表格、图片、层级结构、正文与页眉页脚区分、bbox 和 provenance。这正是 IR 思路的典型实现。  
参考：Docling `DoclingDocument` 官方文档。

#### 原则 C：IR 负责“文档是什么”，Renderer 负责“Word 里怎么长出来”

IR 中描述的是文档本体：

- 它是什么块（标题、正文、图注、表格、图片）
- 它在哪（bbox / polygon）
- 它来自哪一页、哪一个模型
- 它在阅读顺序中排第几

而 DOCX 渲染器负责：

- 这个块在 Word 中应该映射为哪一种段落 / 表格 / 图片 / section
- 采用哪套 Word style
- 用流式排版还是网格近似布局
- 是否需要局部降级为图片 fallback

---

## 2. 总体架构

推荐采用下面这条主流水线：

```text
输入文档
  ↓
预处理层
  ↓
版面分析层
  ↓
区域路由层
  ├─ 文本 OCR
  ├─ 表格识别
  ├─ 公式识别
  ├─ KIE / 表单字段抽取（可选）
  └─ 图片 / 印章 / 图表区域分析（可选）
  ↓
统一文档 IR 构建层
  ↓
后处理 / 纠错 / 合并 / 语义补全层
  ↓
RenderPlan 生成层
  ↓
DOCX Renderer
  ├─ 高层：python-docx
  └─ 低层：OpenXML / WordprocessingML 注入（必要时）
  ↓
输出：DOCX + 调试产物 + 中间 JSON
```

### 2.1 推荐组件组合

#### 主链路推荐

- **预处理**：OCRmyPDF（只用其图像预处理能力也可以）
- **布局分析 / 文档解析主干**：PaddleOCR / DocLayout-YOLO
- **表格识别**：Paddle 表格识别流水线（General Table Recognition Pipeline v2 / TableMagic 相关能力）
- **公式识别**：PP-FormulaNet 或其他公式 OCR 模块
- **多语言 / reading order / 表格 / 公式补强**：Surya（可选）
- **统一文档表示参考**：Docling / 自定义 IR
- **DOCX 生成**：python-docx + 必要时 OpenXML
- **语义导出辅助链路**：Pandoc（可选，仅限适合 reflow 的文档）

- **评测基准**：OmniDocBench

### 2.2 为什么采用“多解析器 + 单一 IR + 单一 Renderer”

推荐的系统边界是：

- **上游可以多引擎**：Paddle、Surya、DocLayout-YOLO、MinerU、Marker 均可接入；
- **中游只有一种 IR**：避免不同上游直接耦合多个输出器；
- **下游尽量只有一个 DOCX Renderer**：统一控制渲染策略与风格映射。

这样做的好处：

1. 更换上游模型时，不必重写 DOCX 输出逻辑。
2. 同一份 IR 可以同时生成 DOCX / HTML / Markdown / 调试视图。
3. 可以对不同页型采用不同解析器，然后汇总到同一种 IR。
4. 出问题时能快速定位到底是预处理、布局、OCR、reading order 还是 Renderer 出错。

---

## 3. 分阶段实现路线

### 3.1 第一阶段：跑通最小可用系统

目标：先把普通扫描文档稳定转换成可编辑 DOCX。

实现范围：

- 预处理
- 布局分析
- 文本 OCR
- 简单表格识别
- 自定义 IR
- 语义优先 DOCX Renderer

此阶段不强求：

- 浮动对象 1:1 复现
- 极复杂杂志页
- 所有公式都可编辑
- 所有双栏都完美复现

### 3.2 第二阶段：加入页型路由与复杂布局支持

新增能力：

- 页面分类（合同页 / 正文页 / 表格页 / 票据页 / 论文页 / 宣传页）
- RenderPlan 三模式：reflow / grid / overlay
- 多栏恢复
- 页眉页脚模板化
- 表格精细渲染
- 局部图片 fallback

### 3.3 第三阶段：精修与可视化调试

新增能力：

- 每一步中间结果持久化
- 置信度阈值与回退策略
- 校对界面 / 人工修订回写
- 回归数据集
- 自动评测与差异报告

---

【预处理这段去掉，假设文档页面已经经过预处理了】

## 4. 输入预处理层设计

### 4.1 目标

输入预处理层要解决的问题是：

- 页面方向错误
- 扫描倾斜
- 低对比度背景
- 脏点和阴影
- 大尺寸页面导致检测失败
- 图片型 PDF 与普通图片的输入格式统一

### 4.2 推荐做法

#### 输入统一

无论输入是 PDF 还是图片，先统一为：

- `DocumentMeta`
- `PageImage[]`
- 每页携带原始宽高、DPI、旋转信息、来源文件路径、页码

建议每页保存两份图：

- `page_raw`：原始图
- `page_preprocessed`：预处理后图

#### 预处理步骤

OCRmyPDF 官方说明其图像处理管线固定按 `rotate -> remove background -> deskew -> clean` 顺序执行，可组合使用 `--deskew`、`--clean`、`--rotate-pages` 等选项。该工具也支持不做 OCR、只做预处理。  
参考：OCRmyPDF Cookbook。

建议的预处理策略：

1. 自动旋转与方向校正
2. 去斜（deskew）
3. 轻量去背景
4. 适度去噪
5. 归一化到合理分辨率（通常保留 200~300 DPI 量级）

#### 注意事项

- 预处理不要过度，否则会损伤细小字符、表格线、印章边缘。
- 所有预处理参数必须可配置，并保留预处理前后对照图。
- 对某些彩色宣传页、复杂背景页，不一定适合强去背景。

---

## 5. 版面分析与区域路由层设计

### 5.1 为什么一定要先做版面分析

如果没有先分块，OCR 只能给你“字”，很难可靠地知道：

- 哪些是标题，哪些是正文
- 哪些是页眉、页脚、图注、脚注
- 哪些区域应该识别成表格
- 哪些区域是图片或图表
- 双栏页面的阅读顺序如何恢复

PP-StructureV3 官方说明其能力覆盖布局检测、表格识别、公式识别、多栏阅读顺序恢复和 Markdown 转换，并强调适配更复杂的文档数据。  
参考：PaddleX / PaddleOCR PP-StructureV3 文档。

### 5.2 区域类型建议

建议统一抽象为以下 block types：

- `title`
- `section_heading`
- `paragraph`
- `list_item`
- `table`
- `figure`
- `caption`
- `formula`
- `header`
- `footer`
- `footnote`
- `page_number`
- `seal`
- `stamp`
- `form_field`
- `unknown`

### 5.3 阅读顺序恢复

阅读顺序必须显式建模，不能只依赖数组顺序。原因：

- 双栏页
- 图文混排页
- 页眉页脚与正文混杂
- 表格、图注位置穿插

Surya 官方仓库将 reading order detection 列为核心能力之一。  
参考：Surya GitHub。

建议在 IR 中显式保存：

- `reading_order`
- `column_id`
- `z_index`（可选）
- `parent_block_id` / `group_id`

### 5.4 区域路由策略

#### 文本区

送文本 OCR 模型。

#### 表格区

送表格结构识别模型，不要退化为普通文本 OCR。

Paddle 的表格识别流水线会输出表格结构与内容，目标形式包括 HTML / 结构化表格表示，这对后续渲染 Word 原生表格至关重要。  
参考：PaddleX General Table Recognition Pipeline v2。

#### 公式区（暂时不考虑）

送公式 OCR，输出 LaTeX 或中间公式表示。

---

## 6. 为什么需要“统一文档 IR”

### 6.1 IR 不是“另一个 JSON 文件”

IR 的重点不在于文件扩展名，而在于它在系统中的职责：

- 不是某个单一模型的原始输出；
- 也不是某个单一渲染器的专用输入；
- 而是整个系统内部的 **统一文档模型**。

### 6.2 三类常见 JSON 的区别

#### A. 原始识别结果 JSON

特点：

- 紧耦合某个模型
- 字段命名受上游算法限制
- 更像“算法输出”而不是“文档抽象”

#### B. 渲染导向 JSON

特点：

- 已经整理成适合 Word 写出的形式
- 例如直接写 `Heading1`、`Normal`、`insert_picture_here`
- 容易丢失页码、bbox、来源模型、置信度等高价值信息

#### C. 文档 IR

特点：

- 与 OCR 引擎解耦
- 与 DOCX 输出解耦
- 同时保存语义、几何、来源信息
- 可支撑多种输出格式和后处理流程

### 6.3 Docling 与 IR 的关系

Docling 官方将 `DoclingDocument` 定义为统一文档表示，支持：

- 文本、表格、图片
- 文档层级结构（sections / groups）
- 主正文与页眉页脚（furniture）区分
- 所有元素的 bbox
- provenance 信息

因此，Docling 与本方案的关系不是“必须替代你的 JSON”，而是：

1. 它证明了“统一文档表示”是成熟路线；
2. 它提供了值得借鉴的字段设计与层级建模方式；
3. 你可以把自己的 IR 设计成与 `DoclingDocument` 思路相近，而不必绑定其具体实现。  
参考：Docling `DoclingDocument` 官方文档。

---

## 7. 文档 IR 设计

### 7.1 设计目标

一个可用于 DOCX 渲染的文档 IR，至少要满足：

1. **多层级**：document / section / page / block / line / span / table / cell
2. **语义明确**：标题、段落、表格、图片、图注等角色清楚
3. **几何信息完整**：bbox / polygon / page size / rotation
4. **来源可追踪**：来源模型、置信度、原始检测框
5. **可多输出**：DOCX、HTML、Markdown、调试视图
6. **避免过早丢信息**

### 7.2 推荐分层

```text
Document
  ├─ metadata
  ├─ style_hints
  ├─ sections[]
  │    ├─ page_setup
  │    ├─ header_ref
  │    ├─ footer_ref
  │    └─ body_flow[]
  ├─ pages[]
  │    ├─ page_meta
  │    ├─ blocks[]
  │    └─ debug_assets[]
  ├─ assets[]
  ├─ tables[]
  ├─ formulas[]
  └─ provenance
```

### 7.3 推荐核心字段

下面给出一份建议的 IR schema（示意，不是最终唯一答案）。

```json
{
  "document_id": "doc_001",
  "meta": {
    "source_path": "input/sample.pdf",
    "source_type": "scanned_pdf",
    "language_hint": ["zh", "en"],
    "created_at": "2026-04-14T10:00:00Z"
  },
  "style_hints": {
    "default_font_family": "SimSun",
    "default_font_size_pt": 10.5,
    "page_numbering": "arabic"
  },
  "pages": [
    {
      "page_no": 1,
      "size": {"width": 2480, "height": 3508, "unit": "px"},
      "rotation": 0,
      "dpi": 300,
      "raw_image": "artifacts/page_001_raw.png",
      "preprocessed_image": "artifacts/page_001_pre.png",
      "blocks": [
        {
          "id": "blk_1",
          "type": "text",
          "role": "title",
          "bbox": [120, 180, 1320, 260],
          "polygon": null,
          "reading_order": 1,
          "column_id": 0,
          "text": "第一章 绪论",
          "lines": [
            {
              "id": "line_1",
              "bbox": [120, 180, 1320, 260],
              "text": "第一章 绪论",
              "spans": [
                {
                  "text": "第一章 绪论",
                  "style": {
                    "bold": true,
                    "italic": false,
                    "underline": false,
                    "font_size_est": 18,
                    "font_family_est": null,
                    "color": "#000000"
                  },
                  "confidence": 0.99
                }
              ]
            }
          ],
          "source": {
            "engine": "paddleocr_pp_structure_v3",
            "confidence": 0.99,
            "detector_box_id": "layout_0001"
          }
        },
        {
          "id": "blk_2",
          "type": "table",
          "role": "table",
          "bbox": [100, 500, 2200, 1200],
          "reading_order": 5,
          "table_ref": "tbl_1",
          "source": {
            "engine": "paddle_table_v2",
            "confidence": 0.94
          }
        }
      ]
    }
  ],
  "tables": [
    {
      "id": "tbl_1",
      "page_no": 1,
      "bbox": [100, 500, 2200, 1200],
      "n_rows": 4,
      "n_cols": 3,
      "cells": [
        {
          "row": 0,
          "col": 0,
          "rowspan": 1,
          "colspan": 2,
          "bbox": [100, 500, 1000, 620],
          "content": [
            {
              "type": "paragraph",
              "text": "表头"
            }
          ],
          "style": {
            "align": "center",
            "valign": "middle",
            "border": true,
            "shading": null
          }
        }
      ],
      "html": "<table>...</table>"
    }
  ],
  "sections": [
    {
      "id": "sec_1",
      "page_range": [1, 3],
      "page_setup": {
        "width_twips": 11906,
        "height_twips": 16838,
        "margin_top_twips": 1440,
        "margin_bottom_twips": 1440,
        "margin_left_twips": 1440,
        "margin_right_twips": 1440,
        "orientation": "portrait",
        "columns": 1
      },
      "header_ref": "hdr_1",
      "footer_ref": "ftr_1",
      "body_flow": ["blk_1", "blk_2"]
    }
  ],
  "headers": [
    {
      "id": "hdr_1",
      "items": [
        {"type": "text", "role": "header", "text": "示例文档"}
      ]
    }
  ],
  "footers": [
    {
      "id": "ftr_1",
      "items": [
        {"type": "page_number", "format": "decimal"}
      ]
    }
  ],
  "assets": [],
  "provenance": {
    "pipeline_version": "1.0.0",
    "ocr_engine": "paddleocr",
    "layout_engine": "pp_structure_v3"
  }
}
```

### 7.4 字段设计原则

#### 字段 1：`role` 与 `type` 分开

- `type`：对象类型，例如 `text / table / figure / formula`
- `role`：语义角色，例如 `title / paragraph / caption / header / footer`

比如一个图注：

- `type = text`
- `role = caption`

这样比只用一个字段更灵活。

#### 字段 2：Word 样式名不要进入 IR 本体

IR 中不要直接写：

- `Heading1`
- `Normal`
- `Caption`

这些属于 Word 渲染阶段的映射结果，而不是文档本体。

IR 中应保留：

- `role = section_heading`
- `level = 1`

到 renderer 再决定映射为 `Heading 1`、`Body Text` 或自定义样式。

#### 字段 3：一定保留 bbox 与 page_no

因为这些信息会影响：

- 阅读顺序修正
- 多栏恢复
- 页眉页脚判定
- 图注挂接
- 调试定位
- page-aware 渲染策略

#### 字段 4：来源信息要单独建模

建议保留：

- `source.engine`
- `source.confidence`
- `source.detector_box_id`
- `source.raw_output_ref`

这是后续调试、人工校正、模型融合的基础。

### 7.5 需要特别建模的对象

#### 页眉 / 页脚

不要让它们混在正文里。

建议：

- `headers[]`
- `footers[]`
- `sections[].header_ref`
- `sections[].footer_ref`

#### 表格

不要压成纯文本。

表格至少保留：

- 行列数
- 单元格合并信息
- 单元格 bbox
- 单元格内容
- 边框 / 对齐 / 阴影等样式提示

#### 图片 / 图表

建议单独建 `figure` 或 `asset`，并允许：

- 原图路径
- 裁剪框
- 图注关联
- 尺寸与显示模式

#### 公式

建议保存：

- 公式 bbox
- LaTeX
- 展示型 / 行内型
- 所属 block 或 paragraph

---

## 8. 从 IR 到 DOCX：总体思路

### 8.1 本质上是“编译器”

IR 到 DOCX 不是简单的数据写出，而是一个“编译”过程：

- **IR**：源语言，描述文档本体
- **RenderPlan**：中间执行计划，描述如何在 Word 中组织内容
- **DOCX / WordprocessingML**：目标语言

微软 Open XML 官方文档说明了 Word 文档的基本结构是 `Document -> Body -> Paragraph -> Run -> Text`，并可继续扩展为 section、table、header/footer、drawing 等更复杂结构。  
参考：Microsoft Open XML 文档。

### 8.2 为什么还需要 RenderPlan

原因是：同一个 IR block，在 Word 中未必只有一种落法。

例如一个正文块：

- 在普通合同中，应作为普通段落输出；
- 在双栏论文中，可能需要放在 multi-column section 中；
- 在试卷题干页中，可能需要放进网格布局的某个单元格里；
- 在极复杂海报页中，可能需要局部图片 fallback。

因此推荐引入中间层：

```text
IR -> Normalizer -> RenderPlan -> DOCX Backend
```

---

## 9. DOCX 渲染器架构设计

### 9.1 模块划分

建议 renderer 至少拆成以下 4 层：

#### 1）Normalizer

职责：

- 合并碎片化行
- 修复 reading order
- 将 header/footer 从正文流中剥离
- 绑定 caption 与 figure/table
- 识别 section 边界
- 统一 style hints

#### 2）StyleMapper

职责：

- 把 IR 中的 `role`、`level`、`style_hint` 转成 Word 样式 / 属性
- 管理 reference template / 样式表

#### 3）LayoutEngine / RenderPlan Builder

职责：

- 决定每页、每个块采用哪种渲染模式
- 生成 body flow、section、headers、footers、floating regions 等信息

#### 4）WordBackend

职责：

- 实际调用 `python-docx` 或更低层 OpenXML API 写出 `.docx`

### 9.2 三种渲染模式

建议在 RenderPlan 中定义三种模式：

#### 模式 A：`reflow`

按阅读顺序流式输出。

适用：

- 合同
- 报告
- 教材正文
- 一般办公文档

优点：

- 最可编辑
- Word 原生兼容性最好

缺点：

- 版式和原页面不一定完全一致

#### 模式 B：`grid`

先把页面分成若干区域，用 Word 表格 / 分栏 / 缩进 / tab stop 去近似布局。

适用：

- 双栏页
- 题目页
- 复杂图文混排页
- 某些宣传页

优点：

- 版式保真度更高

缺点：

- 编辑体验变差
- 渲染规则复杂

#### 模式 C：`overlay`

极复杂页面局部走绝对定位近似，或直接图片 fallback。

适用：

- 封面
- 海报页
- 装饰性特别强的页面
- 极难恢复的复杂版式

优点：

- 最容易保持视觉一致性

缺点：

- 可编辑性最差

### 9.3 模式选择建议

建议按页型路由：

| 页型 | 推荐模式 | 说明 |
|---|---|---|
| 合同 / 报告 / 书籍正文 | reflow | 以编辑性为主 |
| 学术论文双栏正文 | reflow 或 grid | 根据是否要强保双栏决定 |
| 票据 / 表单 / 发票 | grid | 位置关系较重要 |
| 试卷 / 题目页 | grid | 区域布局重要 |
| 封面 / 宣传页 / 海报页 | overlay 或图片 fallback | 视觉保真优先 |

---

## 10. IR 到 DOCX 的具体渲染逻辑

### 10.1 文本块渲染

#### 输入

IR 文本块通常包含：

- `type = text`
- `role`
- `bbox`
- `reading_order`
- `text`
- `lines[]`
- `spans[]`
- `style hints`

#### 处理步骤

1. 根据 `role` 决定目标 paragraph style
2. 根据行与 span 信息重建段落内容
3. 将 span 映射为 runs
4. 应用粗体 / 斜体 / 下划线 / 字号 / 颜色等属性
5. 设置段落对齐、缩进、段前段后、行距

#### 关键点：不要“一行 OCR = 一个段落”

大部分 OCR 输出是“视觉行”，而 Word 的自然编辑单位是“段落”。

因此建议增加 `paragraph reconstruction` 逻辑，根据下列信号判断多行是否应合并成同一段：

- 行间距
- 左右边界对齐程度
- 首行缩进
- 末行长度
- 标点和断句特征
- 原始 block 范围

### 10.2 段落样式映射

建议不要在 IR 里保存 Word 样式名，而是在 StyleMapper 中完成：

```python
ROLE_TO_WORD_STYLE = {
    "title": "Title",
    "section_heading_1": "Heading 1",
    "section_heading_2": "Heading 2",
    "paragraph": "Body Text",
    "caption": "Caption",
    "quote": "Quote",
    "list_item": "List Paragraph"
}
```

Pandoc 的 `reference.docx` 和 custom styles 机制也体现了同样的设计思路：文档语义与目标格式样式映射应分离。  
参考：Pandoc Manual。

### 10.3 表格渲染

#### 原则

表格必须尽量渲染成 **真正的 Word table**，而不是先转纯文本。

`python-docx` 官方支持创建和操作 Word 表格。  
参考：python-docx 官方文档。

#### 推荐流程

1. 创建 Word table
2. 根据 `n_rows`, `n_cols` 初始化网格
3. 按 IR 中 `rowspan`, `colspan` 做单元格合并
4. 填充 cell 内容
5. 设置边框、对齐、单元格宽度和 shading
6. 递归渲染 cell 内的 paragraph / run

#### 表格 IR 的建议最小字段

- `n_rows`
- `n_cols`
- `cells[]`
- `rowspan`
- `colspan`
- `cell.content[]`
- `style.align / valign / border`

### 10.4 图片渲染

#### 普通正文插图

可作为 inline picture 插入。

python-docx 文档明确支持插入图片。  
参考：python-docx 官方文档。

#### 浮动图 / 装饰图 / 复杂图文混排

建议策略：

1. 能 inline 就 inline
2. 需要维持局部相对位置时，优先放进 grid 布局单元格
3. 实在太复杂再考虑更低层 OpenXML 注入或直接图片 fallback

### 10.5 页眉 / 页脚 / section

python-docx 官方文档提供了 `sections` 以及 header/footer 的操作方式。页眉和页脚属于 section 级对象，而不是普通正文块。  
参考：python-docx headers & footers 文档。

#### 推荐规则

当以下情况发生时，开启新 section：

- 页面尺寸变化
- 横竖版切换
- 页边距变化
- 多栏设置变化
- 页眉页脚模板变化

然后按 `section.header_ref`、`section.footer_ref` 渲染页眉页脚。

### 10.6 多栏页面

Word 本身支持多栏 section，但对复杂扫描页不一定总能自然恢复。

推荐策略：

- 学术论文正文：可优先尝试多栏 section
- 如果同页中存在复杂跨栏对象，则考虑 grid 模式
- 对于实在难处理的个别页，允许局部 fallback

### 10.7 公式渲染

公式渲染建议分两个层次：

#### 方案 A：可编辑优先

- 将公式转成 Word 可编辑数学对象（需要专门实现）

#### 方案 B：视觉正确优先

- 先用 LaTeX 渲染成图片再插入
- 或将局部复杂公式作为图片 fallback

工程上建议先实现方案 B，再迭代到可编辑公式。

---

## 11. DOCX 后端实现建议

### 11.1 高层优先：python-docx

python-docx 官方文档说明它可用于创建与更新 `.docx` 文件，并支持段落、文本 runs、表格、图片、sections、页眉页脚等常用对象。  
参考：python-docx 官方文档。

建议第一版 renderer 优先用 python-docx 完成：

- 文本段落
- 样式映射
- 表格
- 图片
- 页眉页脚
- section

### 11.2 低层补丁：OpenXML / WordprocessingML

当你遇到下列情况时，通常要考虑下沉到低层 WordprocessingML：

- 更复杂的浮动布局
- 精细控制某些 Word 原生对象
- python-docx 暴露能力不足的场景

微软 Open XML 官方文档展示了 Word 文档的核心对象模型和 Open XML SDK 对应关系，说明 `.docx` 的底层本质上是可编程的 XML 结构。  
参考：Microsoft Open XML 文档。

### 11.3 推荐策略

- **80% 场景**：python-docx
- **15% 场景**：python-docx + 自定义 hack / template / grid 近似
- **5% 场景**：OpenXML 注入或图片 fallback

---

## 12. 样式系统设计

### 12.1 为什么要独立样式系统

如果没有独立样式系统，Renderer 很容易变成大量硬编码：

- 这里字号 14
- 那里行距 1.5
- 这里宋体
- 那里黑体

长期会非常难维护。

### 12.2 推荐方案

采用三层样式结构：

#### 层 1：IR 语义角色

- `title`
- `section_heading`
- `paragraph`
- `caption`
- `table_header`

#### 层 2：Renderer 抽象样式

- `doc_title`
- `heading_l1`
- `body_text`
- `caption_text`
- `table_header_text`

#### 层 3：Word 样式

- `Title`
- `Heading 1`
- `Body Text`
- `Caption`
- `CustomTableHeader`

这样你可以在不改 IR 的情况下切换不同模板。

### 12.3 样式模板

建议准备 `reference_template.docx`，预置：

- 正文字体
- 中英文字体映射
- 标题层级
- 段前段后
- 表格默认样式
- 图注样式
- 页眉页脚样式

Pandoc 支持 reference docx，这一机制也很适合作为 DOCX 样式模板思路参考。  
参考：Pandoc Manual。

---

## 13. RenderPlan 设计

### 13.1 为什么需要 RenderPlan

IR 是对文档“是什么”的描述；RenderPlan 是“怎么渲染”的执行计划。

建议 RenderPlan 中至少包含：

- section 划分
- section page setup
- header/footer 绑定
- body flow
- 每个 item 的 render mode
- 可能的 fallback 策略

### 13.2 RenderPlan 示例

```json
{
  "sections": [
    {
      "id": "render_sec_1",
      "page_setup": {
        "orientation": "portrait",
        "columns": 1,
        "margin_top_twips": 1440,
        "margin_bottom_twips": 1440,
        "margin_left_twips": 1440,
        "margin_right_twips": 1440
      },
      "header_ref": "hdr_1",
      "footer_ref": "ftr_1",
      "body_flow": [
        {"kind": "heading", "ref": "blk_1", "mode": "reflow"},
        {"kind": "paragraph", "ref": "blk_2", "mode": "reflow"},
        {"kind": "table", "ref": "tbl_1", "mode": "grid"},
        {"kind": "figure", "ref": "fig_1", "mode": "inline"}
      ]
    }
  ]
}
```

### 13.3 RenderPlan 生成规则示例

1. 同一页尺寸与页边距一致的一组页，可合并为同一 section；
2. 多栏页单独成 section；
3. 页眉页脚模板变化时，切 section；
4. 表格页可标记为 `grid` 模式；
5. 极复杂页可标记为 `overlay` 或 `page_image_fallback`。

---

## 14. 推荐代码结构

```text
project/
  ├─ pipeline/
  │   ├─ preprocess.py
  │   ├─ layout.py
  │   ├─ ocr.py
  │   ├─ table.py
  │   ├─ formula.py
  │   └─ router.py
  ├─ ir/
  │   ├─ schema.py
  │   ├─ builders.py
  │   ├─ normalizer.py
  │   └─ validators.py
  ├─ renderer/
  │   ├─ render_plan.py
  │   ├─ style_mapper.py
  │   ├─ word_backend.py
  │   ├─ render_text.py
  │   ├─ render_table.py
  │   ├─ render_figure.py
  │   ├─ render_section.py
  │   └─ render_formula.py
  ├─ exporters/
  │   ├─ to_docx.py
  │   ├─ to_html.py
  │   └─ to_markdown.py
  ├─ eval/
  │   ├─ metrics.py
  │   ├─ regression.py
  │   └─ visual_diff.py
  ├─ configs/
  │   ├─ pipeline.yaml
  │   ├─ styles.yaml
  │   └─ routing.yaml
  └─ tests/
      ├─ samples/
      ├─ expected_ir/
      └─ expected_docx/
```

---

## 15. 推荐伪代码

### 15.1 主流程

```python
from docx import Document


def convert_document(input_path: str, output_docx: str):
    pages = preprocess_input(input_path)

    layout_results = []
    for page in pages:
        layout = run_layout_analysis(page)
        routed_regions = route_regions(layout, page)
        layout_results.append(routed_regions)

    ir = build_ir(layout_results)
    ir = normalize_ir(ir)

    render_plan = build_render_plan(ir)
    doc = render_docx(render_plan, ir)

    doc.save(output_docx)
    return ir
```

### 15.2 Renderer 主入口

```python
def render_docx(render_plan, ir, template_path=None):
    doc = Document(template_path) if template_path else Document()

    for sec in render_plan["sections"]:
        start_section(doc, sec["page_setup"])
        render_header(doc, resolve_header(ir, sec.get("header_ref")))
        render_footer(doc, resolve_footer(ir, sec.get("footer_ref")))

        for item in sec["body_flow"]:
            render_item(doc, ir, item)

    return doc
```

### 15.3 块分派器

```python
def render_item(doc, ir, item):
    ref = item["ref"]
    mode = item.get("mode", "reflow")
    obj = resolve_ref(ir, ref)

    if item["kind"] in ["heading", "paragraph", "text"]:
        render_text_block(doc, obj, mode)
    elif item["kind"] == "table":
        render_table(doc, obj, mode)
    elif item["kind"] == "figure":
        render_figure(doc, obj, mode)
    elif item["kind"] == "formula":
        render_formula(doc, obj, mode)
    else:
        render_unknown(doc, obj)
```

### 15.4 文本块渲染

```python
def render_text_block(doc, block, mode="reflow"):
    style_name = map_role_to_style(block)
    p = doc.add_paragraph(style=style_name)

    for line in reconstruct_paragraph_lines(block):
        for span in line["spans"]:
            run = p.add_run(span["text"])
            apply_run_style(run, span.get("style", {}))
```

### 15.5 表格渲染

```python
def render_table(doc, table_obj, mode="grid"):
    table = doc.add_table(rows=table_obj["n_rows"], cols=table_obj["n_cols"])

    for cell in table_obj["cells"]:
        target = table.cell(cell["row"], cell["col"])
        if cell.get("rowspan", 1) > 1 or cell.get("colspan", 1) > 1:
            merge_cells(table, cell)

        render_cell_content(target, cell["content"])
        apply_cell_style(target, cell.get("style", {}))
```

---

## 16. 错误处理与回退策略

### 16.1 原则

现实系统里，不可能所有页面都完美还原。所以必须从一开始就设计“回退路线”。

### 16.2 推荐回退层级

#### 第一层：块级回退

- 某个公式块太复杂 → 转图片插入
- 某个表格结构识别失败 → 保留 cell 文本并用简化表格输出

#### 第二层：区域级回退

- 某个复杂图文区域难以重建 → 作为区域截图插入，附带可选 OCR 文本注释

#### 第三层：页级回退

- 极复杂宣传页 / 封面页 → 整页图片 fallback

---

## 17. 调试、可视化与可追踪性

### 17.1 每一步都产出调试工件

建议每页保存：

- 布局框可视化图
- OCR 文本框图
- IR JSON
- RenderPlan JSON

### 17.2 为什么必须这么做

当用户说“第 12 页页脚进正文了”，只有保留中间结果，你才能快速判断问题来自：

- 预处理
- 布局分类
- header/footer 判定
- reading order
- 渲染器 section 切分

---

## 18. 评测与回归测试

### 18.1 不要只看 OCR 字准率

建议至少从 4 个维度评测：

1. 文本准确性
2. 表格结构准确性
3. 公式恢复准确性
4. 阅读顺序准确性

OmniDocBench 的评测就覆盖 text、tables、formulas、reading order 四大模块，并支持更细粒度分析。  
参考：OmniDocBench GitHub。

### 18.2 自建回归集

建议建立你自己的核心样本文档集，按文档类型分桶：

- 合同
- 报告
- 教材
- 论文
- 票据
- 表单
- 试卷
- 宣传页

每次改模型、改路由、改渲染器，都跑完整回归。

### 18.3 推荐输出物

- 每个样本一份目标 IR
- 一份预期 DOCX / HTML 结果
- 一份视觉 diff 或截图比对结果

---

## 19. 部署与工程化建议

### 19.1 服务拆分建议

建议至少拆分为以下 worker 或服务：

- `preprocess-worker`
- `layout-worker`
- `ocr-worker`
- `table-worker`
- `formula-worker`
- `ir-builder`
- `render-worker`

### 19.2 为什么不要做单体大脚本

单体脚本会带来：

- 资源调度不灵活
- 某个模块换模型会牵连全局
- 不利于 GPU / CPU 混合部署
- 很难做并发和批处理

### 19.3 建议的持久化对象

- 页面图像
- 中间识别结果
- 统一 IR
- RenderPlan
- 输出 DOCX
- 错误报告
- 评测指标

---

## 20. 你当前实现如何升级

如果你当前已经是：

```text
识别结果 -> 自定义 JSON -> 渲染 DOCX
```

那么最合理的升级路线不是推倒重来，而是：

### 第一步：确认你的 JSON 是否已经接近 IR

检查是否有：

- page_no
- bbox / polygon
- role / type
- reading_order
- source_engine / confidence
- table_structure
- figure / caption 关联
- section / header / footer 概念

### 第二步：把“Word 专属信息”从 JSON 中抽离

例如把：

- `Heading1`
- `Normal`
- `insert_page_break_here`

尽量挪到 renderer 或 RenderPlan 层，不要放在 IR 主体里。

### 第三步：加一个 RenderPlan 层

不要直接 `IR -> python-docx`，而是：

- `IR -> RenderPlan -> DOCX`

### 第四步：引入回退与调试机制

哪怕第一版只支持：

- 文本块正常输出
- 表格块真实表格输出
- 极复杂页图片 fallback

这套结构也会比“全靠渲染器硬写”更稳。

---

## 21. 推荐的最小可行版本（MVP）

### 21.1 MVP 功能清单

- 输入：图片 / 图片型 PDF
- 预处理：旋转、去斜、轻量清理
- 布局：标题 / 正文 / 表格 / 图片 / 页眉 / 页脚
- OCR：中文为主，中英混排
- IR：document / page / block / table / header/footer
- DOCX：
  - 标题 / 正文段落
  - 表格
  - 图片
  - 页眉页脚
  - section
- 回退：整页或区域图片 fallback

### 21.2 不要在 MVP 里强做的内容

- 全量浮动对象 1:1 恢复
- 所有公式都转可编辑数学对象
- 复杂海报页完全可编辑复刻
- 所有字体完全还原

---

## 22. 最推荐的落地组合

如果要从今天开始实施，这里给出一套优先级最高、风险可控的组合：

### 主方案

1. **布局与主解析**：DocLayout-YOLO
2. **文本 OCR**：PaddleOCR
3. **表格识别**：Paddle 表格识别流水线
4. **IR**：自定义 schema，参考 DoclingDocument 思路
5. **Renderer**：python-docx 为主，必要时 OpenXML 补丁
6. **建立离线指标体系**：OmniDocBench

---

## 23. 结论

这类系统要做稳，核心不是“找到一个最强 OCR 模型”，而是建立一条分层明确的流水线：

```text
多解析器
  -> 统一文档 IR
  -> RenderPlan
  -> DOCX Renderer
```

其中最重要的工程决策有三个：

1. **一定要先有统一文档 IR**，不要直接从 OCR 文本拼 Word；
2. **Renderer 一定要和 IR 解耦**，不要把 Word 样式名写进文档本体；
3. **一定要允许局部回退**，不要追求所有页面都 100% 可编辑复刻。

如果按照这个思路推进，系统会同时具备：

- 可替换的上游模型
- 可维护的中间层
- 可持续迭代的渲染器
- 可解释、可评测、可调试的工程结构

这比单点 OCR 工具“一步转 Word”的路线更适合开源、本地部署、长期演进和多文档类型支持。

---

## 24. 参考资料（官方优先）

1. Docling `DoclingDocument` 概念文档  
   https://docling-project.github.io/docling/concepts/docling_document/

2. Docling GitHub 仓库  
   https://github.com/docling-project/docling

3. PaddleX PP-StructureV3 文档  
   https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html

4. PaddleOCR PP-StructureV3 介绍  
   https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html

5. PaddleX General Table Recognition Pipeline v2  
   https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html

6. OCRmyPDF Cookbook  
   https://ocrmypdf.readthedocs.io/en/latest/cookbook.html

7. Surya GitHub 仓库  
   https://github.com/datalab-to/surya

8. DocLayout-YOLO GitHub 仓库  
   https://github.com/opendatalab/DocLayout-YOLO

9. python-docx 官方文档  
   https://python-docx.readthedocs.io/

10. python-docx Headers and Footers 文档  
    https://python-docx.readthedocs.io/en/latest/user/hdrftr.html

11. Microsoft Open XML：Create a word processing document  
    https://learn.microsoft.com/en-us/office/open-xml/word/how-to-create-a-word-processing-document-by-providing-a-file-name

12. Microsoft Open XML：Working with runs  
    https://learn.microsoft.com/en-us/office/open-xml/word/working-with-runs

13. Pandoc User's Guide  
    https://pandoc.org/MANUAL.html

14. MinerU GitHub / 官方文档  
    https://github.com/opendatalab/mineru  
    https://opendatalab.github.io/MinerU/

15. Marker GitHub 仓库  
    https://github.com/datalab-to/marker

16. OmniDocBench GitHub 仓库  
    https://github.com/opendatalab/OmniDocBench

