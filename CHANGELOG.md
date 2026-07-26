# 更新日志

## v0.6.0 — 2026-07-25

- 流式还原重构为 Recognition Evidence → Document Analysis → Reflow Layout Plan → Mechanical Render 四阶段
- PP-DocLayoutV3 Model Order 成为唯一阅读顺序，删除旧 Block、Zone、profile、repair 与策略切换链路
- 每个源页严格映射为一个 Word 页，使用单次预渲染 Page Fit，不做生成—检测—重生成闭环
- Word Safety Factor 只缩减页面高度预算，不再强制缩小本来已满足安全预算的页面
- Word Safety Factor 由 0.80 校准为 0.85，并按文本框累计 1/8 行换行误差，减少统一过缩放
- Page Fit 使用 OCR 原始行数与源 bbox 估算重排高度，恢复结构块间距并计入 Word 分段开销，安全预算校准为 0.90
- 字号联合源行距与文字墨迹高度推断，文本 bbox 映射为可靠的段落缩进和页眉页脚位置，并计入 CJK 与嵌套表格的 Word 行框开销
- Page Fit 固定预留 12pt 跨 Office 引擎分页差异空间，避免 LibreOffice 单页但 Microsoft Word 溢页
- OCR 行高同时驱动 Page Fit 与 DOCX 固定行距，并关闭孤行、段落粘连等隐式分页规则，减少跨 Office 引擎的溢页漂移
- 布局表格不再设置整行禁止跨页，避免 Microsoft Word 将接近页底的分栏或 Grid 容器整体推到下一页
- 原生表格单元格使用与 Page Fit 相同的固定行距，消除 Word 与 LibreOffice 默认表格行框差异
- 原生可编辑表格、公式图像与可编辑编号、Single / Sequential Columns / Grid Flow 纳入统一规划
- Reflow Planner 先按源图几何划分稳定分栏容器，再在容器内保持 PP-DocLayoutV3 Model Order
- 全量验收增加内容来源完整性、DOCX 可打开、语义表格原生化和 PDF Page Budget 检查

## v0.5.0 — 2026-04-23

- 版面还原与渲染管线全面改进：多列布局检测、页边距估算、全局缩放适配
- 报纸/杂志等复杂版面还原准确度提升
- 添加表格行防断裂支持，避免表格在分页处被截断
- 分级修正策略：溢出时先局部修正（段后间距、区块间隙、字号），再全局缩放
- 三遍渲染验证，保证可读性的同时防止内容溢出页面
- 添加渲染计划输出（`render_plan.json`），说明每个样本的渲染策略
- 添加样式推断结果输出，记录页面级别字体大小、行距、页边距等
- 报纸等短横版页面自适应，避免文字溢出

## v0.4.0 — 2026-04-15

- 添加轻量级版面路由功能
- 添加规则命中统计和页面质量指标
- 添加渲染计划输出
- 扩展运行报告，包含版面布局摘要与渲染策略统计
- 改进标题类区块的嵌套去重逻辑
