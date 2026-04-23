# Changelog

## 0.5.0 - 2026-04-23

- 版面还原与渲染管线全面改进：多列布局检测、页边距估算、fit_scale 全局缩放适配
- 改进多列布局检测：排除 TABLE 块干扰列检测聚类，提升报纸/杂志等复杂版面还原准确度
- 添加表格行防断裂（`cantSplit`）支持，避免表格在分页处被截断
- 排除 TABLE 块作为 zone 分隔符，使多列区域划分更准确
- 改进 fit_scale 估算策略：基于内容高度预算选择最佳缩放系数，防止溢页
- 分级修正策略：溢出时先应用局部修正（段后间距、区块间隙、字号），再应用全局 fit_scale
- 两遍/三遍渲染验证：首遍用 8.5pt 字号下限保证可读性，若仍溢页则降低至 7.0pt/6.5pt 兜底
- 添加 `render_plan.json` 渲染计划输出，说明每个样本选择的渲染策略（single_col / multi_col_table / native_columns）
- 添加 `style_inferred` 样式推断结果到输出 JSON，记录页面级别字体大小、行距、页边距等
- 扩展 run/sample manifests，包含 render-plan 路径、版面布局 profile 摘要、渲染策略统计
- 改进 caption 家族区块的嵌套去重逻辑（如短表格号 `表7` 嵌套在长标题中的情况）
- 坐标映射器（CoordMapper）基于页边距自适应缩放，估算后自动失效重建
- 报纸短横版页面页边距下限放宽至 24pt，保证文字不溢出

## 0.4.0 - 2026-04-15

- Added lightweight page layout routing via `layout_profile` in the recovery pipeline.
- Added rule hit statistics and page quality metrics to pipeline page attributes.
- Added `render_plan.json` output for each sample to explain the selected render strategy.
- Expanded run/sample manifests with render-plan paths, layout profile summary, and rendering strategy statistics.
- Added nested duplicate suppression improvements for caption-family blocks such as short table numbers (`表7`) contained in longer captions.
