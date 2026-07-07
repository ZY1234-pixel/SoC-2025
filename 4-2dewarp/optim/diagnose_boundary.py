"""
diagnose_boundary.py
分析 debug_vars.npz 中的边界点数据，验证坐标格式是否正确。
"""
import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data = np.load('optim/debug_vars.npz', allow_pickle=True)
top    = data['top']
right  = data['right']
bottom = data['bottom']
left   = data['left']
u      = data['u']
v      = data['v']

n = 128
print('=== 边界点统计（当前格式：col, row）===')
print(f'top    shape={top.shape}  col(0)=[{top[:,0].min():.1f}, {top[:,0].max():.1f}]  row(1)=[{top[:,1].min():.1f}, {top[:,1].max():.1f}]')
print(f'right  shape={right.shape}  col(0)=[{right[:,0].min():.1f}, {right[:,0].max():.1f}]  row(1)=[{right[:,1].min():.1f}, {right[:,1].max():.1f}]')
print(f'bottom shape={bottom.shape}  col(0)=[{bottom[:,0].min():.1f}, {bottom[:,0].max():.1f}]  row(1)=[{bottom[:,1].min():.1f}, {bottom[:,1].max():.1f}]')
print(f'left   shape={left.shape}  col(0)=[{left[:,0].min():.1f}, {left[:,0].max():.1f}]  row(1)=[{left[:,1].min():.1f}, {left[:,1].max():.1f}]')

print()
print('=== 前3个点 ===')
print(f'top前3:    {top[:3]}')
print(f'right前3:  {right[:3]}')
print(f'bottom前3: {bottom[:3]}')
print(f'left前3:   {left[:3]}')

print()
print('=== 期望值分析（格式为 col, row）===')
print(f'left[:,0](col)均值={left[:,0].mean():.1f}  期望接近0（左边缘）')
print(f'right[:,0](col)均值={right[:,0].mean():.1f}  期望接近{n-1}（右边缘）')
print(f'top[:,1](row)均值={top[:,1].mean():.1f}  期望接近0（上边缘）')
print(f'bottom[:,1](row)均值={bottom[:,1].mean():.1f}  期望接近{n-1}（下边缘）')

print()
print('=== u/v 结果统计 ===')
print(f'u: min={u.min():.3f}  max={u.max():.3f}  mean={u.mean():.3f}  (期望在[0,1]内)')
print(f'v: min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}  (期望在[0,1]内)')

print()
print('=== grid() 调用分析 ===')
print('v = grid(left, right, top[:,::-1], bottom[:,::-1], textline, n)')
print('  left/right 直接传入，第0列作为 x（行索引），第1列作为 y（列索引）')
print('  top[:,::-1] 翻转后，第0列变为 row，第1列变为 col')
print()
print('u = grid(top, bottom, left[:,::-1], right[:,::-1], textline1, n)')
print('  top/bottom 直接传入，第0列作为 x（行索引），第1列作为 y（列索引）')
print()

# 分析 grid() 内部的索引计算
# grid() 中：x,x1,y,y1 = int(left[i,0]), left[i,0]-int(left[i,0]), int(left[i,1]), left[i,1]-int(left[i,1])
# 索引 = x*n+y = left[:,0]*n + left[:,1]
# 如果 left 格式是 (col, row)，则 x=col, y=row，索引 = col*n+row
# 如果 left 格式是 (row, col)，则 x=row, y=col，索引 = row*n+col（正确的行优先索引）

print('=== 索引计算验证 ===')
# 检查 left 的第0列（col）是否在合理范围内作为行索引
print(f'left[:,0] 作为行索引范围: [{left[:,0].min():.1f}, {left[:,0].max():.1f}]')
print(f'left[:,1] 作为列索引范围: [{left[:,1].min():.1f}, {left[:,1].max():.1f}]')
print()
# 如果 left 是 (col, row)，则 x=col（列），y=row（行）
# grid() 用 x*n+y = col*n+row，这是列优先索引，不是行优先
# 正确应该是 row*n+col（行优先）
# 所以如果 left 是 (col, row)，传入 grid() 会导致 col 被当作行索引，row 被当作列索引
# 这会导致 B 矩阵约束错误

# 验证：left 的 col 值（第0列）是否接近 0（左边缘的 col 应该接近 0）
print(f'left[:,0](col) 均值={left[:,0].mean():.1f}，若接近0则格式正确（左边缘col≈0）')
print(f'left[:,1](row) 均值={left[:,1].mean():.1f}，应覆盖整个行范围[0,{n-1}]')

# 如果 left 的 col 接近 0，那么 x=col≈0，索引 = 0*n+row = row，这实际上是正确的！
# 因为左边缘的 col≈0，所以 x*n+y ≈ 0*n+row = row，指向第0列的各行
# 这意味着即使格式是 (col, row)，对于左边缘（col≈0）来说，索引计算是正确的

print()
print('=== 结论 ===')
if left[:,0].mean() < 5:
    print('left 的 col 均值接近 0，说明左边缘点确实在图像左侧')
    print('对于左边缘，col≈0，所以 x*n+y ≈ 0*n+row = row，索引计算基本正确')
else:
    print(f'WARNING: left 的 col 均值={left[:,0].mean():.1f}，远离0，说明左边缘点不在图像左侧！')
    print('这是问题所在：边界点没有正确描述文档边缘位置')

if right[:,0].mean() > n-10:
    print(f'right 的 col 均值接近 {n-1}，说明右边缘点确实在图像右侧')
else:
    print(f'WARNING: right 的 col 均值={right[:,0].mean():.1f}，远离{n-1}，说明右边缘点不在图像右侧！')
