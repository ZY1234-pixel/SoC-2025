from scipy.sparse.linalg import spsolve

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import tqdm
from scipy.interpolate import LinearNDInterpolator
import torch
import itertools
import multiprocessing as mp
import scipy.sparse as sp

global grid

def visualize_interpolation_process(uv, coord_S, n=128, query_points=None):
    """
    可视化插值过程：展示如何从uv和coord_S建立插值器，以及查询结果
    
    参数:
    - uv: 变形坐标（插值器的输入点）
    - coord_S: 均匀坐标（插值器的值）
    - n: 网格分辨率
    - query_points: 查询点，如果为None则使用均匀网格点
    """
    
    # 创建插值器
    transfer_S2T = LinearNDInterpolator(uv, coord_S)
    
    # 如果没有提供查询点，使用均匀网格点
    if query_points is None:
        y_coords = np.linspace(0, 1, n)
        x_coords = np.linspace(0, 1, n)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
        query_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
    
    # 执行插值
    interpolated_values = transfer_S2T(query_points)
    
    # 重塑为网格
    query_grid = query_points.reshape(n, n, 2)
    result_grid = interpolated_values.reshape(n, n, 2)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 左上图：输入数据点
    ax1 = axes[0, 0]
    ax1.set_title('Input Data Points\n(uv as positions, coord_S as values)', 
                  fontsize=11, fontweight='bold')
    
    # 显示uv点（位置），用coord_S的值着色
    scatter = ax1.scatter(uv[::10, 0], uv[::10, 1], 
                         c=coord_S[::10, 0], cmap='viridis', 
                         s=5, alpha=0.6)
    ax1.set_xlabel('u coordinate (position)')
    ax1.set_ylabel('v coordinate (position)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='coord_S X value')
    
    # 右上图：查询点
    ax2 = axes[0, 1]
    ax2.set_title('Query Points\n(Uniform grid)', 
                  fontsize=11, fontweight='bold')
    
    # 显示部分查询点
    step = 8
    ax2.scatter(query_grid[::step, ::step, 0].flatten(),
               query_grid[::step, ::step, 1].flatten(),
               c='blue', s=10, alpha=0.6)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 左下图：插值结果
    ax3 = axes[1, 0]
    ax3.set_title('Interpolation Results\n(grid1 = transfer_S2T(query_points))', 
                  fontsize=11, fontweight='bold')
    
    # 显示结果网格
    step = 8
    for i in range(0, n, step):
        ax3.plot(result_grid[i, :, 0], result_grid[i, :, 1], 
                'r-', alpha=0.5, linewidth=0.5)
    for j in range(0, n, step):
        ax3.plot(result_grid[:, j, 0], result_grid[:, j, 1], 
                'r-', alpha=0.5, linewidth=0.5)
    
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 右下图：插值前后对比
    ax4 = axes[1, 1]
    ax4.set_title('Before vs After Interpolation\n(Sample line at center)', 
                  fontsize=11, fontweight='bold')
    
    center_idx = n // 2
    
    # 查询点（均匀网格的中心线）
    ax4.plot(query_grid[center_idx, :, 1], query_grid[center_idx, :, 0], 
            'b-', linewidth=1.5, label='Query X (uniform)', alpha=0.7)
    
    # 插值结果
    ax4.plot(query_grid[center_idx, :, 1], result_grid[center_idx, :, 0], 
            'r-', linewidth=1.5, label='Result X (deformed)', alpha=0.7)
    
    ax4.set_xlabel('Y coordinate')
    ax4.set_ylabel('X value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes, transfer_S2T

def visualize_uv_coord_mapping(uv, coord_S, n=128, sample_step=16, figsize=(16, 8)):
    """
    可视化LinearNDInterpolator中的uv和coord_S映射关系
    
    参数:
    - uv: 变形后的坐标，形状为(n*n, 2)
    - coord_S: 均匀网格坐标，形状为(n*n, 2)
    - n: 网格分辨率
    - sample_step: 采样步长
    """
    
    # 重塑为网格形式
    uv_grid = uv.reshape(n, n, 2)
    coord_S_grid = coord_S.reshape(n, n, 2)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # ============ 左上图：两个网格的叠加 ============
    ax1 = axes[0, 0]
    ax1.set_title('Grid Overlay\n(Blue: Uniform coord_S, Red: Deformed uv)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 绘制均匀网格 coord_S (蓝色)
    step = sample_step
    for i in range(0, n, step):
        ax1.plot(coord_S_grid[i, :, 0], coord_S_grid[i, :, 1], 
                'b-', alpha=0.4, linewidth=0.5)
    for j in range(0, n, step):
        ax1.plot(coord_S_grid[:, j, 0], coord_S_grid[:, j, 1], 
                'b-', alpha=0.4, linewidth=0.5)
    
    # 绘制变形网格 uv (红色)
    for i in range(0, n, step):
        ax1.plot(uv_grid[i, :, 0], uv_grid[i, :, 1], 
                'r-', alpha=0.6, linewidth=0.8)
    for j in range(0, n, step):
        ax1.plot(uv_grid[:, j, 0], uv_grid[:, j, 1], 
                'r-', alpha=0.6, linewidth=0.8)
    
    from matplotlib.lines import Line2D
    legend_elements1 = [
        Line2D([0], [0], color='blue', linewidth=1, alpha=0.5, label='coord_S (Uniform)'),
        Line2D([0], [0], color='red', linewidth=1, alpha=0.5, label='uv (Deformed)')
    ]
    ax1.legend(handles=legend_elements1, loc='upper right')
    
    # ============ 中上图：映射箭头 ============
    ax2 = axes[0, 1]
    ax2.set_title('Mapping: coord_S → uv\n(Green arrows show displacement)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 采样更稀疏的点来显示箭头
    arrow_step = sample_step * 2
    for i in range(0, n, arrow_step):
        for j in range(0, n, arrow_step):
            start = coord_S_grid[i, j]
            end = uv_grid[i, j]
            dx, dy = end[0] - start[0], end[1] - start[1]
            
            if np.sqrt(dx**2 + dy**2) > 0.0001:
                ax2.arrow(start[0], start[1], dx, dy,
                         head_width=0.008, head_length=0.012,
                         fc='green', ec='green', alpha=0.6, linewidth=0.5)
            
            # 绘制点
            ax2.plot(start[0], start[1], 'bo', markersize=2, alpha=0.5)
            ax2.plot(end[0], end[1], 'r^', markersize=2, alpha=0.5)
    
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=6, label='coord_S points'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markersize=6, label='uv points'),
        plt.Line2D([0], [0], color='green', linewidth=1, label='Displacement')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right')
    
    # ============ 右上图：位移场热图 ============
    ax3 = axes[0, 2]
    ax3.set_title('Displacement Magnitude\n|uv - coord_S|', 
                  fontsize=12, fontweight='bold')
    
    displacement = uv_grid - coord_S_grid
    magnitude = np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2)
    
    im = ax3.imshow(magnitude, origin='lower', extent=[0, 1, 0, 1], 
                   cmap='hot', aspect='auto')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    plt.colorbar(im, ax=ax3, label='Displacement Magnitude')
    
    # ============ 左下图：X分量对比 ============
    ax4 = axes[1, 0]
    ax4.set_title('X-coordinate Component\n(coord_S vs uv)', 
                  fontsize=12, fontweight='bold')
    
    # 沿着中心水平线绘制
    center_y_idx = n // 2
    ax4.plot(coord_S_grid[center_y_idx, :, 1], coord_S_grid[center_y_idx, :, 0], 
            'b-', linewidth=1.5, label='coord_S X', alpha=0.7)
    ax4.plot(coord_S_grid[center_y_idx, :, 1], uv_grid[center_y_idx, :, 0], 
            'r-', linewidth=1.5, label='uv X', alpha=0.7)
    ax4.set_xlabel('Y coordinate')
    ax4.set_ylabel('X value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ============ 中下图：Y分量对比 ============
    ax5 = axes[1, 1]
    ax5.set_title('Y-coordinate Component\n(coord_S vs uv)', 
                  fontsize=12, fontweight='bold')
    
    # 沿着中心垂直线绘制
    center_x_idx = n // 2
    ax5.plot(coord_S_grid[:, center_x_idx, 0], coord_S_grid[:, center_x_idx, 1], 
            'b-', linewidth=1.5, label='coord_S Y', alpha=0.7)
    ax5.plot(coord_S_grid[:, center_x_idx, 0], uv_grid[:, center_x_idx, 1], 
            'r-', linewidth=1.5, label='uv Y', alpha=0.7)
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Y value')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ============ 右下图：散点图 ============
    ax6 = axes[1, 2]
    ax6.set_title('Scatter: coord_S vs uv\n(Sampled points)', 
                  fontsize=12, fontweight='bold')
    
    # 采样一些点
    scatter_step = sample_step * 4
    sample_indices = np.arange(0, n, scatter_step)
    
    for i in sample_indices:
        for j in sample_indices:
            ax6.plot([coord_S_grid[i, j, 0], uv_grid[i, j, 0]], 
                    [coord_S_grid[i, j, 1], uv_grid[i, j, 1]], 
                    'k-', alpha=0.2, linewidth=0.3)
    
    ax6.scatter(coord_S_grid[sample_indices][:, sample_indices, 0].flatten(),
               coord_S_grid[sample_indices][:, sample_indices, 1].flatten(),
               c='blue', s=10, alpha=0.6, label='coord_S')
    ax6.scatter(uv_grid[sample_indices][:, sample_indices, 0].flatten(),
               uv_grid[sample_indices][:, sample_indices, 1].flatten(),
               c='red', s=10, alpha=0.6, label='uv')
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Y coordinate')
    ax6.legend()
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def grid(left,right,top,bottom,line,n):
    n1,n2 = left.shape[0],right.shape[0]
    n3,n4 = top.shape[0],bottom.shape[0]
    num = 0 
    for line_p in line:
        num+=len(line_p)
    N = n*n
    
    z = 0
    row = [];col = [];data = []

    for i in range(1,n-1):
        for j in range(1,n-1):
            row.append(z);col.append(i*n+j-1);data.append(1)
            row.append(z);col.append((i-1)*n+j);data.append(1)
            row.append(z);col.append(i*n+j+1);data.append(1)
            row.append(z);col.append((i+1)*n+j);data.append(1)
            row.append(z);col.append(i*n+j);data.append(-4)
            z+=1
            
    A = sp.coo_matrix((data, (row, col)), shape=(z,N))
    b = np.zeros((n1+n2,1))
    z =   0
    row = [];col = [];data = []

    for i in range(n1): 
        x,x1,y,y1 = int(left[i,0]),left[i,0]-int(left[i,0]),int(left[i,1]),left[i,1]-int(left[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        b[z] = i/(n1-1)
        z = z+1

    for i in range(n2): 
        x,x1,y,y1 = int(right[i,0]),right[i,0]-int(right[i,0]),int(right[i,1]),right[i,1]-int(right[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        b[z] = i/(n2-1)
        z = z+1
    B=sp.coo_matrix((data, (row, col)), shape=(n1+n2,N))

    c = np.zeros((n3+n4,1))
    z = 0
    row = [];col = [];data = []

    for i in range(n3): 
        x,x1,y,y1 = int(top[i,0]),top[i,0]-int(top[i,0]),int(top[i,1]),top[i,1]-int(top[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        z = z+1

    for i in range(n4): 
        x,x1,y,y1 = int(bottom[i,0]),bottom[i,0]-int(bottom[i,0]),int(bottom[i,1]),bottom[i,1]-int(bottom[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        c[z] = 1
        z = z+1

    C=sp.coo_matrix((data, (row, col)), shape=(n3+n4,N))
    
    if num:
        z = 0
        row = [];col = [];data = []
        for line_p in line:
            line_p = line_p[:,::-1]
            for i in range(len(line_p)-1):
                x,x1,y,y1 = int(line_p[i,0]),line_p[i,0]-int(line_p[i,0]),int(line_p[i,1]),line_p[i,1]-int(line_p[i,1])
                w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1

                x_,x1_,y_,y1_ = int(line_p[i+1,0]),line_p[i+1,0]-int(line_p[i+1,0]),int(line_p[i+1,1]),line_p[i+1,1]-int(line_p[i+1,1])
                w1_,w2_,w3_,w4_ = (1-x1_)*(1-y1_),y1_*(1-x1_),x1_*(1-y1_),x1_*y1_

                row.append(z);col.append(x*n+y);data.append(w1)
                row.append(z);col.append(x*n+y+1);data.append(w2)
                row.append(z);col.append(x*n+y+n);data.append(w3)
                row.append(z);col.append(x*n+y+n+1);data.append(w4)
                row.append(z);col.append(x_*n+y_);data.append(-w1_)
                row.append(z);col.append(x_*n+y_+1);data.append(-w2_)
                row.append(z);col.append(x_*n+y_+n);data.append(-w3_)
                row.append(z);col.append(x_*n+y_+n+1);data.append(-w4_)
                z+=1
        D=sp.coo_matrix((data, (row, col)), shape=(num,N))
        
    z = 0
    row = [];col = [];data = []
    for i in range(1,n-1):
        for j in range(1,n-1):
            row.append(z);col.append(i*n+j);data.append(1)
            row.append(z);col.append(i*n+j+1);data.append(-1)
            row.append(z);col.append((i+1)*n+j+1);data.append(1)
            row.append(z);col.append((i+1)*n+j);data.append(-1)

            z = z+1
    E =sp.coo_matrix((data, (row, col)), shape=(z,N))


    row = range(n);col = range(n);data = [1]*n
    I=sp.coo_matrix((data, (row, col)), shape=(n,N))
    x = cp.Variable(N)
    
    
    if num:
        D=sp.csc_matrix(D)
        prob = cp.Problem(cp.Minimize(1*(cp.sum_squares(C @ x-c[:,0])+10*cp.sum_squares(B @ x-b[:,0]))+2*(cp.sum_squares(A @ x)+20*cp.sum_squares(E @ x))+0.00001*cp.sum_squares(x[n:]-x[:-n])+10*cp.sum_squares(D @ x)),
            [])
        
    else:
        prob = cp.Problem(cp.Minimize(1*(cp.sum_squares(C @ x-c[:,0])+10*cp.sum_squares(B @ x-b[:,0]))+2*(cp.sum_squares(A @ x)+20*cp.sum_squares(E @ x))+0.00001*cp.sum_squares(x[n:]-x[:-n])),
                        []) 

    prob.solve(solver=cp.OSQP,
           verbose=True,
           max_iter = 100, 
           eps_abs = 0.1)
    
    return x.value.reshape(n,n)
    
def opt(boundary,line,line1):
    n = 128
    textline = []
    if line:
        for i in range(len(line)):
            line[i] = line[i].astype(np.float32)
            if len(line[i][::3])<=1:
                continue
            textline.append(line[i]/512*(n-1))
            
    textline1 = []
    if line1:
        for i in range(len(line1)):
            line1[i] = line1[i][:,::-1].astype(np.float32)
            if len(line1[i][::3])<=1:
                continue
            textline1.append(line1[i]/512*(n-1))

    N = n*n

    boundary = (boundary.permute(0,2,3,1).cpu().detach().numpy()+1)/2*(n-1)
    top = boundary[0,0,:,:]
    right = boundary[0,:,-1,:][:,::-1]
    bottom = boundary[0,-1,:,:]
    left = boundary[0,:,0,:][:,::-1]
    
    # ── [DEBUG] 步骤1：boundary范围
    print("[DEBUG] boundary范围")
    print("="*60)
    print(f"  boundary top范围{top.shape}): min={top.min():.6f}, max={top.max():.6f}, mean={top.mean():.6f}")
    print(f"  boundary right范围{right.shape}): min={right.min():.6f}, max={right.max():.6f}, mean={right.mean():.6f}")
    print(f"  boundary bottom范围 {bottom.shape}): min={bottom.min():.6f}, max={bottom.max():.6f}, mean={bottom.mean():.6f}")
    print(f"  boundary left范围 {left.shape}): min={left.min():.6f}, max={left.max():.6f}, mean={left.mean():.6f}")

    from multiprocessing import Pool
    pool = Pool(processes=2)

    A = [];B = [] 
    pool.apply_async(grid, args=(left,right,top[:,::-1],bottom[:,::-1],textline,n),callback = A.append)
    pool.apply_async(grid, args=(top,bottom,left[:,::-1],right[:,::-1],textline1,n),callback = B.append) 
    pool.close()
    pool.join()

    v = A[0]
    u = B[0]
    uv = np.stack([u.T,v],-1)
    
    coord_S = torch.Tensor(list(itertools.product(
        np.arange(0, 1 + 0.00001, 1/ (n - 1)),
        np.arange(0, 1 + 0.00001, 1/ (n - 1)),
    )))
    Y, X = coord_S.split(1, dim=1)
    coord_S = torch.cat([X, Y], dim=1)
    coord_S = coord_S.numpy().reshape(n*n,2)#*2-0.5

    uv = uv.reshape(n*n,2)
    transfer_S2T = LinearNDInterpolator(uv,coord_S)
    grid1 = transfer_S2T(coord_S)
    grid1 = 2*grid1-1
        # ── 调试可视化（已关闭，改用 debug_save 保存变量文件）────────
    DEBUG_VISUALIZE = True
    if DEBUG_VISUALIZE:
        visualize_uv_coord_mapping(uv, coord_S, n=n, sample_step=16)
        fig, axes, _ = visualize_interpolation_process(uv, coord_S, n=n)
        plt.show()
    
    return torch.from_numpy(grid1.reshape(n,n,2))