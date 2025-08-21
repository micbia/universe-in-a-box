import numpy as np
import matplotlib.patches as patches
from scipy.linalg import norm

def plot_irregular_grid_cells(r_c, dr, color='white', ax=None):
    """
    To plot the 2D grid with refinement generated with the refine_grid_2d function.
    
    Parameters:
        r_c (np.array): list of the coordinate of each cell.
        dr (np.array): List of the size of each cells.
        ax (plot axes): Axes from subplot to draw the grid. 
    Returns:
        ax (plot axes): Axes for plot
    """
    for i_c in range(r_c.shape[0]):
        rect = patches.Rectangle((r_c[i_c,0]-dr[i_c]/2, r_c[i_c,1]-dr[i_c]/2), 
                                  dr[i_c], dr[i_c], 
                                  linewidth=0.8,
                                  edgecolor='black',
                                  facecolor=color,
                                  alpha=1.0)
        ax.add_patch(rect)
    return ax

def plot_ray(r_s, r_t):
    """
    To plot a ray from a source to a target
    """
    n_st = (r_t - r_s) / norm(r_t - r_s)
    
    t = np.linspace(0., norm(r_t - r_s), 100)
    r_x, r_y = np.zeros_like(t), np.zeros_like(t)
    
    for i_d, d in enumerate(t):
        r_x[i_d] = r_s[0] + d*n_st[0]
        r_y[i_d] = r_s[1] + d*n_st[1]
        
    return r_x, r_y