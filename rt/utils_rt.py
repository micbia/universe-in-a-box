import numpy as np
import matplotlib.patches as patches
from scipy.linalg import norm

def refine_grid_2d(meshsize, refine_cells, levels):
    """
        Generate a 2D grid with refinement at specified cells.
        
        If a cell is refined, it is replaced by its refined sub-cells.

        Parameters:
            meshsize (int): Coarse grid size.
            refine_cells (list): List of (i, j) cells to refine.
            levels (dict): Mapping (i, j) -> refinement level.

        Returns:
            x_grid (np.array): x-coordinates of final cells (lower-left corner).
            y_grid (np.array): y-coordinates of final cells.
            dr (np.array): size of each final cell.
    """
    x_grid = []
    y_grid = []
    dr = []
    
    for i in range(meshsize):
        for j in range(meshsize):
            if (i, j) in refine_cells:
                level = levels[(i, j)]
                sub_divs = 2**level
                dx = 1.0 / sub_divs
                
                # Refine cell into sub_divs × sub_divs sub-cells
                x_sub_cells = np.linspace(i, i + 1, 2**(level+1) + 1)[1:-1]
                y_sub_cells = np.linspace(j, j + 1, 2**(level+1) + 1)[1:-1]

                for x in x_sub_cells:
                    for y in y_sub_cells:
                        if(x % dx != 0 and y % dx != 0):
                            x_grid.append(x)
                            y_grid.append(y)
                            dr.append(dx)
            else:
                # Only keep coarse cell if it was NOT refined
                x_grid.append(i+0.5)
                y_grid.append(j+0.5)
                dr.append(1.0)
    
    return np.array(x_grid), np.array(y_grid), np.array(dr)


class LongChar_Raytracing:
    """ 
        A class that summarize the long charaterisitc raytracing from the notebook: raytracing_2D_longchar.ipynb
    """
    def __init__(self, r_source, r_target, r_cell, dr):
        self.r_source = r_source
        self.r_target = r_target
        self.r_cell = r_cell
        self.dr = dr
        
        print('source:', self.r_source)
        print('target:', self.r_target)
        
        # vector from source to cells
        self.r_st = self.r_target - self.r_source
        self.n_st = self.r_st/norm(self.r_st)
        
        print('source to target:', self.n_st)
        print('distance:', norm(self.r_st))
        
        self.r_touch, self.dr_touch = self.__interacting_cell()

    def __interacting_cell(self):
        # only rays in the direction of the the target
        mask_x = (self.r_cell[:, 0] <= self.r_target[0]) * (self.r_cell[:, 0] >= self.r_source[0])
        mask_y = (self.r_cell[mask_x, 1] <= self.r_target[1]) * (self.r_cell[mask_x, 1] >= self.r_source[1])

        # cells position
        r_c = self.r_cell[mask_x][mask_y]
        dr_grid = self.dr[mask_x][mask_y]

        # source to cells vector
        r_sc = r_c - self.r_source

        # lenght by dot product
        l = r_sc.dot(self.n_st)

        # closest point on the tangent to the cells 
        r_p = self.r_source + l[:,None]*self.n_st[None,:]

        # point onto tanget to cell vector
        r_cp = r_p - r_c

        # select only the cell that are touched by the ray
        mask = np.array([(np.abs(r_cp[i,0]) <= dr_grid[i]/2) and (np.abs(r_cp[i,1]) <= dr_grid[i]/2) for i in range(dr_grid.size)])
        
        # return coordinates and size of the cells interacting with the ray
        r_touch = r_c[mask]
        dr_touch = dr_grid[mask]
        
        return r_touch, dr_touch
    
    def raytracing(self):
        # previous point of interception (init is source position)
        r_inter = np.zeros_like(self.r_touch)
        dist = np.zeros(self.r_touch.shape[0])
        dist_prev = 0

        for idx in range(self.r_touch.shape[0]):  
            # vector to the source to the plane
            r_isp = np.array([self.dr_touch[idx]/2, 0.])
            r_jsp = np.array([0., self.dr_touch[idx]/2])
            #print(r_isp, r_jsp)

            # point on the plane
            p_i0 = self.r_touch[idx] + r_isp
            p_j0 = self.r_touch[idx] + r_jsp

            # normal vector to the surf
            n_iplan = r_isp/norm(r_isp)
            n_jplan = r_jsp/norm(r_jsp)

            # parametric distance of the plan
            d_i = (p_i0-self.r_source).dot(n_iplan)/(self.n_st.dot(n_iplan))
            d_j = (p_j0-self.r_source).dot(n_jplan)/(self.n_st.dot(n_jplan))

            # id of the cell (use this to get the corresponding density)
            if(d_i < d_j):
                r_inter[idx] = self.r_source + d_i*self.n_st
                dist[idx] = d_i - dist_prev
                dist_prev = d_i
            else:
                r_inter[idx] = self.r_source + d_j*self.n_st
                dist[idx] = d_j - dist_prev
                dist_prev = d_j

        # the last cell it stops to the cells coordinate center
        r_inter[-1] = self.r_touch[-1]

        # the distance to the last cell is half that value
        dist[-1] = dist[-1]/2
        return dist, r_inter