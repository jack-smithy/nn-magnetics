from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
import magpylib as magpy
import numpy as np
import scalar_potential

eps = 1e-6

def simulate_demag(a,b,chi, index):

    print(a, b, chi)

    magnet = magpy.magnet.Cuboid(magnetization = (0,0,1), dimension = (a,b,1))
    volume = a*b
    magnet.susceptibility = chi
    cm = mesh_Cuboid(magnet, target_elems=int(1000*volume))
    #cm = mesh_Cuboid(magnet, target_elems=int(10*volume))

    apply_demag(cm, inplace=True)

    cell_pos_all = np.array([cell.position for cell in cm])

    cell_pos = []
    for cell in cm:
        pos = cell.position
        if(pos[0]>=-eps and pos[1]>=-eps and pos[2]>=-eps):
            cell_pos.append(pos)
            
    cell_pos = np.array(cell_pos)


    x_grid = np.linspace(0, a*2.5, 26)
    y_grid = np.linspace(0, b*2.5, 26)
    z_grid = np.linspace(0, 2.5, 26)

    grid = []
    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                if not(x<=a/2+eps and y<=b/2+eps and z<=0.5+eps):
                    grid.append([x,y,z])

    grid = np.array(grid)

    cell_field_all = cm.getB(cell_pos_all)
    cell_H_all = cm.getH(cell_pos_all)
    cell_magnetization_all = cell_field_all - cell_H_all * 4*np.pi/10
    magnetization_mean = np.mean(cell_magnetization_all, axis=0)


    cell_field = cm.getB(cell_pos)
    cell_H = cm.getH(cell_pos)
    cell_magnetization = cell_field - cell_H * 4*np.pi/10
    magnetization_reduced = cell_magnetization - magnetization_mean
    

    grid_field = cm.getB(grid)
    magnet_reduced = magpy.magnet.Cuboid(magnetization = magnetization_mean, dimension = (a,b,1))
    grid_field_magnet_reduced = magnet_reduced.getB(grid)
    grid_field_reduced = grid_field - grid_field_magnet_reduced

    cell_field_magnet_reduced = magnet_reduced.getB(cell_pos)
    cell_field_reduced = cell_field - cell_field_magnet_reduced

    #for scalar potential
    cell_phi = scalar_potential.getPhi_collection(cm, cell_pos)
    cell_phi_magnet_reduced = scalar_potential.getPhi(cell_pos, np.array((a,b,1)), magnetization_mean, np.array((0,0,0)))
    cell_phi_reduced = cell_phi - cell_phi_magnet_reduced

    grid_phi = scalar_potential.getPhi_collection(cm, grid)
    grid_phi_magnet_reduced = scalar_potential.getPhi(grid, np.array((a,b,1)), magnetization_mean, np.array((0,0,0)))
    grid_phi_reduced = grid_phi - grid_phi_magnet_reduced


    np.savez('%d' % index, 
             a=a, 
             b=b, 
             chi=chi, 
             cell_pos=cell_pos, 
             cell_field=cell_field, 
             cell_H=cell_H, 
             grid=grid, 
             grid_field=grid_field, 
             demagnetization_factor=magnetization_mean[2], 
             magnetization_reduced=magnetization_reduced, 
             grid_field_reduced=grid_field_reduced, 
             cell_field_reduced=cell_field_reduced,
             cell_phi = cell_phi,
             cell_phi_reduced = cell_phi_reduced,
             grid_phi = grid_phi,
             grid_phi_reduced = grid_phi_reduced
             )

    
if __name__ == "__main__":

    np.random.seed(seed=0)

    for index in range(0,1):
        chi = np.random.random()
        a = np.random.random()*2.7+0.3
        b = np.random.random()*2.7+0.3

        if a > b:
            a,b = b,a

        simulate_demag(a,b,chi,index)

