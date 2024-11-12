import numpy as np

def magnet_cuboid_scalar_potential(
    observers: np.ndarray,
    dimensions: np.ndarray,
    polarizations: np.ndarray,
):

    assert len(observers) == len(dimensions) and len(dimensions) == len(polarizations), "wrong dimensions"
    
    a, b, c = dimensions.T / 2
    x, y, z = np.copy(observers).T

    eps = 1e-6

    # can be extended for vectorized computation of dimensions
    scalar_potential = np.zeros(len(observers))

    sign = -1

    for z1 in [-c,c]:
        zmz1 = z-z1

        for x1 in [-a,a]:
            xmx1 = x-x1
        
            for y1 in [-b,b]:
                ymy1 = y-y1
                mask1 = (np.abs(xmx1) > eps) & (np.abs(ymy1) > eps)
                mask2 = (np.abs(zmz1) > eps) & (np.abs(xmx1) > eps) & (np.abs(ymy1) > eps)
                sr = np.sqrt(xmx1**2+ymy1**2+zmz1**2)

                #print(xmx1, ymy1, zmz1, sr)

                scalar_potential[mask1] += sign * xmx1[mask1] * np.arctanh(ymy1[mask1]/sr[mask1])
                scalar_potential[mask1] += sign * ymy1[mask1] * np.arctanh(xmx1[mask1]/sr[mask1])
                scalar_potential[mask2] += -sign * zmz1[mask2] * np.arctan(xmx1[mask2]*ymy1[mask2]/zmz1[mask2]/sr[mask2])

                sign *= -1

            sign *= -1
        
        sign *= -1

    scalar_potential *= polarizations / 4 / np.pi

    return scalar_potential

def magnet_cuboid_scalar_potential_all_components(
    observers: np.ndarray,
    dimensions: np.ndarray,
    polarizations: np.ndarray,
):
    
    x_component = magnet_cuboid_scalar_potential(np.column_stack((observers[:,2], -observers[:,1], observers[:,0])), np.column_stack((dimensions[:,2], dimensions[:,1], dimensions[:,0])), polarizations[:,0])
    y_component = magnet_cuboid_scalar_potential(np.column_stack((observers[:,2], observers[:,0], observers[:,1])), np.column_stack((dimensions[:,2], dimensions[:,0], dimensions[:,1])), polarizations[:,1])
    z_component = magnet_cuboid_scalar_potential(observers, dimensions, polarizations[:,2])

    scalar_potential = x_component + y_component + z_component

    return scalar_potential

def getPhi(
    observers: np.ndarray,
    dimensions: np.ndarray,
    polarizations: np.ndarray,
    positions: np.ndarray,
):
    #check if dimensions are okay
    assert len(dimensions) == len(polarizations) and len(polarizations) == len(positions), "wrong dimensions"

    #check if arrays are only 1D - extend them if necessary
    if len(observers.shape) < 2:
        observers = np.expand_dims(observers, axis=0)

    if len(dimensions.shape) < 2:
        dimensions = np.expand_dims(dimensions, axis=0)
        polarizations = np.expand_dims(polarizations, axis=0)
        positions = np.expand_dims(positions, axis=0)



    len_obs = len(observers)
    len_dim = len(dimensions)

    observers = np.tile(observers, (len_dim, 1))
    dimensions = np.repeat(dimensions, len_obs, axis=0)
    polarizations = np.repeat(polarizations, len_obs, axis=0)
    positions = np.repeat(positions, len_obs, axis=0)

    scalar_potential = magnet_cuboid_scalar_potential_all_components(observers-positions, dimensions, polarizations)

    if(len_dim > 1):
        scalar_potential = np.reshape(scalar_potential, (len_dim, len_obs))


    return scalar_potential

def getPhi_collection(collection, observers):
    dimensions = np.zeros((len(collection), 3))
    polarizations = np.zeros((len(collection), 3))
    positions = np.zeros((len(collection), 3))

    #print('length', len(collection))

    for i in range(len(collection)):

        # print('dim',collection[i].dimension)
        # print('mag',collection[i].magnetization)
        # print('pos',collection[i].position)

    
        dimensions[i,:] = collection[i].dimension
        polarizations[i,:] = collection[i].magnetization
        positions[i,:] = collection[i].position

    #print(dimensions, polarizations, positions)

    scalar_potential = getPhi(observers, dimensions, polarizations, positions)
    scalar_potential = np.sum(scalar_potential, axis=0)

    return scalar_potential

def derivative(fun, x, args, eps=1e-6):

    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=0)

    deltax = np.zeros(x.shape)
    deltax[:,0] = eps
    deltay = np.zeros(x.shape)
    deltay[:,1] = eps
    deltaz = np.zeros(x.shape)
    deltaz[:,2] = eps
    dfdx = (fun(x+deltax, *args) - fun(x-deltax, *args)) / 2 / eps
    dfdy = (fun(x+deltay, *args) - fun(x-deltay, *args)) / 2 / eps
    dfdz = (fun(x+deltaz, *args) - fun(x-deltaz, *args)) / 2 / eps

    return (dfdx, dfdy, dfdz)


if __name__ == "__main__":

    # observers = (np.random.rand(10,3)-0.5)*10
    # #observers = np.array(((0,0,-1), (0,0,1)))
    # dimensions = np.array((1,1,1))
    # polarizations = 1

    # scalar_potential = magnet_cuboid_scalar_potential(observers, dimensions, polarizations)

    # print(scalar_potential)

    # eps = 1e-6

    # deltax = np.zeros(observers.shape)
    # deltax[:,0] = eps
    # deltay = np.zeros(observers.shape)
    # deltay[:,1] = eps
    # deltaz = np.zeros(observers.shape)
    # deltaz[:,2] = eps
    # Bx = -(magnet_cuboid_scalar_potential(observers+deltax, dimensions, polarizations) - magnet_cuboid_scalar_potential(observers-deltax, dimensions, polarizations)) / 2 / eps
    # By = -(magnet_cuboid_scalar_potential(observers+deltay, dimensions, polarizations) - magnet_cuboid_scalar_potential(observers-deltay, dimensions, polarizations)) / 2 / eps
    # Bz = -(magnet_cuboid_scalar_potential(observers+deltaz, dimensions, polarizations) - magnet_cuboid_scalar_potential(observers-deltaz, dimensions, polarizations)) / 2 / eps

    # print('Bx', Bx)
    # print('By', By)
    # print('Bz', Bz)


    # import magpylib as magpy
    # magnet = magpy.magnet.Cuboid(magnetization=(0,0,polarizations), dimension=dimensions)
    # print(magnet.getB(observers))

    ##################

    # observers = (np.random.rand(10,3)-0.5)*10
    # #observers = np.array(((0,0,-1), (0,0,1)))
    # dimensions = np.array((1,1,1))
    # polarizations = np.array((1,2,3))

    # scalar_potential = magnet_cuboid_scalar_potential_all_components(observers, dimensions, polarizations)

    # print(scalar_potential)

    # eps = 1e-6

    # deltax = np.zeros(observers.shape)
    # deltax[:,0] = eps
    # deltay = np.zeros(observers.shape)
    # deltay[:,1] = eps
    # deltaz = np.zeros(observers.shape)
    # deltaz[:,2] = eps
    # Bx = -(magnet_cuboid_scalar_potential_all_components(observers+deltax, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltax, dimensions, polarizations)) / 2 / eps
    # By = -(magnet_cuboid_scalar_potential_all_components(observers+deltay, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltay, dimensions, polarizations)) / 2 / eps
    # Bz = -(magnet_cuboid_scalar_potential_all_components(observers+deltaz, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltaz, dimensions, polarizations)) / 2 / eps

    # print('Bx', Bx)
    # print('By', By)
    # print('Bz', Bz)


    # import magpylib as magpy
    # magnet = magpy.magnet.Cuboid(magnetization=polarizations, dimension=dimensions)
    # print(magnet.getB(observers))

    #########################

    # #observers = (np.random.rand(10,3)-0.5)*10
    # observers = np.array(((0.1,0.1,0.1), (-10,4,3)))
    # dimensions = np.array(((1,1,1), (1,2,3)))
    # polarizations = np.array(((1,2,-3), (1,-4,-2)))

    # scalar_potential = magnet_cuboid_scalar_potential_all_components(observers, dimensions, polarizations)

    # print(scalar_potential)

    # eps = 1e-6

    # deltax = np.zeros(observers.shape)
    # deltax[:,0] = eps
    # deltay = np.zeros(observers.shape)
    # deltay[:,1] = eps
    # deltaz = np.zeros(observers.shape)
    # deltaz[:,2] = eps
    # Bx = -(magnet_cuboid_scalar_potential_all_components(observers+deltax, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltax, dimensions, polarizations)) / 2 / eps
    # By = -(magnet_cuboid_scalar_potential_all_components(observers+deltay, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltay, dimensions, polarizations)) / 2 / eps
    # Bz = -(magnet_cuboid_scalar_potential_all_components(observers+deltaz, dimensions, polarizations) - magnet_cuboid_scalar_potential_all_components(observers-deltaz, dimensions, polarizations)) / 2 / eps

    # print('Bx', Bx)
    # print('By', By)
    # print('Bz', Bz)


    # import magpylib as magpy

    # for i in range(len(dimensions)):
    #     magnet = magpy.magnet.Cuboid(magnetization=polarizations[i], dimension=dimensions[i])
    #     #print(magnet.getB(observers))
    #     print(magnet.getH(observers[i])*4*np.pi/10)

##################

    #observers = np.logspace(np.array((0,0,0)),np.array((20,0,0)),21)
    #observers = np.array(((0.1,0.1,0.5), (-10,4,0.5), (0,-4,0.5)))
    # observers = np.array(((0.1,0.1,0.5), (-10,4,0.5), (0,-4,0.5)))
    # dimensions = np.array(((1,1,1), (1,2,3)))
    # polarizations = np.array(((1,2,-3), (1,-4,-2)))
    # positions = np.array(((0,0,0), (-1,-0.5,-2)))

    observers = np.array(((-3,-3,-3),(3,-3,-3),(-3,3,-3),(3,3,-3),(-3,-3,3),(3,-3,3),(-3,3,3),(3,3,3)))*1e-3
    dimensions = np.array(((1,2,3)))*1e-3
    polarizations = np.array(((0,0,1)))*1e-3
    positions = np.array(((0,0,0)))*1e-3

    scalar_potential = getPhi(observers, dimensions, polarizations, positions)

    print(scalar_potential)


    res = derivative(getPhi, observers, (dimensions, polarizations, positions), eps=1e-6)
    Bx = -res[0]
    By = -res[1]
    Bz = -res[2]

    print('Bx', Bx)
    print('By', By)
    print('Bz', Bz)

###############

    import magpylib as magpy
    from scipy.constants import mu_0

    #check if arrays are only 1D - extend them if necessary
    if len(observers.shape) < 2:
        observers = np.expand_dims(observers, axis=0)

    if len(dimensions.shape) < 2:
        dimensions = np.expand_dims(dimensions, axis=0)
        polarizations = np.expand_dims(polarizations, axis=0)
        positions = np.expand_dims(positions, axis=0)

    for i in range(len(dimensions)):
        magnet = magpy.magnet.Cuboid(magnetization=polarizations[i], dimension=dimensions[i], position=positions[i])
        print(magnet.getB(observers)/mu_0)
        print(magnet.getH(observers))