# cython: language_level = 3str
# cython: embedsignature = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from Utility import collapseMeshGridFeatures, reverseOldGridShape
from Preprocess.Tensor import contractSymmetricTensor, expandSymmetricTensor
cimport cython

cpdef nparr[flt, ndim=3] computeDivDevR_2D(nparr bij_mesh, nparr[flt, ndim=2] tke_mesh, nparr[flt, ndim=3] ddev_ri3_dz_mesh, double dx=1., double dy=1.):
    """
    Compute -div(dev(ui'uj')) vector uniform 2D meshgrid of a horizontal or vertical slice, 
    given anisotropy tensor bij_mesh, TKE tke_mesh, d(dev(ui'u3'))/dz vector ddev_ri3_dz_mesh -- all uniform 2D meshgrid -- and dx and dy.
    dev(ui'uj') = ui'uj' - 1/3tr(ui'uj') = 1/3*2TKE*I + 2TKE*bij - 1/3*2TKE*I = 2TKE*bij.
    -div(dev(ui'uj')) = -div(2TKE*bij).
    d(dev(ui'u3'))/dz vector 2D meshgrid needs to be given since z or 3rd D information is missing to derive "/dz" components from given bij_mesh.
    
    z can be interchanged with y: 
        Example 1: nx x ny meshgrid given (horizontal slice), dy is literally dy and d(dev(ui'u3'))/dz is literally about z -- 3rd D.
                   Output -div(dev(ui'uj')) has order of x, y, z.
        Example 2: nx x nz meshgrid given (vertical slice), dy is dz and d(dev(ui'u3'))/dz is about y -- 3rd D.
                   Output -div(dev(ui'uj')) has order of x, z, y.
    
    :param bij_mesh: Anisotropy tensor uniform 2D meshgrid.
    :type bij_mesh: 3/4D array of shape (nx, ny, 6/9) or (nx, ny, 3, 3)
    :param tke_mesh: TKE uniform 2D meshgrid.
    :type tke_mesh: 2D array of shape (nx, ny)
    :param ddev_ri3_dz_mesh: d(dev(ui'u3'))/dz vector uniform 2D meshgrid. dz refers to either literally dz or 3rd D.
    :type ddev_ri3_dz_mesh: 3D array of shape (nx, ny, 3)
    :param dx: Mesh spacing in x or 1st D.
    :type dx: float, optional (default=1.)
    :param dy: Mesh spacing in y or 2nd D.
    :type dy: float, optional (default=1.)
    
    :return: -div(dev(ui'uj')) vector uniform 2D meshgrid of a horizontal or vertical slice.
    :rtype: 3D array of shape (nx, ny, 3)
    """

    cdef nparr[flt, ndim=3] dev_rij_mesh, ddev_ri1_dx_mesh, ddev_ri2_dy_mesh
    cdef unsignint i

    # If bij's last dim is 9, reduce duplicate elements from 9 to 6
    bij_mesh = contractSymmetricTensor(bij_mesh)
    # dev(Rij) = (2/3*tke*I + 2tke*bij) - 2/3*tke*I = 2tke*bij
    # div(dev(Rij)) = -[ddev(R11)/dx + ddev(R12)/dy + ddev(R13)/dz,
    #                   ddev(R21)/dx + ddev(R22)/dy + ddev(R23)/dz,
    #                   ddev(R31)/dx + ddev(R32)/dy + ddev(R33)/dz]
    # Go through every unique component, except R33 as only ddev(R33)/dz is only provided
    # dev(Rij) has shape (nx, ny, 5)
    dev_rij_mesh = 2.*np.atleast_3d(tke_mesh)*bij_mesh[:, :, :5]
    # ddev(Rij)/dx and ddev(Rij)/dy has shape (nx, ny, 3)
    # ddev(Rij)/dx is [ddev(R11)/dx, ddev(R12)/dx, ddev(R13)/dx]
    # ddev(Rij)/dy is [ddev(R12)/dy, ddev(R22)/dy, ddev(R23)/dy]
    ddev_ri1_dx_mesh, ddev_ri2_dy_mesh = (np.empty((bij_mesh.shape[0], bij_mesh.shape[1], 3)),)*2
    for i in range(5):
        # Only dev(R12) has derivative w.r.t. both x and y
        if i == 1:
            ddev_ri2_dy_mesh[:, :, 0], ddev_ri1_dx_mesh[:, :, i] = np.gradient(dev_rij_mesh[:, :, i], dy, dx)
        # For dev(R11) and dev(R13), don't care about "/dy"
        elif i in (0, 2):
            ddev_ri1_dx_mesh[:, :, i] = np.gradient(dev_rij_mesh[:, :, i], dx, axis=1)
        # For dev(R22), dev(R23), don't care about "/dx", i in (3, 4)
        else:
            ddev_ri2_dy_mesh[:, :, i - 2] = np.gradient(dev_rij_mesh[:, :, i], dy, axis=0)

    # div(dev(Rij)) has shape (nx, ny, 3), 3 corresponds to 3 dir
    # So far, there's no ddev(Ri3)/dz info.
    # Therefore, ddev(Ri3)/dz is supplied by ddevRi3_dz_mesh which comes from OpenFOAM
    print("\nFinished -div(dev(ui'uj')) computation")
    return -ddev_ri1_dx_mesh - ddev_ri2_dy_mesh - ddev_ri3_dz_mesh


