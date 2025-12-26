
import numpy as np
import ufl
from dolfinx import mesh, fem, io, default_scalar_type, nls, log
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import math
import time
import os

def relaxation(celltype, nx, dt, m_func, prefix="temp", output=True, maxsteps=1000, tol=1e-8):
    if celltype == 'tri':
        domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    elif celltype == 'quad':
        domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.quadrilateral)
    else:
        raise ValueError("Unsupported cell type")

    degree = 2
    
    # phi
    V1 = fem.functionspace(domain, ("Lagrange", degree))
    # sigma, hessian
    V2 = fem.functionspace(domain, ("Lagrange", degree, (2, 2)))
    # monitor
    P1 = fem.functionspace(domain, ("Lagrange", 1))
    # intergals over cells
    P0 = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    # grad(phi) continuous
    W_cts = fem.functionspace(domain, ("Lagrange", 1, (2, )))

    # integration measures
    dx = ufl.dx

    # functions
    phiold = fem.Function(V1)
    phinew = fem.Function(V1)
    sigmaold = fem.Function(V2)
    sigmanew = fem.Function(V2)
    
    coord_dim = domain.geometry.x.shape[1]
    V_coord = fem.functionspace(domain, ("Lagrange", 1, (coord_dim,)))
    x = fem.Function(V_coord)
    xi = fem.Function(V_coord)    
    xi.interpolate(lambda x : x[:coord_dim])
    x.interpolate(lambda x : x[:coord_dim])

    dtc = fem.Constant(domain, default_scalar_type(dt))
    m = fem.Function(P1)
    theta = fem.Constant(domain, default_scalar_type(0.0))
    cellint = fem.Function(P0)

    # prepare for output
    if output:
        vtx = io.VTXWriter(domain.comm, f"{prefix}.bp", [cellint], engine="BP4")

    ### EQUATION DEFINITIONS ###
    phi = ufl.TrialFunction(V1)
    v = ufl.TestFunction(V1)
    sigma = ufl.TrialFunction(V2)
    tau = ufl.TestFunction(V2)

    I = ufl.Identity(2)

    aphi = ufl.dot(ufl.grad(v), ufl.grad(phi))*dx
    Lphi = ufl.dot(ufl.grad(v), ufl.grad(phiold))*dx + dtc*v*(m*ufl.det(I + sigmaold) - theta)*dx

    asigma = ufl.inner(tau, sigma)*dx
    Lsigma = -ufl.dot(ufl.div(tau), ufl.grad(phinew))*dx

    thetaform = m*ufl.det(I + sigmaold)*dx
    resiform = m*ufl.det(I + sigmaold) - theta

    ### SOLVER OPTIONS ###
    
    probphi = LinearProblem(aphi, Lphi, u=phinew, petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    probsigma = LinearProblem(asigma, Lsigma, u=sigmanew, petsc_options={"ksp_type": "cg"})

    ### RESIDUAL DEFINITION ###

    resi_l2_form = v*resiform*dx
    norm_l2_form = v*theta*dx

    ### MISC EQUATION SETUP ###

    # generate continuous version of grad(phi) by L^2 projection
    u_cts = ufl.TrialFunction(W_cts)
    v_cts = ufl.TestFunction(W_cts)

    gradphi_cts = fem.Function(W_cts)

    a_cts = ufl.dot(v_cts, u_cts)*dx
    L_cts = ufl.dot(v_cts, ufl.grad(phiold))*dx

    probgradphi = LinearProblem(a_cts, L_cts, u=gradphi_cts, petsc_options={"ksp_type": "cg"})

    gradphi_dg = fem.Function(V_coord) # Using V_coord instead of DG for simplicity in update

    # cell average
    v_p0 = ufl.TestFunction(P0)
    L_p0 = v_p0*m*dx

    # original cell area
    L_area = v_p0*dx
    orig_area = fem.assemble_vector(fem.form(L_area))
    # orig_area.assemble()

    # get mesh area
    total_area = domain.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(domain, default_scalar_type(1.0))*dx)), op=MPI.SUM)

    ### SET UP INITIAL sigma ###

    sigma_ = ufl.TrialFunction(V2)
    tau_ = ufl.TestFunction(V2)
    sigma_temp = fem.Function(V2)

    asigmainit = ufl.inner(tau_, sigma_)*dx
    Lsigmainit = -ufl.dot(ufl.div(tau_), ufl.grad(phiold))*dx

    probsigmainit = LinearProblem(asigmainit, Lsigmainit, u=sigma_temp, petsc_options={"ksp_type": "cg"})
    probsigmainit.solve()

    sigmaold.x.array[:] = sigma_temp.x.array[:]

    norms = []

    initial_norm = 1.0

    for ii in range(maxsteps):
        # Make continuous grad(phi)
        probgradphi.solve()
        
        # We can just use gradphi_cts directly if spaces match
        # x = xi + gradphi_cts
        xi_vals = xi.x.array.reshape((-1, coord_dim))
        g_vals = gradphi_cts.x.array.reshape((-1, 2))
        
        # Ensure shapes match
        if coord_dim > 2:
            g_vals_padded = np.zeros_like(xi_vals)
            g_vals_padded[:, :2] = g_vals
            x.x.array[:] = (xi_vals + g_vals_padded).flatten()
        else:
            x.x.array[:] = (xi_vals + g_vals).flatten()
                
        # Update mesh coordinates to x
        domain.geometry.x[:] = x.x.array.reshape((-1, coord_dim))
        
        # Update m
        m.interpolate(m_func)
        
        # cellint
        cellint_vec = fem.assemble_vector(fem.form(L_p0))     
        c_vals = cellint_vec.array
        o_vals = orig_area.array
        o_vals[o_vals==0] = 1.0
        cellint.x.array[:] = c_vals / o_vals
        
        if output:
            vtx.write(ii * dt)
            
        # Restore mesh coordinates to xi for next step?
        # Original: mesh.coordinates.assign(xi)
        domain.geometry.x[:] = xi.x.array.reshape((-1, coord_dim))

        # Evaluate theta
        theta_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(thetaform)), op=MPI.SUM) / total_area
        theta.value = theta_val

        # Residuals
        r_vec = fem.assemble_vector(fem.form(resi_l2_form))
        local_size = r_vec.index_map.size_local
        local_norm_sq = np.sum(r_vec.array[:local_size]**2)
        resi_l2 = np.sqrt(domain.comm.allreduce(local_norm_sq, op=MPI.SUM))
        
        n_vec = fem.assemble_vector(fem.form(norm_l2_form))
        local_size = n_vec.index_map.size_local
        local_norm_sq = np.sum(n_vec.array[:local_size]**2)
        norm_l2 = np.sqrt(domain.comm.allreduce(local_norm_sq, op=MPI.SUM))
        
        if norm_l2 == 0: norm_l2 = 1.0
        resi_l2_norm = resi_l2/norm_l2
        
        if ii == 0:
            initial_norm = resi_l2_norm

        minmax = np.min(cellint.x.array)/np.max(cellint.x.array)
        equi = np.std(cellint.x.array)/np.mean(cellint.x.array)

        norms.append((resi_l2_norm, equi))

        if ii % 10 == 0:
            print(f"{ii}, {minmax}, {resi_l2_norm}, {equi}")

        if resi_l2_norm < tol or resi_l2_norm > 2.0*initial_norm:
            break

        probphi.solve()
        probsigma.solve()

        phiold.x.array[:] = phinew.x.array[:]
        sigmaold.x.array[:] = sigmanew.x.array[:]

    if output:
        vtx.close()

    # Output final mesh to VTU
    # Ensure mesh is in physical configuration
    domain.geometry.x[:] = x.x.array.reshape((-1, coord_dim))
    with io.VTKFile(domain.comm, f"{prefix}_final.vtu", "w") as vtk:
        vtk.write_mesh(domain)

    #np.save(f"{prefix}.npy", np.asarray(norms))


if __name__ == "__main__":
    # Example usages
    
    # Plane - Ring
    if True:
        def mexpr_ring(x):
            # x is (3, N) or (2, N)
            # We need to handle shape
            a = 0.25
            alpha1 = 10
            alpha2 = 200
            r2 = (x[0]-0.5)**2 + (x[1]-0.5)**2
            val = 1.0 + alpha1 * np.cosh(alpha2 * (r2 - a*a))**(-2)
            return val
            
        relaxation("quad", 60, 0.1, mexpr_ring, prefix="plane-ring", output=True)

    # Plane - Bell
    if False:
        def mexpr_bell(x):
            a = 0.0
            alpha1 = 50
            alpha2 = 100
            r2 = (x[0]-0.5)**2 + (x[1]-0.5)**2
            val = 1.0 + alpha1 * np.cosh(alpha2 * (r2 - a*a))**(-2)
            return val
            
        relaxation("tri", 60, 0.04, mexpr_bell, prefix="plane-bell", output=True)

