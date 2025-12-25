from firedrake import *
import math
import numpy as np

parameters["pyop2_options"]["lazy_evaluation"] = False


def newton(cell, nx, mfile, prefix="temp", logging=False, output=False, p=2, tol=1e-8):
    assert cell in ("tri", "quad")
    quads = (cell == "quad")

    m_mesh = PeriodicUnitSquareMesh(100, 100)
    m_FS = FunctionSpace(m_mesh, "DG", 1)
    m_FS_cts = FunctionSpace(m_mesh, "CG", 1)
    q_dg = Function(m_FS)
    q_dg.dat.data[:] = np.load(mfile)[:]
    q = Function(m_FS_cts).project(q_dg)

    def interpolate_onto_m(coords, m, q):
        # ensure coords fall within [0, 1]^2
        processed_coords = np.mod(coords.dat.data[:] + 1.0, 1.0)

        m.dat.data[:] = q.at(processed_coords, tolerance=1e-10)
        m.dat.data[:] = np.square(m.dat.data[:])

        # set minimum
        m.dat.data[m.dat.data < 0.005] = 0.005

    if output:
        pvdout = File(prefix + "/pvd/newton.pvd")  # output file name

    mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral=quads)

    V1 = FunctionSpace(mesh, "Q" if quads else "P", p)
    V2 = TensorFunctionSpace(mesh, "Q" if quads else "P", p)
    V = V1*V2
    P1 = FunctionSpace(mesh, "Q" if quads else "P", 1)  # for representing m
    P0 = FunctionSpace(mesh, "DQ" if quads else "DG", 0)  # for representing integrals over cells
    W_cts = VectorFunctionSpace(mesh, "Q" if quads else "P", 1)  # for continuous grad(phi)
    DP1 = FunctionSpace(mesh, "DQ" if quads else "DP", 1)

    # function objects
    phisigma = Function(V); phi, sigma = split(phisigma)
    phisigma_temp = Function(V); phi_temp, sigma_temp = split(phisigma_temp)
    x = Function(mesh.coordinates)
    x_cts = Function(W_cts)
    xi = Function(mesh.coordinates)

    m = Function(P1)
    count = Function(P1)
    m_dg = Function(DP1)

    theta = Constant(0.0)
    cellint = Function(P0)

    ### EQUATION DEFINITIONS ###

    v, tau = TestFunctions(V)

    I = Identity(2)

    F = inner(sigma, tau)*dx + dot(div(tau), grad(phi))*dx - (m*det(I + sigma) - theta)*v*dx

    thetaform = m*det(I + sigma_temp)*dx
    resiform = m*det(I + sigma_temp) - theta

    ### RESIDUAL DEFINITION ###

    v = TestFunction(V1)
    resi_l2_form = v*resiform*dx
    norm_l2_form = v*theta*dx

    ### MONITOR FUNCTION GENERATION ###

    def generate_m(cursol):
        with phisigma_temp.dat.vec as v:
            cursol.copy(v)

        # Make continuous grad(phi)
        solvgradphi.solve()

        # Generate coordinates

        # Copy CG grad(phi) into DG field
        par_loop("""
for (int i=0; i<cg.dofs; i++) {
    for (int j=0; j<2; j++) {
        dg[i][j] = cg[i][j];
    }
}
""", dx, {'cg': (gradphi_cts, READ),
          'dg': (gradphi_dg, WRITE)})

        x.assign(xi + gradphi_dg)  # x = xi + grad(phi)

        mesh.coordinates.assign(x)

        # make "continuous" x to avoid doing 4 unstructured interpolations per
        # vertex
        par_loop("""
for (int i=0; i<cg.dofs; i++) {
    for (int j=0; j<2; j++) {
        cg[i][j] = dg[i][j];
    }
}
""", dx, {'cg': (x_cts, WRITE),
          'dg': (x, READ)})

        interpolate_onto_m(x_cts, m, q)

        mesh.coordinates.assign(xi)

        theta_new = assemble(thetaform)/total_area
        theta.assign(theta_new)

    ### SOLVER OPTIONS ###

    phi__, sigma__ = TrialFunctions(V)
    v__, tau__ = TestFunctions(V)

    # Custom preconditioning matrix
    Jp = inner(sigma__, tau__)*dx + phi__*v__*dx + dot(grad(phi__), grad(v__))*dx

    prob = NonlinearVariationalProblem(F, phisigma, Jp=Jp)
    V1_nullspace = VectorSpaceBasis(constant=True)
    nullspace = MixedVectorSpaceBasis(V, [V1_nullspace, V.sub(1)])

    params = {"ksp_type": "gmres",
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "multiplicative",
              "pc_fieldsplit_off_diag_use_amat": True,
              "fieldsplit_0_pc_type": "gamg",
              "fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_mg_levels_ksp_max_it": 5,
              # "fieldsplit_0_mg_levels_pc_type": "bjacobi",  # parallel
              # "fieldsplit_0_mg_levels_sub_ksp_type": "preonly",  # parallel
              # "fieldsplit_0_mg_levels_sub_pc_type": "ilu",  # parallel
              "fieldsplit_0_mg_levels_pc_type": "ilu",  # serial
              # "fieldsplit_1_pc_type": "bjacobi",  # parallel
              # "fieldsplit_1_sub_ksp_type": "preonly",  # parallel
              # "fieldsplit_1_sub_pc_type": "ilu",  # parallel
              "fieldsplit_1_pc_type": "ilu",  # serial
              "fieldsplit_1_ksp_type": "preonly",
              "ksp_max_it": 200,
              "snes_max_it": 125,
              "ksp_gmres_restart": 200,
              "snes_rtol": tol,
              "snes_linesearch_type": "l2",
              "snes_linesearch_max_it": 5,
              "snes_linesearch_maxstep": 1.05,
              "snes_linesearch_damping": 0.8,
              # "ksp_monitor": True,
              # "snes_monitor": True,
              # "snes_linesearch_monitor": True,
              "snes_lag_preconditioner": -1}

    solv = NonlinearVariationalSolver(prob,
                                      nullspace=nullspace,
                                      transpose_nullspace=nullspace,
                                      pre_jacobian_callback=generate_m,
                                      pre_function_callback=generate_m,
                                      solver_parameters=params)

    ### MISC EQUATION SETUP ###

    # generate continuous version of grad(phi) by L^2 projection
    u_cts = TrialFunction(W_cts)
    v_cts = TestFunction(W_cts)

    gradphi_cts = Function(W_cts)

    a_cts = dot(v_cts, u_cts)*dx
    L_cts = dot(v_cts, grad(phi_temp))*dx

    probgradphi = LinearVariationalProblem(a_cts, L_cts, gradphi_cts)
    solvgradphi = LinearVariationalSolver(probgradphi,
                                          solver_parameters={'ksp_type': 'cg'})

    gradphi_dg = Function(mesh.coordinates)

    # cell average
    v_p0 = TestFunction(P0)
    L_p0 = v_p0*m*dx

    # original cell area
    L_area = v_p0*dx
    orig_area = assemble(L_area)

    # get mesh area
    total_area = assemble(Constant(1.0)*dx(domain=mesh))

    ### SET UP INITIAL sigma (important on sphere) ###

    sigma_ = TrialFunction(V2)
    tau_ = TestFunction(V2)
    sigma_ini = Function(V2)

    asigmainit = inner(sigma_, tau_)*dx
    Lsigmainit = -dot(div(tau_), grad(phi))*dx

    solve(asigmainit == Lsigmainit, sigma_ini, solver_parameters={'ksp_type': 'cg'})

    phisigma.sub(1).assign(sigma_ini)

    def fakemonitor(snes, it, rnorm):
        cursol = snes.getSolution()
        generate_m(cursol)  # updates m and theta

        mesh.coordinates.assign(x)
        assemble(L_p0, tensor=cellint)  # For equidistribution measure
        cellint.dat.data[:] /= orig_area.dat.data[:]  # Normalise
        if output:
            pvdout.write(cellint)
        mesh.coordinates.assign(xi)

        resi_l2 = assemble(resi_l2_form).dat.norm
        norm_l2 = assemble(norm_l2_form).dat.norm
        resi_l2_norm = resi_l2/norm_l2

        minmax = min(cellint.dat.data)/max(cellint.dat.data)
        equi = np.std(cellint.dat.data)/np.mean(cellint.dat.data)

        if logging:
            norms.append((resi_l2_norm, equi))

        print(str(it), minmax, resi_l2_norm, equi)

    if logging:
        norms = []

    solv.snes.setMonitor(fakemonitor)

    try:
        solv.solve()
    except ConvergenceError:
        pass

    if logging:
        np.save(prefix + "/norms-newt.npy", np.asarray(norms))
        # np.save(prefix + "/cellints.npy", np.asarray(cellint.dat.data))


### Plane - QG from file ###
mfile = "qg100.npy"
newton("quad", 60, mfile, prefix="qg", output=True, tol=1e-4)
