from firedrake import *
import math
import time
import numpy as np

parameters["pyop2_options"]["lazy_evaluation"] = False


def newton(meshtype, cell, nx, mexpr, prefix="temp", logging=False, output=False, knownmesh=False, knownmeshfile=None, p=2, sph_deg=2, tol=1e-8):
    assert meshtype in ("plane", "sphere")
    assert cell in ("tri", "quad")
    quads = (cell == "quad")

    if output:
        pvdout = File(prefix + "/pvd/newton.pvd")  # output file name

    if meshtype == "plane":
        mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral=quads)
    else:
        if quads:
            mesh = UnitCubedSphereMesh(refinement_level=nx, degree=sph_deg)
        else:
            mesh = UnitIcosahedralSphereMesh(refinement_level=nx, degree=sph_deg)

    V1 = FunctionSpace(mesh, "Q" if quads else "P", p)
    V2 = TensorFunctionSpace(mesh, "Q" if quads else "P", p)
    V = V1*V2
    P1 = FunctionSpace(mesh, "Q" if quads else "P", 1)  # for representing m
    P0 = FunctionSpace(mesh, "DQ" if quads else "DG", 0)  # for representing integrals over cells
    W_cts = VectorFunctionSpace(mesh, "Q" if quads else "P", 1 if meshtype == "plane" else sph_deg)  # for continuous grad(phi)

    if meshtype == "sphere":
        dxdeg = dx(degree=4*p)  # quadrature degree for nasty terms
    else:
        dxdeg = dx

    # function objects
    phisigma = Function(V); phi, sigma = split(phisigma)
    phisigma_temp = Function(V); phi_temp, sigma_temp = split(phisigma_temp)
    x = Function(mesh.coordinates)
    xi = Function(mesh.coordinates)

    if knownmesh:
        xe = Function(mesh.coordinates)
        xe.dat.data[:] = np.load(prefix + "/" + knownmeshfile)[:]

    m = Function(P1)
    theta = Constant(0.0)
    cellint = Function(P0)

    ### EQUATION DEFINITIONS ###

    v, tau = TestFunctions(V)

    if meshtype == "plane":
        I = Identity(2)

        F = inner(sigma, tau)*dx + dot(div(tau), grad(phi))*dx - (m*det(I + sigma) - theta)*v*dx

        thetaform = m*det(I + sigma_temp)*dx
        resiform = m*det(I + sigma_temp) - theta

    elif meshtype == "sphere":
        modgphi = sqrt(dot(grad(phi), grad(phi)) + 1e-12)
        expxi = xi*cos(modgphi) + grad(phi)*sin(modgphi)/modgphi
        projxi = Identity(3) - outer(xi, xi)

        modgphi_temp = sqrt(dot(grad(phi_temp), grad(phi_temp)) + 1e-12)
        expxi_temp = xi*cos(modgphi_temp) + grad(phi_temp)*sin(modgphi_temp)/modgphi_temp

        F = inner(sigma, tau)*dxdeg + dot(div(tau), expxi)*dxdeg - (m*det(outer(expxi, xi) + dot(sigma, projxi)) - theta)*v*dxdeg

        thetaform = m*det(outer(expxi_temp, xi) + dot(sigma_temp, projxi))*dxdeg
        resiform = m*det(outer(expxi_temp, xi) + dot(sigma_temp, projxi)) - theta

    ### RESIDUAL DEFINITION ###

    v = TestFunction(V1)
    resi_l2_form = v*resiform*dxdeg
    norm_l2_form = v*theta*dxdeg

    ### MONITOR FUNCTION GENERATION ###

    def generate_m(cursol):
        with phisigma_temp.dat.vec as v:
            cursol.copy(v)

        # Make continuous grad(phi)
        solvgradphi.solve()

        # "Fix grad(phi) on sphere"
        if meshtype == "sphere":
            # Ensures that grad(phi).x = 0, assuming |x| = 1
            par_loop("""
for (int i=0; i<vnew.dofs; i++) {
    double dot = 0.0;
    for (int j=0; j<3; j++) {
        dot += x[i][j]*v[i][j];
    }
    for (int j=0; j<3; j++) {
        vnew[i][j] = v[i][j] - dot*x[i][j];
    }
}
""", dx, {'x': (mesh.coordinates, READ),
          'v': (gradphi_cts, READ),
          'vnew': (gradphi_cts2, WRITE)})

        # Generate coordinates
        if meshtype == "plane":
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
        else:
            # Generate new coordinate field using exponential map
            par_loop("""
for (int i=0; i<xi.dofs; i++) {
    double norm = 0.0;
    for (int j=0; j<3; j++) {
        norm += u[i][j]*u[i][j];
    }
    norm = sqrt(norm) + 1e-12;

    for (int j=0; j<3; j++) {
        x[i][j] = xi[i][j]*cos(norm) + (u[i][j]/norm)*sin(norm);
    }
}
""", dx, {'xi': (xi, READ),
          'u': (gradphi_cts2, READ),
          'x': (x, WRITE)})

        mesh.coordinates.assign(x)
        m.interpolate(mexpr)
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

    if meshtype == "plane":
        gradphi_dg = Function(mesh.coordinates)
    if meshtype == "sphere":
        gradphi_cts2 = Function(W_cts)  # extra, as gradphi_cts not necessarily tangential

    # cell average
    v_p0 = TestFunction(P0)
    L_p0 = v_p0*m*dxdeg

    # original cell area
    L_area = v_p0*dxdeg
    orig_area = assemble(L_area)

    # get mesh area
    total_area = assemble(Constant(1.0)*dxdeg(domain=mesh))

    ### SET UP INITIAL sigma (important on sphere) ###

    sigma_ = TrialFunction(V2)
    tau_ = TestFunction(V2)
    sigma_ini = Function(V2)

    asigmainit = inner(sigma_, tau_)*dxdeg
    if meshtype == "plane":
        Lsigmainit = -dot(div(tau_), grad(phi))*dx
    else:
        Lsigmainit = -dot(div(tau_), expxi)*dxdeg

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

        # Exact mesh calculations (where relevant)
        if knownmesh:
            temp = np.arccos(np.minimum(1.0, np.einsum('ij,ij->i', xe.dat.data, x.dat.data)))
            meshl2 = np.linalg.norm(temp)/sqrt(temp.shape[0])
        else:
            meshl2 = 0.0

        resi_l2 = assemble(resi_l2_form).dat.norm
        norm_l2 = assemble(norm_l2_form).dat.norm
        resi_l2_norm = resi_l2/norm_l2

        minmax = min(cellint.dat.data)/max(cellint.dat.data)
        equi = np.std(cellint.dat.data)/np.mean(cellint.dat.data)

        if logging:
            norms.append((resi_l2_norm, equi, meshl2))

        print(str(it), minmax, resi_l2_norm, equi, meshl2)

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


if False:
    ### Plane - Ring ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 60, mexpr, prefix="plane-ring", logging=True)

if False:
    ### Plane - Bell ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.0, alpha1=50, alpha2=100)
    newton("plane", "quad", 60, mexpr, prefix="plane-bell", logging=True)

if False:
    ### Sphere - Ringler tanh function 1 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.5**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-1-quad", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=2)

if False:
    ### Sphere - Ringler tanh function 1 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.5**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-1-lin", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 2 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.25**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-2-lin", logging=True, output=True, sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 2 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.25**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-2-quad", logging=True, output=True, sph_deg=2)

if False:
    ### Sphere - Ringler tanh function 3 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.125**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-3-lin", logging=True, output=True, sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 3 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.125**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-3-quad", logging=True, output=True, sph_deg=2)

if False:
    ### Sphere - Ringler tanh function 4 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.0625**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-4-lin", logging=True, output=True, sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 4 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.0625**4, x0=0.86602540378, x1=0.0, x2=0.5)
    newton("sphere", "quad", 4, mexpr, prefix="ringler-4-quad", logging=True, output=True, sph_deg=2)

if False:
    ### Cross ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 4, mexpr, prefix="cross", output=True)

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 60, mexpr)
    t1 = time.time()
    print("60: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 90, mexpr)
    t1 = time.time()
    print("90: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 120, mexpr)
    t1 = time.time()
    print("120: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 180, mexpr)
    t1 = time.time()
    print("180: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 240, mexpr)
    t1 = time.time()
    print("240: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 3, mexpr)
    t1 = time.time()
    print("3: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 4, mexpr)
    t1 = time.time()
    print("4: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 5, mexpr)
    t1 = time.time()
    print("5: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 6, mexpr)
    t1 = time.time()
    print("6: " + str(t1 - t0))

if False:
    ### Close-up of ring ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 240, mexpr, prefix="plane-ring240", output=True)

if False:
    ### Close-up of cross ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 6, mexpr, prefix="cross240", output=True)

if False:
    ### Ring 30 60 120 240 ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    newton("plane", "quad", 30, mexpr, prefix="plane-ring-equi1", logging=True)
    newton("plane", "quad", 60, mexpr, prefix="plane-ring-equi2", logging=True)
    newton("plane", "quad", 120, mexpr, prefix="plane-ring-equi3", logging=True)
    newton("plane", "quad", 240, mexpr, prefix="plane-ring-equi4", logging=True)

if False:
    ### Cross 3 4 5 6 ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    newton("sphere", "tri", 3, mexpr, prefix="cross-equi1", logging=True)
    newton("sphere", "tri", 4, mexpr, prefix="cross-equi2", logging=True)
    newton("sphere", "tri", 5, mexpr, prefix="cross-equi3", logging=True)
    newton("sphere", "tri", 6, mexpr, prefix="cross-equi4", logging=True)
