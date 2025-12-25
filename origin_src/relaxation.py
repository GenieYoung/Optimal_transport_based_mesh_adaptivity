from firedrake import *
import math
import time
import numpy as np

parameters["pyop2_options"]["lazy_evaluation"] = False


def relaxation(meshtype, cell, nx, dt, mexpr, prefix="temp", logging=False, output=False, knownmesh=False, knownmeshfile=None, p=2, sph_deg=2, maxsteps=1000, tol=1e-8):
    assert meshtype in ("plane", "sphere")
    assert cell in ("tri", "quad")
    quads = (cell == "quad")

    if output:
        pvdout = File(prefix + "/pvd/relaxation.pvd")  # output file name

    if meshtype == "plane":
        mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral=quads)
    else:
        if quads:
            mesh = UnitCubedSphereMesh(refinement_level=nx, degree=sph_deg)
        else:
            mesh = UnitIcosahedralSphereMesh(refinement_level=nx, degree=sph_deg)

    V1 = FunctionSpace(mesh, "Q" if quads else "P", p)
    V2 = TensorFunctionSpace(mesh, "Q" if quads else "P", p)
    P1 = FunctionSpace(mesh, "Q" if quads else "P", 1)  # for representing m
    P0 = FunctionSpace(mesh, "DQ" if quads else "DP", 0)  # for representing integrals over cells
    W_cts = VectorFunctionSpace(mesh, "Q" if quads else "P", 1 if meshtype == "plane" else sph_deg)  # for continuous grad(phi)

    if meshtype == "sphere":
        dxdeg = dx(degree=4*p)  # quadrature degree for nasty terms
    else:
        dxdeg = dx

    # function objects
    phiold = Function(V1); phinew = Function(V1)
    sigmaold = Function(V2); sigmanew = Function(V2)
    x = Function(mesh.coordinates)
    xi = Function(mesh.coordinates)

    if knownmesh:
        xe = Function(mesh.coordinates)
        xe.dat.data[:] = np.load(prefix + "/" + knownmeshfile)[:]

    dtc = Constant(dt)
    m = Function(P1)
    theta = Constant(0.0)
    cellint = Function(P0)

    ### EQUATION DEFINITIONS ###

    phi = TrialFunction(V1)
    v = TestFunction(V1)
    sigma = TrialFunction(V2)
    tau = TestFunction(V2)

    if meshtype == "plane":
        I = Identity(2)

        aphi = dot(grad(v), grad(phi))*dx
        Lphi = dot(grad(v), grad(phiold))*dx + dtc*v*(m*det(I + sigmaold) - theta)*dx

        asigma = inner(tau, sigma)*dx
        Lsigma = -dot(div(tau), grad(phinew))*dx

        thetaform = m*det(I + sigmaold)*dx
        resiform = m*det(I + sigmaold) - theta

    elif meshtype == "sphere":
        modgphi = sqrt(dot(grad(phiold), grad(phiold)) + 1e-12)
        expxi = xi*cos(modgphi) + grad(phiold)*sin(modgphi)/modgphi
        projxi = Identity(3) - outer(xi, xi)
        modgphinew = sqrt(dot(grad(phinew), grad(phinew)) + 1e-12)
        expxinew = xi*cos(modgphinew) + grad(phinew)*sin(modgphinew)/modgphinew

        aphi = dot(grad(v), grad(phi))*dxdeg
        Lphi = dot(grad(v), grad(phiold))*dxdeg + dtc*v*(m*det(outer(expxi, xi) + dot(sigmaold, projxi)) - theta)*dxdeg

        asigma = inner(tau, sigma)*dxdeg
        Lsigma = -dot(div(tau), expxinew)*dxdeg

        thetaform = m*det(outer(expxi, xi) + dot(sigmaold, projxi))*dxdeg
        resiform = m*det(outer(expxi, xi) + dot(sigmaold, projxi)) - theta

    ### SOLVER OPTIONS ###

    probphi = LinearVariationalProblem(aphi, Lphi, phinew)
    nullspace = VectorSpaceBasis(constant=True)
    solvphi = LinearVariationalSolver(probphi,
                                      nullspace=nullspace,
                                      transpose_nullspace=nullspace,
                                      solver_parameters={'ksp_type': 'cg',
                                                         'pc_type': 'gamg'})

    probsigma = LinearVariationalProblem(asigma, Lsigma, sigmanew)
    solvsigma = LinearVariationalSolver(probsigma,
                                        solver_parameters={'ksp_type': 'cg'})

    ### RESIDUAL DEFINITION ###

    resi_l2_form = v*resiform*dxdeg
    norm_l2_form = v*theta*dxdeg

    ### MISC EQUATION SETUP ###

    # generate continuous version of grad(phi) by L^2 projection
    u_cts = TrialFunction(W_cts)
    v_cts = TestFunction(W_cts)

    gradphi_cts = Function(W_cts)

    a_cts = dot(v_cts, u_cts)*dxdeg
    L_cts = dot(v_cts, grad(phiold))*dxdeg

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
    sigma_temp = Function(V2)

    asigmainit = inner(tau_, sigma_)*dxdeg
    if meshtype == "plane":
        Lsigmainit = -dot(div(tau_), grad(phiold))*dx
    else:
        Lsigmainit = -dot(div(tau_), expxi)*dxdeg

    solve(asigmainit == Lsigmainit, sigma_temp, solver_parameters={'ksp_type': 'cg'})

    sigmaold.assign(sigma_temp)

    if logging:
        norms = []

    for ii in range(maxsteps):
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

        # Update m
        mesh.coordinates.assign(x)
        m.interpolate(mexpr)
        assemble(L_p0, tensor=cellint)  # For equidistribution measure
        cellint.dat.data[:] /= orig_area.dat.data[:]  # Normalise
        if output:
            pvdout.write(cellint)
        mesh.coordinates.assign(xi)

        # Evaluate theta
        theta_new = assemble(thetaform)/total_area
        theta.assign(theta_new)

        # Exact mesh calculations (where relevant)
        if knownmesh:
            temp = np.arccos(np.minimum(1.0, np.einsum('ij,ij->i', xe.dat.data, x.dat.data)))
            meshl2 = np.linalg.norm(temp)/sqrt(temp.shape[0])
        else:
            meshl2 = 0.0

        resi_l2 = assemble(resi_l2_form).dat.norm
        norm_l2 = assemble(norm_l2_form).dat.norm
        resi_l2_norm = resi_l2/norm_l2
        if ii == 0:
            initial_norm = resi_l2_norm  # store to check for divergence

        minmax = min(cellint.dat.data)/max(cellint.dat.data)
        equi = np.std(cellint.dat.data)/np.mean(cellint.dat.data)

        if logging:
            norms.append((resi_l2_norm, equi, meshl2))

        if ii % 10 == 0:
            print(ii, minmax, resi_l2_norm, equi, meshl2)

        if resi_l2_norm < tol or resi_l2_norm > 2.0*initial_norm:
            break

        solvphi.solve()  # obtain new phi
        solvsigma.solve()  # obtain new sigma

        phiold.assign(phinew)
        sigmaold.assign(sigmanew)

    if logging:
        np.save(prefix + "/norms-relax.npy", np.asarray(norms))
        # np.save(prefix + "/cellints.npy", np.asarray(cellint.dat.data))


if False:
    ### Plane - Ring ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 60, 0.1, mexpr, prefix="plane-ring", logging=True, output=True)

if False:
    ### Plane - Bell ###
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.0, alpha1=50, alpha2=100)
    relaxation("plane", "quad", 60, 0.04, mexpr, prefix="plane-bell", logging=True, output=True)

if False:
    ### Sphere - Ringler tanh function 1 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.5**4, x0=0.86602540378, x1=0.0, x2=0.5)
    relaxation("sphere", "quad", 4, 2.0, mexpr, prefix="ringler-1-lin", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 4 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.0625**4, x0=0.86602540378, x1=0.0, x2=0.5)
    relaxation("sphere", "quad", 4, 2.0, mexpr, prefix="ringler-4-lin", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=1)

if False:
    ### Sphere - Ringler tanh function 1 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.5**4, x0=0.86602540378, x1=0.0, x2=0.5)
    relaxation("sphere", "quad", 4, 2.0, mexpr, prefix="ringler-1-quad", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=2)

if False:
    ### Sphere - Ringler tanh function 4 ###
    mexpr = Expression("sqrt(((1-gamma)/2)*(tanh((beta - acos(x0*x[0] + x1*x[1] + x2*x[2]))/alpha) + 1.0) + gamma)", alpha=math.pi/20, beta=math.pi/6, gamma=0.0625**4, x0=0.86602540378, x1=0.0, x2=0.5)
    relaxation("sphere", "quad", 4, 2.0, mexpr, prefix="ringler-4-quad", logging=True, output=True, knownmesh=True, knownmeshfile="exact.npy", sph_deg=2)

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 60, 0.1, mexpr)
    t1 = time.time()
    print("60: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 90, 0.1, mexpr)
    t1 = time.time()
    print("90: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 120, 0.1, mexpr)
    t1 = time.time()
    print("120: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 180, 0.1, mexpr)
    t1 = time.time()
    print("180: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - a*a)), -2)", a=0.25, alpha1=10, alpha2=200)
    relaxation("plane", "quad", 240, 0.1, mexpr)
    t1 = time.time()
    print("240: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    relaxation("sphere", "tri", 3, 0.12, mexpr)
    t1 = time.time()
    print("3: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    relaxation("sphere", "tri", 4, 0.12, mexpr)
    t1 = time.time()
    print("4: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    relaxation("sphere", "tri", 5, 0.12, mexpr)
    t1 = time.time()
    print("5: " + str(t1 - t0))

if False:
    t0 = time.time()
    mexpr = Expression("1.0 + alpha1*pow(cosh(alpha2*(pow(acos(x0*x[0] + x1*x[1] + x2*x[2]), 2) - a*a)), -2) + alpha3*pow(cosh(alpha4*(pow(acos(y0*x[0] + y1*x[1] + y2*x[2]), 2) - a*a)), -2)", a=math.pi/2, alpha1=10, alpha2=5, alpha3=10, alpha4=5, x0=0.86602540378, x1=0.0, x2=0.5, y0=-0.86602540378, y1=0.0, y2=0.5)
    relaxation("sphere", "tri", 6, 0.12, mexpr)
    t1 = time.time()
    print("6: " + str(t1 - t0))
