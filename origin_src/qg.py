import numpy as np
from firedrake import *

n0 = 100
mesh = PeriodicUnitSquareMesh(n0, n0)

Vdg = FunctionSpace(mesh, "DG", 1)       # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh, "CG", 1)       # CG elements for Streamfunction

# generate random q
np.random.seed(0)
temp = Function(Vcg)
temp.dat.data[:] = np.random.uniform(-1.0, 1.0, size=len(temp.dat.data))
q0 = Function(Vdg)
q0.interpolate(temp)
del temp

qplusdq = Function(Vdg)  # PV fields for different time steps
q1 = Function(Vdg)

psi0 = Function(Vcg)     # Streamfunctions for different time steps
psi1 = Function(Vcg)

F = Constant(1.0)        # Rotational Froude number
Dt = 0.1                 # Time step
dt = Constant(Dt)

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)

Apsi = (dot(grad(psi), grad(phi)) + F*psi*phi)*dx
Lpsi = -q1*phi*dx

psi_problem = LinearVariationalProblem(Apsi, Lpsi, psi0)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={'ksp_type': 'cg',
                                                        'pc_type': 'gamg'})


def gradperp(u):
    return as_vector((-u.dx(1), u.dx(0)))


n = FacetNormal(mesh)
u = gradperp(psi0)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

q = TrialFunction(Vdg)
p = TestFunction(Vdg)
a_mass = p*q*dx
a_int = -dot(grad(p), u*q)*dx
a_flux = jump(p)*(un('+')*q('+') - un('-')*q('-'))*dS
arhs = a_mass - dt*(a_int + a_flux)

q_problem = LinearVariationalProblem(a_mass, action(arhs, q1), qplusdq)

q_solver = LinearVariationalSolver(q_problem,
                                   solver_parameters={'ksp_type': 'preonly',
                                                      'pc_type': 'bjacobi',
                                                      'sub_pc_type': 'ilu'})

t = 0.0
T = 500.0

outfile = File("qg/pvd/qg.pvd")
outfile.write(q0)
dumpfreq = 50

dumpn = 0
while t < T - 0.5*Dt:
    # Compute the streamfunction for the known value of q0
    q1.assign(q0)
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(1)
    q1.assign(qplusdq)
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(2)
    q1.assign(0.75*q0 + 0.25*qplusdq)
    psi_solver.solve()
    q_solver.solve()

    # Find new solution q^(n+1)
    q0.assign((1.0/3.0)*q0 + (2.0/3.0)*qplusdq)

    t += Dt
    dumpn += 1
    if dumpn == dumpfreq:
        dumpn = 0
        outfile.write(q0)

    print("t = " + str(t))

np.save("qg"+str(n0)+".npy", q0.dat.data)
