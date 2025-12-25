from firedrake import *
import numpy as np
from scipy import integrate, optimize

tol = 1e-6


def meshgen(mesh, gamma, prefix="temp"):
    xc = np.array([0.86602540378, 0.0, 0.5])
    alpha = np.pi/20
    beta = np.pi/6

    func = lambda foo: np.sqrt(((1-gamma)/2)*(np.tanh((beta - foo)/alpha) + 1.0) + gamma)

    theta, err = integrate.quad(func, 0, np.pi, weight='sin', wvar=1.0, epsrel=1e-2*tol)
    theta /= 2.0

    new_coords = Function(mesh.coordinates)
    len_coords = len(new_coords.dat.data)

    for ii in range(len_coords):
        if ii % 100 == 0:
            print "vertex " + str(ii) + " of " + str(len_coords)
        xi = mesh.coordinates.dat.data[ii]
        t = np.arccos(np.dot(xc, xi))
        intfunc = lambda foo: integrate.quad(func, 0, foo, weight='sin', wvar=1.0, epsrel=0.1*tol)[0] - theta*(1 - np.cos(t))
        s = optimize.bisect(intfunc, 0, np.pi, xtol=tol)
        v = xi - xc
        w = v - np.dot(v, xc)*xc  # project out component in dir of xc
        norm = np.linalg.norm(w)
        if norm > 1e-8:
            w = w/norm
            x = xc*np.cos(s) + w*np.sin(s)
            new_coords.dat.data[ii, :] = x[:]
        else:
            # Point unexpectedly close to axis. If axis coincides with
            # mesh vertex then this is expected; uncomment below line.
            assert False
            # new_coords.dat.data[ii, :] = xi[:]

    np.save(prefix + "/exact.npy", new_coords.dat.data)


mesh = UnitCubedSphereMesh(4)
meshgen(mesh, 0.5**4, "ringler-1-lin")

mesh = UnitCubedSphereMesh(4, degree=2)
meshgen(mesh, 0.5**4, "ringler-1-quad")

mesh = UnitCubedSphereMesh(4)
meshgen(mesh, 0.0625**4, "ringler-4-lin")

mesh = UnitCubedSphereMesh(4, degree=2)
meshgen(mesh, 0.0625**4, "ringler-4-quad")
