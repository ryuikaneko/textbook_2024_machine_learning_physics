import numpy as np
from numba import jit
import scipy.sparse.linalg

@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

@jit(nopython=True)
def flip_spin(state,site):
    return state^(1<<site)

def make_ham(L,g,Nstate,vecnew):
    @jit(nopython=True)
    def get_vec(vec,vecnew):
        vecnew[:] = 0.0
        for a in range(Nstate):
            for i in range(L):
                if get_spin(a,i) == get_spin(a,(i+1)%L):
                    vecnew[a] -= vec[a]
                else:
                    vecnew[a] += vec[a]
            for i in range(L):
                b = flip_spin(a,i)
                vecnew[a] -= g*vec[b]
        return vecnew
    return get_vec

def main():
    g = 1.0
    Ls = np.arange(2,21)
    Enes = []
    for L in Ls:
        Nstate = 2**L
        vecnew = np.zeros(Nstate,dtype=np.float64)
        get_vec = make_ham(L,g,Nstate,vecnew)
        Ham = scipy.sparse.linalg.LinearOperator((Nstate,Nstate),matvec=lambda vec: get_vec(vec,vecnew))
        Neig = 1
        Ene = scipy.sparse.linalg.eigsh(Ham,which="SA",k=min(Neig,Nstate-1),return_eigenvectors=False)
        Enes.append(Ene[::-1][0])
    Enes = np.array(Enes)
    Dats = np.vstack((Ls,Enes)).T
    np.savetxt("dat_ed_1d_tfising_linear_operator",Dats)
    print(Dats)

main()
