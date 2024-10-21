import numpy as np
from numba import jit
import scipy.sparse

@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

@jit(nopython=True)
def flip_spin(state,site):
    return state^(1<<site)

@jit(nopython=True)
def make_ham_child(L,g,Nstate):
    listki = np.array([i for k in range(L+1) for i in range(Nstate)],dtype=np.int64)
    loc = np.zeros((L+1)*Nstate,dtype=np.int64)
    elemnt = np.zeros((L+1)*Nstate,dtype=np.float64)
    for a in range(Nstate):
        for i in range(L):
            loc[L*Nstate+a] = a
            if get_spin(a,i) == get_spin(a,(i+1)%L):
                elemnt[L*Nstate+a] -= 1.0
            else:
                elemnt[L*Nstate+a] += 1.0
        for i in range(L):
            b = flip_spin(a,i)
            loc[i*Nstate+a] = b
            elemnt[i*Nstate+a] -= g
    return elemnt, listki, loc

def make_ham(Nstate,elemnt,listki,loc):
    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nstate,Nstate),dtype=np.float64)

def main():
    g = 1.0
    Ls = np.arange(2,21)
    Enes = []
    for L in Ls:
        Nstate = 2**L
        elemnt, listki, loc = make_ham_child(L,g,Nstate)
        Ham = make_ham(Nstate,elemnt,listki,loc)
        Neig = 1
        Ene = scipy.sparse.linalg.eigsh(Ham,which="SA",k=min(Neig,Nstate-1),return_eigenvectors=False)
        Enes.append(Ene[::-1][0])
    Enes = np.array(Enes)
    Dats = np.vstack((Ls,Enes)).T
    np.savetxt("dat_ed_1d_tfising_sparse_matrix",Dats)
    print(Dats)

main()
