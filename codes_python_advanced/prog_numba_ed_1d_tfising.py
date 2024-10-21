import numpy as np
from numba import jit

@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

@jit(nopython=True)
def flip_spin(state,site):
    return state^(1<<site)

@jit(nopython=True)
def make_ham(L,g):
    Nstate = 2**L
    Ham = np.zeros((Nstate,Nstate),dtype=np.float64)
    for a in range(Nstate):
        for i in range(L):
            if get_spin(a,i) == get_spin(a,(i+1)%L):
                Ham[a,a] -= 1.0
            else:
                Ham[a,a] += 1.0
        for i in range(L):
            b = flip_spin(a,i)
            Ham[a,b] -= g
    return Ham

@jit(nopython=True)
def main():
    L = 10; g = 1.0
    Ham = make_ham(L,g)
    print("Hamiltonian:\n",Ham)
    Ene, Vec = np.linalg.eigh(Ham)
    print("Ground state energy:\n",Ene[0])
    print("Ground state vector:\n",Vec[:,0])

main()