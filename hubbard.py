# Copyright 2016 - 2019 Deepak Vaid

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import sympy as sp

# # Hubbard Model
#
# ## Hamiltonian
#
# $$ H_{h} = -t \sum_{<i,j>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow}n_{j\downarrow} - \mu \sum_i (n_{i\uparrow} + n_{i\downarrow}) $$
#
# where the first term is the kinetic energy. The sum is over both possible spin at each lattice site $ \sigma \in \{\uparrow,\downarrow\}$. Second term is the potential energy due to repulsion of electrons at a site containing two electrons. The last term is the chemical potential associated with adding particles to the system.
#
# The partition function for a system in a thermal state is given by:
#
# $$ Z = \text{Tr} \left[ e^{-\beta H} \right] = \sum_\alpha \langle \alpha \vert e^{-\beta H} \vert \alpha \rangle  $$
#
# ## Partition Function
#
# For $ H_{h} $ on a single site, the partition function becomes:
#
# $$ Z_{h} = 1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)} $$
#
# If we redefine the chemical potential $ \mu \rightarrow \mu + U/2$, then the Hubbard hamiltonian becomes:
#
# $$ H = -t \sum_{<i,j>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i \left(n_{i\uparrow} - \frac{1}{2}\right) \left(n_{j\downarrow} - \frac{1}{2}\right) - \mu \sum_i (n_{i\uparrow} + n_{i\downarrow}) $$

# ## Energy
#
# The energy for the single site Hubbard model is:
#
# \begin{align}
#     E & = \langle H + \mu n \rangle = \text{Tr} \left[ (H + \mu n) e^{-\beta H} \right] \\
#       & = \frac{1}{Z} \sum_\alpha \langle \alpha \vert (H + \mu n) e^{-\beta H} \vert \alpha \rangle \\
#       & = \frac{U e^{2\beta(t+\mu - U/2)}}{1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)}}
# \end{align}
#
# ## Occupation Number
#
# and the occupation number is:
#
# \begin{align}
#     \rho & = \langle n \rangle = \text{Tr} \left[ n e^{-\beta H} \right] \\
#          & = \frac{2 e^{\beta(\mu + t)} + 2 e^{2\beta(t+\mu-U/2)}}{1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)}}
# \end{align}

# # Operators and States in QuTiP

# Rather than using the analytical expression for the partition function, energy and other properties of the one-site Hubbard model, we will now use the Python package [QuTiP](http://qutip.org/) to define the Hubbard model Hamiltonian on $N$ sites. For this we will need to define creation, annihilation and number operators.
#
# For a single site system a creation operator is simply defined using the `create()` function:
#
# $$\text{create(2)} \Rightarrow c^\dagger$$
#
# which returns the matrix:
#
# \begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.0\\1.0 & 0.0\\\end{array}\right)\end{equation*}

# In[8]:


# create(2)


# ## Operators for many-body systems in 1D
#
# When we have more than one site, the operator must be defined as an operator acting of the *full* Hilbert space of the system:
# $$ \mc{H} = \otimes_{i=1}^N \mc{H}_i $$
# where $\mc{H}_i$ is the Hilbert space corresponding to a single site. Thus the creation operator for the $j^\text{th}$ site is given by:
# $$ \mb{1}_1 \otimes \ldots \mb{1}_{j-1} \otimes c^\dagger_j \otimes \mb{1}_{j+1} \ldots \mb{1}_N $$
# where $\mb{1}_i$ is the identity operator acting on the $i^\text{th}$ site.
#
# The code for this is as follows:

# ## Utility Functions

# In[3]:

def anticommutator(X,Y):
    return X*Y + Y*X

def qtAntiCommutator(X,Y):
    if type(X) == Qobj and type(Y) == Qobj:
        if len(X.shape) == 2 and len(Y.shape) == 2:
            if X.shape[1] == Y.shape[0]:
                return qutip.commutator(X,Y,kind='anti')
            else:
                raise ValueError("Number of columns of first argument must be equal to number of rows of second argument")
        else:
            raise ValueError("Arguements must be QuTiP arrays of shape M x N")
    else:
        raise ValueError("Arguments must be QuTiP arrays of compatible dimensions")
        
def npAntiCommutator(X,Y):
    if type(X) == np.ndarray and type(Y) == np.ndarray:
        if len(X.shape) == 2 and len(Y.shape) == 2:
            if X.shape[1] == Y.shape[0]:
                return X*Y + Y*X
            else:
                raise ValueError("Number of columns of first argument must be equal to number of rows of second argument")
        else:
            raise ValueError("Arguements must be numpy arrays of shape M x N")
    else:
        raise ValueError("Arguments must be numpy arrays of compatible dimensions")

def zeroOpN(N = 1, dims = 2):
    '''Returns zero operator for an N site chain with a Hilbert space of dimensions dims at each site.
    '''
    zeroket = qutip.zero_ket(2**N,dims=[[2]*N,[2]*N])
    return zeroket * zeroket.dag()

def identityList(N = 1,dims = 2):
    '''Returns a list of N identity operators for a N site spin-system with a
    Hilber space at each state of dimensionality dims
    '''

    iden = identity(dims)

    iden_list = []

    [iden_list.append(iden) for i in range(N)]

    return iden_list


# ## Position Representation Operators

# In[4]:


def posOperatorN(oper, i = 0, N = 1):
    '''Returns the operator given by oper, in the position representation, for the i^th site of an N site spin-chain
    with a Hilbert space, of dimension compatible with the dimension of the operator, at each site'''

    if not isinstance(oper, Qobj):
        raise TypeError('oper must of type qutip.Qobj')

    if not oper.isoper:
        raise ValueError('oper must be a qutip operator')

    shape = oper.shape

    if shape[0] == shape[1]:
        dims = shape[0]
    else:
        raise ValueError('oper must be a square matrix')

    if oper == identity(oper.shape[0]):
        return tensor(identityList(N,oper.shape[0]))
    else:
        iden_list = identityList(N, oper.shape[0])
        iden_list[i] = oper
        return tensor(iden_list)


# In[5]:


def posCreationOpN(i=0, N=10):
    '''Returns the creation operator in the position representation for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''

    return posOperatorN(create(2),i,N)


# In[6]:


def posDestructionOpN(i=0, N=10):
    '''Returns the destruction operator in the position representation for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''

    return posOperatorN(destroy(2),i,N)


# In[7]:


def posOperHamiltonian(oper, N = 10, coefs = [1]*10, dims=2):
    ''' Returns the Hamiltonian, in position representation, given by the sum of oper acting on each site
    with a weight given by the values in coefs
    '''

    if not isinstance(oper, Qobj):
        raise ValueError('oper must be of type Qobj')
    else:
        if not oper.isoper:
            raise ValueError('oper must be an operator')

    H = 0

    for i in range(N):
        op_list = identityList(N, dims)
        op_list[i] = oper
        H += coefs[i]*tensor(op_list)

    return H


# In[8]:


# To get the number operator at the 3rd site, in a N=5 site system, we do the following
# numOp = create(2)*destroy(2)
# posOperatorN(sigmaz(), N=5, i=4)


# ## Operators for Many Body Systems in 2D

# ## Utility Functions

# In[9]:


def idOpList2D(latt_size=(1,1), dims = 2):
    '''Returns a 2d list of identity operators for a 2D spin-system of size given by latt_size with a
    Hilbert space at each state of dimensionality dims.
    '''

    iden = identity(dims)

#    iden_list = [[0 for x in range(latt_size[0])] for y in range(latt_size[1])] # initialize empty list

    iden_list = np.ndarray(tuple(latt_size),dtype=Qobj)

#    [iden_list[i].append(iden) for i in range(latt_size[0]) for j in range(latt_size[1])] # assign identity op to each element of list

    for i in range(latt_size[0]):
        for j in range(latt_size[1]):
            iden_list[i][j] = iden

    return iden_list


# In[10]:


# idOpList2D((2,2))


# In[11]:


def posOp2d(oper, pos = (0,0), latt_size = (1,1) ):
    '''Returns QuTiP operator in position representation defined on site (i,j) passed as argument pos, of 2d lattice
    of size specified in tuple latt_size.
    '''

    if not isinstance(oper, Qobj):
        raise TypeError('oper must of type qutip.Qobj')

    if not oper.isoper:
        raise ValueError('oper must be a qutip operator')

    shape = oper.shape

    if shape[0] == shape[1]:
        dims = shape[0]
    else:
        raise ValueError('oper must be a square matrix')

    if (not isinstance(pos, tuple)) or (not isinstance(latt_size, tuple)):
        raise TypeError('pos and latt_size must be tuples')

    if len(pos)!=len(latt_size):
        raise ValueError('pos and latt_size must be of same size')

    if pos[0]>=latt_size[0] or pos[1]>=latt_size[1]:
        raise IndexError('Lattice position out of bounds')

    idOp = idOpList2D(latt_size,dims)

    idOp[pos[0]][pos[1]] = oper

    op_list = []

    for i in range(latt_size[0]):
        for j in range(latt_size[1]):
            op_list.append(idOp[i][j])

#    if oper == identity(dims):
#        return tensor(op_list)
#    else:
#        op_list[pos[0]*latt_size[1]+pos[1]] = oper
    return tensor(op_list)


def posToMomentumOpN(oper, k = 0, N = 1):
    '''Returns the momentum space represenation of the operator given by oper for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''

    momOp = tensor(identityList(N,oper.shape[0]))
    invrtn = 1/np.sqrt(N)

#    type(invrtn)

    for i in range(N):
        momOp += invrtn * np.exp(1j*i*k/(2*np.pi*N)) * posOperatorN(oper,i,N)

    return momOp


# In[17]:


def momCreationOpN(k = 0, N = 1):
    '''Returns the momentum space represenation of the creation operator for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''

    return posToMomentumOpN(create(2),k,N)


def momDestructionOpN(k = 0, N = 1):
    '''Returns the momentum space represenation of the operator given by oper for the k^th momentum
    of an N site spin-chain with a Hilbert space of dimensionality dims at each site'''

    return qutip.dag(momCreationOpN(k,N))


def matrixPosToMom(N = 10):
    ''' Returns a QuTiP object corresponding to a matrix whose elements are:

    U_{k,l} = e^{-i*k*l}

    '''

    matrix = np.zeros([N,N],dtype=complex)

    for k in range(N):
        for l in range(N):
            matrix[k][l] = np.exp(1j*k*l)

    return Qobj(matrix)


def hamiltonianHubbard(N = 10, t = 1, U = 1, mu = 0, periodic = True, shift = False, dims = 2):
    '''Returns operator corresponding to Hubbard Hamiltonian on N sites.
    Default value of N is 10. t, U and mu are the kinetic energy, potential energy and
    chemical potential respectively.
    If shift is False then first version of Hamiltonian is returned, else the second
    version (where the chemical potential is shifted $\mu \rightarrow \mu - U/2$) is
    returned.
    dims is the dimension of the Hilbert space for each electron. Default is 2'''

    # two sets of creation/destruction operators, labeled A and B.

    destroyA_list = []
    createA_list = []

    destroyB_list = []
    createB_list = []

    cOp = create(dims)
    dOp = destroy(dims)
    nOp = cOp * dOp

    nA_list = []
    nB_list = []

    idOp = identity(dims)

    idOp_list = []

    [idOp_list.append(idOp) for i in range(N)]

    superid = tensor(idOp_list) # identity operator for whole system

    H = 0

    for i in range(N):
        # Create list containing creation/destruction/number operators for each site

        createA_list.append(posOperatorN(cOp,N=N,i=i))
        createB_list.append(posOperatorN(cOp,N=N,i=i))

        destroyA_list.append(posOperatorN(dOp,N=N,i=i))
        destroyB_list.append(posOperatorN(dOp,N=N,i=i))

        nA_list.append(posOperatorN(nOp,N=N,i=i))
        nB_list.append(posOperatorN(nOp,N=N,i=i))

    if periodic == True:
        for i in range(N):
            H += - t * (createA_list[i%N] * destroyA_list[(i+1)%N] + createB_list[i%N] * destroyB_list[(i+1)%N])
    else:
        for i in range(N-1):
            H += - t * (createA_list[i] * destroyA_list[i+1] + createB_list[i] * destroyB_list[i+1])

    for i in range(N):
        H += - mu * (nA_list[i] + nB_list[i])
        if shift == True:
            H += U * (nA_list[i] - 0.5 * superid) * (nB_list[i] - 0.5 * superid)
        else:
            H += U * nA_list[i] * nB_list[i]

    return H


def hamiltonianIsing(N = 10, J = 1, B = 0.0, periodic = True, spin=0.5):
    '''Returns operator corresponding to Ising Hamiltonian for give spin on N sites.
    Default value of N is 10. j is the coupling strength. Default is -1 for
    ferromagnetic interaction.
    Default value of spin is 0.5
    '''

    op_list = []
    op_list2 = []

    jz = 2*jmat(spin,'z')
    jx = 2*jmat(spin,'x')

    dimj = 2*spin + 1

    H = 0

    idlist = identityList(N, dimj)

    for i in range(N):
        # Create list containing spin-z operators for each site:

        op_list.append(posOperatorN(jz,i=i,N=N))
        op_list2.append(posOperatorN(jx,i=i,N=N))

    if periodic == True:
        for i in range(N):
            H += -J*op_list[i%N]*op_list[(i+1)%N] + B*op_list2[i%N]
    else:
        for i in range(N-1):
            H += -J*op_list[i]*op_list[i+1] + B*op_list2[i]
        H += B*op_list2[N-1]

    return H


def hamiltonianIsingTransverse(N = 10, J = 1.0, B = 1.0, periodic = True, spin=0.5):
    '''Returns operator corresponding to Ising Hamiltonian for give spin on N sites.
    Default value of N is 10. jcoef is the coupling strength. Default is -1 for
    ferromagnetic interaction.
    Default value of spin is 0.5
    '''

    op_list = []
    op_list2 = []

    jz = 2*jmat(spin,'z')
    jx = 2*jmat(spin,'x')

    dimj = 2*spin + 1

    H = 0

    idlist = identityList(N, dimj)

    for i in range(N):
        # Create list containing spin-z operators for each site:

        op_list.append(posOperatorN(jz,i=i,N=N))
        op_list2.append(posOperatorN(jx,i=i,N=N))

    if periodic == True:
        for i in range(N):
            H += -J*op_list[i%N]*op_list[(i+1)%N] - B*op_list2[i%N]
    else:
        for i in range(N-1):
            H += -J*op_list[i]*op_list[i+1] - B*op_list2[i]
        H += B*op_list2[N-1]

    return H


def hamiltonianHeisenberg(N = 5, J = 1, periodic = True):
    '''Returns operator corresponding to the Heisenberg 1D spin-chain on N sites.
    $$ H = - J \sum_{i=1}^N S_n \cdot S_{n+1} $$
    where $ S_n (S_x, S_y, S_z) $ is the spin-operator acting on the n^{th} site.

    H can be written in terms of spin-flip operators $S^+, S^-$ as:

    $$ H = -J \sum_{i=1}^N \left[ \frac{1}{2} (S_n^+ S_{n+1}^- + S_n^- S_{n+1}^+) + S_n^z S_{n+1}^z \right] $$

    '''

    spinp_list = []
    spinm_list = []
    spinz_list = []

    opSpinP = sigmap()
    opSpinM = sigmam()
    opSpinZ = sigmaz()

    idOp = identity(2)

    idOp_list = []

    [idOp_list.append(idOp) for i in range(N)]

    superid = tensor(idOp_list) # identity operator for whole system

    H = 0

    for i in range(N):
        # Create list containing creation/destruction/number operators for each site

        spinp_list.append(posOperatorN(opSpinP,N=N,i=i))
        spinm_list.append(posOperatorN(opSpinM,N=N,i=i))
        spinz_list.append(posOperatorN(opSpinZ,N=N,i=i))

    if periodic == True:
        for i in range(N):
            H += - J * ( 0.5*(spinp_list[i%N] * spinp_list[(i+1)%N] + spinm_list[i%N] * spinm_list[(i+1)%N])                     + spinz_list[i%N] * spinz_list[(i+1)%N] )
    else:
        for i in range(N-1):
            H += - J * ( 0.5*(spinp_list[i] * spinp_list[i+1] + spinm_list[i] * spinm_list[i+1])                     + spinz_list[i] * spinz_list[i+1] )

    return H


class Hamiltonian(Qobj):

    _hamTypes = ['NumberOp', 'Ising', 'Heisenberg','Hubbard']
    _hamType = ''
    _maxSites = 100
    _numSites = 1
    _dims = 2
    _label = None
    _data = None

    _hamiltonian = Qobj()
    _eigenenergies = []
    _eigenstates = []
    _isHermitian = True

    def __init__(self, label=None, dims=2, isHermitian=True,                 numSites=1, hamType=None,data=None):
#        try:
#            from qutip import *
#        except:
#            raise NameError('QuTiP is not installed')

        if numSites<1 or not isinstance(numSites,int):
            raise ValueError('numSites must be an integer greater than or equal to 1')
        if numSites>self._maxSites:
            raise ValueError('numSites cannot be greater than ' + str(self._maxSites))
        else:
            self._numSites = numSites

        if label!=None and isinstance(label, str):
            self._label = label

        if data!=None:
            self._data = data

        self._isHermitian = isHermitian

        if hamType != None:
            if hamType not in self._hamTypes:
                from string import join
                raise ValueError('hamType must be one of ' + join(self._hamTypes,', '))
            else:
                self._hamType = hamType
                self.createHamiltonian()
        else:
            self._hamiltonian = Qobj()

        if dims < 2 or not isinstance(dims, int):
            raise ValueError('dim must be an integer greater than or equal to 2')
        else:
            self._dims = dims

        Qobj.__init__(self._hamiltonian)

        return

    def createHamiltonian(self):

        if self._hamType == 'Ising':

            self._hamiltonian = hamiltonianIsing(self._numSites,self._data['jcoefs'],self._data['spin'])

        elif self._hamType == 'Hubbard':

            self._hamiltonian = hamiltonianHubbard(self._numSites, self._data['t'],                                self._data['U'], self._data['mu'],                                 self._data['shift'], self._dims)

        elif self._hamType == 'NumberOp':

            numOp = create(self._dims)*destroy(self._dims)

            self._hamiltonian = posOperHamiltonian(numOp, self._numSites,                                     self._data['coefs'], self._dims)

        elif self._hamType == 'Heisenberg':

            self._hamiltonian = hamiltonianHeisenberg()

        return

    @property
    def hermitian(self):
        return self._isHermitian

    @hermitian.setter
    def hermitian(self, value):
        if isinstance(value, bool):
            self._isHermitian = value
        else:
            raise ValueError('hermitian must be a boolean data type')


# # Vacuum State, Basis States

# In[ ]:


def stateVacuum(N = 10):
    ''' Returns a QuTiP object representing the vacuum state for a spin-chain with N sites '''

    state = []

    for i in range(N):
        state.append(basis(2,0))

    return tensor(state)


# In[ ]:


# stateVacuum(N=3)


# In[ ]:


def stateUpK(k = 0, N = 10):
    ''' Returns a QuTiP object representating a state with the k^th spin pointing up and the rest pointing
    down for a N site spin-chain. '''

    state = []

    for i in range(N):
        state.append(basis(2,0))

    state[k] = basis(2,1)

    return tensor(state)


# In[ ]:


# stateUpK(2,3)


# In[ ]:


def stateDownK(k = 0, N = 10):
    ''' Returns a QuTiP object representating a state with the k^th spin pointing down and the rest pointing
    up for a N site spin-chain. '''

    state = []

    for i in range(N):
        state.append(basis(2,1))

    state[k] = basis(2,0)

    return tensor(state)


# In[ ]:


# stateDownK(2,3)


# In[ ]:


def stateListDown(N = 10):
    ''' Returns a list of QuTiP objects, whose elements are the mutually orthogonal states, with mostly
    down spins, of a spin-chain with N sites. '''

    stateList = []

    for i in range(N):
        stateList.append(stateDownK(i,N))

    return stateList

def stateListUp(N = 10):
    ''' Returns a list of QuTiP objects, whose elements are the mutually orthogonal states, with mostly
    up spins, of a spin-chain with N sites. '''

    stateList = []

    for i in range(N):
        stateList.append(stateUpK(i,N))

    return stateList


# In[ ]:


stateListDown(N=3)


# # Jordan-Wigner Transformation
#
# Given a n-qubit system, with Pauli operator $ X_i, Y_i, Z_i $ acting on each qubit, we can define a set of ***fermionic*** operators $ \{a_j\} $
#
# $$ a_j = -\left(\otimes_{i=1}^{j-1} Z_i \right) \otimes \sigma_j $$
#
# The function `jordanWignerDestroyI(i,N)` returns the fermionic destruction operator $a_i$ for the $i^\text{th}$ site of a N-site qubit chain.
#

# In[ ]:


def jordanWignerDestroyI(i = 0, N = 1):
    ''' Returns the fermionic annihilation operator for the i^th site of a spin 1/2 N site spin-chain:
    a_i = Z_1 x Z_2 ... x Z_{i-1} x sigma_i
    where Z_i is the Pauli sigma_z matrix acting on the i^th site and sigma_i is the density
    matrix acting on the i^th qubit, given by:
    sigma_i = id_1 x id_2 ... x id_{i-1} x |0><1| x id_{i+1} ... x id_n
    where id_i is the identity operator. Note that this works only for a spin 1/2 chain. For higher spins,
    one will have to generalize this transformation.

    Reference: Nielsen, Fermionic CCR and Jordan Wigner Transformation
    '''

#     dims = 2*spin + 1
    
#     if !isinstance(dims,int) and dims > 0:
#         raise ValueError("The spin argument must be positive half integer valued")
    
    # create zop, assign to it identity operator for N site system
    zop = tensor([qeye(dims)]*N)

    # for k in (0..i-1), create Z_k operator for N-site chain
    # zop is product of Z_1 * Z_2 ... * Z_{i-1}

    for k in range(i):
#         zop *= posOperatorN(sigmaz(),i = k, N = N)
        zop *= posOperatorN(spin_Jz(spin), i = k, N = N)

    # create single qubit density matrix |0><1|

    sigmai = ket([0])*bra([1])

    # create list of N single-site identity operators

    op_list = identityList(N = N)

    # assign single qubit density matrix |0><1| to i^th item of above list

    op_list[i] = sigmai

    # take the tensor product of resulting list to obtain operator for density matrix |0><1|
    # acting on i^th qubit of N-site chain.

    sigmaop = tensor(op_list)

    # return zop*sigmaop = Z_1 x Z_2 .. x Z_{i-1} x sigma_i, which is fermionic annihilation
    # operator for i^th site of N-site spin-chain.

    return zop * sigmaop


# In[ ]:


def offDiagBelow(N,a=1):
    """
    Returns a NxN array with the elements below the diagonal set to a
    """
    return a*(np.tri(N,N,-1) - np.tri(N,N,-2))

def offDiagAbove(N,a=1):
    """
    Returns a NxN array with the elements above the diagonal set to a
    """
    return a*(np.tri(N,N,1) - np.tri(N,N))


# # Schmidt Decomposition (Singular Value Decomposition)
#
# We need to be able to perform the following tests:
#
# 1. Determine whether a given state is entangled or not
# 2. Determine whether a given density matrix represents a pure or a mixed state
#
# In order to test of entanglement, we can use the Schmidt decomposition or the Singular Value Decomposition (SVD). Numpy implements the SVD in the module numpy.linalg.svd, which given a matrix $A$ as argument, returns the SVD of $A$, in the form of two unitary matrices $U$, $V$ and a diagonal matrix $S$:
#
# $$ A = U \cdot S \cdot V $$
#
# Since $S$ is diagonal ($S_{ij} \equiv s_i \delta_{ij}$), the elements of $A$ can be written as:
#
# \begin{align}
# A_{ij} & = U_{ik} \cdot S_{km} \cdot V_{mj} \\
#        & = U_{ik} \cdot s_k \delta_{km} \cdot V_{mj} \\
#        & = s_k U_{ik} V_{kj}
# \end{align}
#
# $$  $$

# Construct entangled state:
# $$ \fullket{\psi} = \frac{1}{\sqrt{2}} ( \fullket{00} + \fullket{11} ) $$

# The state $\fullket{\psi}$ can be written in the form:
# $$ \psi = \alpha_{ij} u_i \otimes v_j $$
# where $u_i, v_i \in \{\fullket{0},\fullket{1}\}$ are basis vectors for the two Hilbert spaces $H_1$ and $H_2$. For the given state, the matrix $\alpha_{ij}$ has the form:
#
# $$
# \alpha = \begin{pmatrix} 0.707 & 0 \\
#                          0     & 0.707 \end{pmatrix}
# $$
#
# If $\fullket{\psi}$ is **not** entangled then in the SVD of the matrix $\alpha_{ij}$:
# $$ \alpha = U \, S \, V $$
# the diagonal matrix $S$ will have only one non-zero element.
