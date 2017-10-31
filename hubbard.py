
# coding: utf-8

# ##### Mathjax custom macros #####
# 
# $ \newcommand{\opexpect}[3]{\langle #1 \vert #2 \vert #3 \rangle} $
# 
# $ \newcommand{\rarrow}{\rightarrow} $
# $ \newcommand{\bra}{\langle} $
# $ \newcommand{\ket}{\rangle} $
# 
# $ \newcommand{\mb}[1]{\mathbf{#1}} $
# $ \newcommand{\mc}[1]{\mathcal{#1}} $
# $ \newcommand{\mbb}[1]{\mathbb{#1}} $
# $ \newcommand{\mf}[1]{\mathfrak{#1}} $
# 
# $ \newcommand{\vect}[1]{\boldsymbol{\mathrm{#1}}} $
# $ \newcommand{\expect}[1]{\langle #1\rangle} $
# 
# $ \newcommand{\innerp}[2]{\langle #1 \vert #2 \rangle} $
# $ \newcommand{\fullbra}[1]{\langle #1 \vert} $
# $ \newcommand{\fullket}[1]{\vert #1 \rangle} $
# $ \newcommand{\supersc}[1]{$^{\text{#1}}$} $
# $ \newcommand{\subsc}[1]{$_{\text{#1}}$} $
# $ \newcommand{\sltwoc}{SL(2,\mathbb{C})} $
# $ \newcommand{\sltwoz}{SL(2,\mathbb{Z})} $
# 
# $ \newcommand{\utilde}[1]{\underset{\sim}{#1}} $

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from sympy.matrices.densetools import isHermitian
get_ipython().magic(u'matplotlib inline')


# #### Hubbard Model
# 
# ##### Hamiltonian:
# 
# $$ H_{h} = -t \sum_{<i,j>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow}n_{j\downarrow} - \mu \sum_i (n_{i\uparrow} + n_{i\downarrow}) $$
# 
# where the first term is the kinetic energy. The sum is over both possible spin at each lattice site $ \sigma \in \{\uparrow,\downarrow\}$. Second term is the potential energy due to repulsion of electrons at a site containing two electrons. The last term is the chemical potential associated with adding particles to the system.
# 
# The partition function for a system in a thermal state is given by:
# 
# $$ Z = \text{Tr} \left[ e^{-\beta H} \right] = \sum_\alpha \langle \alpha \vert e^{-\beta H} \vert \alpha \rangle  $$
# 
# For $ H_{h} $ on a single site, the partition function becomes:
# 
# $$ Z_{h} = 1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)} $$
# 
# If we redefine the chemical potential $ \mu \rightarrow \mu + U/2$, then the Hubbard hamiltonian becomes:
# 
# $$ H = -t \sum_{<i,j>\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i \left(n_{i\uparrow} - \frac{1}{2}\right) \left(n_{j\downarrow} - \frac{1}{2}\right) - \mu \sum_i (n_{i\uparrow} + n_{i\downarrow}) $$

# ### Operators and States in QuTiP

# Rather than using the analytical expression for the partition function, energy and other properties of the one-site Hubbard model, we will now use the Python package [QuTiP](http://qutip.org/) to define the Hubbard model Hamiltonian on $N$ sites. For this we will need to define creation, annihilation and number operators.
# 
# For a single site system a creation operator is simply defined using the `create()` function:
# 
# $$\text{create(2)} \Rightarrow c^\dagger$$
# 
# which returns the matrix:
# 
# \begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.0\\1.0 & 0.0\\\end{array}\right)\end{equation*}

# When we have more than one site, the operator must be defined as an operator acting of the *full* Hilbert space of the system:
# $$ \mc{H} = \otimes_{i=1}^N \mc{H}_i $$
# where $\mc{H}_i$ is the Hilbert space corresponding to a single site. Thus the creation operator for the $j^\text{th}$ site is given by:
# $$ \mb{1}_1 \otimes \ldots \mb{1}_{j-1} \otimes c^\dagger_j \otimes \mb{1}_{j+1} \ldots \mb{1}_N $$
# where $\mb{1}_i$ is the identity operator acting on the $i^\text{th}$ site.

#===============================================================================
# With definition of operatorN, the following three functions have become obsolete  
#
# def createN(i = 1, N = 1, dims = 2):
#     '''Returns the creation operator for the i^th site of an N site spin-chain with a 
#     Hilbert space of dimensionality dims at each site'''
#     
#     cdag = create(dims)
#     c = destroy(dims)
#     cid = identity(dims)
#     
#     id_list = []
#     c_list = []
#     
#     [c_list.append(cid) for n in range(N)]
#     
# #    print len(c_list)
#     
#     c_list[i-1] = cdag
#     
#     return tensor(c_list)
# 
#     
# def destroyN(i = 1, N = 1, dims = 2):
#     '''Returns the destruction operator for the i^th site of an N site spin-chain with a 
#     Hilbert space of dimensionality dims at each site'''
#     
#     c = destroy(dims)
#     cid = identity(dims)
#     
#     id_list = []
#     c_list = []
#     
#     [c_list.append(cid) for n in range(N)]
#     
# #    print len(c_list)
#     
#     c_list[i-1] = c
#     
#     return tensor(c_list)
# 
# 
# def numberN(i = 1, N = 1, dims = 2):
#     '''Returns the number operator for the i^th site of an N site spin-chain with a 
#     Hilbert space of dimensionality dims at each site'''
#     
#     cdag = create(dims)
#     c = destroy(dims)
#     numop = cdag * c
#     cid = identity(dims)
#     
#     id_list = []
#     op_list = []
#     
#     [op_list.append(cid) for n in range(N)]
#     
# #    print len(op_list)
#     
#     op_list[i-1] = numop
#     
#     return tensor(op_list)
#===============================================================================

def identityList(N = 1,dims = 2):
    '''Returns a list of N identity operators for a N site spin-system with a
    Hilber space at each state of dimensionality dims 
    '''
    try:
        from qutip import *
    except:
        raise NameError('Cannot import QuTiP')
    
    iden = identity(dims)
    
    iden_list = []

    [iden_list.append(iden) for i in range(N)]
    
    return iden_list

def operatorN(oper, i = 1, N = 1):
    '''Returns the operator given by oper for the i^th site of an N site spin-chain
    with a Hilbert space of dimensionality dims at each site'''
    
    try:
        from qutip import *
    except:
        raise NameError('Cannot import QuTiP')
    
    if not isinstance(oper, Qobj):
        raise TypeError('oper must of type qutip.Qobj')
    
    if not oper.isoper():
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
        iden_list[i-1] = oper
        return tensor(iden_list)

# Now at each site, one can have two electrons with spin up and down respectively. Thus our creation/annihilation operators also have a spin index $\sigma$:
# $$ \mb{1}_1 \otimes \ldots \mb{1}_{j-1} \otimes c^\dagger_{j\sigma} \otimes \mb{1}_{j+1} \ldots \mb{1}_N $$
# In order to implement we need two sets of creation/annihilation operators. One set for up-spin and one set for down-spin.

def jordanWignerDestroyI(i = 0, N = 1):
    ''' Returns the fermionic annihilation operator for the i^th site of a N site spin-chain:
    a_i = Z_1 x Z_2 ... x Z_{i-1} x sigma_i
    where Z_i is the Pauli sigma_z matrix acting on the i^th site and sigma_i is the density
    matrix acting on the i^th qubit, given by:
    sigma_i = id_1 x id_2 ... x id_{i-1} x |0><1| x id_{i+1} ... x id_n
    where id_i is the identity operator
    
    Reference: Nielsen, Fermionic CCR and Jordan Wigner Transformation
    '''
    
    # create zop, assign to it identity operator for N site system
    zop = tensor([qeye(2)]*N)
    
    # for k in (0..i-1), create Z_k operator for N-site chain
    # zop is product of Z_1 * Z_2 ... * Z_{i-1}
    
    for k in range(i):
        zop *= operatorN(sigmaz(),i = k, N = N)
    
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
    

def hubbardHamiltonian(N = 10, t = 1, U = 1, mu = 0, shift = False, dims = 2):
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
        # Create list containing identity operator for each site

        createA_list.append(operatorN(cOp,N=N,i=i))
        createB_list.append(operatorN(cOp,N=N,i=i))

        destroyA_list.append(operatorN(dOp,N=N,i=i))
        destroyB_list.append(operatorN(dOp,N=N,i=i))

        nA_list.append(operatorN(nOp,N=N,i=i))
        nB_list.append(operatorN(nOp,N=N,i=i))
        
    for i in range(N):
        H += - t * (createA_list[i] * destroyA_list[i] + createB_list[i] * destroyB_list[i]) \
            - mu * (nA_list[i] + nB_list[i])
        if shift == True:
            H += U * (nA_list[i] - 0.5 * superid) * (nB_list[i] - 0.5 * superid)
        else:
            H += U * nA_list[i] * nB_list[i]
    
    return H

def isingHamiltonian(N = 10, jcoefs=[-1]*10,spin=0.5):
    '''Returns operator corresponding to Ising Hamiltonian for give spin on N sites.
    Default value of N is 10. jcoef is the coupling strength. Default is -1 for
    ferromagnetic interaction.
    Default value of spin is 0.5
    '''
    
    op_list = []
    
    jz = jmat(spin,'z')
    
    dimj = 2*spin + 1
    
    H = 0
    
    for i in range(N-1):
        op_list = identityList(N, dimj)
        op_list[i] = op_list[i+1] = jz
        H += jcoefs[i]*tensor(op_list)
    
    return H

def operHamiltonian(oper, N = 10, coefs = [1]*10, dims=2):
    ''' Returns the Hamiltonian given by the sum of oper acting on each site
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

class Hamiltonian():
    
    _hamTypes = ['NumberOp', 'Ising', 'Heisenberg']
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
    
    def __init__(self, label=None, dims=2, isHermitian=True,\
                 numSites=1, hamType='NumberOp',data=None):
        try:
            from qutip import *
        except:
            raise NameError('QuTiP is not installed')
        
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
        
        if hamType not in self._hamTypes:
            from string import join
            raise ValueError('hamType must be one of ' + join(self._hamTypes,', '))
        else:
            self._hamType = hamType
        
        if dims < 2 or not isinstance(dims, int):
            raise ValueError('dim must be an integer greater than or equal to 2')
        else:
            self._dims = dims
        
        self.createHamiltonian()
            
        return

    def createHamiltonian(self):
        
        if self._hamType == 'Ising':
            
            self._hamiltonian = isingHamiltonian(self._numSites,self._data['jcoefs'],self._data['spin'])
        
        elif self._hamType == 'Hubbard':
            
            self._hamiltonian = hubbardHamiltonian(self._numSites, self._data['t'],\
                                self._data['U'], self._data['mu'], \
                                self._data['shift'], self._dims)
            
        elif self._hamType == 'NumberOp':
            
            numOp = create(self._dims)*destroy(self._dims)
            
            self._hamiltonian = operHamiltonian(numOp, self._numSites, \
                                    self._data['coefs'], self._dims)
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


def partition_hubbard(temp = 1., t = 0., U = 1., mu = 0.,kB = 1.):
    ''' Partition function for single site Hubbard model.
        t is the kinetic energy
        U is the potential energy
        mu is the chemical potential
        temp is the temperature
        kB is Boltzmann's constant, set to 1 by default
    '''
#    temp = temp + 10**(-5)
    beta = 1/(temp*kB)
    exp1 = np.exp(beta*(mu+t))
    exp2 = np.exp(2.0*beta*(mu+t-U/2.0))
    return 1 + 2*exp1 + exp2


# In[3]:

partition_hubbard(mu=np.arange(0.1,2,0.5))


# In[ ]:

temp_values = np.arange(0.1,5,0.1)
plot_num = 1
for mu in np.arange(0,0.6,0.1):
    z_values = partition_hubbard(temp=temp_values,mu=mu)
    plt.subplot(2,3,plot_num)
    plt.title('mu = '+ str(mu))
    plt.plot(temp_values,z_values)
    plot_num+=1
 


# In[4]:

# plt.tight_layout() ensures that neighboring subplots do not overlap
plt.tight_layout()
plt.show()


# In[ ]:

plot_num = 1
for mu in np.arange(-0.5,0.5,0.25):
    z_values = partition_hubbard(temp=temp_values,mu=mu)
    plt.subplot(2,3,plot_num)
    plt.title('mu = '+ str(mu))
    plt.plot(temp_values,z_values)
    plot_num+=1


# In[5]:

# plt.tight_layout() ensures that neighboring subplots do not overlap
plt.tight_layout()
plt.show()


# In[ ]:

plot_num = 1
for mu in np.arange(0,0.09,0.01):
    z_values = partition_hubbard(temp=temp_values,mu=mu)
    plt.subplot(3,3,plot_num)
    plt.title('mu = '+ str(mu))
    plt.plot(temp_values,z_values)
    plot_num+=1


# In[6]:

# plt.tight_layout() ensures that neighboring subplots do not overlap
plt.tight_layout()
plt.show()


# The energy for the single site Hubbard model is:
# 
# \begin{align}
#     E & = \langle H + \mu n \rangle = \text{Tr} \left[ (H + \mu n) e^{-\beta H} \right] \\
#       & = \frac{1}{Z} \sum_\alpha \langle \alpha \vert (H + \mu n) e^{-\beta H} \vert \alpha \rangle \\ 
#       & = \frac{U e^{2\beta(t+\mu - U/2)}}{1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)}}
# \end{align}
# 
# and the occupation number is:
# 
# \begin{align}
#     \rho & = \langle n \rangle = \text{Tr} \left[ n e^{-\beta H} \right] \\
#          & = \frac{2 e^{\beta(\mu + t)} + 2 e^{2\beta(t+\mu-U/2)}}{1 + e^{\beta(\mu + t)} + e^{\beta(2t + 2\mu - U)}}
# \end{align}

# In[7]:

def plot_hubbard(mu):
    z_values = partition_hubbard(temp=temp_values,mu=mu)
    plt.title('mu = '+ str(mu))
    plt.plot(temp_values,z_values)


# In[8]:

from IPython.html.widgets import *


# In[10]:

def energy_hubbard(temp = 1, t = 0, U = 1, mu = 0,kB = 1):
    ''' Partition function for single site Hubbard model.
        t is the kinetic energy
        U is the potential energy
        mu is the chemical potential
        temp is the temperature
        kB is Boltzmann's constant, set to 1 by default
    '''
#    temp = temp + 10**(-5)
    beta = 1/(temp*kB)
    exp1 = np.exp(beta*(mu+t))
    exp2 = np.exp(2.*beta*(mu+t-U/2.))
    return (-2*t*exp1 + (U - 2*t) * exp2)/(1 + 2*exp1 + exp2 )


# In[11]:

def filling_hubbard(temp = 1, t = 0, U = 1, mu = 0,kB = 1):
    ''' Partition function for single site Hubbard model.
        t is the kinetic energy
        U is the potential energy
        mu is the chemical potential
        temp is the temperature
        kB is Boltzmann's constant, set to 1 by default
    '''
#    temp = temp + 10**(-5)
    beta = 1/(temp*kB)
    exp1 = np.exp(beta*(mu+t))
    exp2 = np.exp(2.*beta*(mu+t-U/2.))
    return (2*exp1 + 2*exp2)/(1 + 2*exp1 + exp2 )


# In[12]:

mu_values = np.arange(0,10,0.1)


# In[14]:

import matplotlib as mpl

mpl.rc('text', usetex=False)
mpl.rc('font', family='serif')


# In[15]:

def plot_filling(temperature):
    f_values = filling_hubbard(temp=temperature,mu=mu_values,t=0,U=4.0)
    plt.title('U = 4.0')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\langle n \rangle $')
    plt.plot(mu_values,f_values,label='T = '+str(temperature))
    plt.legend(loc='lower right', shadow=True)


# In[16]:

def plot_energy_mu(temperature):
    e_values = energy_hubbard(temp=temperature,mu=mu_values)
    plt.title('U = 4.0')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\langle E \rangle $')
    plt.legend()
    plt.plot(mu_values,e_values,label='T = '+str(temperature))
    plt.legend(loc='upper center', shadow=True)

