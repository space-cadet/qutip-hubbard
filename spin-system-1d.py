import array
import numpy as np

def zeroCoefs(coefs,N):
    ''' Creates N * N size matrix, sets all values to zero and returns matrix '''
    try:
        coefs = np.zeros((N,N))
        return coefs
    except Exception:
        return false

def identityCoefs(coefs, N):
    ''' Creates N * N size matrix, sets diagonals to one, all other entries to zero, and returns matrix '''
    try:
        # use np.diagflat method to create diagonal array
        # use list comprehension to generate list of 1s to pass to np.diagflat
        coefs = np.diagflat([1 for i in range(N)])
        return coefs
    except Exception:
        return false
    
def randCoefs(N, max_val = 1.0, min_val = -1.0, symmetric = True):
    ''' Creates N * N size matrix and randomly initializes all entries to a value between
    max_val and min_val. If not given, these default to 1.0 and -1.0 respectively. Unless specified
    using the variable symmetric=FALSE, the generated matrix will be symmetric'''
    try:
        coefs = np.zeros((N,N))
        delta = max_val - min_val
        
        if symmetric:
            for i in range(N):
                for j in range(0,i+1):
                    coefs[i][j] = max_val - delta * np.random.random_sample()
                    coefs[j][i] = coefs[i][j]
        else:
            for i in range(N):
                for j in range(N):
                    coefs[i][j] = max_val - delta * np.random.random_sample()
        return coefs
    except Exception:
        return false

def randVec(N, values=[0,1]):
    ''' Returns size N array with each element assigned a member of values[] array at random '''
    try:
        vect = np.zeros(N)
        lenval = len(values)
        for i in range(N):
            vect[i] = values[np.random.randint(0,lenval)]
        return vect
    except Exception:
        return false
    
def expectation(inputVec,coefs):
    ''' Accepts 1d vector of length N & matrix of size N x N as input.
    Returns expectation value of matrix acting on the input vector '''
    n = len(inputVec)
    (a,b) = np.shape(coefs)
    if cmp((a,b),(n,n)) != 0:
        print "Dimensions of vector ({0}) are not compatible with dimensions of matrix ({1})".format(n,(a,b))
        return false
#    if np.shape(inputVec) != (chainN,):
#        print "Vector length is not equal to: ", chainN
#        return
#    if np.shape(coefs) != (chainN,chainN):
#        print "Coefficient matrix dimensions are not:", chainN, "x", chainN
    # ensure that val is initialized to float value by adding decimal
    val = 0.
    for i in range(n):
        for j in range(n):
           val = val + coefs[i][j]*inputVec[i]*inputVec[j]
    return val