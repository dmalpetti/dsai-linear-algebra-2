# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:21:12 2023

@author: dmalpetti
"""

from sympy import Matrix, symbols, factor, pprint, I, eye, exp, Rational


#### Initialization 

# initialization of numeric matrix
A = Matrix([[4, 1], [1, 4]])
pprint(A)

# initialization of numeric matrix containing complex numbers
# use I for the imaginary unit
B = Matrix([[1+I, 2], [I, 2-I]])
pprint(B)

# initialization of a matrix with a parameter, k in this case
# you can change the name of the parameter of course
k = symbols('k')
C = Matrix([[4*k, 3], [1, 3+k]])
pprint(C)

# identity matrix (e.g. 3x3)
Id = eye(3)
pprint(Id)

# initialization of a vector
v = Matrix([[1], [1]])
pprint(v)



#### Basic matrix operations

# matrix multiplication, remember to use @.
# notice we can multiply a numeric matrix and one containing a parameter
pprint(A @ B)
pprint(A @ C)

# matrix by vector multiplication
pprint(A @ v)

# transpose of a matrix
At = A.T
pprint(At)

# inverse of a matrix
Ainv = A.inv()
pprint(Ainv)

# trace of a matrix
trA = A.trace()
pprint(trA)

# determinant of a matrix
detA = A.det()
pprint(detA)



#### Powers and exponentials

# matrix power (e.g. square)
Asq = A**2
pprint(Asq)

# matrix power with parameter
n = symbols('n')
An = A**n
pprint(An)

# matrix exponential
Aexp = exp(A)
pprint(Aexp)



#### Eigenstuff

# characteristic polynomial
lamda = symbols('lamda')
p = A.charpoly(lamda)
pprint(p)
# factorized characteristic polynomial
p_fact = factor(p.as_expr(), extension=[I])
pprint(p_fact)

# eigenvalues
# returns: {eigenvalue, algebraic_multiplicity}
Aeigvals = A.eigenvals()
pprint(Aeigvals)

# eigenvalues and associated eigenvectors
# returns: (eigenvalue, algebraic_multiplicity, [eigenvectors])
Aeigvecs = A.eigenvects()
pprint(Aeigvecs)

# full diagonalization with matrices
def diagonalize(matrix):
    s, diagonal = matrix.diagonalize(matrix)
    return s, diagonal, s.inv()
Afulldiag = diagonalize(A)
pprint(Afulldiag)



#### Applications chapter

# norm (notice the 2, since we want Euclidean norm)
Anorm = A.norm(2)
pprint(Anorm)

# condition number
Acond = A.condition_number()
pprint(Acond)

# singular values
sigmas = A.singular_values()
pprint(sigmas)

# to homogeneous coordinates
def to_homogeneous(matrix):
     """Convert a 2x2 or 3x3 matrix to homogeneous coordinates."""
     return matrix.row_join(Matrix.zeros(matrix.rows,1)).col_join(Matrix([[0] * matrix.cols + [1]]))

A_hom = to_homogeneous(A)
pprint(A)