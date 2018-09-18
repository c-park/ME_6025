"""
Amir Beck provides MATLAB code at 
https://sites.google.com/site/amirbeck314/books
including
+ gradient_method_backtracking
+ gradient_method_quadratic
+ gradient_scaled_quadratic
+ newton_backtracking

This code implements the same algorithms in python. 

Braxton Osting
9/3/2018
"""

import numpy as np
from numpy.linalg import norm

def gradient_method_backtracking(f,g,x0,s,alpha,beta,epsilon):
    """
    Gradient method with backtracking stepsize rule
    INPUT
    =======================================
    f ......... objective function
    g ......... gradient of the objective function
    x0......... initial point
    s ......... initial choice of stepsize
    alpha ..... tolerance parameter for the stepsize selection
    beta ...... the constant in which the stepsize is multiplied 
                at each backtracking step (0<beta<1)
    epsilon ... tolerance parameter for stopping rule
    OUTPUT
    =======================================
    x ......... optimal solution (up to a tolerance) 
                of min f(x)
    fun_val ... optimal function value
    """
    x=x0
    grad=g(x)
    fun_val=f(x)
    iter=0
    while (norm(grad)>epsilon):
        iter=iter+1
        t=s
        while (fun_val-f(x-t*grad)<alpha*t*norm(grad)**2):
            t=beta*t
        x=x-t*grad
        fun_val=f(x)
        grad=g(x)
        print('iter_number = '+ str(iter) + ' norm_grad = ' + str(norm(grad)) + ' fun_val = ' + str(fun_val))
    return x,fun_val



def gradient_method_quadratic(A,b,x0,epsilon):
    """
    INPUT
    ======================
    A ....... the positive definite matrix associated with the objective function
    b ....... a column vector associated with the linear part of the objective function 
    x0 ...... starting point of the method
    epsilon . tolerance parameter
    OUTPUT
    =======================
    x ....... an optimal solution (up to a tolerance) of min(x^T A x+2 b^T x)
    fun_val . the optimal function value up to a tolerance
    """
    x=x0
    iter=0
    grad=2*(A@x+b)
    while (norm(grad)>epsilon):
        iter=iter+1
        t=norm(grad)**2/(2*grad@A@grad)
        x=x-t*grad
        grad=2*(A@x+b)
        fun_val=x@A@x+2*b@x
        print('iter_number = '+ str(iter) + ' norm_grad = ' + str(norm(grad)) + ' fun_val = ' + str(fun_val))
    return x,fun_val



def gradient_scaled_quadratic(A,b,D,x0,epsilon):
    """
    INPUT
    ======================
    A ....... the positive definite matrix associated with the objective function
    b ....... a column vector associated with the linear part of the objective function 
    D ....... scaling matrix
    x0 ...... starting point of the method
    epsilon . tolerance parameter
    OUTPUT
    =======================
    x ....... an optimal solution (up to a tolerance) of min(x^T A x+2 b^T x)
    fun_val . the optimal function value up to a tolerance
    """

    x=x0
    iter=0
    grad=2*(A@x+b)
    while (norm(grad)>epsilon):
        iter=iter+1
        t=grad@D@grad/(2*(grad@D.transpose())@A@(D@grad))
        x=x-t*D@grad
        grad=2*(A@x+b)
        fun_val=x@A@x+2*b@x
        print('iter_number = '+ str(iter) + ' norm_grad = ' + str(norm(grad)) + ' fun_val = ' + str(fun_val))
    return x,fun_val



def newton_backtracking(f,g,h,x0,alpha,beta,epsilon):
    """
    Newton's method with backtracking
    
    INPUT
    =======================================
    f ......... objective function
    g ......... gradient of the objective function
    h ......... hessian of the objective function
    x0......... initial point
    alpha ..... tolerance parameter for the stepsize selection strategy
    beta ...... the proportion in which the stepsize is multiplied
                at each backtracking step (0<beta<1)
    epsilon ... tolerance parameter for stopping rule
    OUTPUT
    =======================================
    x ......... optimal solution (up to a tolerance)
                of min f(x)
    fun_val ... optimal function value
    """

    x=x0
    gval=g(x)
    hval=h(x)
    d=np.linalg.solve(hval,gval)
    iter=0
    while ((norm(gval)>epsilon) and (iter<10000)):
        iter=iter+1
        t=1
        while(f(x-t*d)>f(x)-alpha*t*gval@d):
            t=beta*t
        x=x-t*d
        fun_val = f(x)
        print('iter_number = '+ str(iter) + ' fun_val = ' + str(fun_val))
        gval=g(x)
        hval=h(x)
        d=np.linalg.solve(hval,gval)

    if (iter==10000):
        print('did not converge')

    return x,fun_val
