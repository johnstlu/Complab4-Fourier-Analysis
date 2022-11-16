import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 
from scipy import signal
pi = np.pi

dt = 0.01
def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
 
def quadratic(x):
    return x**2

def simpson_integration(lower_limit,upper_limit,function,n_steps):
    if n_steps % 2 ==1:
        string = "n-steps must be even. Try again"
        print(string)
        return
        
    h = (upper_limit-lower_limit)/n_steps
    
    f_0 = function(lower_limit)
    print(f_0,'=f_0')
    f_n = function(upper_limit)
    print(f_n,'=f_n')
    even_values = 0
    print('even js =',np.arange(1,n_steps/2,1))
    for j in np.arange(1,n_steps/2,1):
        x_even = lower_limit + 2*j*h
        print(x_even, 'x_even')
        print(function(x_even), 'f(x_even), j=',j)
        even_values += 2*function(x_even)
    
    odd_values = 0
    print('odd js =',np.arange(1,n_steps/2+1,1) )
    for j in np.arange(1,n_steps/2+1,1):
        x_odd = lower_limit +(2*j-1)*h
        print(x_odd, 'x_odd')
        print(function(x_odd), 'f(x_odd),j =',j)
        odd_values += 4*function(x_odd)
        
    print(even_values , '=even values')
    print(odd_values , '=odd values')    
    integral = (h/3)*(f_0 + even_values + odd_values + f_n)
    return integral


"""
Ex1.4 Fourier Series
"""

def coefficent_calculator(function,period,n_terms,integration_steps,plot_limit):
    
    xs = np.arange(0,period,dt)
    yf = function(xs)
    
    a_0 = (1/period)*integrate.simpson(yf,xs)
    series_expansion = a_0    
    
    print(a_0, '=a0')
    
    omega = 2*np.pi/period
    
    a_ks = [0]
    b_ks = [0]
    for k in np.arange(1,n_terms+1,1):

        def sin_times_function(x):
            return function(x)*np.sin(k*omega*x)
        
        def cos_times_function(x):
            return function(x)*np.cos(k*omega*x)
        
        ya = cos_times_function(xs)
        yb = sin_times_function(xs)
        
        a_k = np.round_((2/period)*integrate.simpson(ya,xs),4)
        b_k = np.round((2/period)*integrate.simpson(yb,xs),4)
        
        a_ks.append(a_k)
        b_ks.append(b_k)
    
    def series_expansion(x):
        cos_terms = 0
        for n in np.arange(1,len(a_ks),1):
            cos_terms += a_ks[n]*np.cos(n*omega*x)
        sin_terms = 0
        for n in np.arange(1,len(b_ks),1):
            sin_terms += b_ks[n]*np.sin(n*omega*x)
        
        return a_0 + sin_terms + cos_terms
        
    print(a_ks,'=a_ks')
    print('')
    print(b_ks,'=b_ks')
    print('')
        
    ts = np.arange(0,plot_limit,dt)
    fs = [series_expansion(t) for t in ts] 
    
    plt.plot(ts,fs)
    plt.title("First 100 Terms of the Fourier Series of a Square Wave ")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.savefig('Ex2.5_100term.png',dpi = 200)

    return [a_0,a_ks,b_ks]

def test_function1(x):
    return np.sin(x)
def test_function2(x):
    return cos(x) +3*cos(2*x) -4*cos(3*x)
def test_function3(x):
    return sin(x) +3*sin(3*x) +5*sin(5*x)
def test_function4(x):
    return sin(x) +2*cos(3*x) +3*sin(5*x)
