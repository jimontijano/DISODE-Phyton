import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
      Fc=0.4;
      f=np.array([y[1], -y[0]-Fc*np.sign(y[1])])
      return f
#
#  Definition of the switching surface
#
def gfun(t,y):
      g=y[1]
      isterminal=0
      direction=0
      return g,isterminal,direction
  
def gradfun(t,y, ind):
      grad = np.array([0, 1])
      return grad
  
#
#  Call to disode45
#
y0=[3, 0]
#tspan=[0, 20]
# options = disodeset('RelTol',1.e-6, 'AbsTol',1.e-6)
# options = disodeset('Refine',10)
#tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0,options)
#tspan=np.linspace(0,20,100)
#tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0)
options = disodeset('Refine',10,'Gradient',gradfun)
tspan=np.linspace(0,20,100)
tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0, options)
#
#  display of the results
#
print("\n tdis=",tdis)
print("\n ydis=",ydis)
print("\n idis=",idis)
#
#  Plot of the figures
#
plt.figure(1)
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'k--',tdis,ydis[:,1],'ro')
plt.savefig("Figure1a.pdf")
plt.figure(2)
plt.plot(yout[:,0],yout[:,1],'k-',ydis[:,0],ydis[:,1],'ro')
plt.savefig("Figure2a.pdf")
plt.show()
