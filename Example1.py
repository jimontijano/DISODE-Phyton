import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun1(t,y):
   E=1.0
   if y[0] > 0:
      mu1=1.0
   else:
      mu1=0.6

   if y[1] > 0:
      mu2=0.2
   else:
      mu2=0.5
      
   f=np.array([y[2],y[3],-E*(y[0]-y[1])-mu1*np.sign(y[2]), 
                    -E*(y[1]-y[0])-mu2*np.sign(y[3])] )
   return f
#
#  Definition of the switching surface
#
def gfun1(t,y):
   g=np.array([y[2], y[3], y[0], y[1]])
   isterminal=[0,0,0,0]
   direction=[0,0,0,0]
   return g, isterminal,direction

  
#
#  Call to disode45
#
y0 = np.array([-2, 3, 0, 0])
tout,yout,tdis,ydis,idis,stats=disode45(fun1, gfun1,[0, 12], y0)
#tspan=[0, 20]
# options = disodeset('RelTol',1.e-6, 'AbsTol',1.e-6)
# options = disodeset('Refine',10)
#tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0,options)
#tspan=np.linspace(0,20,100)
#tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0)
#options = disodeset('Refine',10,'Gradient',gradfun)
#tspan=np.linspace(0,20,100)
#tout,yout,tdis,ydis,idis,stats =disode45(fun, gfun,tspan, y0, options)
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
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'b',tout,yout[:,2],'k--',
     tout,yout[:,3],'b--',tdis,ydis[:,0],'ro')
plt.savefig("Figure1a.pdf")
plt.show()