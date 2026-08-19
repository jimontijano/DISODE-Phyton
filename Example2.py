import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
      k=1.0
      r=0.2
      a0=1.0
      w=0.7
      Fc=0.4
      v=np.cos(t)+0.7
      f=np.array([y[1],-k*y[0]-2*r*y[1]+a0*np.cos(w*t)-Fc*np.sign(y[1]-v)])
      return  f
#
#  Definition of the switching surface
#
def gfun(t,y):
    v=np.cos(t)+0.7;
    g=y[1]-v;
    isterminal=0;
    direction=0;
    return  g,isterminal,direction



#
#  Call to disode45
#
y0 = np.array([3, 0])
tout,yout,tdis,ydis,idis,stats=disode45(fun, gfun,[0, 30], y0)
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
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'b',
     tdis,ydis[:,0],'ro')
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'k--', tdis,
                                          ydis[:,0],'ro')
plt.savefig("Figure3a.pdf")
plt.figure(2)
plt.plot(tout,yout[:,1]-0.7-np.cos(tout),[0,30],[0,0],'r')
plt.savefig("Figure3b.pdf")
plt.show()