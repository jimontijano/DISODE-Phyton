import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
      k=210.125
      c=2.47e+6
      nu=0.005
      r=2*np.sin(14*t)
      if y[0]>nu:
        if y[1]>0:
           u=c*(y[0]-nu)**(1.5) + 1.98*np.sqrt(2*c*np.sqrt(y[0]-nu))*y[1]                                     
        else:
           u=c*(y[0]-nu)**(1.5)
      else:
        u=0;
      f=np.array([y[1], (-4.1*y[1]-k*y[0]-u-r)/2])
      return f
  
#
#  Definition of the switching surface
#
def gfun(t,y):
    if y[0]<0.005:
#       g=np.array([y[0]-0.005, y[1]])
       g=np.array([y[0]-0.005, 1])
    else:
       g=np.array([y[0]-0.005, y[1]])
    isterminal=[0,0]
    direction=[0,-1]
    return g,isterminal,direction

#
#  Call to disode45
#
options=disodeset('AbsTol',1.e-4,'RelTol',1.e-4,'Refine',10)
y0 = [0, 0]
tout,yout,tdis,ydis,idis,stats = disode45(fun,
                                    gfun,[0,3.0], y0,options)


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
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'b')
plt.savefig("Figure4a.pdf")
plt.figure(2)
plt.plot(yout[:,0], yout[:,1],[0.005,0.005],[-0.15,0.25],'r--',[0.005, 0.008],[0,0],'r--')
plt.savefig("Figure4b.pdf")
plt.show()

