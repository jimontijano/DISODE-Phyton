import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
     if y[1]==-1:
       ydot=np.array([-0.1*(y[0]-18), 0])
     else:
       ydot=np.array([-0.1*(y[0]-18)+2, 0])
     return ydot
#
#  Definition of the switching surface
#
def gfun(t,y):
      g=[y[0]-23.5, y[0]-22]
      isterminal=[-1,-1]     #  Call to actionatswitch when found
      direction=[1,-1]       #  From negative to positive the first one
      return g,isterminal,direction
#
#  Output switch function
#
def actionatswitch(t,y):
     if y[1]==1:
       yswitch=[y[0], -1]
     else:
       yswitch=[y[0], 1];
     return yswitch

#
#  Call to disode45
#
options=disodeset('AbsTol',1.e-4,'RelTol',1.e-4,
                 'ActionSwitch', actionatswitch);
y0=np.array([15, 1])
tout,yout,tdis,ydis,idis,stats = disode45(fun,
                              gfun, [0,20], y0, options);


#
#  display of the results
#
print("\n tdis=",tdis)
print("\n ydis=",ydis)
print("\n idis=",idis)


