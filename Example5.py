import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
      ydot=np.array([y[1], -9.8])
      return ydot
#
#  Definition of the switching surface
#
def gfun(t,y):
      g=y[0]
      isterminal=-1    #  Call to gswitch when a switching point is found
      direction=-1
      return g,isterminal,direction
#
#  Output switch function
#
def actionatswitch(t,y):
      ysw=np.array([0, -0.9*y[1]])
      return ysw
#
#  Call to disode45
#
options=disodeset('AbsTol',1.e-4,'RelTol',1.e-4,
                 'Refine',10, 'ActionSwitch', actionatswitch);
y0=np.array([10, 0])
tout,yout,tdis,ydis,idis,stats = disode45(fun,
                              gfun, [0,20], y0, options);


#
#  display of the results
#
print("\n tdis=",tdis)
print("\n ydis=",ydis)
print("\n idis=",idis)


