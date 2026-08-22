import numpy as np
import matplotlib.pyplot as plt
from  disode45 import *

#
#  Definition of the vector field
#
def fun(t,y):
    m1=2
    m2=1
    r1=0.6
    r2=0.6
    k1=30;
    k2=20;
    a0=30;
    b0=35;
    d=1.0;
    ee=0.7
    w=1.38
    u1=-(r1/m1)*y[2]-(k1/m1)*y[0]+b0/m1+(a0/m1)*np.cos(w*t);
    u2=-(r2/m2)*y[3]-(k2/m2)*y[1];
    if y[0]-y[1]>= d/2 and u1>=u2:   # and   y[2] - y[3] >= -1e-4 
       aux = -((r1 + r2) / (m1 + m2)) * y[2] - ((k1 + k2) / (m1 + m2)) * y[0]+ b0 / (m1 + m2) + (k2 * d / (2 * m1 + 2 * m2))+ (a0 / (m1 + m2)) * np.cos(w * t)
       f=np.array([y[2], y[2], aux, aux]) 
    elif y[1]-y[0]>= d/2 and u2>=u1:   #    and Y[2] - Y[3] <= 1e-4
       aux = -((r1 + r2) / (m1 + m2)) * y[2] - ((k1 + k2) / (m1 + m2)) * y[0]+ b0 / (m1 + m2) - (k2 * d / (2 * m1 + 2 * m2)) + (a0 / (m1 + m2)) * np.cos(w * t)       
       f=np.array([y[2], y[2], aux, aux])
    else:
       f=np.array([y[2], y[3], u1, u2])
        
    return f
#
#  Definition of the switching surface
#
def gfun(t,y):
    m1=2;
    m2=1;
    r1=0.6;
    r2=0.6;
    k1=30;
    k2=20;
    a0=30;
    b0=35;
    d=1.0;
    ee=0.7;
    w=1.38;
    u1=-(r1/m1)*y[2]-(k1/m1)*y[0]+b0/m1+(a0/m1)*np.cos(w*t);
    u2=-(r2/m2)*y[3]-(k2/m2)*y[1];
    
    if np.abs(y[0] - y[1] - d / 2) < 1e-12:
        isterminal=np.array([-1, -1, -1])
        direction=np.array([1, 1, -1])
        g = np.array([(y[0] - y[1] - d / 2), y[1] - y[0] - d / 2, u1 - u2])
    elif np.abs(y[1] - y[0] - d / 2) < 1e-12:
        isterminal=np.array([-1, -1, 1])
        direction=np.array([1, 1, 1])
        g = np.array([(y[0] - y[1] - d / 2), y[1] - y[0] - d / 2, u1 - u2])
    else:
        isterminal = np.array([-1, -1, 0])
        direction = np.array([1, 1, 0])
        g = np.array([(y[0] - y[1] - d / 2), y[1] - y[0] - d / 2, 1.0])
    return g,isterminal,direction

#
#  Definition of the action at switch function
#
def actionatswitch(t,y):
    m1=2
    m2=1
    r1=0.6
    r2=0.6
    k1=30
    k2=20
    a0=30
    b0=35
    d=1.0
    ee=0.7
    w=1.38
    ysw=y.copy()
    ysw[2]=((m1-m2*ee)/(m1+m2))*y[2]+((1+ee)*m2/(m1+m2))*y[3]
    ysw[3]=((m2-m1*ee)/(m1+m2))*y[3]+((1+ee)*m1/(m1+m2))*y[2]
    
    if abs(y[0]-y[1]-d/2) <1.e-5:
           ysw[0]=y[1]+d/2-1.e-8;
    elif abs(y[1]-y[0]-d/2)<1.e-5:
           ysw[1]=y[0]+d/2-1.e-8;
    if abs(y[2]-y[3])<1.e-5:
           ysw[3]=y[3]
           ysw[2]=ysw[3]

    return ysw


#
# Call to disode45
#

options=disodeset('RelTol',1.e-5,'AbsTol',1.e-5,
                        'ActionSwitch', actionatswitch,'Verbose',1)
y0 = np.array([0.2,0.3,0,0])
[tout,yout,tdis,ydis,idis,stats]=disode45(fun, gfun,[0,1.15], y0,options);

print("pasos=",tout.shape)

#
#  Plot of the figures
#
plt.figure(1)
plt.plot(tout,yout[:,0],'k',tout,yout[:,1],'b')
plt.savefig("Figure4a.pdf")
plt.figure(2)
plt.plot(tout,yout[:,0]-yout[:,1],[0,10],[0.5,0.5],'r--')
plt.savefig("Figure4b.pdf")
plt.show()