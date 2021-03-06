
#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import random
import time
import copy

from matplotlib import animation, rc
from IPython.display import HTML


def _update_plot (i,fig,scat,qax) :
    scat.set_offsets(P[i])
    qax.set_offsets(P[i])

    VVV=[list(x) for x in zip(*nV[i])]

    qax.set_UVC(VVV[0],VVV[1])
    # qax.quiver(PPP[0],PPP[1],VVV[0],VVV[1],angles='xy')

    
    # print ('Frames:%d' %i)
    return scat,qax

def psi(s,b):
    a = pow((1+s**2),-b)
    return a

def csm(X,V,k):
    a=[0,0]
    for i in range(N):
        s=pow((X[k][0]-X[i][0])**2+(X[k][1]-X[i][1])**2,0.5)
        ps = psi(s,beta)
        a[0]+= ps * (V[i][0]-V[k][0])
        a[1]+= ps * (V[i][1]-V[k][1])

    a[0]=a[0]/N
    a[1]=a[1]/N 
    
    return a

# def ode_rk(X,):    

if __name__ == '__main__':
    N=int(input("N=?"))
    beta=float(input("beta=?"))
    T=int(input("T=?"))


    Pinit=[]
    Vinit=[]

    for n in range(N):
        x = random.uniform(-100,100)
        y = random.uniform(-100,100)
        Pinit.append([x,y])
        
        x = random.uniform(5,70)
        xd=random.randint(0,1)
        y = random.uniform(5,70)
        yd=random.randint(0,1)
        Vinit.append([x*(2*xd-1),y*(2*yd-1)])
        
   
    P=[Pinit]
    V=[Vinit]
    nV=[[[0,0] for row in range(N)]]

    print(P[0])
    print(V[0])

    for n in range(N):
        s=pow(V[0][n][0]**2+V[0][n][1]**2,0.5)
        if (s==0) :
            s=1
        nV[0][n][0]=V[0][n][0]/s
        nV[0][n][1]=V[0][n][1]/s
 

    # P 위치 V 초기가ㅄ 설정, nV:V를 normalize -> 이거 np 사용하면 더 간단해질 수도 있을 듯

    h=0.025

    for t in range(1,T):
        # print ("start %dth loop" %t)
        # print (P[0])

        Pnow=copy.deepcopy(P[t-1])
        Vnow=copy.deepcopy(V[t-1])
        nVnow=[[0,0] for row in range(N)]

        K1=[]
        K2=[]
        K3=[]
        K4=[]

        # K1-K4가 runge kutta 에서 그 h*k1-h*k4를 가ㄱ각 k별로 구해서 list로 만든 것.

        Phk1=copy.deepcopy(Pnow)
        Vhk1=copy.deepcopy(Vnow)
        for n in range(N):

            k1=csm(Pnow,Vnow,n)
            k1[0]*=h
            k1[1]*=h

            Phk1[n][0]+=Vnow[n][0]*h/2
            Phk1[n][1]+=Vnow[n][1]*h/2
            Vhk1[n][0]+=k1[0]/2 
            Vhk1[n][1]+=k1[1]/2

            K1.append([Vnow[n],k1])
        #Vhk1 = y+h*k1/2

        Phk2=copy.deepcopy(Pnow)
        Vhk2=copy.deepcopy(Vnow)
        for n in range(N):

            k2=csm(Phk1,Vhk1,n)
            k2[0]*=h
            k2[1]*=h

            Phk2[n][0]+=Vhk1[n][0]*h/2
            Phk2[n][1]+=Vhk1[n][1]*h/2
            Vhk2[n][0]+=k2[0]/2
            Vhk2[n][1]+=k2[1]/2

            K2.append([Vhk1[n],k2])
        #Vhk2 = y+h*k2/2

        Phk3=copy.deepcopy(Pnow)
        Vhk3=copy.deepcopy(Vnow)
        for n in range(N):

            k3=csm(Phk2,Vhk2,n)
            k3[0]*=h
            k3[1]*=h

            Phk3[n][0]+=Vhk2[n][0]*h
            Phk3[n][1]+=Vhk2[n][1]*h
            Vhk3[n][0]+=k3[0]
            Vhk3[n][1]+=k3[1]

            K3.append([Vhk2[n],k3])
        #Vhk3 = y+h*k3

        for n in range(N):

            k4=csm(Phk3,Vhk3,n)
            k4[0]*=h
            k4[1]*=h

            K4.append([Vhk3[n],k4])

        for n in range(N):

            Pnow[n][0]+=(K1[n][0][0]+2*K2[n][0][0]+2*K3[n][0][0]+K4[n][0][0])*h/6
            Pnow[n][1]+=(K1[n][0][1]+2*K2[n][0][1]+2*K3[n][0][1]+K4[n][0][1])*h/6
            
            Vnow[n][0]+=(K1[n][1][0]+2*K2[n][1][0]+2*K3[n][1][0]+K4[n][1][0])/6
            Vnow[n][1]+=(K1[n][1][1]+2*K2[n][1][1]+2*K3[n][1][1]+K4[n][1][1])/6
            
            s=pow(Vnow[n][0]**2+Vnow[n][1]**2,0.5)
            if (s==0):
                s=1
            nVnow[n][0]=Vnow[n][0]/s
            nVnow[n][1]=Vnow[n][1]/s            

        P.append(Pnow)
        V.append(Vnow)
        nV.append(nVnow)
        # print(P[0])

    # print ("end")
    print (Pnow)
    print (Vnow)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim([-1000,1000])
    ax.set_ylim([-1000,1000])

    PP=[list(x) for x in zip(*P[0])]
    VV=[list(x) for x in zip(*nV[0])]
    scat=plt.scatter(PP[0],PP[1],s=20)
    scat.set_alpha(0.2)
    qax=ax.quiver(PP[0],PP[1],VV[0],VV[1],angles='xy',width=0.001,scale=70)


    ani = animation.FuncAnimation(fig,_update_plot,fargs=(fig,scat,qax),frames=T-1,interval=10,save_count=T-1)

    #interval이 너무 작으니깐 save가 안됨-파일을 열때 에러남.
    
    plt.show()
    

    ani.save('csm-ode45-simu.mp4')

    print("DONE")




# %%
