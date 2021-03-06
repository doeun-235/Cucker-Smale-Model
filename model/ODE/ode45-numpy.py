
#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import time

from matplotlib import animation, rc
from IPython.display import HTML

#np 참고 http://taewan.kim/post/numpy_cheat_sheet/

def _update_plot (i,fig,scat,qax) :

    ax.set_title('time : %i' %i)

    scat.set_offsets(P[i])
    qax.set_offsets(P[i])

    qax.set_UVC(nV[i].T[0],nV[i].T[1])
    # qax.quiver(PPP[0],PPP[1],VVV[0],VVV[1],angles='xy') -> 이렇게 하면 화살표가 위에 새로 계속 찍힘
    
    # print ('Frames:%d' %i)
    return scat,qax

def psi(s,b):
    a = np.power((1+np.power(s,2)),-b)
    return a

def csm(X,V):

    k=np.zeros((N,2))

    for n in range(N):

        s=np.sqrt(np.power(X[n][0]-X[:,0],2)+np.power(X[n][1]-X[:,1],2))
        ps=psi(s,beta)

        a=np.sum((V-V[n])*np.array([ps]).T,axis=0)/N
        #행렬의 각 행별로 array에 저장된 scalar를 곱하려고 하려면 dimension이 같아야함
        #즉 여기선 n*2 행렬이 있고 거기에서 각각의 행에 곱할 scalar가 ps에 저장되어 있는데,
        #1dimension으로 n짜리 array로는 못하고, n*1 사이즈의 2차원 행렬이어야 함.

        k[n]=a
   
    return k

if __name__ == '__main__':
    
    N=int(input("N=?"))
    beta=float(input("beta=?"))
    T=int(input("T=?"))

    Pinit=np.random.rand(N,2)*200-100 #자동으로 모든 component에 100씩 빼짐
    Vinit=(np.random.rand(N,2)*65+5)*(np.random.randint(0,2,size=(N,2))*2-1) #np.array끼리 그냥 곱하면 component 사이의 곱

    P=np.array([Pinit])
    V=np.array([Vinit])

    Vimean=np.array(np.mean(Vinit,axis=0))

    print(P[0])
    print(V[0])
    print(Vimean)

    s=np.sqrt(np.power(Vinit[:,0],2)+np.power(Vinit[:,1],2))
    nVinit=np.copy(Vinit)/np.array([s]).T
    nV=np.array([nVinit])

    Vinorm=np.sum(np.power(s,2))
    Vnorm=np.array(Vinorm)

    #Rv(t) = 시간 t에서 v_i - v_j의 max

    maxi=0
    for i in range(N) :
        psmax = np.max(np.power(Vinit[i][0]-Vinit[:,0],2)+np.power(Vinit[i][1]-Vinit[:,1],2))
        maxi=max(maxi,psmax)

    Rv=np.zeros(T)
    Rv[0]=np.sqrt(maxi) 

    # P 위치 V 초기가ㅄ 설정, nV:V를 normalize -> 이거 np 사용하면 더 간단해질 수도 있을 듯

    h=0.025

    for t in range(1,T):
        # print ("start %dth loop" %t)
        
        Pnow=np.copy(P[t-1])
        Vnow=np.copy(V[t-1])

        # K1-K4가 runge kutta 에서 가ㄱ각 k별로 구해서 list로 만든 것.

        K1=np.array([Vnow,csm(Pnow,Vnow)])

        K2=np.array([Vnow+K1[1]*h/2,csm(Pnow+K1[0]*h/2,Vnow+K1[1]*h/2)])

        K3=np.array([Vnow+K2[1]*h/2,csm(Pnow+K2[0]*h/2,Vnow+K2[1]*h/2)])

        K4=np.array([Vnow+K3[1]*h,csm(Pnow+K3[0]*h,Vnow+K3[1]*h)])
        
        Pnext=np.copy(Pnow)
        Vnext=np.copy(Vnow)

        Pnext+=(K1[0]+2*K2[0]+2*K3[0]+K4[0])*h/6
        Vnext+=(K1[1]+2*K2[1]+2*K3[1]+K4[1])*h/6

        s=np.sqrt(np.power(Vnext[:,0],2)+np.power(Vnext[:,1],2))
        nVnext=np.copy(Vnext)/np.array([s]).T

        Vnnorm=np.sum(np.power(s,2))

        P=np.append(P,np.array([Pnext]),axis=0)
        V=np.append(V,np.array([Vnext]),axis=0)
        nV=np.append(nV,np.array([nVnext]),axis=0)
        Vnorm=np.append(Vnorm,Vnnorm)

        maxi=0
        for i in range(N) :
            psmax = np.max(np.power(Vnext[i][0]-Vnext[:,0],2)+np.power(Vnext[i][1]-Vnext[:,1],2))
            maxi=max(maxi,psmax)
        Rv[t]=np.sqrt(maxi) 


        if t % 500 ==0 :
            print("end %d" %t)
        
    # print ("end")
    print (Pnow)
    print (Vnow)
    print (Vimean)

    # print("Rv")
    # print(Rv)

    # print ("V0")
    # print(V[0])
    # print("V1")
    # print(V[1])
    # print("V3")
    # print(V[3])
    # print("V1000")
    # print(V[1000])
    # print("V1003")
    # print(V[1003])

    fig = plt.figure()


    # ax = fig.add_subplot(131) #(1,1,1) 대신에 (111)이라고 써도 돼서... 

    # bx = fig.add_subplot(132)
    
    # #bx.set_yscale('log') #bx.yscale('~')는 에러남

    # bx.plot(range(0,T),np.log(Vnorm))

    # cx = fig.add_subplot(133)


    ax = fig.add_subplot(121)

    cx = fig.add_subplot(122)


    ax.set_xlim([-1000,1000])
    ax.set_ylim([-1000,1000])

    scat=ax.scatter(P[0].T[0], P[0].T[1],s=20) #P[0].T[1] = (transpose of P[0])[1]
    scat.set_alpha(0.2)
    qax=ax.quiver(P[0].T[0],P[0].T[1],nV[0].T[0],nV[0].T[1],angles='xy',width=0.001,scale=70)

    ani = animation.FuncAnimation(fig,_update_plot,fargs=(fig,scat,qax),frames=T-1,interval=10,save_count=T-1)

    #interval이 너무 작으니깐 save가 안됨-파일을 열때 에러남.

    cx.set_title("max|v_i(t)-v_j(t)| in log scale")
    cx.set_yscale('log')
    cx.plot(range(0,T),Rv)

    plt.show()
    

#    ani.save('csm-ode45-simu.mp4')

    print("DONE")




# %%
