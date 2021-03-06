
#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import time

from matplotlib import animation, rc
from IPython.display import HTML

#np 참고 http://taewan.kim/post/numpy_cheat_sheet/

def _update_plot (i,fig,scat,qax,ax,reddot) :

    ax.set_title('time : %i' %i)
    
    if limT < i :
        Xmax = np.max(P[i,:,0])
        Ymax = np.max(P[i,:,1])
        Xmin = np.min(P[i,:,0])
        Ymin = np.min(P[i,:,1])

        xylen = max(Xmax-Xmin,Ymax-Ymin)
        ax.set_xlim([Xmin-xylen*0.15,Xmin+xylen*1.15])
        ax.set_ylim([Ymin-xylen*0.15,Ymin+xylen*1.15])


    scat.set_offsets(P[i])
    qax.set_offsets(P[i])

    qax.set_UVC(nV[i].T[0],nV[i].T[1])
    # qax.quiver(PPP[0],PPP[1],VVV[0],VVV[1],angles='xy') -> 이렇게 하면 화살표가 위에 새로 계속 찍힘. 만약 ax.clear() 혹은 .remove() 했으면 달라졌을지도 
    
    # print ('Frames:%d' %i)
    reddot.set_data(i,Rv[i])

    return scat,qax,ax,reddot

def psi(s,b):
    a = np.power((1+np.power(s,2)),-b)
    return a

def csmmp(X,V):

    K=np.zeros((N,2))

    for i in range(N):

        J= np.array(range(N)) != i 

        s=np.sqrt(np.power(X[i][0]-X[:,0],2)+np.power(X[i][1]-X[:,1],2))[J]
        
        ps=psi(s,alpha)

        a=np.sum((V[J]-V[i])*np.array([ps]).T,axis=0) * kapa
        #행렬의 각 행별로 array에 저장된 scalar를 곱하려고 하려면 dimension이 같아야함
        #즉 여기선 n*2 행렬이 있고 거기에서 각각의 행에 곱할 scalar가 ps에 저장되어 있는데,
        #1dimension으로 n짜리 array로는 못하고, n*1 사이즈의 2차원 행렬이어야 함.

        a/=N

        u=np.array([0.0,0.0])
        if i!=0 :
            u+=psi(np.sqrt(np.power(X[i-1,0]-X[i,0]-Z[i-1,0],2)+np.power(X[i-1,1]-X[i,1]-Z[i-1,1],2)),beta)*(X[i-1]-X[i]-Z[i-1])
        if i!=N-1 :
            u-=psi(np.sqrt(np.power(X[i,0]-X[i+1,0]-Z[i,0],2)+np.power(X[i,1]-X[i+1,1]-Z[i,1],2)),beta)*(X[i]-X[i+1]-Z[i])

        a+=M*u

        K[i]=a
   
    return K

def Curve (t) :
    
    # X = np.cos(t) 
    # Y = np.sin(t)

    X = 17/31 *np.sin(235/57 - 32 *t) + 19/17 *np.sin(192/55 - 30 *t) + 47/32 *np.sin(69/25 - 29 *t) + 35/26 *np.sin(75/34 - 27 *t) + 6/31 *np.sin(23/10 - 26 *t) + 35/43 *np.sin(10/33 - 25 *t) + 126/43 *np.sin(421/158 - 24 *t) + 143/57 *np.sin(35/22 - 22 *t) + 106/27 *np.sin(84/29 - 21 *t) + 88/25 *np.sin(23/27 - 20 *t) + 74/27 *np.sin(53/22 - 19 *t) + 44/53 *np.sin(117/25 - 18 *t) + 126/25 *np.sin(88/49 - 17 *t) + 79/11 *np.sin(43/26 - 16 *t) + 43/12 *np.sin(41/17 - 15 *t) + 47/27 *np.sin(244/81 - 14 *t) + 8/5 *np.sin(79/19 - 13 *t) + 373/46 *np.sin(109/38 - 12 *t) + 1200/31 *np.sin(133/74 - 11 *t) + 67/24 *np.sin(157/61 - 10 *t) + 583/28 *np.sin(13/8 - 8 *t) + 772/35 *np.sin(59/16 - 7 *t) + 3705/46 *np.sin(117/50 - 6 *t) + 862/13 *np.sin(19/8 - 5 *t) + 6555/34 *np.sin(157/78 - 3 *t) + 6949/13 *np.sin(83/27 - t) - 6805/54 *np.sin(2 *t + 1/145) - 5207/37 *np.sin(4 *t + 49/74) - 1811/58 *np.sin(9 *t + 55/43) - 63/20 *np.sin(23 *t + 2/23) - 266/177 *np.sin(28 *t + 13/18) - 2/21 *np.sin(31 *t + 7/16)
    Y = 70/37 *np.sin(65/32 - 32 *t) + 11/12 *np.sin(98/41 - 31 *t) + 26/29 *np.sin(35/12 - 30 *t) + 54/41 *np.sin(18/7 - 29 *t) + 177/71 *np.sin(51/19 - 27 *t) + 59/34 *np.sin(125/33 - 26 *t) + 49/29 *np.sin(18/11 - 25 *t) + 151/75 *np.sin(59/22 - 24 *t) + 52/9 *np.sin(118/45 - 22 *t) + 52/33 *np.sin(133/52 - 21 *t) + 37/45 *np.sin(61/14 - 20 *t) + 143/46 *np.sin(144/41 - 19 *t) + 254/47 *np.sin(19/52 - 18 *t) + 246/35 *np.sin(92/25 - 17 *t) + 722/111 *np.sin(176/67 - 16 *t) + 136/23 *np.sin(3/19 - 15 *t) + 273/25 *np.sin(32/21 - 13 *t) + 229/33 *np.sin(117/28 - 12 *t) + 19/4 *np.sin(43/11 - 11 *t) + 135/8 *np.sin(23/10 - 10 *t) + 205/6 *np.sin(33/23 - 8 *t) + 679/45 *np.sin(55/12 - 7 *t) + 101/8 *np.sin(11/12 - 6 *t) + 2760/59 *np.sin(40/11 - 5 *t) + 1207/18 *np.sin(21/23 - 4 *t) + 8566/27 *np.sin(39/28 - 3 *t) + 12334/29 *np.sin(47/37 - 2 *t) + 15410/39 *np.sin(185/41 - t) - 596/17 *np.sin(9 *t + 3/26) - 247/28 *np.sin(14 *t + 25/21) - 458/131 *np.sin(23 *t + 21/37) - 41/36 *np.sin(28 *t + 7/8)

    return X,Y


if __name__ == '__main__':
    
    N=int(input("N=?"))

    alpha=float(input("alpha=?"))
    beta=float(input("beta=? (if alpha = beta, input -1)"))

    if beta==-1.0 :
        beta = alpha

    kapa=float(input("kapa=?"))

    M=float(input("M=?"))

    if beta==-1 :
        beta = alpha

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

    #Rv(t) = 시간 t에서 v_i - v_j의 max

    maxi=0
    for i in range(N) :
        psmax = np.max(np.power(Vinit[i][0]-Vinit[:,0],2)+np.power(Vinit[i][1]-Vinit[:,1],2))
        maxi=max(maxi,psmax)

    Rv=np.zeros(T)
    Rv[0]=np.sqrt(maxi)

    #target pattern 만들기
    Domain=np.linspace(0,2*np.pi,N,endpoint=False)

    TagP=np.zeros((2,N))
    TagP[0],TagP[1]=Curve(Domain) #sin x 모양으로
    Z=np.zeros((N-1,2))
    
    for i in range(N-1) :
        Z[i]=TagP.T[i]-TagP.T[i+1] #i+1 th - i th 하면 그림이 뒤집혀 나옴

    print(Z)

    # P 위치 V 초기가ㅄ 설정, nV:V를 normalize -> 이거 np 사용하면 더 간단해질 수도 있을 듯

    h=0.025
    limT=T

    for t in range(1,T):
        # print ("start %dth loop" %t)
        
        Pnow=np.copy(P[t-1])
        Vnow=np.copy(V[t-1])

        # K1-K4가 runge kutta 에서 가ㄱ각 k별로 구해서 list로 만든 것.

        K1=np.array([Vnow,csmmp(Pnow,Vnow)])

        K2=np.array([Vnow+K1[1]*h/2,csmmp(Pnow+K1[0]*h/2,Vnow+K1[1]*h/2)])

        K3=np.array([Vnow+K2[1]*h/2,csmmp(Pnow+K2[0]*h/2,Vnow+K2[1]*h/2)])

        K4=np.array([Vnow+K3[1]*h,csmmp(Pnow+K3[0]*h,Vnow+K3[1]*h)])
        
        Pnext=np.copy(Pnow)
        Vnext=np.copy(Vnow)

        Pnext+=(K1[0]+2*K2[0]+2*K3[0]+K4[0])*h/6
        Vnext+=(K1[1]+2*K2[1]+2*K3[1]+K4[1])*h/6

        s=np.sqrt(np.power(Vnext[:,0],2)+np.power(Vnext[:,1],2))
        nVnext=np.copy(Vnext)/np.array([s]).T

        P=np.append(P,np.array([Pnext]),axis=0)
        V=np.append(V,np.array([Vnext]),axis=0)
        nV=np.append(nV,np.array([nVnext]),axis=0)

        maxi=0
        for i in range(N) :
            psmax = np.max(np.power(Vnext[i][0]-Vnext[:,0],2)+np.power(Vnext[i][1]-Vnext[:,1],2))
            maxi=max(maxi,psmax)
        Rv[t]=np.sqrt(maxi) 

        if t % 500 ==0 :
            print("end %d" %t)

        if t >= 500 and limT==T:
            if Rv[t] < 10 :
                limT=t


    # print ("end")
    print (Pnow)
    print (Vnow)
    print (Vimean)
    print (limT)
    print (Z)
    print (TagP)

    #그림을 어떻게 그릴지 : (Xmax-Xmin)+(Ymax-Ymin)이 제일 클 때 까지는 개중에 Xinf~Xsup,Yinf~Ysup으로
    #-> 뭔가 사이즈는 limsup같은 느낌으로 하고 중앙점은 움직임 따라가는 식으로 할 수 있을 것 같은데
    # 프레임 사이즈가 큰 차이 안나면 고정으로 가야지 덜 어지러울 듯.
    #아 몰라 이건 나중에 하고 가계부나 하자


    limT-=1

    fig = plt.figure()
    ax = fig.add_subplot(121)
    cx = fig.add_subplot(122)

    Xmax = np.max(P[:limT,:,0])
    Ymax = np.max(P[:limT,:,1])
    Xmin = np.min(P[:limT,:,0])
    Ymin = np.min(P[:limT,:,1])

    xylen=max(Xmax-Xmin,Ymax-Ymin)
    ax.set_xlim([Xmin-xylen*0.15,Xmin+xylen*1.15])
    ax.set_ylim([Ymin-xylen*0.15,Ymin+xylen*1.15])
    ax.set_aspect('equal')

    scat=ax.scatter(P[0].T[0], P[0].T[1],s=20,c=Domain) #P[0].T[1] = (transpose of P[0])[1]. 이유는 모르겠는데 앞번호일수록 보라, 뒷번호일수록 노랑으로 나옴.
    scat.set_alpha(0.2)
    qax=ax.quiver(P[0].T[0],P[0].T[1],nV[0].T[0],nV[0].T[1],angles='xy',width=0.001,scale=70)


    cx.set_title("max|v_i(t)-v_j(t)| in log scale")
    cx.set_yscale('log')
    
    cx.plot(range(0,T),Rv)
    reddot, =cx.plot([0],[Rv[0]],'r.') #왠지 모르겠는데 reddot뒤의 ,을 빼면 animation 부분에서 에러남. 

    ani = animation.FuncAnimation(fig,_update_plot,fargs=(fig,scat,qax,ax,reddot),frames=T-1,interval=10,save_count=T-1)

    #interval이 너무 작으니깐(fps가 너무 커지니깐) save가 안됨-파일을 열때 에러남.

      
    plt.show()
    
    pprr=int(input("Do you want to save? Yes=1, No=else "))
    
    if pprr==1 :
        curvename=input("Curve name=?")
        savename='csm-ode45-simu-'+curvename+'.mp4'
        print("Saving...")
        ani.save(savename,dpi=300,fps=60)

    print("DONE")




# %%
