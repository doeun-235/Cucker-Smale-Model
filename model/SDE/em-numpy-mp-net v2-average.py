
#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import time

from matplotlib import animation, rc
from IPython.display import HTML

import pandas as pd

import os
import glob

#np 참고 http://taewan.kim/post/numpy_cheat_sheet/

# def _update_plot (i,fig,scat,qax,ax,reddot,reddot2) :

#     global xylen

#     ax.set_title('alpha=%.3f,beta=%.3f \n K=%.2f,M=%.2f, L=%.2f \n mean(V_0)=(%.3f,%.3f) \n time : %d' %(alpha,beta,kapa,M,L,Vimean[0],Vimean[1],i))
    
#     if limT < i :

#         Xmin = np.min(P[i,:,0])
#         Ymin = np.min(P[i,:,1])

#         xylen=np.max(Pdiff[i:])

#         if xylen>=10*Pdiff[i] :
#             xylen=Pdiff[i] #이걸 좀 더 세련되게 하면 덜 울렁거릴 듯.

#         ax.set_xlim([Xmin-xylen*0.15,Xmin+xylen*1.15])
#         ax.set_ylim([Ymin-xylen*0.15,Ymin+xylen*1.15])
        
#         frame_v=((np.min(P[i+1,:,0])-np.min(P[i,:,0]))/h,(np.min(P[i+1,:,1])-np.min(P[i,:,1]))/h)
#         ax.set_xlabel('(frame_len)=%f\n (frame_v)=(%.3f,%.3f)\n vmean=(%.6f,%.6f)' %(xylen,frame_v[0],frame_v[1],np.mean(V[i,:,0]),np.mean(V[i,:,1]))) #globla 선언이 안되어서 꼬인다는데 이유는 모르겠음. 해주니깐 해결되긴 함



#     scat.set_offsets(P[i])
#     qax.set_offsets(P[i])

#     qax.set_UVC(nV[i].T[0],nV[i].T[1])
#     # qax.quiver(PPP[0],PPP[1],VVV[0],VVV[1],angles='xy') -> 이렇게 하면 화살표가 위에 새로 계속 찍힘. 만약 ax.clear() 혹은 .remove() 했으면 달라졌을지도 
    
#     # print ('Frames:%d' %i)
#     reddot.set_data(i,Vnrm[i])
#     reddot2.set_data(i,Xsnrm[i])
    
#     return scat,qax,ax,reddot,reddot2

def psi(s,b,LB):
    a = np.power((1+np.power(s,2)),-b) + LB
    return a

def csmmp(X,V):

    K=np.zeros((N,2))

    for i in range(N):

        J= A[i,:] == 1

        s=np.sqrt(np.power(X[i][0]-X[:,0],2)+np.power(X[i][1]-X[:,1],2))[J]
        
        ps=psi(s,alpha,psLB)

        a=np.sum((V[J]-V[i])*np.array([ps]).T,axis=0) * kapa
        #행렬의 각 행별로 array에 저장된 scalar를 곱하려고 하려면 dimension이 같아야함
        #즉 여기선 n*2 행렬이 있고 거기에서 각각의 행에 곱할 scalar가 ps에 저장되어 있는데,
        #1dimension으로 n짜리 array로는 못하고, n*1 사이즈의 2차원 행렬이어야 함.

        a/=N

        u=np.array([0.0,0.0])

        ss=np.sqrt(np.power(X[i][0]-Z[i][0]-X[:,0]+Z[:,0],2)+np.power(X[i][1]-Z[i][1]-X[:,1]+Z[:,1],2))[J]
        pss=psi(ss,beta,0)
        u=np.sum((X[J]-Z[J]-X[i]+Z[i])*np.array([pss]).T,axis=0) * M

        a+=u

        K[i]=a
   
    K=np.nan_to_num(K)

    return K

def brown(V) :

    W=np.zeros((N,2))

    for i in range(N) :
        J= A[i,:]==1
        W[i]=np.sum((V[J]-V[i]))*L

    return W

def dW(dt) :
    return np.random.normal(0,np.sqrt(dt))

def Curve (t) :
    
    # X = np.cos(t) 
    # Y = np.sin(t)

    X = 17/31 *np.sin(235/57 - 32 *t) + 19/17 *np.sin(192/55 - 30 *t) + 47/32 *np.sin(69/25 - 29 *t) + 35/26 *np.sin(75/34 - 27 *t) + 6/31 *np.sin(23/10 - 26 *t) + 35/43 *np.sin(10/33 - 25 *t) + 126/43 *np.sin(421/158 - 24 *t) + 143/57 *np.sin(35/22 - 22 *t) + 106/27 *np.sin(84/29 - 21 *t) + 88/25 *np.sin(23/27 - 20 *t) + 74/27 *np.sin(53/22 - 19 *t) + 44/53 *np.sin(117/25 - 18 *t) + 126/25 *np.sin(88/49 - 17 *t) + 79/11 *np.sin(43/26 - 16 *t) + 43/12 *np.sin(41/17 - 15 *t) + 47/27 *np.sin(244/81 - 14 *t) + 8/5 *np.sin(79/19 - 13 *t) + 373/46 *np.sin(109/38 - 12 *t) + 1200/31 *np.sin(133/74 - 11 *t) + 67/24 *np.sin(157/61 - 10 *t) + 583/28 *np.sin(13/8 - 8 *t) + 772/35 *np.sin(59/16 - 7 *t) + 3705/46 *np.sin(117/50 - 6 *t) + 862/13 *np.sin(19/8 - 5 *t) + 6555/34 *np.sin(157/78 - 3 *t) + 6949/13 *np.sin(83/27 - t) - 6805/54 *np.sin(2 *t + 1/145) - 5207/37 *np.sin(4 *t + 49/74) - 1811/58 *np.sin(9 *t + 55/43) - 63/20 *np.sin(23 *t + 2/23) - 266/177 *np.sin(28 *t + 13/18) - 2/21 *np.sin(31 *t + 7/16)
    Y = 70/37 *np.sin(65/32 - 32 *t) + 11/12 *np.sin(98/41 - 31 *t) + 26/29 *np.sin(35/12 - 30 *t) + 54/41 *np.sin(18/7 - 29 *t) + 177/71 *np.sin(51/19 - 27 *t) + 59/34 *np.sin(125/33 - 26 *t) + 49/29 *np.sin(18/11 - 25 *t) + 151/75 *np.sin(59/22 - 24 *t) + 52/9 *np.sin(118/45 - 22 *t) + 52/33 *np.sin(133/52 - 21 *t) + 37/45 *np.sin(61/14 - 20 *t) + 143/46 *np.sin(144/41 - 19 *t) + 254/47 *np.sin(19/52 - 18 *t) + 246/35 *np.sin(92/25 - 17 *t) + 722/111 *np.sin(176/67 - 16 *t) + 136/23 *np.sin(3/19 - 15 *t) + 273/25 *np.sin(32/21 - 13 *t) + 229/33 *np.sin(117/28 - 12 *t) + 19/4 *np.sin(43/11 - 11 *t) + 135/8 *np.sin(23/10 - 10 *t) + 205/6 *np.sin(33/23 - 8 *t) + 679/45 *np.sin(55/12 - 7 *t) + 101/8 *np.sin(11/12 - 6 *t) + 2760/59 *np.sin(40/11 - 5 *t) + 1207/18 *np.sin(21/23 - 4 *t) + 8566/27 *np.sin(39/28 - 3 *t) + 12334/29 *np.sin(47/37 - 2 *t) + 15410/39 *np.sin(185/41 - t) - 596/17 *np.sin(9 *t + 3/26) - 247/28 *np.sin(14 *t + 25/21) - 458/131 *np.sin(23 *t + 21/37) - 41/36 *np.sin(28 *t + 7/8)

    return X,Y


if __name__ == '__main__':
    
    # N=int(input("N=?"))

    # alpha=float(input("alpha=?"))
    # beta=float(input("beta=? (if alpha = beta, input -1)"))

    # if beta==-1.0 :
    #     beta = alpha

    N=30
    alpha=0.25
    beta=alpha

    psLB=0.3

    # kapa=float(input("kapa=?"))
    kapa=100

    # M=float(input("M=?"))
    M=5

    # L=float(input("L=?"))
    L=0.1

    # T=int(input("T=?"))
    T=1500

    #target pattern 만들기
    Domain=np.linspace(0,2*np.pi,N,endpoint=False)

    TagP=np.zeros((2,N))
    TagP[0],TagP[1]=Curve(Domain) #주어진 curve 모양으로
    Z=np.zeros((N,2))
    Z=TagP.T

    #P, V 초기가ㅄ 설정

    Pinit=np.random.rand(N,2)*400-200 #자동으로 모든 component에 100씩 빼짐
    # Pinit[int(round(N/2,0))]=-np.array(np.sum(Pinit[:int(round(N/2,0))]-Z[:int(round(N/2,0))],axis=0))+Z[int(round(N/2,0))]
    # Pinit[N-1]=-np.array(np.sum(Pinit[int(round(N/2,0))+1:N-1]-Z[int(round(N/2,0))+1:N-1],axis=0))+Z[N-1]
    Xsimean=np.array(np.mean(Pinit[:]-Z[:],axis=0))

    print("\nX*imean={}".format(Xsimean))

    Vinit=(np.random.rand(N,2)*50)-25 #np.array끼리 그냥 곱하면 component 사이의 곱
    Vimean=np.array(np.mean(Vinit[:],axis=0))
    Vinit-=Vimean
    Vimean=np.array(np.mean(Vinit[:],axis=0))

    print("Vimean={}\n".format(Vimean))

    Pdiff=np.zeros(T)

    # Pdiff[0]=max(np.max(Pinit[:,0])-np.min(Pinit[:,0]),np.max(Pinit[:,1])-np.min(Pinit[:,1]))

    # print(Pinit)
    # print(Vinit)

    #Vnrm(t) = 시간 t에서 Vnrm = sum i |v_i|^2
    #nV(t) = V를 normalize
    s=np.sqrt(np.power(Vinit[:,0],2)+np.power(Vinit[:,1],2))
    nVinit=np.copy(Vinit)/np.array([s]).T
    nV=np.array([nVinit])

    Vnrmi=np.sum(np.power(s,2))

    #Xsnrm(t) = sum i |x_i*|^2
    
    Xsnrmi=np.sum(np.power(Pinit[:,0]-Z[:,0],2)+np.power(Pinit[:,1]-Z[:,1],2))

    # P 위치 V 초기가ㅄ 설정, nV:V를 normalize -> 이거 np 사용하면 더 간단해질 수도 있을 듯

    A=np.zeros((30,30))

    net = 1 
    if net==0 :
        A[:,:]=1
        for i in range(30) :
            A[i,i]=0
    elif net==1 :
        for i in range(29) :
            A[i,i+1] = 1

        A[0,1:30]=1
        A[0:14,14]=1
        A[14,15:30]=1
        A[:29,29]=1

        A=A+A.T

    # A : adjacency matrix

    h=0.025
    # limT=T
    Trial=500
    cut=10

    #이렇게 계속 append 하는 방식 말고 cum 미리 array 만들어 놓는게 빠른지 함 확인 해봐야 + 공간 어떤 방식이 더 많이 확보 가능한가

    Pcum=np.array([np.zeros((T,N,2))])
    Vcum=np.array([np.zeros((T,N,2))])
    nVcum=np.array([np.zeros((T,N,2))])

    Vnrmcum=np.array([np.zeros(T)])
    Xsnrmcum=np.array([np.zeros(T)])
    dBcum=np.array([np.zeros(T)])

    #지금 수정하기 귀찮다고 대충 했더니 cum ndarray들은 [0]=0 이고 [1]부터 가ㅄ이들어가는 상황 

    for trial in range (0,Trial):

        P=np.zeros((T,N,2))
        V=np.zeros((T,N,2))
        nV=np.zeros((T,N,2))
        Vnrm=np.zeros(T)
        Xsnrm=np.zeros(T)
        dB=np.zeros(T)
                
        P[0]=np.array([Pinit])
        V[0]=np.array([Vinit])
        nV[0]=np.array([nVinit])
        Vnrm[0]=Vnrmi
        Xsnrm[0]=Xsnrmi

        for t in range(1,T):
            # print ("start %dth loop" %t)
            
            Pnow=np.copy(P[t-1])
            Vnow=np.copy(V[t-1])

            # euler maruyama .

            K=np.array([Vnow,csmmp(Pnow,Vnow)])

            # if np.isnan(K).any() == True :
            #     print("NaN-K")
            #     print(t)
            #     print (Pnow)
            #     print (Vnow)
            #     print (K)
            #     break

            Pnext=np.copy(Pnow)
            Vnext=np.copy(Vnow)

            Pnext+=K[0]*h
            Vnext+=K[1]*h

            Br=brown(Vnow)

            dBt=dW(h)
            Vnext+=Br*dBt

            # if np.isnan(Pnext).any() == True :
            #     print("NaN-P")
            #     print (Pnow)
            #     print (Pnext)
            #     break
            
            # if np.isnan(Vnext).any() == True :
            #     print ("NaN-V")
            #     print (Pnow)
            #     print (Vnow)
            #     print ("next")
            #     print (Pnext)
            #     print (Vnext)
            #     print ("Br")
            #     print (Br)
            #     break

            Pnext=np.nan_to_num(Pnext)
            Vnext=np.nan_to_num(Vnext)

            s=np.sqrt(np.power(Vnext[:,0],2)+np.power(Vnext[:,1],2))
            nVnext=np.copy(Vnext)/np.array([s]).T


            # if np.isnan(nVnext).any() == True :
            #     print ("NaN-nV")
            #     print (Pnow)
            #     print (Vnow)
            #     print ("next")
            #     print (Pnext)
            #     print (Vnext)
            #     print ("Br")
            #     print (Br)ㅍ
            #     print (nVnext)
            #     break


            nVnext=np.nan_to_num(nVnext)

            P[t]=np.array([Pnext])
            V[t]=np.array([Vnext])
            nV[t]=np.array([nVnext])
            Vnrm[t]=np.sum(np.power(s,2))
            Xsnrm[t]=np.sum(np.power(Pnext[:,0]-Z[:,0],2)+np.power(Pnext[:,1]-Z[:,1],2))
            dB[t]=dBt


        Pcum=np.append(Pcum,np.array([P]),axis=0)
        Vcum=np.append(Vcum,np.array([V]),axis=0)
        nVcum=np.append(nVcum,np.array([nV]),axis=0)
        Vnrmcum=np.append(Vnrmcum,np.array([Vnrm]),axis=0)
        Xsnrmcum=np.append(Xsnrmcum,np.array([Xsnrm]),axis=0)
        dBcum=np.append(dBcum,np.array([dB]),axis=0)

        if trial % 10 ==0 :
            print("end %d" %trial)


    print ("End\n")

    medVnrm=np.median(Vnrmcum[1:],axis=0)
    stdVnrm=np.std(Vnrmcum[1:],axis=0)

    select=0
    while select==0:
        checkI= Vnrmcum-medVnrm > cut*stdVnrm
        selcount = np.sum(checkI,axis=1)
        selI = selcount > 0
        
        print ('cut:%.2f select:%d' %(cut,np.sum(selI)))
        
        if np.sum(selI)<=Trial*0.01 :
            cut-=0.25
        else :
            select=1
            print('select average')
            print(np.mean(selcount))
            break
    
    # print (P)
    # print (V)
    # print ("Pcum")
    # print (Pcum)
    # print ("Vcum")
    # print (Vcum)
    # print ("Vnrmcum")
    # print (Vnrmcum)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    bx = fig.add_subplot(222)
    cx = fig.add_subplot(223)
    dx = fig.add_subplot(224)

    #trial=1에서 v1,1 ... vn,1 의 그래프
    ax.set_title("velocity of particle 1 ~30 of\nthe 1st realization in x-axis")
    ax.set_xlabel('K=%.2f,M=%.2f, sigma=%.2f' %(kapa,M,L))

    for i in range(N):
        ax.plot(range(0,T),Vcum[1,:,i,0],linewidth=0.5)

    #모든 trial에서 v1,1의 그래프
    bx.set_title("velocity of particle 1 in\nx-axis for every realization")
    bx.set_xlabel('K=%.2f,M=%.2f, sigma=%.2f' %(kapa,M,L))

    for i in range(Trial) :
        bx.plot(range(0,T),Vcum[i+1,:,0,0],linewidth=0.5)

    #E(Vnrm), E(X*nrm) 그래프
    
    EVnrm=np.mean(Vnrmcum[1:],axis=0)
    EXsnrm=np.mean(Xsnrmcum[1:],axis=0)
    VVnrmcum=np.sort(Vnrmcum[1:],axis=0)
    print(VVnrmcum)

    cx.set_title(r'$\mathrm{\mathbb{E}}||v(t)||^2$')
    cx.set_xlabel('K=%.2f,M=%.2f, sigma=%.2f' %(kapa,M,L))

    # cx.set_yscale('log')
    
    cx.plot(range(0,T),EVnrm,label=r'$\mathrm{\mathbb{E}}$')
    cx.plot(range(0,T),medVnrm,c='paleturquoise',linewidth=0.7,alpha=0.7,label='median')
    cx.plot(range(0,T),VVnrmcum[int(Trial*0.85),:],c='lightsteelblue',linewidth=0.5,alpha=0.7,label='85%')
    cx.legend(loc='best')

    dx.set_title(r'$\mathrm{\mathbb{E}}\sum |\bar{x_i}(t)|^2$')
    # dx.set_yscale('log')
    dx.set_xlabel('K=%.2f,M=%.2f, sigma=%.2f' %(kapa,M,L))

    dx.plot(range(0,T),EXsnrm)

    fig.tight_layout()

    # plt.show()

    # pprr=(input("Do you want to save? Yes=else, No=0 "))

    pprr=1    
    if pprr!='0' :
        plt.savefig('graph_k%.2fm%.2fs%.2fnet%dtrial%d.png' %(kapa,M,L,net,Trial),dpi=200)

    #파일 저장은 어떻게 할지. 데이터 P,V,nV : Trial*T*N*2 / Vnrm, Xsnrm, dB : Trial*T
    #전자 : numpy를 썼을때 data모양 재조합하기 쉬운 방식으로 모양을 바꿔주면 될 듯.
    
    print("image saved")

    savemode=0 #0은 아웃라이어만, 1은 전체

    if savemode==0 :

        print("data reshaping")

        selP = Pcum[selI,:,:,:].reshape(-1,T*N*2)
        selV = Vcum[selI,:,:,:].reshape(-1,T*N*2)
        selnV = nVcum[selI,:,:,:].reshape(-1,T*N*2)

        selVnrm = Vnrmcum[selI,:]
        selXsnrm = Xsnrmcum[selI,:]
        seldB = dBcum[selI,:]

        selP_df=pd.DataFrame(selP)
        selV_df=pd.DataFrame(selV)
        selnV_df=pd.DataFrame(selnV)

        selVnrm_df=pd.DataFrame(selVnrm)
        selXsnrm_df=pd.DataFrame(selXsnrm)
        seldB_df=pd.DataFrame(seldB)

        selcount_df=pd.DataFrame(selcount)

        print("saving...")

        selP_df.to_csv('selP_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        selV_df.to_csv('selV_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        selnV_df.to_csv('selnV_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        selVnrm_df.to_csv('selVnrm_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        selXsnrm_df.to_csv('selXsnrm_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        seldB_df.to_csv('seldB_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        selcount_df.to_csv('selcount_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")

    else :
        print("data reshaping")

        reP = Pcum.reshape(-1,T*N*2)
        reV = Vcum.reshape(-1,T*N*2)
        renV = nVcum.reshape(-1,T*N*2)

        selVnrm = Vnrmcum[:,:]
        selXsnrm = Xsnrmcum[selI,:]
        seldB = dBcum[selI,:]

        P_df=pd.DataFrame(reP)
        V_df=pd.DataFrame(reV)
        nV_df=pd.DataFrame(renV)

        Vnrm_df=pd.DataFrame(Vnrmcum)
        Xsnrm_df=pd.DataFrame(Xsnrmcum)
        
        print("saving...")

        P_df.to_csv('P_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        V_df.to_csv('V_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        nV_df.to_csv('nV_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        Vnrm_df.to_csv('Vnrm_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
        Xsnrm_df.to_csv('Xsnrm_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")

    # setting 을 df로 만들어서 입출력 편하게 저장하는 법 확인하기. 
    # dB, Pinit, Vinit, Z, A

    # N=30
    # alpha=0.25
    # beta=alpha
    # psLB=0.3
    # kapa=100
    # M=5
    # L=0.1
    # T=1500
    # h=0.025
    # Trial=500
    # cut=8
    # net

    # 이정도 인 듯.

    dB_df=pd.DataFrame(dBcum)
    Pinit_df=pd.DataFrame(Pinit)
    Vinit_df=pd.DataFrame(Vinit)
    Z_df=pd.DataFrame(Z)
    A_df=pd.DataFrame(A)

    dB_df.to_csv('dB_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
    Pinit_df.to_csv('Pinit_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
    Vinit_df.to_csv('Vinit_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
    Z_df.to_csv('Z_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")
    A_df.to_csv('A_k%.2fm%.2fs%.2fnet%dtrial%d_cut%0.2fstd.csv' %(kapa,M,L,net,Trial,cut),index=False,sep="\t")

    print("DONE")
    


# %%
