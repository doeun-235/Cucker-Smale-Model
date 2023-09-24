
#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import time

from matplotlib import animation, rc
from IPython.display import HTML

from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import os
import glob
import pandas as pd


#np 참고 http://taewan.kim/post/numpy_cheat_sheet/

def _update_plot (i,fig,scat,qax,ax,reddot,reddot2) :

    global xylen

    ax.set_title(r'$\alpha=%.3f,\beta=%.3f$'%(alpha,beta)+'\n'+ r'$K=%.2f,M=%.2f,\sigma=%.4f$'%(K,M,L)+'\n'+r'$v_0^{ave}=(%.3f,%.3f)$'%(Vimean[0],Vimean[1])+'\n'+'time : %d' %i)
    
    ax.set_xlabel('(frame_len)=%.3f\n dBt=%.3f' %(xylen,dB[i]))

    if limT < i :

        Xmin = np.min(P[i,:,0])
        Ymin = np.min(P[i,:,1])

        xylen=np.max(Pdiff[i:])

        if xylen>=10*Pdiff[i] :
            xylen=Pdiff[i] #이걸 좀 더 세련되게 하면 덜 울렁거릴 듯.

        ax.set_xlim([Xmin-xylen*0.15,Xmin+xylen*1.15])
        ax.set_ylim([Ymin-xylen*0.15,Ymin+xylen*1.15])
        
        frame_v=((np.min(P[i+1,:,0])-np.min(P[i,:,0]))/h,(np.min(P[i+1,:,1])-np.min(P[i,:,1]))/h)
        ax.set_xlabel('(frame_len)=%.3f\n (frame_v)=(%.3f,%.3f)\n'%(xylen,frame_v[0],frame_v[1]) +r'$v_t^{ave}=(%.6f,%.6f)$' %(np.mean(V[i,:,0]),np.mean(V[i,:,1]))+'\ndBt=%.3f' %dB[i]) #globla 선언이 안되어서 꼬인다는데 이유는 모르겠음. 해주니깐 해결되긴 함

    
    scat.set_offsets(P[i])
    qax.set_offsets(P[i])

    qax.set_UVC(nV[i].T[0],nV[i].T[1])
    # qax.quiver(PPP[0],PPP[1],VVV[0],VVV[1],angles='xy') -> 이렇게 하면 화살표가 위에 새로 계속 찍힘. 만약 ax.clear() 혹은 .remove() 했으면 달라졌을지도 
    
    # print ('Frames:%d' %i)
    reddot.set_data(i,Vnrm[i])
    reddot2.set_data(i,Xbnrm[i])
    
    return scat,qax,ax,reddot,reddot2

def phiEest(ssq,b):
    phE=(np.power(1+ssq,1-b)-1)/(1-b)
    return phE

def psi(s,b,LB):
    a = np.power((1+np.power(s,2)),-b) + LB
    return a

def csmmp(X,V):

    Kem=np.zeros((N,2))

    for i in range(N):

        J= A_ps[i,:] == 1

        s=np.sqrt(np.power(X[i][0]-X[:,0],2)+np.power(X[i][1]-X[:,1],2))[J]
        
        ps=psi(s,alpha,psLB)

        a=np.sum((V[J]-V[i])*np.array([ps]).T,axis=0) * K
        #행렬의 각 행별로 array에 저장된 scalar를 곱하려고 하려면 dimension이 같아야함
        #즉 여기선 n*2 행렬이 있고 거기에서 각각의 행에 곱할 scalar가 ps에 저장되어 있는데,
        #1dimension으로 n짜리 array로는 못하고, n*1 사이즈의 2차원 행렬이어야 함.

        a/=N

        J= A_ph[i,:] == 1

        u=np.array([0.0,0.0])

        ss=np.sqrt(np.power(X[i][0]-Z[i][0]-X[:,0]+Z[:,0],2)+np.power(X[i][1]-Z[i][1]-X[:,1]+Z[:,1],2))[J]
        pss=psi(ss,beta,phLB)
        u=np.sum((X[J]-Z[J]-X[i]+Z[i])*np.array([pss]).T,axis=0) * M

        a+=u

        Kem[i]=a
   
    Kem=np.nan_to_num(Kem)

    return Kem

def brown(V) :

    W=np.zeros((N,2))

    for i in range(N) :
        J= A_b[i,:]==1
        W[i]=np.sum((V[J]-V[i]))*L

    return W

def dW(dt) :
    return np.random.normal(0,np.sqrt(dt))

def Curve (dom) :

    num=N
    if jump==1:
        num*=2

    t=np.linspace(0,dom,num,endpoint=False)

    if curvename=='circle':
        X = np.cos(t) 
        Y = np.sin(t)
    elif curvename=='pi':
        X = 17/31 *np.sin(235/57 - 32 *t) + 19/17 *np.sin(192/55 - 30 *t) + 47/32 *np.sin(69/25 - 29 *t) + 35/26 *np.sin(75/34 - 27 *t) + 6/31 *np.sin(23/10 - 26 *t) + 35/43 *np.sin(10/33 - 25 *t) + 126/43 *np.sin(421/158 - 24 *t) + 143/57 *np.sin(35/22 - 22 *t) + 106/27 *np.sin(84/29 - 21 *t) + 88/25 *np.sin(23/27 - 20 *t) + 74/27 *np.sin(53/22 - 19 *t) + 44/53 *np.sin(117/25 - 18 *t) + 126/25 *np.sin(88/49 - 17 *t) + 79/11 *np.sin(43/26 - 16 *t) + 43/12 *np.sin(41/17 - 15 *t) + 47/27 *np.sin(244/81 - 14 *t) + 8/5 *np.sin(79/19 - 13 *t) + 373/46 *np.sin(109/38 - 12 *t) + 1200/31 *np.sin(133/74 - 11 *t) + 67/24 *np.sin(157/61 - 10 *t) + 583/28 *np.sin(13/8 - 8 *t) + 772/35 *np.sin(59/16 - 7 *t) + 3705/46 *np.sin(117/50 - 6 *t) + 862/13 *np.sin(19/8 - 5 *t) + 6555/34 *np.sin(157/78 - 3 *t) + 6949/13 *np.sin(83/27 - t) - 6805/54 *np.sin(2 *t + 1/145) - 5207/37 *np.sin(4 *t + 49/74) - 1811/58 *np.sin(9 *t + 55/43) - 63/20 *np.sin(23 *t + 2/23) - 266/177 *np.sin(28 *t + 13/18) - 2/21 *np.sin(31 *t + 7/16)
        Y = 70/37 *np.sin(65/32 - 32 *t) + 11/12 *np.sin(98/41 - 31 *t) + 26/29 *np.sin(35/12 - 30 *t) + 54/41 *np.sin(18/7 - 29 *t) + 177/71 *np.sin(51/19 - 27 *t) + 59/34 *np.sin(125/33 - 26 *t) + 49/29 *np.sin(18/11 - 25 *t) + 151/75 *np.sin(59/22 - 24 *t) + 52/9 *np.sin(118/45 - 22 *t) + 52/33 *np.sin(133/52 - 21 *t) + 37/45 *np.sin(61/14 - 20 *t) + 143/46 *np.sin(144/41 - 19 *t) + 254/47 *np.sin(19/52 - 18 *t) + 246/35 *np.sin(92/25 - 17 *t) + 722/111 *np.sin(176/67 - 16 *t) + 136/23 *np.sin(3/19 - 15 *t) + 273/25 *np.sin(32/21 - 13 *t) + 229/33 *np.sin(117/28 - 12 *t) + 19/4 *np.sin(43/11 - 11 *t) + 135/8 *np.sin(23/10 - 10 *t) + 205/6 *np.sin(33/23 - 8 *t) + 679/45 *np.sin(55/12 - 7 *t) + 101/8 *np.sin(11/12 - 6 *t) + 2760/59 *np.sin(40/11 - 5 *t) + 1207/18 *np.sin(21/23 - 4 *t) + 8566/27 *np.sin(39/28 - 3 *t) + 12334/29 *np.sin(47/37 - 2 *t) + 15410/39 *np.sin(185/41 - t) - 596/17 *np.sin(9 *t + 3/26) - 247/28 *np.sin(14 *t + 25/21) - 458/131 *np.sin(23 *t + 21/37) - 41/36 *np.sin(28 *t + 7/8)
    elif curvename=='b.simpson':
        X = ((-5/37 *np.sin(61/42 - 8 *t) - 112/41 *np.sin(36/23 - 7 *t) - 62/37 *np.sin(14/9 - 6 *t) - 31/7 *np.sin(69/44 - 5 *t) - 275/13 *np.sin(47/30 - 3 *t) - 23/38 *np.sin(48/31 - 2 *t) + 461/10 *np.sin(t + 107/68) + 8/23 *np.sin(4 *t + 179/38) - 2345/18) *np.heaviside(71 *np.pi - t,0.5) *np.heaviside(t - 67 *np.pi,0.5) + (-41/74 *np.sin(112/75 - 17 *t) - 274/37 *np.sin(17/11 - 11 *t) - 907/30 *np.sin(36/23 - 5 *t) + 623/15 *np.sin(t + 47/30) + 684/29 *np.sin(2 *t + 212/45) + 3864/25 *np.sin(3 *t + 63/40) + 1513/78 *np.sin(4 *t + 19/12) + 269/45 *np.sin(6 *t + 14/9) + 66/25 *np.sin(7 *t + 155/33) + 121/35 *np.sin(8 *t + 27/17) + 44/13 *np.sin(9 *t + 35/22) + 71/7 *np.sin(10 *t + 30/19) + 87/40 *np.sin(12 *t + 57/37) + 61/62 *np.sin(13 *t + 47/28) + 43/65 *np.sin(14 *t + 31/20) + 54/37 *np.sin(15 *t + 127/27) + 56/27 *np.sin(16 *t + 30/19) + 11843/35) *np.heaviside(67 *np.pi - t,0.5) *np.heaviside(t - 63 *np.pi,0.5) + (-119/55 *np.sin(65/43 - 13 *t) - 78/11 *np.sin(82/55 - 12 *t) - 25/31 *np.sin(36/23 - 11 *t) - 70/9 *np.sin(31/20 - 9 *t) - 707/65 *np.sin(47/30 - 7 *t) - 953/22 *np.sin(45/29 - 5 *t) + 962/15 *np.sin(t + 113/72) + 1091/30 *np.sin(2 *t + 212/45) + 3177/26 *np.sin(3 *t + 27/17) + 1685/22 *np.sin(4 *t + 43/27) + 143/27 *np.sin(6 *t + 49/32) + 203/27 *np.sin(8 *t + 27/17) + 411/37 *np.sin(10 *t + 50/31) + 65/27 *np.sin(14 *t + 33/20) + 8/19 *np.sin(15 *t + 13/7) + 3/11 *np.sin(16 *t + 43/33) - 11597/35) *np.heaviside(63 *np.pi - t,0.5) *np.heaviside(t - 59 *np.pi,0.5) + (-1/7 *np.sin(41/81 - 30 *t) - 8/27 *np.sin(3/28 - 28 *t) - 10/23 *np.sin(3/26 - 26 *t) + 2377/13 *np.sin(t + 33/28) + 43/15 *np.sin(2 *t + 26/7) + 131/18 *np.sin(3 *t + 3/25) + 45/41 *np.sin(4 *t + 105/32) + 43/14 *np.sin(5 *t + 87/23) + 135/136 *np.sin(6 *t + 51/20) + 51/14 *np.sin(7 *t + 118/43) + 19/18 *np.sin(8 *t + 23/18) + 49/23 *np.sin(9 *t + 25/12) + 14/19 *np.sin(10 *t + 63/55) + 54/49 *np.sin(11 *t + 68/41) + 32/37 *np.sin(12 *t + 30/29) + 5/12 *np.sin(13 *t + 43/24) + 34/45 *np.sin(14 *t + 15/17) + 13/30 *np.sin(15 *t + 67/23) + 21/31 *np.sin(16 *t + 43/60) + 25/62 *np.sin(17 *t + 89/34) + 9/20 *np.sin(18 *t + 11/26) + 4/17 *np.sin(19 *t + 55/28) + 26/51 *np.sin(20 *t + 4/17) + 2/33 *np.sin(21 *t + 247/62) + 14/31 *np.sin(22 *t + 9/44) + 5/26 *np.sin(23 *t + 113/34) + 9/17 *np.sin(24 *t + 3/10) + 4/25 *np.sin(25 *t + 99/32) + 6/23 *np.sin(27 *t + 548/183) + 10/33 *np.sin(29 *t + 129/37) + 5/12 *np.sin(31 *t + 127/39) - 9719/87) *np.heaviside(59 *np.pi - t,0.5) *np.heaviside(t - 55 *np.pi,0.5) + (228/65 *np.sin(t + 116/33) + 353/40 *np.sin(2 *t + 33/19) + 107/24 *np.sin(3 *t + 58/33) + 58/21 *np.sin(4 *t + 519/130) + 19/15 *np.sin(5 *t + 45/37) + 13/12 *np.sin(6 *t + 145/38) + 43/42 *np.sin(7 *t + 25/99) + 11/19 *np.sin(8 *t + 105/44) + 203/19) *np.heaviside(55 *np.pi - t,0.5) *np.heaviside(t - 51 *np.pi,0.5) + (-23/10 *np.sin(22/17 - 4 *t) - 159/17 *np.sin(156/125 - 3 *t) + 523/112 *np.sin(t + 80/21) + 111/23 *np.sin(2 *t + 25/24) + 92/79 *np.sin(5 *t + 57/32) + 58/37 *np.sin(6 *t + 159/35) + 18/31 *np.sin(7 *t + 27/43) - 7563/28) *np.heaviside(51 *np.pi - t,0.5) *np.heaviside(t - 47 *np.pi,0.5) + (-76/17 *np.sin(42/41 - 14 *t) - 154/31 *np.sin(37/38 - 11 *t) + 10820/41 *np.sin(t + 25/34) + 1476/31 *np.sin(2 *t + 36/19) + 595/12 *np.sin(3 *t + 67/43) + 3568/67 *np.sin(4 *t + 282/77) + 974/59 *np.sin(5 *t + 40/19) + 427/18 *np.sin(6 *t + 47/25) + 454/23 *np.sin(7 *t + 20/27) + 41/40 *np.sin(8 *t + 9/2) + 139/22 *np.sin(9 *t + 99/26) + 276/37 *np.sin(10 *t + 37/29) + 113/25 *np.sin(12 *t + 61/30) + 37/29 *np.sin(13 *t + 37/31) + 51/19 *np.sin(15 *t + 127/34) + 115/72 *np.sin(16 *t + 7/38) + 162/43 *np.sin(17 *t + 67/21) + 26/33 *np.sin(18 *t + 194/45) - 3614/99) *np.heaviside(47 *np.pi - t,0.5) *np.heaviside(t - 43 *np.pi,0.5) + (347/17 *np.sin(t + 3/13) + 9951/41) *np.heaviside(43 *np.pi - t,0.5) *np.heaviside(t - 39 *np.pi,0.5) + (760/29 *np.sin(t + 23/25) - 6059/28) *np.heaviside(39 *np.pi - t,0.5) *np.heaviside(t - 35 *np.pi,0.5) + (-106/41 *np.sin(7/13 - 18 *t) - 55/38 *np.sin(13/29 - 16 *t) - 173/19 *np.sin(34/29 - 7 *t) - 484/31 *np.sin(13/16 - 5 *t) - 1193/17 *np.sin(97/83 - 2 *t) + 6885/26 *np.sin(t + 41/48) + 99/5 *np.sin(3 *t + 5/16) + 751/36 *np.sin(4 *t + 73/18) + 129/40 *np.sin(6 *t + 83/18) + 327/31 *np.sin(8 *t + 17/23) + 498/47 *np.sin(9 *t + 123/88) + 298/49 *np.sin(10 *t + 54/25) + 82/15 *np.sin(11 *t + 153/35) + 106/27 *np.sin(12 *t + 3/32) + 171/43 *np.sin(13 *t + 433/173) + 36/11 *np.sin(14 *t + 98/33) + 39/22 *np.sin(15 *t + 97/25) + 68/37 *np.sin(17 *t + 157/34) - 227/29) *np.heaviside(35 *np.pi - t,0.5) *np.heaviside(t - 31 *np.pi,0.5) + (-2/15 *np.sin(66/47 - 14 *t) - 45/23 *np.sin(5/9 - 11 *t) - 151/43 *np.sin(13/32 - 8 *t) - 31/36 *np.sin(24/19 - 7 *t) + 2121/32 *np.sin(t + 45/38) + 2085/47 *np.sin(2 *t + 299/88) + 1321/43 *np.sin(3 *t + 72/25) + 557/37 *np.sin(4 *t + 74/21) + 205/17 *np.sin(5 *t + 27/23) + 13/9 *np.sin(6 *t + 113/32) + 35/17 *np.sin(9 *t + 7/22) + 93/26 *np.sin(10 *t + 112/25) + 11/14 *np.sin(12 *t + 58/17) + 8/15 *np.sin(13 *t + 47/30) + 33/20 *np.sin(15 *t + 32/25) + 31/94 *np.sin(16 *t + 192/59) + 35/31 *np.sin(17 *t + 51/77) + 9473/34) *np.heaviside(31 *np.pi - t,0.5) *np.heaviside(t - 27 *np.pi,0.5) + (-33/29 *np.sin(27/55 - 12 *t) + 388/13 *np.sin(t + 5/13) + 2087/30 *np.sin(2 *t + 193/55) + 1311/49 *np.sin(3 *t + 133/30) + 993/41 *np.sin(4 *t + 134/31) + 175/17 *np.sin(5 *t + 73/29) + 83/23 *np.sin(6 *t + 28/33) + 9/19 *np.sin(7 *t + 73/36) + 101/32 *np.sin(8 *t + 57/28) + 51/25 *np.sin(9 *t + 106/39) + 47/28 *np.sin(10 *t + 129/47) + 17/29 *np.sin(11 *t + 33/17) + 27/22 *np.sin(13 *t + 155/86) + 108/65 *np.sin(14 *t + 8/27) + 9/16 *np.sin(15 *t + 44/13) + 11/14 *np.sin(16 *t + 3/19) + 11/23 *np.sin(17 *t + 106/23) + 9/64 *np.sin(18 *t + 97/22) - 10004/35) *np.heaviside(27 *np.pi - t,0.5) *np.heaviside(t - 23 *np.pi,0.5) + (-18/13 *np.sin(7/50 - 18 *t) - 7/5 *np.sin(1/10 - 16 *t) - 51/25 *np.sin(18/19 - 12 *t) - 219/35 *np.sin(7/30 - 10 *t) - 158/43 *np.sin(40/37 - 6 *t) - 512/25 *np.sin(13/16 - 4 *t) - 289/29 *np.sin(68/67 - 2 *t) + 18315/101 *np.sin(t + 29/18) + 664/31 *np.sin(3 *t + 61/23) + 48/11 *np.sin(5 *t + 84/67) + 489/49 *np.sin(7 *t + 11/25) + 397/33 *np.sin(8 *t + 8/19) + 73/12 *np.sin(9 *t + 9/53) + 194/41 *np.sin(11 *t + 17/14) + 2/3 *np.sin(13 *t + 149/50) + 43/29 *np.sin(14 *t + 91/31) + 61/35 *np.sin(15 *t + 131/56) + 29/37 *np.sin(17 *t + 1/19) + 49/43 *np.sin(19 *t + 65/24) + 15/19 *np.sin(20 *t + 88/21) + 11/38 *np.sin(21 *t + 217/50) + 3917/10) *np.heaviside(23 *np.pi - t,0.5) *np.heaviside(t - 19 *np.pi,0.5) + (-8/9 *np.sin(12/23 - 16 *t) - 504/53 *np.sin(8/29 - 8 *t) - 635/43 *np.sin(32/37 - 4 *t) - 307/41 *np.sin(8/27 - 3 *t) - 20292/91 *np.sin(16/19 - t) + 483/19 *np.sin(2 *t + 41/13) + 108/23 *np.sin(5 *t + 70/29) + 74/35 *np.sin(6 *t + 145/34) + 287/43 *np.sin(7 *t + 69/16) + 254/39 *np.sin(9 *t + 5/27) + 19/4 *np.sin(10 *t + 37/30) + 129/46 *np.sin(11 *t + 75/32) + 24/19 *np.sin(12 *t + 71/46) + 125/52 *np.sin(13 *t + 87/44) + 46/27 *np.sin(14 *t + 40/31) + 26/29 *np.sin(15 *t + 106/27) + 25/63 *np.sin(17 *t + 53/12) + 23/22 *np.sin(18 *t + 9/29) + 3/35 *np.sin(19 *t + 205/103) + 200/201 *np.sin(20 *t + 22/25) + 8/31 *np.sin(21 *t + 77/25) - 15195/29) *np.heaviside(19 *np.pi - t,0.5) *np.heaviside(t - 15 *np.pi,0.5) + (-15/23 *np.sin(22/23 - 35 *t) - 21/26 *np.sin(13/40 - 30 *t) - 71/64 *np.sin(16/19 - 29 *t) - 97/29 *np.sin(15/41 - 23 *t) - 57/17 *np.sin(54/35 - 16 *t) - 79/25 *np.sin(41/39 - 14 *t) - 24/11 *np.sin(3/8 - 13 *t) - 149/17 *np.sin(21/62 - 6 *t) - 613/31 *np.sin(16/17 - 2 *t) + 6033/20 *np.sin(t + 24/17) + 631/16 *np.sin(3 *t + 127/30) + 463/31 *np.sin(4 *t + 71/28) + 94/23 *np.sin(5 *t + 98/25) + 45/11 *np.sin(7 *t + 31/10) + 39/23 *np.sin(8 *t + 163/39) + 23/22 *np.sin(9 *t + 42/17) + 167/44 *np.sin(10 *t + 232/231) + 233/49 *np.sin(11 *t + 29/45) + 194/129 *np.sin(12 *t + 3/5) + 166/37 *np.sin(15 *t + 83/29) + 123/35 *np.sin(17 *t + 136/35) + 47/26 *np.sin(18 *t + 64/25) + 72/35 *np.sin(19 *t + 41/14) + 56/31 *np.sin(20 *t + 48/35) + 63/25 *np.sin(21 *t + 2/5) + 100/37 *np.sin(22 *t + 13/15) + 4/3 *np.sin(24 *t + 59/19) + 17/25 *np.sin(25 *t + 15/38) + 51/19 *np.sin(26 *t + 68/19) + 11/27 *np.sin(27 *t + 228/91) + 19/14 *np.sin(28 *t + 31/9) + 4/13 *np.sin(31 *t + 14/55) + 31/37 *np.sin(32 *t + 2/31) + 150/151 *np.sin(33 *t + 58/21) + 41/32 *np.sin(34 *t + 26/11) + 4/3 *np.sin(36 *t + 25/18) - 6956/53) *np.heaviside(15 *np.pi - t,0.5) *np.heaviside(t - 11 *np.pi,0.5) + (4337/36 *np.sin(t + 45/29) + 265/18) *np.heaviside(11 *np.pi - t,0.5) *np.heaviside(t - 7 *np.pi,0.5) + (-23/21 *np.sin(31/61 - t) - 1152/11) *np.heaviside(7 *np.pi - t,0.5) *np.heaviside(t - 3 *np.pi,0.5) + (3314/27 *np.sin(t + 30/31) + 65/31 *np.sin(2 *t + 26/23) - 1467/5) *np.heaviside(3 *np.pi - t,0.5) *np.heaviside(t + np.pi,0.5)) *np.heaviside(np.sin(t/2),0.0)
        Y = ((-9/23 *np.sin(38/25 - 6 *t) - 67/38 *np.sin(36/23 - 3 *t) + 31/30 *np.sin(t + 14/9) + 409/9 *np.sin(2 *t + 74/47) + 493/141 *np.sin(4 *t + 85/54) + 14/17 *np.sin(5 *t + 75/16) + 5/46 *np.sin(7 *t + 21/13) + 33/23 *np.sin(8 *t + 74/47) + 14536/41) *np.heaviside(71 *np.pi - t,0.5) *np.heaviside(t - 67 *np.pi,0.5) + (-89/29 *np.sin(59/38 - 17 *t) - 5/11 *np.sin(14/9 - 16 *t) - 99/40 *np.sin(58/37 - 15 *t) - 59/7 *np.sin(25/16 - 11 *t) - 2/35 *np.sin(8/41 - 10 *t) - 381/26 *np.sin(25/16 - 9 *t) - 67/21 *np.sin(17/11 - 8 *t) - 1706/37 *np.sin(36/23 - 5 *t) - 29/9 *np.sin(29/19 - 4 *t) - 851/29 *np.sin(58/37 - 3 *t) + 1991/30 *np.sin(t + 96/61) + 528/17 *np.sin(2 *t + 85/54) + 89/67 *np.sin(6 *t + 37/24) + 102/13 *np.sin(7 *t + 80/17) + 17/16 *np.sin(12 *t + 91/58) + 35/12 *np.sin(13 *t + 37/23) + 127/27 *np.sin(14 *t + 27/17) - 26576/29) *np.heaviside(67 *np.pi - t,0.5) *np.heaviside(t - 63 *np.pi,0.5) + (-29/14 *np.sin(47/33 - 16 *t) - 35/22 *np.sin(75/52 - 15 *t) - 236/63 *np.sin(16/11 - 14 *t) - 41/6 *np.sin(34/23 - 13 *t) - 236/29 *np.sin(46/31 - 12 *t) - 167/28 *np.sin(55/37 - 11 *t) - 259/33 *np.sin(76/51 - 10 *t) - 414/73 *np.sin(56/37 - 9 *t) - 121/28 *np.sin(17/11 - 7 *t) - 177/32 *np.sin(61/41 - 6 *t) - 1499/41 *np.sin(48/31 - 5 *t) - 647/23 *np.sin(25/16 - 3 *t) + 610/13 *np.sin(t + 30/19) + 1474/31 *np.sin(2 *t + 30/19) + 807/41 *np.sin(4 *t + 41/26) + 208/31 *np.sin(8 *t + 43/27) - 16147/17) *np.heaviside(63 *np.pi - t,0.5) *np.heaviside(t - 59 *np.pi,0.5) + (-12/41 *np.sin(1/4 - 28 *t) - 11/43 *np.sin(9/14 - 26 *t) - 17/41 *np.sin(14/13 - 24 *t) - 22/31 *np.sin(17/67 - 22 *t) - 7/10 *np.sin(64/63 - 19 *t) - 69/41 *np.sin(39/31 - 14 *t) - 86/25 *np.sin(22/41 - 12 *t) - 87/52 *np.sin(31/27 - 9 *t) - 23/15 *np.sin(13/33 - 7 *t) - 25/17 *np.sin(22/25 - 3 *t) + 159/28 *np.sin(t + 249/248) + 571/20 *np.sin(2 *t + 23/26) + 109/36 *np.sin(4 *t + 29/18) + 161/58 *np.sin(5 *t + 31/23) + 147/26 *np.sin(6 *t + 31/19) + 199/35 *np.sin(8 *t + 37/42) + 96/19 *np.sin(10 *t + 17/47) + 64/27 *np.sin(11 *t + 337/75) + 15/7 *np.sin(13 *t + 157/44) + np.sin(15 *t + 101/33) + 5/38 *np.sin(16 *t + 1/28) + 11/56 *np.sin(17 *t + 23/37) + 6/11 *np.sin(18 *t + 8/9) + 91/136 *np.sin(20 *t + 3/19) + 55/54 *np.sin(21 *t + 102/25) + 15/16 *np.sin(23 *t + 118/31) + 22/27 *np.sin(25 *t + 49/15) + 3/8 *np.sin(27 *t + 27/8) + 22/43 *np.sin(29 *t + 57/16) + 10/19 *np.sin(30 *t + 50/83) + 5/31 *np.sin(31 *t + 121/38) + 2727/23) *np.heaviside(59 *np.pi - t,0.5) *np.heaviside(t - 55 *np.pi,0.5) + (-41/31 *np.sin(23/21 - 4 *t) - 85/14 *np.sin(17/32 - t) + 407/35 *np.sin(2 *t + 75/22) + 21/10 *np.sin(3 *t + 41/14) + 53/54 *np.sin(5 *t + 54/25) + 31/61 *np.sin(6 *t + 124/27) + 5/36 *np.sin(7 *t + 3/19) + 19/31 *np.sin(8 *t + 144/31) + 10393/23) *np.heaviside(55 *np.pi - t,0.5) *np.heaviside(t - 51 *np.pi,0.5) + (-36/41 *np.sin(5/18 - 6 *t) + 83/35 *np.sin(t + 95/28) + 43/37 *np.sin(2 *t + 66/17) + 165/13 *np.sin(3 *t + 27/53) + 79/19 *np.sin(4 *t + 9/17) + 37/24 *np.sin(5 *t + 190/63) + 57/58 *np.sin(7 *t + 267/100) + 13545/31) *np.heaviside(51 *np.pi - t,0.5) *np.heaviside(t - 47 *np.pi,0.5) + (-123/47 *np.sin(19/15 - 18 *t) - 59/29 *np.sin(1/49 - 16 *t) - 213/37 *np.sin(29/22 - 13 *t) - 381/40 *np.sin(4/29 - 11 *t) - 168/29 *np.sin(6/11 - 10 *t) - 1233/44 *np.sin(3/19 - 3 *t) - 711/7 *np.sin(1/39 - 2 *t) - 5171/26 *np.sin(12/19 - t) + 2965/57 *np.sin(4 *t + 89/28) + 347/21 *np.sin(5 *t + 23/93) + 1087/69 *np.sin(6 *t + 4/31) + 760/37 *np.sin(7 *t + 172/53) + 333/19 *np.sin(8 *t + 7/13) + 325/81 *np.sin(9 *t + 96/55) + 53/17 *np.sin(12 *t + 138/49) + 73/40 *np.sin(14 *t + 92/67) + 47/31 *np.sin(15 *t + 81/19) + 7/11 *np.sin(17 *t + 29/30) - 3017/19) *np.heaviside(47 *np.pi - t,0.5) *np.heaviside(t - 43 *np.pi,0.5) + (-713/27 *np.sin(22/17 - t) - 36840/41) *np.heaviside(43 *np.pi - t,0.5) *np.heaviside(t - 39 *np.pi,0.5) + (-675/23 *np.sin(13/16 - t) - 17750/19) *np.heaviside(39 *np.pi - t,0.5) *np.heaviside(t - 35 *np.pi,0.5) + (-39/29 *np.sin(11/16 - 17 *t) - 102/37 *np.sin(8/49 - 11 *t) - 95/34 *np.sin(4/13 - 9 *t) - 71/22 *np.sin(7/12 - 8 *t) - 194/17 *np.sin(29/23 - 7 *t) - 2531/25 *np.sin(13/36 - t) + 601/19 *np.sin(2 *t + 264/61) + 232/5 *np.sin(3 *t + 53/13) + 309/40 *np.sin(4 *t + 29/10) + 266/39 *np.sin(5 *t + 3/16) + 71/95 *np.sin(6 *t + 50/37) + 281/44 *np.sin(10 *t + 33/43) + 29/15 *np.sin(12 *t + 105/29) + 39/25 *np.sin(13 *t + 109/36) + 24/11 *np.sin(14 *t + 51/38) + 19/9 *np.sin(15 *t + 38/23) + 43/29 *np.sin(16 *t + 4) + 53/74 *np.sin(18 *t + 74/25) - 45956/91) *np.heaviside(35 *np.pi - t,0.5) *np.heaviside(t - 31 *np.pi,0.5) + (-25/32 *np.sin(4/13 - 15 *t) - 40/43 *np.sin(11/19 - 13 *t) - 12727/115 *np.sin(83/84 - t) + 1762/31 *np.sin(2 *t + 66/29) + 905/78 *np.sin(3 *t + 46/25) + 209/25 *np.sin(4 *t + 104/37) + 103/27 *np.sin(5 *t + 32/17) + 121/60 *np.sin(6 *t + 143/37) + 29/7 *np.sin(7 *t + 45/13) + 41/36 *np.sin(8 *t + 271/58) + 125/62 *np.sin(9 *t + 152/33) + 118/79 *np.sin(10 *t + 56/25) + 41/24 *np.sin(11 *t + 108/25) + 22/45 *np.sin(12 *t + 116/41) + 43/35 *np.sin(14 *t + 68/19) + 1/15 *np.sin(16 *t + 26/11) + 13/43 *np.sin(17 *t + 53/25) - 29541/41) *np.heaviside(31 *np.pi - t,0.5) *np.heaviside(t - 27 *np.pi,0.5) + (-235/21 *np.sin(5/46 - 5 *t) - 133/13 *np.sin(3/29 - 4 *t) - 437/37 *np.sin(50/37 - 3 *t) - 2785/19 *np.sin(5/4 - t) + 724/17 *np.sin(2 *t + 68/29) + 211/141 *np.sin(6 *t + 83/44) + 41/14 *np.sin(7 *t + 135/32) + 83/20 *np.sin(8 *t + 135/38) + 123/62 *np.sin(9 *t + 136/33) + 304/203 *np.sin(10 *t + 166/47) + 59/44 *np.sin(11 *t + 5/29) + 25/36 *np.sin(12 *t + 102/49) + 13/12 *np.sin(13 *t + 101/41) + 23/13 *np.sin(14 *t + 73/26) + 5/32 *np.sin(15 *t + 85/27) + 41/61 *np.sin(16 *t + 56/25) + 1/7 *np.sin(17 *t + 10/17) + 7/18 *np.sin(18 *t + 134/51) - 8059/11) *np.heaviside(27 *np.pi - t,0.5) *np.heaviside(t - 23 *np.pi,0.5) + (-32/23 *np.sin(20/27 - 18 *t) - 31/20 *np.sin(19/17 - 17 *t) - 89/38 *np.sin(30/23 - 13 *t) - 529/122 *np.sin(22/15 - 10 *t) - 151/35 *np.sin(2/27 - 8 *t) - 417/28 *np.sin(43/29 - 4 *t) - 851/35 *np.sin(3/14 - 3 *t) - 13229/88 *np.sin(31/52 - t) + 425/12 *np.sin(2 *t + 37/18) + 397/30 *np.sin(5 *t + 37/17) + 299/31 *np.sin(6 *t + 122/41) + 301/38 *np.sin(7 *t + 58/35) + 240/43 *np.sin(9 *t + 118/27) + 39/28 *np.sin(11 *t + 27/34) + 82/165 *np.sin(12 *t + 58/27) + 29/26 *np.sin(14 *t + 77/27) + 47/19 *np.sin(15 *t + 7/4) + 46/17 *np.sin(16 *t + 79/22) + 46/35 *np.sin(19 *t + 43/21) + 23/28 *np.sin(20 *t + 105/31) + 27/23 *np.sin(21 *t + 184/41) - 12036/55) *np.heaviside(23 *np.pi - t,0.5) *np.heaviside(t - 19 *np.pi,0.5) + (-16/37 *np.sin(42/43 - 19 *t) - 21/23 *np.sin(37/26 - 18 *t) - 23/17 *np.sin(25/56 - 17 *t) - 46/61 *np.sin(34/45 - 16 *t) - 161/22 *np.sin(1/2 - 6 *t) - 472/43 *np.sin(15/23 - 5 *t) - 620/29 *np.sin(43/60 - 3 *t) + 2821/25 *np.sin(t + 167/39) + 2605/88 *np.sin(2 *t + 89/30) + 449/43 *np.sin(4 *t + 66/25) + 37/24 *np.sin(7 *t + 37/33) + 107/13 *np.sin(8 *t + 175/52) + 341/128 *np.sin(9 *t + 188/41) + 32/15 *np.sin(10 *t + 12/19) + 208/43 *np.sin(11 *t + 44/73) + 122/53 *np.sin(12 *t + 41/39) + 69/40 *np.sin(13 *t + 9/32) + 34/23 *np.sin(14 *t + 208/45) + 19/11 *np.sin(15 *t + 11/36) + 17/19 *np.sin(20 *t + 111/26) + 4/15 *np.sin(21 *t + 26/25) - 10055/37) *np.heaviside(19 *np.pi - t,0.5) *np.heaviside(t - 15 *np.pi,0.5) + (-59/44 *np.sin(173/172 - 36 *t) - 73/31 *np.sin(21/53 - 30 *t) - 23/11 *np.sin(13/12 - 29 *t) - 133/50 *np.sin(23/19 - 28 *t) - 125/29 *np.sin(108/77 - 24 *t) - 122/33 *np.sin(1/19 - 21 *t) - 238/79 *np.sin(4/7 - 16 *t) - 141/16 *np.sin(34/37 - 9 *t) - 45/8 *np.sin(16/27 - 7 *t) + 11594/23 *np.sin(t + 1768/589) + 1582/37 *np.sin(2 *t + 28/25) + 771/38 *np.sin(3 *t + 107/31) + 863/22 *np.sin(4 *t + 87/22) + 485/29 *np.sin(5 *t + 63/25) + 27/8 *np.sin(6 *t + 75/76) + 106/19 *np.sin(8 *t + 20/23) + 54/17 *np.sin(10 *t + 10/49) + 206/61 *np.sin(11 *t + 106/29) + 65/14 *np.sin(12 *t + 81/29) + 80/11 *np.sin(13 *t + 49/43) + 41/29 *np.sin(14 *t + 1/114) + 17/38 *np.sin(15 *t + 97/43) + 97/20 *np.sin(17 *t + 98/23) + 77/30 *np.sin(18 *t + 49/19) + 44/13 *np.sin(19 *t + 53/16) + 44/19 *np.sin(20 *t + 95/23) + 135/29 *np.sin(22 *t + 27/25) + 243/121 *np.sin(23 *t + 23/17) + 15/4 *np.sin(25 *t + 10/17) + 50/13 *np.sin(26 *t + 75/32) + 308/47 *np.sin(27 *t + 253/76) + 65/19 *np.sin(31 *t + 7/15) + 92/33 *np.sin(32 *t + 26/11) + 17/15 *np.sin(33 *t + 74/23) + 8/15 *np.sin(34 *t + 64/27) + 17/27 *np.sin(35 *t + 215/72) + 16757/30) *np.heaviside(15 *np.pi - t,0.5) *np.heaviside(t - 11 *np.pi,0.5) + (1805/16 *np.sin(t + 1/303) + 19936/43) *np.heaviside(11 *np.pi - t,0.5) *np.heaviside(t - 7 *np.pi,0.5) + (374/65 *np.sin(t + 149/47) + 11537/27) *np.heaviside(7 *np.pi - t,0.5) *np.heaviside(t - 3 *np.pi,0.5) + (-15391/135 *np.sin(35/71 - t) + 112/53 *np.sin(2 *t + 66/29) + 13507/30) *np.heaviside(3 *np.pi - t,0.5) *np.heaviside(t + np.pi,0.5)) *np.heaviside(np.sin(t/2),0.0)

    Target = np.array([X,Y]).T

    if jump==1:
        
        Jcheck0=np.zeros(num)
        for i in range(num-1) :
            Jcheck0[i]=np.count_nonzero(Target[i+1:] == Target[i])

        Jcheck1= Target[:] != [[0,0]]

        Jcheck = np.logical_and(Jcheck0[:]==0,Jcheck1[:,0])

        Target=Target[Jcheck]
        t=np.linspace(0,dom,Jcheck.sum(),endpoint=False)
        print(Jcheck.sum())

    return Target,t


if __name__ == '__main__':
    
    # N=int(input("N=?"))

    # alpha=float(input("alpha=?"))
    # beta=float(input("beta=? (if alpha = beta, input -1)"))

    # if beta==-1.0 :
    #     beta = alpha

    N=500
    alpha=0.25
    beta=alpha

    psLB=0.5
    phLB=0.1

    # K=float(input("K=?"))

    # M=float(input("M=?"))

    # L=float(input("L=?"))

    # T=int(input("T=?"))

    K=150
    M=5
    L=0.0005
    T=1000

    #target pattern 만들기

    curvetype=2

    if curvetype==0:
        curvename='circle'
        domain=2*np.pi

        jump=0

    elif curvetype==1:
        curvename='pi'
        domain=2*np.pi

        jump=0

    elif curvetype==2:
        curvename='b.simpson'
        domain=72*np.pi

        jump=1

    Z,Domain=Curve(domain) #주어진 curve 모양으로

    # A : adjacency matrix

    zeta=['ps','ph','b']
    nettype=[0,4,3]

    for ind in range(3):

        A=np.zeros((N,N))
        
        if nettype[ind]==0 :
            A[:,:]=1
            for i in range(N) :
                A[i,i]=0

        elif nettype[ind]==1 :
            for i in range(N-1) :
                A[i,i+1] = 1

            A[0,1:N]=1
            A[0:int(N/2)-1,int(N/2)-1]=1
            A[int(N/2)-1,int(N/2):N]=1
            A[:N-1,N-1]=1

            A=A+A.T

        elif nettype[ind]==2 :
            for i in range(N-1) :
                A[i,i+1] = 1

            A=A+A.T

        elif nettype[ind]==3 :
            if N % 2 != 0 :
                step=2
            elif N % 4 == 0 :
                step=N/2-1
            else :
                step=N/2-2

            for i in range(N) :
                A[i,int((i+step)%N)]=1
                A[i,int((i-step)%N)]=1

        elif nettype[ind]==4 :
            for i in range(N-1) :
                A[i,i+1] = 1

            A=A+A.T

            for i in range(int(N/10)) :
                A[i*10,:]=1
                A[:,i*10]=1
                A[i*10,i*10]=0


        globals()['A_{}'.format(zeta[ind])]=np.copy(A)
   

    #P, V 초기가ㅄ 설정

    PVinitset=0

    if PVinitset==0:
        Pinit=np.random.rand(N,2)*(np.max(Z)-np.min(Z))*20-(np.max(Z)-np.min(Z))*10 #자동으로 모든 component에 100씩 빼짐

        Vinit=(np.random.rand(N,2)*50)-15 #np.array끼리 그냥 곱하면 component 사이의 곱
        Vimean=np.array(np.mean(Vinit[:],axis=0))

        # Vinit-=Vimean

    else :
        print(os.getcwd())
        Pinit_df=pd.read_csv('./Pinit.csv',delimiter='\t')
        Vinit_df=pd.read_csv('./Vinit.csv',delimiter='\t')

        Pinit=Pinit_df.to_numpy()
        Vinit=Vinit_df.to_numpy()

    Xsimean=np.array(np.mean(Pinit[:]-Z[:],axis=0))
    print("\nX*imean={}".format(Xsimean))
    Vimean=np.array(np.mean(Vinit[:],axis=0))        
    print("\nVimean={}".format(Vimean))

    P=np.array([Pinit])
    V=np.array([Vinit])
    Pdiff=np.zeros(T)
    Pdiff[0]=max(np.max(Pinit[:,0])-np.min(Pinit[:,0]),np.max(Pinit[:,1])-np.min(Pinit[:,1]))

    print(P[0])
    print(V[0])
    print(Vimean)

    #Vnrm(t) = Vnrm = sum_i |v_t^i|^2
    #nV(t) = V를 normalize
    s=np.sqrt(np.power(Vinit[:,0],2)+np.power(Vinit[:,1],2))
    nVinit=np.copy(Vinit)/np.array([s]).T
    nV=np.array([nVinit])

    Vnrm=np.zeros(T)
    Vnrm[0]=np.sum(np.power(s,2))

    #Xbnrm(t) = sum_i,j |x*_i-x*_j|^2 = 2N sum_i |x*_i|^2 - 2|sum_i x*_i|^2 = \sum_{i,j} |\bar{x}_t^i-\bar{x}_t^j|^2
    Xbnrm=np.zeros(T)
    Xbave=np.sum(Pinit,axis=0)-np.sum(Z,axis=0)
    Xbnrm[0]=2*(N*np.sum(np.power(Pinit[:,0]-Z[:,0],2)+np.power(Pinit[:,1]-Z[:,1],2))-(np.power(Xbave[0],2)+np.power(Xbave[1],2)))

    #phE(t)=\sum_{i,j\in\calE} \int_0^|\bar{x}_t^{ij}|^2 \phi(r)dr
    phE=np.zeros(T)
    for i in range(N):
        J = A_ph[i,:]==1
        ssq=(np.power(P[0,i,0]-Z[i,0]-P[0,:,0]+Z[:,0],2)+np.power(P[0,i,1]-Z[i,1]-P[0,:,1]+Z[:,1],2))[J]

        phE[0]+=np.sum(phiEest(ssq,beta))

    phE[0]*=M/2

    # P 위치 V 초기가ㅄ 설정, nV:V를 normalize -> 이거 np 사용하면 더 간단해질 수도 있을 듯

    dB=np.zeros(T)

    h=0.0125
    limT=T

    for t in range(1,T):
        # print ("start %dth loop" %t)
        
        Pnow=np.copy(P[t-1])
        Vnow=np.copy(V[t-1])

        # euler maruyama .

        Kem=np.array([Vnow,csmmp(Pnow,Vnow)])

        # if np.isnan(K).any() == True :
        #     print("NaN-K")
        #     print(t)
        #     print (Pnow)
        #     print (Vnow)
        #     print (K)
        #     break

        Pnext=np.copy(Pnow)
        Vnext=np.copy(Vnow)

        Pnext+=Kem[0]*h
        Vnext+=Kem[1]*h

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

        P=np.append(P,np.array([Pnext]),axis=0)
        V=np.append(V,np.array([Vnext]),axis=0)
        nV=np.append(nV,np.array([nVnext]),axis=0)
        Pdiff[t]=max(np.max(Pnext[:,0])-np.min(Pnext[:,0]),np.max(Pnext[:,1])-np.min(Pnext[:,1]))

        Vnrm[t]=np.sum(np.power(s,2))

        Xbave=np.sum(Pnext,axis=0)-np.sum(Z,axis=0)
        Xbnrm[t]=2*(N*np.sum(np.power(Pnext[:,0]-Z[:,0],2)+np.power(Pnext[:,1]-Z[:,1],2))-(np.power(Xbave[0],2)+np.power(Xbave[1],2)))
        dB[t]=dBt    

        for i in range(N):
            J = A_ph[i,:]==1
            ssq=(np.power(P[t,i,0]-Z[i,0]-P[t,:,0]+Z[:,0],2)+np.power(P[t,i,1]-Z[i,1]-P[t,:,1]+Z[:,1],2))[J]

            phE[t]+=np.sum(phiEest(ssq,beta))

        phE[t]*=M/2

        if t % 500 ==0 :
            print("end %d" %t)


    # print ("end")
    print (Pnow)
    print (Vnow)
    print (Vimean)
    print (Z)

    #그림을 어떻게 그릴지 : (Xmax-Xmin)+(Ymax-Ymin)이 제일 클 때 까지는 개중에 Xinf~Xsup,Yinf~Ysup으로
    #-> 뭔가 사이즈는 limsup같은 느낌으로 하고 중앙점은 움직임 따라가는 식으로 할 수 있을 것 같은데
    # 프레임 사이즈가 큰 차이 안나면 고정으로 가야지 덜 어지러울 듯.
    #아 몰라 이건 나중에 하고 가계부나 하자

    # X,Y차이값이 max일때 까지는 scale 유지하는 것으로.
    limT=np.argmax(Pdiff)
    #np.argmax(np.amax(np.amax(P,axis=1)-np.amin(P,axis=1),axis=1)) 로도 찾을 수 있지만 이미 for 돌리기도 했고 len구해야 하기 때문에 걍 Pdiff 만듦
    #근데 어차피 매번 Xmin Ymin 구할거면 것도 걍 미리 구해놓지? 이건 지금 바꾸는건 귀찮아 나중에.
      
    print(limT)

    #즉 limT까지는 축 고정하고 그 이후로는 원래는 scale 고정하려고 했는데 의미 불분명하니 걍 scale decreasing 하게 잡는 것으로

    fig = plt.figure()
    ax = fig.add_subplot(131)
    cx = fig.add_subplot(132)
    dx = fig.add_subplot(133)

    Xmax = np.max(P[:limT+1,:,0])
    Ymax = np.max(P[:limT+1,:,1])
    Xmin = np.min(P[:limT+1,:,0])
    Ymin = np.min(P[:limT+1,:,1])

    xylen=np.nan_to_num(max(Xmax-Xmin,Ymax-Ymin))

    if xylen >= 10*Pdiff[limT] or limT >= T*0.3:
        xylen=Pdiff[0]
        limT=0
        Xmin=np.min(P[0,:,0])
        Ymin=np.min(P[0,:,1])

    ax.set_xlim([np.nan_to_num(Xmin-xylen*0.15),np.nan_to_num(Xmin+xylen*1.15)])
    ax.set_ylim([np.nan_to_num(Ymin-xylen*0.15),np.nan_to_num(Ymin+xylen*1.15)])
    ax.set_aspect('equal')
    ax.set_xlabel('(frame_len)=%.3f\n dBt=%.3f' %(xylen,dB[0]))

    dotsize=600/len(Domain)
    scat=ax.scatter(P[0].T[0], P[0].T[1],s=dotsize,c=-Domain, cmap='nipy_spectral') #P[0].T[1] = (transpose of P[0])[1]. cmap=따라 컬러 스펙트럼 바뀜
    scat.set_alpha(0.6)
    qax=ax.quiver(P[0].T[0],P[0].T[1],nV[0].T[0],nV[0].T[1],angles='xy',width=0.001,scale=70)


    cx.set_title(r'$\|\|v_t\|\|^2$')
    # cx.set_yscale('log')
    cx.plot(range(0,T),Vnrm,linewidth=0.7, c='teal', label=r'$\|\|v_t\|\|^2$')
    cx.plot(range(0,T),[Vnrm[0]+phE[0] for tt in range(T)],linewidth=0.5,c='gold',label=r'$E_0$')
    cx.plot(range(0,T),Vnrm[0]+phE[0]-phE, alpha=0.8,linewidth=0.5,c='forestgreen',label=r'$E_0-E_\phi(t)$')
    cx.legend(loc='best')

    reddot, =cx.plot([0],[Vnrm[0]],'r.') #왠지 모르겠는데 reddot뒤의 ,을 빼면 animation 부분에서 에러남. 

    dx.set_title(r'$\sum_{i,j=1}^N |\bar{x}_t^i-\bar{x}_t^j|^2$')
    # dx.set_yscale('log')
    
    dx.plot(range(0,T),Xbnrm)
    reddot2, =dx.plot([0],[Xbnrm[0]],'r.') #왠지 모르겠는데 reddot뒤의 ,을 빼면 animation 부분에서 에러남. 

    ani = animation.FuncAnimation(fig,_update_plot,fargs=(fig,scat,qax,ax,reddot,reddot2),frames=T-1,interval=100/6,save_count=T-1)

    #interval이 너무 작으니깐(fps가 너무 커지니깐) save가 안됨-파일을 열때 에러남.

    fig.tight_layout()

    plt.show()
    
    pprr=(input("Do you want to save? Yes=else, No=0 "))
    
    if pprr!='0' :
        trialname=input("trial name=?")
        savename='scs-em-simu-%s-k%.2fm%.2fsigma%.4f-net%d.%d.%d.-%s.mp4' %(curvename,K,M,L,nettype[0],nettype[1],nettype[2],trialname)
        print("Saving...")
        ani.save(savename,dpi=300,fps=60)

    print("DONE")

#sine 그래프일때는 fluctuation 하는 진폭이 logscale로 봤을때 거의 일정했음. 그리고 빙빙 돌면서 수렴한다는 느낌이었음.
#K, M을 작게하고(각각 1) 이렇게 멀리 떨어뜨려놔도 초기 속도차이 10^2~10^3 스케일까지는 뜀
#12만 프레임 dpi300 fps60 기준으로 33분 나오고 저장시간은 9시간 정도 걸림.

# 중간에 물결선 넣기
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html 물결선 넣기 -> 결국 그래프를 두개로 쪼개야함
# https://matplotlib.org/3.1.0/tutorials/intermediate/gridspec.html ->그래프를 격자 위에 임의로 배치할 수 있음. 조각보처럼
# https://stackoverflow.com/questions/53642861/broken-axis-slash-marks-inside-bar-chart-in-matplotlib 


# %%
