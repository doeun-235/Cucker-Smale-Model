
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, TransformedBbox,BboxPatch, BboxConnector)
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import os
import glob
import pandas as pd


#np 참고 http://taewan.kim/post/numpy_cheat_sheet/

def sto_ratio(start,end,below):
    if np.min(Vcum[1:,start:end,0,0]) >= below :
        ratio = abs(np.max(np.max(Vcum[1:,start:end,0,0],axis=0)-np.min(Vcum[1:,start:end,0,0],axis=0))/(np.max(Vcum[1:,start:end,0,0])-np.min(Vcum[1:,start:end,0,0])))
    else :
        ratio=0
    return ratio

def makeplot():

    global fig

    # fig=plt.figure(constrained_layout=True,figsize=(10,5))
    # spec=gridspec.GridSpec(ncols=9,nrows=6,figure=fig)
    # ax=fig.add_subplot(spec[0:3,0:4])
    # bx=fig.add_subplot(spec[0:3,4:8])
    # bx_larged=fig.add_subplot(spec[0:3,8])

    # dx=fig.add_subplot(spec[3:,6:])


    ## plot1 : graph of (v_t^1)_1 ... (v_t^n)_1 on trial =1
    fig_size=(5.1,3.4)
    fig=plt.figure(figsize=fig_size)

    plt.plot(np.array(range(0,T))*h,[0 for tt in range(T)],linewidth=0.8,c='k')

    for i in range(N):
        plt.plot(np.array(range(0,T))*h,Vcum[1,:,i,0],alpha=0.6,linewidth=0.3)

    axes = plt.axes()
    axes.set_xlim(right=(T-2)*h,left=0)

    axes.set_xlabel('t')
    axes.set_ylabel('velocity',rotation=90)

    saveplot(1,plotsave)


    ## plot2 : graph of (v_t^1)_1 on all trial
    fig=plt.figure(figsize=fig_size)

    ## plot 안에 그래프 그리기 : inset_axes,mark_inset/https://data-newbie.tistory.com/447
    axes=plt.axes()

    plt.plot(np.array(range(0,T))*h,[0 for tt in range(T)],linewidth=0.8,c='k')

    for i in range(Trial) :
        plt.plot(np.array(range(0,T))*h,Vcum[i+1,:,0,0],linewidth=0.3, alpha=0.6)

    axes.set_xlim(right=(T-2)*h,left=0)

    ax_max=max(np.max(Vcum[1:,:,0,0]),-1*np.min(Vcum[1:,:,0,0]))
    if np.max(Vcum[1:,:,0,0]) > (-1)*np.min(Vcum[1:,:,0,0]):
        # ax_cut=int(np.median(np.argmax(Vcum[1:,:,0,0],axis=1)))
        cut_sign=1
        plotlen=abs(np.min(Vcum[1:,:,0,0]))/(abs(np.max(Vcum[1:,:,0,0]))+abs(np.min(Vcum[1:,:,0,0])))

    else :
        # ax_cut=int(np.median(np.argmin(Vcum[1:,:,0,0],axis=1)))
        cut_sign=-1
        plotlen=abs(np.max(Vcum[1:,:,0,0]))/(abs(np.max(Vcum[1:,:,0,0]))+abs(np.min(Vcum[1:,:,0,0])))

    # ax_ycut=np.max(Vcum[1:,ax_cut,0,0])
    # ax_ycut2=np.min(Vcum[1:,ax_cut,0,0])

    # # print(np.argmax(Vcum[1:,:,0,0],axis=1))
    # # print(Vcum[1,:,0,0])
    # # print(Vcum[2,:,0,0])
    # # print(Vcum[3,:,0,0])

    # print("ax_ycut : %f" %ax_ycut)
    # print("ax_ycut2 : %f" %ax_ycut2)

    # lar_len=min(12*(ax_ycut-ax_ycut2),abs(ax_ycut)*0.3)
    # cutunit=3
    # print("ax_cut : %d" %ax_cut)
    # print("lar_len : %.2f" %lar_len)

    # # select=0
    # # bounded_l=ax_cut
    # # bounded_r=T-1
    # # ccut=ax_cut*3+50
    # # while select==0 :
    # #     if cut_sign*(ax_ycut-np.mean(Vcum[:,ccut,0,0]))>lar_len :
    # #         bounded_r=ccut
    # #         ccut=int((bounded_l+ccut)/2)
    # #     if cut_sign*(ax_ycut-np.mean(Vcum[:,ccut,0,0]))<lar_len :
    # #         bounded_l=ccut
    # #         ccut=int((bounded_r+ccut)/2)

    # #     if bounded_r-bounded_l<cutunit:
    # #         select=1            

    # # ax_rightcut=min(int(ax_cut*4),bounded_r+1)

    # # print("bound l : %d" %bounded_l)
    # # print("bound r : %d" %bounded_r)
    # # print("ax_rightcut : %d" %ax_rightcut)

    # # select=0
    # # bounded_l=0
    # # bounded_r=ax_cut
    # # ccut=int(ax_cut*0.5)
    # # while select==0 :
    # #     if cut_sign*(ax_ycut-np.mean(Vcum[:,ccut,0,0]))>lar_len :
    # #         bounded_l=ccut
    # #         ccut=int((bounded_r+ccut)/2)
    # #     if cut_sign*(ax_ycut-np.mean(Vcum[:,ccut,0,0]))<lar_len :
    # #         bounded_r=ccut
    # #         ccut=int((bounded_l+ccut)/2)

    # #     if bounded_r-bounded_l<cutunit:
    # #         select=1            

    # # ax_leftcut=max(bounded_l,int(ax_cut-(ax_rightcut-ax_cut)),0)

    # # print("bound l : %d" %bounded_l)
    # # print("bound r : %d" %bounded_r)
    # # print("ax_leftcut : %d" %ax_leftcut)

    below=0.1
    if T>900 :
        enlarge_len=6
    else :
        enlarge_len=3
    
    I_start=range(0,T-enlarge_len)
    ratio=np.zeros(T-enlarge_len)

    for i in I_start:
        ratio[i] = sto_ratio(i,i+enlarge_len,below*ax_max)

    ax_leftcut=np.argmax(ratio)
    ax_rightcut=ax_leftcut+enlarge_len

    print("ax_leftcut : %d" %ax_leftcut)
    print("cut_sign : %d" %cut_sign)
    print("plotlen : %.2f" %plotlen )

    ax_larged=inset_axes(axes,"100%","100%",bbox_to_anchor=[0.9-0.35,0.05+(0.1+plotlen)*(1+cut_sign)/2,0.08+0.35,0.8-plotlen+0.02*(cut_sign-1)/2],bbox_transform=axes.transAxes,borderpad=0) #x축 시작 위치, y축 시작 위치, 너비, 높이

    for i in range(Trial) :
        ax_larged.plot(np.array(range(ax_leftcut,ax_rightcut))*h,Vcum[i+1,ax_leftcut:ax_rightcut,0,0],linewidth=0.3, alpha=0.5)
    # ax_larged.set_xticks([])
    ax_larged.set_yticks([])

    # ax_larged.set_ylim(bottom=np.max(Vcum[:,ax_leftcut:ax_rightcut,0,0])-lar_len*1.5,top=np.max(Vcum[:,ax_leftcut:ax_rightcut,0,0])+lar_len*0.05)
    # ax_larged.set_ylim(bottom=np.max(Vcum[:,ax_cut,0,0])-lar_len*1.5,top=np.max(Vcum[:,ax_cut,0,0])+lar_len*0.05)

    my_mark_inset(axes,ax_larged,loc1a=2,loc1b=1,loc2a=3,loc2b=4,fc="none",ec="0.2",boxlw=0.6,connectlw=0.4) #우상부터 반시계로 1~4

    axes.set_xlabel('t')
    axes.set_ylabel('velocity',rotation=90)
        
    saveplot(2,plotsave)


    # plot3 : graph of E(Vnrm)
    fig=plt.figure(constrained_layout=True,figsize=fig_size)
    spec=gridspec.GridSpec(ncols=18,nrows=6,figure=fig)

    EVnrm=np.mean(Vnrmcum[1:],axis=0)
    EphE=np.mean(phEcum[1:],axis=0)
    VVnrmcum=np.sort(Vnrmcum[1:],axis=0)

    E0=EVnrm[0]+EphE[0]
    SupVnrm=max(np.max(VVnrmcum[int(Trial*0.9),:]), np.max(EVnrm))
    print (EVnrm)

    scale_const=int(np.log10(E0))
    scale=10**(-scale_const)
    
    if E0>= SupVnrm:
        cx_up=fig.add_subplot(spec[:2,:])
        cx_down=fig.add_subplot(spec[2:,:])

        # cx_down.plot(np.array(range(0,T))*h,[0 for tt in range(T)],linewidth=0.8,c='k')

        cx_up.plot(np.array(range(0,T))*h,EVnrm*scale,linewidth=0.7,alpha=0.8,label=r'$\mathrm{\mathbb{E}}\Vert {\bf v}_t \Vert$')
        cx_up.plot(np.array(range(0,T))*h,[E0*scale for tt in range(T)],linewidth=0.5,c='gold',label=r'$E_0$',ls='--')
        cx_up.plot(np.array(range(0,T))*h,(E0-EphE)*scale, alpha=0.8,linewidth=0.7,c='forestgreen',label=r'$E_0-E^{\phi}_t$',ls='-.')
        cx_up.plot(np.array(range(0,T))*h,(EphE+EVnrm)*scale, alpha=0.8,linewidth=0.5,c='firebrick',label=r'$E_t$',ls=':')

        # cx_up.plot(range(0,T),medVnrm,c='paleturquoise',linewidth=0.7,alpha=0.7,label='median of ')
        # cx_up.plot(range(0,T),VVnrmcum[int(Trial*0.9),:],c='deeppink',linewidth=0.5,alpha=0.7,label='90%')

        cx_down.plot(np.array(range(0,T))*h,EVnrm*scale,linewidth=0.95,alpha=0.8,label=r'$\mathrm{\mathbb{E}}$')
        cx_down.plot(np.array(range(0,T))*h,[E0*scale for tt in range(T)],linewidth=0.5,c='gold')
        cx_down.plot(np.array(range(0,T))*h,(E0-EphE)*scale, alpha=0.8,linewidth=0.7,c='forestgreen',ls='-.')
        cx_down.plot(np.array(range(0,T))*h,(EphE+EVnrm)*scale, alpha=0.8,linewidth=0.5,c='firebrick',ls=':')

        # cx_down.plot(range(0,T),medVnrm,c='paleturquoise',linewidth=0.7,alpha=0.7,label='median')
        # cx_down.plot(range(0,T),VVnrmcum[int(Trial*0.9),:],c='deeppink',linewidth=0.5,alpha=0.7,label='90%')

        cx_up.set_ylim(0.9*E0*scale,E0*scale*1.01)
        
        # cxcut=E0-min(EphE[np.argmax(VVnrmcum[int(Trial*0.9),:])], EphE[np.argmax(EVnrm)])
        
        cx_top=SupVnrm*1.2*scale
        # cx_bottom=min(np.min(VVnrmcum[0,:])*(-0.5),SupVnrm*(-0.05))*scale
        cx_bottom=-0.001*2
        cx_down.set_ylim(bottom=cx_bottom,top=cx_top)
        
        # cx_down.set_ylim(top=cxcut*1.2)


        cx_up.legend(fontsize=9.5*0.83,loc='best')
        # cx_down.legend(fontsize=10,loc='best')

        cx_up.spines['bottom'].set_visible(False)
        cx_down.spines['top'].set_visible(False)
        cx_up.xaxis.set_visible(False)
        cx_down.xaxis.tick_bottom()

        cx_up.set_xlim(right=(T-2)*h, left=-.01/9)
        cx_down.set_xlim(right=(T-2)*h,left=-.01/9)
        axes=cx_down

        d = .004  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=cx_up.transAxes, color='k', clip_on=False,lw=0.8)

        curvyline_x=np.linspace(-d,1+d,250)
        curvyline_y=1.5*2*d*np.sin(2*np.pi/d*8*curvyline_x)
        cx_up.plot(curvyline_x,curvyline_y, **kwargs)        # top-left diagonal

        kwargs.update(transform=cx_down.transAxes)  # switch to the bottom axes
        cx_down.plot(curvyline_x, 1+0.7*curvyline_y, **kwargs)  # bottom-right diagonal

        cx_up.annotate(r'$\times$10$^{%i}$' %(scale_const),xy=(0.003,0.83),xycoords='axes fraction')
        #축의 가ㅄ 표시 형식 -> 이것 저것 해보다가 결국 수동으로 하기로
        #https://stackoverflow.com/questions/39620700//positioning-the-exponent-of-tick-labels-when-using-scientific-notation-in-matplo

    else :
        cx=fig.add_subplot(spec[:,:])
   
        # cx.set_title(r'$\mathrm{\mathbb{E}}||v(t)||^2$')
        # cx.set_xlabel('K=%.2f,M=%.2f, sigma=%.2f' %(K,M,L))

        cx.plot(np.array(range(0,T))*h,EVnrm,linewidth=0.7,alpha=0.8,label=r'$\mathrm{\mathbb{E}}||v||$')
        cx.plot(np.array(range(0,T))*h,[E0 for tt in range(T)],linewidth=0.5,c='gold',label=r'$E_0$')
        cx.plot(np.array(range(0,T))*h,E0-EphE, alpha=0.8,linewidth=0.5,c='forestgreen',label=r'$E_0-E^{\phi}_t$',ls='-.')
        cx.plot(np.array(range(0,T))*h,EphE+EVnrm, alpha=0.8,linewidth=0.5,c='firebrick',label=r'$E_t$',ls=':')

        # cx.plot(range(0,T),medVnrm,c='paleturquoise',linewidth=0.7,alpha=0.7,label='median')
        # cx.plot(range(0,T),VVnrmcum[int(Trial*0.9),:],c='deeppink',linewidth=0.5,alpha=0.7,label='90%')
        cx.legend(fontsize=10,loc='best')
        
        cx_top=E0
        cx_bottom=-0.001
        axes=cx

    plotlen=abs(np.min(EVnrm))/(E0+abs(np.min(EVnrm)))
    cx_larged=inset_axes(axes,"100%","100%",bbox_to_anchor=[0.7,plotlen+0.15,0.28,0.8-plotlen],bbox_transform=axes.transAxes,borderpad=0) #x축 시작 위치, y축 시작 위치, 너비, 높이

    cx_larged.plot(np.array(range(0,T))*h,EVnrm*scale,label=r'$\mathrm{\mathbb{E}}$')
    cx_larged.plot(np.array(range(0,T))*h,(E0-EphE)*scale, alpha=0.8,linewidth=0.7,c='forestgreen',label=r'$E_0-E_\phi(t)$',ls='-.')
    # cx_larged.plot(range(0,T),medVnrm,c='paleturquoise',linewidth=0.7,alpha=0.7)
    # cx_larged.plot(range(0,T),VVnrmcum[int(Trial*0.9),:],c='deeppink',linewidth=0.5,alpha=0.7)

    # cx_rightcut=np.argmax(EVnrm)
    cx_rightcut=int(T/250*3)
    # cx_larged.set_ylim(bottom=min(np.min(VVnrmcum[0,:])*(-0.5),SupVnrm*(-0.01)),top=SupVnrm*1.05)
    # cx_larged.set_ylim(bottom=min(np.min(VVnrmcum[0:cx_rightcut])*(-0.5),np.max(EVnrm[0:cx_rightcut])*(-0.1)),top=(E0-np.min(EphE[:int(cx_rightcut*1.05)]))*0.65)
    # cx_larged.set_ylim(bottom=max(cx_bottom,np.max(EVnrm[:int(cx_rightcut*1.05)+1])*(-0.025)*scale),top=min(scale*((E0-np.min(EphE[:int(cx_rightcut*1.05)+1]))*0.35+np.max(EVnrm[0:int(cx_rightcut*1.05)+1])*0.65),cx_top*0.9))
    cx_larged.set_ylim(bottom=max(cx_bottom,np.min(EVnrm[:int(cx_rightcut*1.05)+1])*scale),top=min(scale*((E0-np.min(EphE[:int(cx_rightcut*1.05)+1]))*0.35+np.max(EVnrm[0:int(cx_rightcut*1.05)+1])*0.65),cx_top*0.9))
    # cx_larged.set_xlim(left=cx_rightcut*(-0.05)*h,right=cx_rightcut*1.05*h)
    cx_larged.set_xlim(left=0,right=cx_rightcut*1.05*h)
    # cx_larged.legend(fontsize=5,loc='best')
    cx_larged.set_yticks([])

    my_mark_inset(axes,cx_larged,loc1a=2,loc1b=1,loc2a=3,loc2b=4,fc="none",ec="0.2",boxlw=0.4,connectlw=0.3) #우상부터 반시계로 1~4

    axes.set_xlabel('t')

    saveplot(3,plotsave)


    ## dx : graph of E(X*nrm_t)
    fig=plt.figure(figsize=fig_size)

    EXbnrm=np.mean(Xbnrmcum[1:],axis=0)
    plt.plot(np.array(range(0,T))*h,EXbnrm)
    axes=plt.axes()
    axes.set_xlim(left=0,right=(T-2)*h)
    axes.set_ylim(bottom=-0.001*2400)

    axes.set_xlabel('t')

    saveplot(4,plotsave)

    #fig.tight_layout()

def my_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2,boxlw=0.5,connectlw=0.5, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    pp = BboxPatch(rect, fill=False,lw=boxlw, **kwargs)
    parent_axes.add_patch(pp)
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b,lw=connectlw, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b,lw=connectlw, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)
    return pp, p1, p2

def saveplot(plotnum,save) : 
    if save!=0 :
        # plt.savefig('graph_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%dh%.5f-%d.pdf' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,h,plotnum),dpi=2000,bbox_inches='tight',pad_inches=0.02)
        if plotnum == 2 :
            plt.savefig('graph_k%.1fm%.1fs%.5fnet%d.%d.%dN%d-%d.eps' %(K,M,L,nettype[0],nettype[1],nettype[2],N,plotnum),dpi=3000,bbox_inches='tight',pad_inches=0.02)

        plt.savefig('graph_k%.1fm%.1fs%.5fnet%d.%d.%dN%d-%d.pdf' %(K,M,L,nettype[0],nettype[1],nettype[2],N,plotnum),dpi=3000,bbox_inches='tight',pad_inches=0.02)
                
        print("image saved_%d" %plotnum)

    else :
        plt.show()



def savedata(savemode) :
    ## savemode==1 : save just outlier data and initial setting
    ## savemode==2 : save all data and initial setting
    ## savemode==0 : don't save

    if savemode==1 :

        print("data reshaping")

        selP = Pcum[selI,:,:,:].reshape(-1,T*N*2)
        selV = Vcum[selI,:,:,:].reshape(-1,T*N*2)

        selVnrm = Vnrmcum[selI,:]
        selXbnrm = Xbnrmcum[selI,:]
        seldB = dBcum[selI,:]

        selP_df=pd.DataFrame(selP)
        selV_df=pd.DataFrame(selV)

        selVnrm_df=pd.DataFrame(selVnrm)
        selXbnrm_df=pd.DataFrame(selXbnrm)
        seldB_df=pd.DataFrame(seldB)

        selcount_df=pd.DataFrame(selcount)

        print("saving...")

        selP_df.to_csv('selP_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        selV_df.to_csv('selV_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        selVnrm_df.to_csv('selVnrm_k%.2fm%.2fs%.4fnet%d%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        selXbnrm_df.to_csv('selXbnrm_k%.2fm%.2fs%.4fnet%d%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        seldB_df.to_csv('seldB_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        selcount_df.to_csv('selcount_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")

    elif savemode==2 :
        print("data reshaping")

        reP = Pcum.reshape(-1,T*N*2)
        reV = Vcum.reshape(-1,T*N*2)

        Vnrm = Vnrmcum[:,:]
        Xbnrm = Xbnrmcum[:,:]

        P_df=pd.DataFrame(reP)
        V_df=pd.DataFrame(reV)

        Vnrm_df=pd.DataFrame(Vnrmcum)
        Xbnrm_df=pd.DataFrame(Xbnrmcum)
        phE_df=pd.DataFrame(phEcum)
        
        print("saving...")

        P_df.to_csv('P_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        V_df.to_csv('V_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        Vnrm_df.to_csv('Vnrm_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        Xbnrm_df.to_csv('Xbnrm_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        phE_df.to_csv('phE_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")

    # setting 을 df로 만들어서 입출력 편하게 저장하는 법 확인하기. 
    # dB, Pinit, Vinit, Z, A


    if savemode!=0 :
        variables=['N','alpha','beta','psLB','phLB','K','M','L','T','h','Trial','cut','nettype','curvetype','version']
        setting={}
        for index in variables:
            setting[index]=globals()[index]

        setting_df = pd.DataFrame(setting)

        dB_df=pd.DataFrame(dBcum)
        Pinit_df=pd.DataFrame(Pinit)
        Vinit_df=pd.DataFrame(Vinit)
        Z_df=pd.DataFrame(Z)
        A_df=pd.DataFrame(A)

        setting_df.to_csv('setting_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        dB_df.to_csv('dB_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        Pinit_df.to_csv('Pinit_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        Vinit_df.to_csv('Vinit_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        Z_df.to_csv('Z_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")
        A_df.to_csv('A_k%.2fm%.2fs%.4fnet%d.%d.%dN%dpsLB%.2fphLB%.2ftrial%d_cut%0.2fstd.csv' %(K,M,L,nettype[0],nettype[1],nettype[2],N,psLB,phLB,Trial,cut),index=False,sep="\t")

def makenet(net_type):
    A=np.zeros((N,N))
    
    if net_type==0 :
        A[:,:]=1
        for i in range(N) :
            A[i,i]=0

    elif net_type==1 :
        for i in range(N-1) :
            A[i,i+1] = 1

        A[0,1:N]=1
        A[0:int(N/2)-1,int(N/2)-1]=1
        A[int(N/2)-1,int(N/2):N]=1
        A[:N-1,N-1]=1

        A=A+A.T

    elif net_type==2 :
        for i in range(N-1) :
            A[i,i+1] = 1

        A=A+A.T

    elif net_type==3 :
        if N % 2 != 0 :
            step=2
        elif N % 4 == 0 :
            step=N/2-1
        else :
            step=N/2-2

        for i in range(N) :
            A[i,int((i+step)%N)]=1
            A[i,int((i-step)%N)]=1

    elif net_type==4 :
        for i in range(N-1) :
            A[i,i+1] = 1

        A=A+A.T

        for i in range(int(N/10)) :
            A[i*10,:]=1
            A[:,i*10]=1
            A[i*10,i*10]=0

    elif net_type==5 :
        for i in range(N-1) :
            A[i,i+1] = 1
        A[N-1,0]=1

        A=A+A.T


    return A

def phiEest(ssq,b,LB):
    phE=(np.power(1+ssq,1-b)-1)/(1-b)+ssq*LB
    return phE

def psi(s,b,LB):
    a = np.power((1+s),-b) + LB
    return a

def csmpf(X,V):

    Kem=np.zeros((N,2))

    for i in range(N):

        J= A_ps[i,:] == 1

        s=(np.power(X[i][0]-X[:,0],2)+np.power(X[i][1]-X[:,1],2))[J]
        
        ps=psi(s,alpha,psLB)

        a=np.sum((V[J]-V[i])*np.array([ps]).T,axis=0) * K
        #행렬의 각 행별로 array에 저장된 scalar를 곱하려고 하려면 dimension이 같아야함
        #즉 여기선 n*2 행렬이 있고 거기에서 각각의 행에 곱할 scalar가 ps에 저장되어 있는데,
        #1dimension으로 n짜리 array로는 못하고, n*1 사이즈의 2차원 행렬이어야 함.

        # a/=N

        J= A_ph[i,:] == 1

        u=np.array([0.0,0.0])

        ss=(np.power(X[i][0]-Z[i][0]-X[:,0]+Z[:,0],2)+np.power(X[i][1]-Z[i][1]-X[:,1]+Z[:,1],2))[J]
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

def theta(x):
    a=np.heaviside(x,0)
    return a

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
    elif curvename=='einstein':
        X = ((-38/9 *np.sin(11/7 - 3 *t) + 156/5 *np.sin(t + 47/10) + 91/16 *np.sin(2 *t + 21/13) + 555/2) *theta(91 *np.pi - t) *theta(t - 87 *np.pi) + (-12/11 *np.sin(35/23 - 11 *t) + 4243/12 *np.sin(t + 11/7) + 678/11 *np.sin(2 *t + 33/7) + 401/6 *np.sin(3 *t + 47/10) + 59/3 *np.sin(4 *t + 11/7) + 238/25 *np.sin(5 *t + 47/10) + 85/11 *np.sin(6 *t + 51/11) + 57/4 *np.sin(7 *t + 61/13) + 28/29 *np.sin(8 *t + 22/5) + 52/9 *np.sin(9 *t + 14/3) + 286/57 *np.sin(10 *t + 11/7) + 19/11 *np.sin(12 *t + 32/7) + 30/11 *np.sin(13 *t + 60/13) + 95/14 *np.sin(14 *t + 89/19) + 32/7 *np.sin(15 *t + 11/7) + 43/10 *np.sin(16 *t + 65/14) + 19/7 *np.sin(17 *t + 32/7) + 13/10 *np.sin(18 *t + 77/17) + 11/9 *np.sin(19 *t + 85/19) + 1/5 *np.sin(20 *t + 4) + 3/11 *np.sin(21 *t + 28/9) + 29/11 *np.sin(22 *t + 60/13) + 80/27 *np.sin(23 *t + 50/11) + 19/12 *np.sin(24 *t + 60/13) + 1/5 *np.sin(25 *t + 12/5) + 82/13 *np.sin(26 *t + 51/11) + 3/11 *np.sin(27 *t + 19/8) + 32/9 *np.sin(28 *t + 10/7) + 41/7 *np.sin(29 *t + 22/15) + 9/11 *np.sin(30 *t + 11/8) + 2881/6) *theta(87 *np.pi - t) *theta(t - 83 *np.pi) + (-46/31 *np.sin(20/13 - 22 *t) - 22/9 *np.sin(14/9 - 6 *t) - 5/4 *np.sin(3/2 - 4 *t) + 399/5 *np.sin(t + 11/7) + 16/9 *np.sin(2 *t + 3/2) + 116/13 *np.sin(3 *t + 14/9) + 8/5 *np.sin(5 *t + 14/9) + 11/7 *np.sin(7 *t + 8/5) + 9/11 *np.sin(8 *t + 14/3) + 28/13 *np.sin(9 *t + 11/7) + 7/8 *np.sin(10 *t + 11/7) + 23/12 *np.sin(11 *t + 17/11) + 11/12 *np.sin(12 *t + 19/13) + 35/23 *np.sin(13 *t + 3/2) + 13/7 *np.sin(14 *t + 20/13) + 19/9 *np.sin(15 *t + 3/2) + 11/5 *np.sin(16 *t + 3/2) + 27/13 *np.sin(17 *t + 34/23) + 3 *np.sin(18 *t + 26/17) + 6/5 *np.sin(19 *t + 7/5) + 19/12 *np.sin(20 *t + 29/19) + 20/13 *np.sin(21 *t + 21/13) + 8/9 *np.sin(23 *t + 32/7) + 22/23 *np.sin(24 *t + 23/5) + 17/11 *np.sin(25 *t + 61/13) + 13021/30) *theta(83 *np.pi - t) *theta(t - 79 *np.pi) + (-15/31 *np.sin(11/7 - 8 *t) + 1/15 *np.sin(t + 11/6) + 55/14 *np.sin(2 *t + 19/12) + 88/13 *np.sin(3 *t + 19/12) + 17/9 *np.sin(4 *t + 8/5) + 1/18 *np.sin(5 *t + 16/9) + 4/7 *np.sin(6 *t + 21/13) + 9/8 *np.sin(7 *t + 8/5) + 8/15 *np.sin(9 *t + 8/5) + 3053/7) *theta(79 *np.pi - t) *theta(t - 75 *np.pi) + (-20/3 *np.sin(11/7 - 4 *t) - 117/8 *np.sin(11/7 - 3 *t) - 647/27 *np.sin(11/7 - 2 *t) + 559/15 *np.sin(t + 11/7) + 2/13 *np.sin(5 *t + 13/8) + 6/17 *np.sin(6 *t + 18/11) + 5/8 *np.sin(7 *t + 8/5) + 22549/41) *theta(75 *np.pi - t) *theta(t - 71 *np.pi) + (-11/9 *np.sin(17/11 - 10 *t) - 40/13 *np.sin(14/9 - 8 *t) - 254/23 *np.sin(11/7 - 4 *t) - 62/7 *np.sin(11/7 - 2 *t) + 11 *np.sin(t + 11/7) + 255/16 *np.sin(3 *t + 11/7) + 137/10 *np.sin(5 *t + 19/12) + 111/8 *np.sin(6 *t + 19/12) + 29/19 *np.sin(7 *t + 8/5) + 2/9 *np.sin(9 *t + 26/17) + 11/12 *np.sin(11 *t + 19/12) + 1/24 *np.sin(12 *t + 41/9) + 8/9 *np.sin(14 *t + 13/8) + 1313/3) *theta(71 *np.pi - t) *theta(t - 67 *np.pi) + (-5/8 *np.sin(14/9 - 8 *t) - 11/13 *np.sin(14/9 - 7 *t) - 12/5 *np.sin(11/7 - 6 *t) - 7/9 *np.sin(14/9 - 3 *t) - 272/13 *np.sin(11/7 - 2 *t) + 7/2 *np.sin(t + 11/7) + 3/4 *np.sin(4 *t + 14/9) + 7/9 *np.sin(5 *t + 11/7) + 3/13 *np.sin(9 *t + 11/7) + 4876/9) *theta(67 *np.pi - t) *theta(t - 63 *np.pi) + (-22/9 *np.sin(11/7 - t) + 177/7 *np.sin(2 *t + 11/7) + 21/10 *np.sin(3 *t + 11/7) + 11/7 *np.sin(4 *t + 11/7) + 1/14 *np.sin(5 *t + 17/10) + 66/19 *np.sin(6 *t + 11/7) + 1/22 *np.sin(7 *t + 12/7) + 20/13 *np.sin(8 *t + 11/7) + 3561/10) *theta(63 *np.pi - t) *theta(t - 59 *np.pi) + (-9/17 *np.sin(25/17 - 11 *t) - 1/2 *np.sin(25/17 - 10 *t) - 1/5 *np.sin(9/7 - 9 *t) - 1/3 *np.sin(4/3 - 8 *t) - 7/3 *np.sin(14/9 - 7 *t) - 208/25 *np.sin(14/9 - 4 *t) + 139/3 *np.sin(t + 11/7) + 186/5 *np.sin(2 *t + 11/7) + 19/6 *np.sin(3 *t + 8/5) + 19/12 *np.sin(5 *t + 8/5) + 3/13 *np.sin(6 *t + 7/4) + 2/5 *np.sin(12 *t + 13/8) + 1/9 *np.sin(13 *t + 65/14) + 6/13 *np.sin(14 *t + 18/11) + 1/8 *np.sin(15 *t + 5/3) + 1/8 *np.sin(16 *t + 7/4) + 1/18 *np.sin(17 *t + 24/11) + 1737/4) *theta(59 *np.pi - t) *theta(t - 55 *np.pi) + (-6/13 *np.sin(23/15 - 21 *t) - 3/10 *np.sin(10/7 - 20 *t) - 7/8 *np.sin(26/17 - 19 *t) - 1/4 *np.sin(19/13 - 18 *t) - 11/17 *np.sin(17/11 - 17 *t) - 1/8 *np.sin(11/9 - 16 *t) - 7/8 *np.sin(17/11 - 15 *t) - 38/39 *np.sin(11/7 - 13 *t) - 57/10 *np.sin(14/9 - 7 *t) - 1/7 *np.sin(3/5 - 6 *t) - 201/10 *np.sin(14/9 - 5 *t) - 28/11 *np.sin(17/11 - 4 *t) - 303/10 *np.sin(14/9 - 3 *t) + 1084/9 *np.sin(t + 11/7) + 39/7 *np.sin(2 *t + 14/9) + 23/14 *np.sin(8 *t + 14/9) + 22/23 *np.sin(9 *t + 47/10) + 8/13 *np.sin(10 *t + 11/7) + 1/8 *np.sin(11 *t + 22/13) + 10/19 *np.sin(12 *t + 11/7) + 9/13 *np.sin(14 *t + 21/13) + 1/8 *np.sin(22 *t + 11/7) + 1319/3) *theta(55 *np.pi - t) *theta(t - 51 *np.pi) + (-3/2 *np.sin(11/7 - 17 *t) - 9/8 *np.sin(14/9 - 15 *t) - 12/7 *np.sin(14/9 - 14 *t) - 8/7 *np.sin(14/9 - 12 *t) - 6/19 *np.sin(3/2 - 11 *t) - 296/11 *np.sin(11/7 - 5 *t) - 163/25 *np.sin(11/7 - 4 *t) - 721/20 *np.sin(11/7 - 3 *t) - 85/4 *np.sin(11/7 - 2 *t) + 1353/7 *np.sin(t + 11/7) + 31/11 *np.sin(6 *t + 8/5) + 113/10 *np.sin(7 *t + 33/7) + 27/7 *np.sin(8 *t + 14/9) + 23/8 *np.sin(9 *t + 33/7) + 7/6 *np.sin(10 *t + 13/8) + 5/12 *np.sin(13 *t + 37/8) + 2/3 *np.sin(16 *t + 51/11) + 3/8 *np.sin(18 *t + 8/5) + 7126/15) *theta(51 *np.pi - t) *theta(t - 47 *np.pi) + (-2/9 *np.sin(1/3 - 4 *t) + 791/5 *np.sin(t + 11/7) + 10/19 *np.sin(2 *t + 9/14) + 118/7 *np.sin(3 *t + 14/9) + 21/4 *np.sin(5 *t + 11/7) + 1/9 *np.sin(6 *t + 117/58) + 30/11 *np.sin(7 *t + 14/9) + 5/13 *np.sin(8 *t + 17/14) + 7/4 *np.sin(9 *t + 28/19) + 3/14 *np.sin(10 *t + 15/16) + 12/13 *np.sin(11 *t + 19/12) + 1/15 *np.sin(12 *t + 43/13) + 11/16 *np.sin(13 *t + 13/8) + 2251/5) *theta(47 *np.pi - t) *theta(t - 43 *np.pi) + (3724/25 *np.sin(t + 11/7) + 1/3 *np.sin(2 *t + 16/9) + 266/17 *np.sin(3 *t + 11/7) + 10/13 *np.sin(4 *t + 19/11) + 34/7 *np.sin(5 *t + 19/12) + 5/12 *np.sin(6 *t + 5/3) + 20/11 *np.sin(7 *t + 8/5) + 1/5 *np.sin(8 *t + 11/7) + 7/5 *np.sin(9 *t + 19/12) + 2/7 *np.sin(10 *t + 5/3) + 7/8 *np.sin(11 *t + 14/9) + 1/51 *np.sin(12 *t + 47/16) + 7/9 *np.sin(13 *t + 13/8) + 1/10 *np.sin(14 *t + 50/11) + 12403/28) *theta(43 *np.pi - t) *theta(t - 39 *np.pi) + (-4/7 *np.sin(5/9 - 19 *t) + 4341/11 *np.sin(t + 17/11) + 595/6 *np.sin(2 *t + 14/3) + 1286/17 *np.sin(3 *t + 37/8) + 314/9 *np.sin(4 *t + 23/15) + 121/3 *np.sin(5 *t + 37/8) + 222/17 *np.sin(6 *t + 21/5) + 103/9 *np.sin(7 *t + 23/5) + 29/5 *np.sin(8 *t + 25/6) + 127/9 *np.sin(9 *t + 49/11) + 11/6 *np.sin(10 *t + 37/19) + 23/3 *np.sin(11 *t + 23/5) + 77/13 *np.sin(12 *t + 23/12) + 97/7 *np.sin(13 *t + 41/9) + 29/7 *np.sin(14 *t + 17/8) + 39/7 *np.sin(15 *t + 49/11) + 5/8 *np.sin(16 *t + 19/11) + 5/11 *np.sin(17 *t + 17/9) + 2/3 *np.sin(18 *t + 27/7) + 19/13 *np.sin(20 *t + 37/12) + 84/13 *np.sin(21 *t + 25/6) + 11/23 *np.sin(22 *t + 41/14) + 45/13 *np.sin(23 *t + 31/32) + 3/14 *np.sin(24 *t + 41/20) + 49/13 *np.sin(25 *t + 41/10) + 16/11 *np.sin(26 *t + 17/11) + 12/7 *np.sin(27 *t + 22/5) + 37/13 *np.sin(28 *t + 48/13) + 4/3 *np.sin(29 *t + 3) + 31/11 *np.sin(30 *t + 3/10) + 79/15 *np.sin(31 *t + 10/11) + 10753/21) *theta(39 *np.pi - t) *theta(t - 35 *np.pi) + (-16/9 *np.sin(13/9 - 8 *t) - 108/19 *np.sin(8/11 - 6 *t) + 17/13 *np.sin(t + 8/7) + 7/3 *np.sin(2 *t + 21/10) + 24/7 *np.sin(3 *t + 20/9) + 26/7 *np.sin(4 *t + 32/7) + 26/11 *np.sin(5 *t + 11/4) + 105/19 *np.sin(7 *t + 30/7) + 6/7 *np.sin(9 *t + 5/11) + 23/15 *np.sin(10 *t + 7/5) + 11/6 *np.sin(11 *t + 11/3) + 12822/23) *theta(35 *np.pi - t) *theta(t - 31 *np.pi) + (-5/8 *np.sin(11/12 - 10 *t) - 64/13 *np.sin(13/14 - 6 *t) + 7/5 *np.sin(t + 45/11) + 74/21 *np.sin(2 *t + 1/7) + 52/15 *np.sin(3 *t + 39/10) + 5/8 *np.sin(4 *t + 3/5) + 17/11 *np.sin(5 *t + 7/6) + 39/8 *np.sin(7 *t + 51/13) + 15/8 *np.sin(8 *t + 29/8) + 16/9 *np.sin(9 *t + 14/3) + 97/48 *np.sin(11 *t + 5/9) + 3401/10) *theta(31 *np.pi - t) *theta(t - 27 *np.pi) + (-12/25 *np.sin(17/13 - 6 *t) - 7/11 *np.sin(4/7 - 4 *t) - 14/27 *np.sin(3/13 - 2 *t) + 351/10 *np.sin(t + 11/8) + 17/6 *np.sin(3 *t + 28/27) + 9/8 *np.sin(5 *t + 10/13) + 3921/7) *theta(27 *np.pi - t) *theta(t - 23 *np.pi) + (431/8 *np.sin(t + 4/5) + 199/25 *np.sin(2 *t + 40/9) + 2328/7) *theta(23 *np.pi - t) *theta(t - 19 *np.pi) + (-2/3 *np.sin(5/4 - 9 *t) - 11/9 *np.sin(4/3 - 5 *t) - 74/21 *np.sin(1/13 - 4 *t) + 107/6 *np.sin(t + 8/17) + 73/10 *np.sin(2 *t + 12/11) + 53/12 *np.sin(3 *t + 48/11) + 4/9 *np.sin(6 *t + 31/13) + 4/11 *np.sin(7 *t + 5/13) + 5/14 *np.sin(8 *t + 127/42) + 5/16 *np.sin(10 *t + 17/9) + 2/5 *np.sin(11 *t + 29/7) + 2378/13) *theta(19 *np.pi - t) *theta(t - 15 *np.pi) + (194/13 *np.sin(t + 51/14) + 93/23 *np.sin(2 *t + 43/12) + 13/8 *np.sin(3 *t + 57/17) + 9/5 *np.sin(4 *t + 32/13) + 14050/21) *theta(15 *np.pi - t) *theta(t - 11 *np.pi) + (-19/18 *np.sin(1/11 - 16 *t) - 8/11 *np.sin(1/6 - 14 *t) - 13/11 *np.sin(1 - 7 *t) - 9/8 *np.sin(7/11 - 5 *t) - 148/9 *np.sin(1/7 - 2 *t) + 19/6 *np.sin(t + 37/8) + 625/11 *np.sin(3 *t + 8/5) + 241/24 *np.sin(4 *t + 1/6) + 16/17 *np.sin(6 *t + 7/5) + 95/47 *np.sin(8 *t + 1/4) + 20/9 *np.sin(9 *t + 12/7) + 11/5 *np.sin(10 *t + 1/4) + 3/7 *np.sin(11 *t + 2/3) + 9/19 *np.sin(12 *t + 28/9) + 3/5 *np.sin(13 *t + 25/6) + 2/11 *np.sin(15 *t + 13/9) + 1/3 *np.sin(17 *t + 1/6) + 3925/7) *theta(11 *np.pi - t) *theta(t - 7 *np.pi) + (-31/12 *np.sin(11/12 - 6 *t) - 244/9 *np.sin(15/11 - 4 *t) - 186/5 *np.sin(7/6 - 2 *t) + 911/26 *np.sin(t + 74/21) + 317/7 *np.sin(3 *t + 1/3) + 28/9 *np.sin(5 *t + 52/15) + 33/17 *np.sin(7 *t + 12/5) + 7/10 *np.sin(8 *t + 13/7) + 6/7 *np.sin(9 *t + 9/5) + 6/7 *np.sin(10 *t + 11/4) + 13/5 *np.sin(11 *t + 4/7) + 2721/8) *theta(7 *np.pi - t) *theta(t - 3 *np.pi) + (-10/7 *np.sin(14/9 - 12 *t) - 11/7 *np.sin(7/9 - 11 *t) - 51/19 *np.sin(3/2 - 4 *t) - 89/4 *np.sin(18/13 - 3 *t) - 81/10 *np.sin(12/25 - 2 *t) + 2029/8 *np.sin(t + 3/2) + 3 *np.sin(5 *t + 3/5) + 23/15 *np.sin(6 *t + 29/10) + 74/15 *np.sin(7 *t + 51/25) + 10/11 *np.sin(8 *t + 32/21) + 13/6 *np.sin(9 *t + 8/5) + 2/7 *np.sin(10 *t + 16/7) + 4407/10) *theta(3 *np.pi - t) *theta(t +np.pi)) *theta(np.sin(t/2))
        Y = ((41/2 *np.sin(t + 61/13) + 163/18 *np.sin(2 *t + 14/3) + 1/2 *np.sin(3 *t + 41/9) + 3802/5) *theta(91 *np.pi - t) *theta(t - 87 *np.pi) + (-12/7 *np.sin(7/5 - 17 *t) - 41/11 *np.sin(11/7 - 9 *t) - 3/7 *np.sin(11/8 - 4 *t) + 1175/14 *np.sin(t + 47/10) + 9961/40 *np.sin(2 *t + 33/7) + 555/8 *np.sin(3 *t + 11/7) + 39/5 *np.sin(5 *t + 14/9) + 11/5 *np.sin(6 *t + 3/2) + 25/2 *np.sin(7 *t + 47/10) + 155/12 *np.sin(8 *t + 14/9) + 33/10 *np.sin(10 *t + 19/12) + 14/5 *np.sin(11 *t + 51/11) + 64/7 *np.sin(12 *t + 14/3) + 45/7 *np.sin(13 *t + 11/7) + 1/14 *np.sin(14 *t + 49/13) + 1/2 *np.sin(15 *t + 16/13) + 76/25 *np.sin(16 *t + 19/12) + 23/5 *np.sin(18 *t + 26/17) + 191/38 *np.sin(19 *t + 47/10) + 47/13 *np.sin(20 *t + 23/15) + 62/9 *np.sin(21 *t + 33/7) + 31/9 *np.sin(22 *t + 37/25) + 31/4 *np.sin(23 *t + 16/11) + 18/7 *np.sin(24 *t + 4/3) + 91/15 *np.sin(25 *t + 3/2) + 29/7 *np.sin(26 *t + 14/3) + 49/25 *np.sin(27 *t + 47/10) + 9/4 *np.sin(28 *t + 23/5) + 57/56 *np.sin(29 *t + 6/5) + 83/10 *np.sin(30 *t + 16/11) + 18532/29) *theta(87 *np.pi - t) *theta(t - 83 *np.pi) + (-9/7 *np.sin(4/3 - 25 *t) - 106/11 *np.sin(16/11 - 22 *t) - 11/3 *np.sin(17/11 - 11 *t) - 1/17 *np.sin(1/16 - 9 *t) - 2/9 *np.sin(3/2 - 8 *t) - 2/9 *np.sin(11/9 - 6 *t) + 38/39 *np.sin(t + 14/3) + 9/5 *np.sin(2 *t + 61/13) + 19/7 *np.sin(3 *t + 8/5) + 22/5 *np.sin(4 *t + 33/7) + 8/11 *np.sin(5 *t + 3/2) + 95/94 *np.sin(7 *t + 14/9) + 25/13 *np.sin(10 *t + 13/8) + 3/5 *np.sin(12 *t + 14/3) + 2/11 *np.sin(13 *t + 17/4) + 35/11 *np.sin(14 *t + 14/3) + 17/5 *np.sin(15 *t + 51/11) + 84/13 *np.sin(16 *t + 89/19) + 51/8 *np.sin(17 *t + 51/11) + 5/8 *np.sin(18 *t + 17/5) + 35/6 *np.sin(19 *t + 61/13) + 11/9 *np.sin(20 *t + 9/2) + 21/13 *np.sin(21 *t + 27/16) + 77/12 *np.sin(23 *t + 8/5) + 151/14 *np.sin(24 *t + 21/13) + 2152/7) *theta(83 *np.pi - t) *theta(t - 79 *np.pi) + (-14/11 *np.sin(20/13 - 7 *t) - 47/8 *np.sin(14/9 - 3 *t) - 388/7 *np.sin(11/7 - t) + 18/11 *np.sin(2 *t + 3/2) + 4/3 *np.sin(4 *t + 19/12) + 19/14 *np.sin(5 *t + 47/10) + 3/11 *np.sin(6 *t + 25/17) + 1/24 *np.sin(8 *t + 9/14) + 1/3 *np.sin(9 *t + 47/10) + 5435/13) *theta(79 *np.pi - t) *theta(t - 75 *np.pi) + (-5/2 *np.sin(14/9 - 5 *t) - 42/11 *np.sin(11/7 - 3 *t) - 237/19 *np.sin(11/7 - t) + 86/3 *np.sin(2 *t + 11/7) + 14/15 *np.sin(4 *t + 11/7) + 17/8 *np.sin(6 *t + 11/7) + 15/16 *np.sin(7 *t + 8/5) + 4683/10) *theta(75 *np.pi - t) *theta(t - 71 *np.pi) + (-5/7 *np.sin(14/9 - 13 *t) - 11/16 *np.sin(14/9 - 9 *t) - 13/6 *np.sin(11/7 - 5 *t) - 2/7 *np.sin(20/13 - 4 *t) - np.sin(11/7 - 3 *t) - 341/34 *np.sin(11/7 - t) + 5/3 *np.sin(2 *t + 11/7) + 19/8 *np.sin(6 *t + 19/12) + 1/11 *np.sin(7 *t + 55/12) + 7/6 *np.sin(8 *t + 19/12) + 3/8 *np.sin(10 *t + 11/7) + 1/10 *np.sin(11 *t + 5/3) + 1/2 *np.sin(12 *t + 19/12) + 7/10 *np.sin(14 *t + 8/5) + 469/2) *theta(71 *np.pi - t) *theta(t - 67 *np.pi) + (-3/10 *np.sin(14/9 - 8 *t) + 16/11 *np.sin(t + 75/16) + 63/2 *np.sin(2 *t + 11/7) + 5/7 *np.sin(3 *t + 8/5) + 2/3 *np.sin(4 *t + 13/8) + 1/33 *np.sin(5 *t + 9/2) + 23/7 *np.sin(6 *t + 11/7) + 1/29 *np.sin(7 *t + 14/3) + 1/5 *np.sin(9 *t + 61/13) + 3265/9) *theta(67 *np.pi - t) *theta(t - 63 *np.pi) + (-16/13 *np.sin(11/7 - 4 *t) - 59/12 *np.sin(11/7 - t) + 183/5 *np.sin(2 *t + 11/7) + 5/4 *np.sin(3 *t + 14/9) + 8/7 *np.sin(5 *t + 47/10) + 80/27 *np.sin(6 *t + 11/7) + 14/13 *np.sin(7 *t + 19/12) + 20/19 *np.sin(8 *t + 33/7) + 10934/29) *theta(63 *np.pi - t) *theta(t - 59 *np.pi) + (-7/9 *np.sin(29/19 - 15 *t) - 121/60 *np.sin(17/11 - 5 *t) - 742/11 *np.sin(11/7 - t) + 494/11 *np.sin(2 *t + 19/12) + 74/15 *np.sin(3 *t + 19/12) + 78/7 *np.sin(4 *t + 21/13) + 47/10 *np.sin(6 *t + 13/8) + 35/17 *np.sin(7 *t + 27/16) + 17/7 *np.sin(8 *t + 19/12) + 5/16 *np.sin(9 *t + 19/8) + 22/9 *np.sin(10 *t + 11/7) + 2/11 *np.sin(11 *t + 39/10) + 10/11 *np.sin(12 *t + 19/12) + 5/13 *np.sin(13 *t + 12/7) + 3/7 *np.sin(14 *t + 23/14) + 1/4 *np.sin(16 *t + 18/11) + 1/12 *np.sin(17 *t + 15/7) + 4470/11) *theta(59 *np.pi - t) *theta(t - 55 *np.pi) + (-2/9 *np.sin(17/11 - 21 *t) - 9/7 *np.sin(3/2 - 18 *t) - 3/10 *np.sin(22/15 - 17 *t) - 23/7 *np.sin(14/9 - 8 *t) + 18/11 *np.sin(t + 8/5) + 155/4 *np.sin(2 *t + 11/7) + 9/7 *np.sin(3 *t + 28/17) + 173/10 *np.sin(4 *t + 11/7) + 14/13 *np.sin(5 *t + 75/16) + 22/9 *np.sin(6 *t + 8/5) + 1/7 *np.sin(7 *t + 16/9) + 5/3 *np.sin(9 *t + 8/5) + 9/8 *np.sin(10 *t + 8/5) + 16/9 *np.sin(11 *t + 8/5) + 8/3 *np.sin(12 *t + 8/5) + 3/13 *np.sin(13 *t + 14/3) + 29/30 *np.sin(14 *t + 11/7) + 1/6 *np.sin(15 *t + 16/9) + 7/8 *np.sin(16 *t + 28/17) + 5/16 *np.sin(19 *t + 18/11) + 11/12 *np.sin(20 *t + 18/11) + 1/7 *np.sin(22 *t + 9/7) + 3262/11) *theta(55 *np.pi - t) *theta(t - 51 *np.pi) + (-7/8 *np.sin(17/11 - 18 *t) - 7/6 *np.sin(17/11 - 17 *t) - 3/10 *np.sin(23/15 - 15 *t) - 17/10 *np.sin(11/7 - 10 *t) - 24/7 *np.sin(14/9 - 9 *t) - 24/25 *np.sin(14/9 - 8 *t) - 40/11 *np.sin(11/7 - 7 *t) + 19/10 *np.sin(t + 33/7) + 39/7 *np.sin(2 *t + 11/7) + 162/19 *np.sin(3 *t + 11/7) + 123/8 *np.sin(4 *t + 11/7) + 33/7 *np.sin(5 *t + 11/7) + 77/9 *np.sin(6 *t + 11/7) + 21/22 *np.sin(11 *t + 8/5) + 9/17 *np.sin(12 *t + 21/13) + 31/12 *np.sin(13 *t + 19/12) + 1/20 *np.sin(14 *t + 23/5) + 5/14 *np.sin(16 *t + 8/5) + 16814/23) *theta(51 *np.pi - t) *theta(t - 47 *np.pi) + (-102/11 *np.sin(11/7 - t) + 29/7 *np.sin(2 *t + 33/7) + 27/10 *np.sin(3 *t + 19/12) + 17/7 *np.sin(4 *t + 47/10) + 2/11 *np.sin(5 *t + 13/7) + 37/14 *np.sin(6 *t + 47/10) + 1/13 *np.sin(7 *t + 85/21) + 51/26 *np.sin(8 *t + 47/10) + 5/6 *np.sin(9 *t + 8/5) + 3/5 *np.sin(10 *t + 47/10) + 8/13 *np.sin(11 *t + 8/5) + 9/13 *np.sin(12 *t + 47/10) + 1/10 *np.sin(13 *t + 53/12) + 9028/13) *theta(47 *np.pi - t) *theta(t - 43 *np.pi) + (-1/2 *np.sin(17/11 - 14 *t) - 15/11 *np.sin(14/9 - 10 *t) - 29/11 *np.sin(14/9 - 8 *t) - 29/9 *np.sin(14/9 - 6 *t) - 1/3 *np.sin(13/9 - 5 *t) - 108/13 *np.sin(14/9 - 4 *t) - 12/7 *np.sin(14/9 - t) + 4/13 *np.sin(2 *t + 8/5) + 15/11 *np.sin(3 *t + 11/7) + 7/6 *np.sin(7 *t + 11/7) + 1/15 *np.sin(9 *t + 5/4) + 1/9 *np.sin(11 *t + 16/11) + 1/10 *np.sin(12 *t + 12/7) + 3/8 *np.sin(13 *t + 8/5) + 5872/9) *theta(43 *np.pi - t) *theta(t - 39 *np.pi) + (-6/7 *np.sin(38/25 - 30 *t) - 6/5 *np.sin(1/21 - 28 *t) - 13/8 *np.sin(7/9 - 18 *t) + 275/3 *np.sin(t + 23/5) + 3929/11 *np.sin(2 *t + 14/3) + 219/4 *np.sin(3 *t + 27/16) + 421/11 *np.sin(4 *t + 47/10) + 101/6 *np.sin(5 *t + 26/17) + 242/9 *np.sin(6 *t + 4/3) + 153/13 *np.sin(7 *t + 1) + 73/6 *np.sin(8 *t + 5/4) + 65/9 *np.sin(9 *t + 13/11) + 47/14 *np.sin(10 *t + 9/7) + 51/11 *np.sin(11 *t + 25/6) + 25/7 *np.sin(12 *t + 17/13) + 13/2 *np.sin(13 *t + 9/8) + 40/17 *np.sin(14 *t + 16/17) + 36/7 *np.sin(15 *t + 46/47) + 2 *np.sin(16 *t + 2/7) + 52/21 *np.sin(17 *t + 10/7) + 55/12 *np.sin(19 *t + 6/5) + 17/8 *np.sin(20 *t + 1/3) + 17/6 *np.sin(21 *t + 58/57) + 37/12 *np.sin(22 *t + 35/8) + 3/4 *np.sin(23 *t + 12/13) + 28/13 *np.sin(24 *t + 4/5) + 37/19 *np.sin(25 *t + 19/5) + 7/10 *np.sin(26 *t + 55/13) + 89/14 *np.sin(27 *t + 7/8) + 15/7 *np.sin(29 *t + 23/6) + 7/11 *np.sin(31 *t + 11/14) + 8933/13) *theta(39 *np.pi - t) *theta(t - 35 *np.pi) + (-17/9 *np.sin(9/14 - 11 *t) - 4/3 *np.sin(1/5 - 8 *t) - 29/6 *np.sin(3/8 - 7 *t) + 13/8 *np.sin(t + 11/6) + 6/5 *np.sin(2 *t + 30/7) + 8/7 *np.sin(3 *t + 31/11) + 13/6 *np.sin(4 *t + 1/11) + 4/5 *np.sin(5 *t + 31/7) + 31/9 *np.sin(6 *t + 8/11) + 1/3 *np.sin(9 *t + 39/20) + 9/5 *np.sin(10 *t + 13/4) + 7555/14) *theta(35 *np.pi - t) *theta(t - 31 *np.pi) + (-11/10 *np.sin(10/9 - 8 *t) - 9/2 *np.sin(2/5 - 7 *t) - 18/11 *np.sin(10/11 - 3 *t) + 17/9 *np.sin(t + 32/7) + 6/5 *np.sin(2 *t + 38/13) + 19/14 *np.sin(4 *t + 28/9) + 13/9 *np.sin(5 *t + 3) + 15/4 *np.sin(6 *t + 3/4) + 60/17 *np.sin(9 *t + 1/14) + 10/9 *np.sin(10 *t + 5/4) + 13/7 *np.sin(11 *t + 30/13) + 9899/18) *theta(31 *np.pi - t) *theta(t - 27 *np.pi) + (-2/11 *np.sin(2/9 - 5 *t) + 110/7 *np.sin(t + 35/12) + 16/9 *np.sin(2 *t + 68/15) + 3/14 *np.sin(3 *t + 36/13) + 1/2 *np.sin(4 *t + 7/2) + 1/7 *np.sin(6 *t + 45/13) + 2682/5) *theta(27 *np.pi - t) *theta(t - 23 *np.pi) + (157/9 *np.sin(t + 69/34) + 19/3 *np.sin(2 *t + 20/7) + 2169/4) *theta(23 *np.pi - t) *theta(t - 19 *np.pi) + (-3/2 *np.sin(3/13 - 7 *t) - 13/4 *np.sin(11/12 - 4 *t) - 131/7 *np.sin(5/4 - 2 *t) + 370/7 *np.sin(t + 74/17) + 31/3 *np.sin(3 *t + 47/16) + 11/4 *np.sin(5 *t + 50/11) + 43/11 *np.sin(6 *t + 19/7) + 23/14 *np.sin(8 *t + 33/10) + 3/5 *np.sin(9 *t + 21/11) + 1/10 *np.sin(10 *t + 1/16) + 1/3 *np.sin(11 *t + 62/25) + 5541/11) *theta(19 *np.pi - t) *theta(t - 15 *np.pi) + (171/4 *np.sin(t + 37/8) + 7/9 *np.sin(2 *t + 18/13) + 41/10 *np.sin(3 *t + 40/9) + 6/11 *np.sin(4 *t + 15/11) + 5012/11) *theta(15 *np.pi - t) *theta(t - 11 *np.pi) + (-12/13 *np.sin(7/5 - 12 *t) - 13/8 *np.sin(13/11 - 10 *t) + 43/12 *np.sin(t + 7/9) + 279/35 *np.sin(2 *t + 9/2) + 201/14 *np.sin(3 *t + 2/9) + 23/9 *np.sin(4 *t + 8/7) + 64/9 *np.sin(5 *t + 14/5) + 83/6 *np.sin(6 *t + 14/3) + 103/17 *np.sin(7 *t + 13/4) + 36/13 *np.sin(8 *t + 46/11) + 22/7 *np.sin(9 *t + 2/7) + 8/9 *np.sin(11 *t + 11/4) + 20/11 *np.sin(13 *t + 74/25) + 5/7 *np.sin(14 *t + 42/13) + 7/9 *np.sin(15 *t + 4/7) + 9/11 *np.sin(16 *t + 17/4) + 7/12 *np.sin(17 *t + 36/11) + 3437/6) *theta(11 *np.pi - t) *theta(t - 7 *np.pi) + (-22/7 *np.sin(7/9 - 9 *t) - 36/7 *np.sin(1 - 5 *t) - 181/26 *np.sin(6/5 - 3 *t) + 28/9 *np.sin(t + 5/6) + 131/22 *np.sin(2 *t + 26/7) + 127/13 *np.sin(4 *t + 23/5) + 21/4 *np.sin(6 *t + 1/10) + 40/3 *np.sin(7 *t + 22/23) + 88/13 *np.sin(8 *t + 23/5) + 115/38 *np.sin(10 *t + 3/7) + 11/9 *np.sin(11 *t + 11/8) + 8493/14) *theta(7 *np.pi - t) *theta(t - 3 *np.pi) + (-8/7 *np.sin(16/13 - 10 *t) - 23/10 *np.sin(4/3 - 7 *t) - 3961/12 *np.sin(1/19 - t) + 55/3 *np.sin(2 *t + 13/11) + 9/17 *np.sin(3 *t + 31/13) + 81/7 *np.sin(4 *t + 9/2) + 113/17 *np.sin(5 *t + 13/4) + 40/9 *np.sin(6 *t + 12/11) + 24/23 *np.sin(8 *t + 53/21) + 19/8 *np.sin(9 *t + 3/7) + 3/13 *np.sin(11 *t + 18/5) + 45/44 *np.sin(12 *t + 5/7) + 6798/13) *theta(3 *np.pi - t) *theta(t +np.pi)) *theta(np.sin(t/2))
    else :
        print("error : there is not such curve name")
        quit()

    Target = np.array([X,Y]).T

    if jump==1:
        
        Jcheck0=np.zeros(num)
        for i in range(num-1) :
            Jcheck0[i]=np .count_nonzero(Target[i+1:] == Target[i])

        Jcheck1= Target[:] != [[0,0]]

        Jcheck = np.logical_and(Jcheck0[:]==0,Jcheck1[:,0])

        Target=Target[Jcheck]
        t=np.linspace(0,dom,Jcheck.sum(),endpoint=False)
        print(Jcheck.sum())

    return Target,t


if __name__ == '__main__':
    
    version=1.85

    ##inputs and settings

    # N=int(input("N=?"))
    # alpha=float(input("alpha=?"))
    # beta=float(input("beta=? (if alpha = beta, input -1)"))
    # if beta==-1.0 :
    #     beta = alpha

    ##ein set

    N=500
    alpha=0.25
    beta=alpha

    psLB=0.3
    phLB=0.1

    # K=float(input("K=?"))
    # M=float(input("M=?"))
    # L=float(input("L=?"))
    # T=int(input("T=?"))

    K=0.5
    M=7
    L=0.00001 #L=sigma
    T=400 #T : number of steps for solving the DE
    T=180

    curvetype=3 
    # nettype=[0,0,0]
    nettype=[3,1,0]
    nettype=[4,4,0]
    # nettype=[4,3,0]

    PVinitset=1 #whether making or loading the initial data of P,V
    fname='_ein'
    h=0.025/2 #h=\Delta t ~ dt
    Trial=100
    cut=10
    dataset=0

    # ##pi set

    # N=30
    # alpha=0.25
    # beta=alpha

    # psLB=0.31
    # phLB=0.1

    # # K=float(input("K=?"))
    # # M=float(input("M=?"))
    # # L=float(input("L=?"))
    # # T=int(input("T=?"))

    # K=5
    # M=7
    # L=0.001 #L=sigma
    # T=400 #T : number of steps for solving the DE
    # T=1400

    # curvetype=1 
    # # nettype=[0,0,0]
    # nettype=[3,1,0]
    # # nettype=[4,3,0]

    # PVinitset=1 #whether making or loading the initial data of P,V
    # fname='_pi'
    # h=0.025 #h=\Delta t ~ dt
    # Trial=100
    # cut=100
    # dataset=0 #whether making or loading the solutions of DE


    ##Setting target pattern. /curvetype
    #jump==1 then the "curve" is disconnected
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

    elif curvetype==3:
        curvename='einstein'
        domain=92*np.pi

        jump=1


    Z,Domain=Curve(domain) #Set Z as given curve 

    # A : adjacency matrix / nettype
    zeta=['ps','ph','b']

    for ind in range(3):

        A=makenet(nettype[ind])

        globals()['A_{}'.format(zeta[ind])]=np.copy(A)
   
    print(A_ps)

    ##Setting a initial data of P,V / PVinitset
    # P : positions, V : velocity. i.e. P=\bx, V=\bv 

    if PVinitset==0:
        Pinitlen=int((np.max(Z)-np.min(Z)))
        # Pinitlen=20
        coeff=0.25
        # coeff=0.5
        # coeff=1.25
        
        P_range=Pinitlen*coeff
        print("P_range = %f" %P_range)
        Pinit=np.random.rand(N,2)*P_range-P_range/2 

        Pinit-=np.array(np.mean(Pinit[:],axis=0))
        Pinit+=np.array(np.mean(Z[:],axis=0))

        v_range=50
        v_s=np.random.randint(-10,10,size=2)
        v_s=np.array([0,0])
        print("v_s = %.1f,%.1f" %(v_s[0],v_s[1]))
        Vinit=(np.random.rand(N,2)*v_range)-v_range/2 + v_s #np.array끼리 그냥 곱하면 component 사이의 곱
        Vimean=np.array(np.mean(Vinit[:],axis=0))

        Vinit-=Vimean

    else :
        print(os.getcwd())
        Pinit_df=pd.read_csv('./Pinit%s.csv' %fname,delimiter='\t')
        Vinit_df=pd.read_csv('./Vinit%s.csv' %fname,delimiter='\t')

        Pinit=Pinit_df.to_numpy()
        Vinit=Vinit_df.to_numpy()
        print(Pinit.shape)

    Xbimean=np.array(np.mean(Pinit[:]-Z[:],axis=0))
    print("\nbarX_i mean={}".format(Xbimean))
    Vimean=np.array(np.mean(Vinit[:],axis=0))        
    print("\nVimean={}".format(Vimean))

    P=np.array([Pinit])
    V=np.array([Vinit])

    print(P[0])
    print(V[0])
    print(Vimean)

    ## Vnrm(t) = Vnrm = sum_i |v_t^i|^2
    s=np.power(Vinit[:,0],2)+np.power(Vinit[:,1],2)
    Vnrmi=np.sum(s)

    ##Xbnrm(t) = max |x*_i| . cf. sum_i,j |x*_i-x*_j|^2 = 2N sum_i |x*_i|^2 - 2|sum_i x*_i|^2 = \sum_{i,j} |\bar{x}_t^i-\bar{x}_t^j|^2
    Xbnrm=np.zeros(T)
    # Xbave=np.sum(Pinit,axis=0)-np.sum(Z,axis=0)
    Xbnrmi=np.power(np.max(np.power(Pinit[:,0]-Z[:,0],2)+np.power(Pinit[:,1]-Z[:,1],2)),0.5)

    ##phE(t)=\sum_{i,j\in\calE} \int_0^|\bar{x}_t^{ij}|^2 \phi(r)dr
    phEi=0    
    for i in range(N):
        J = A_ph[i,:]==1
        ssq=(np.power(P[0,i,0]-Z[i,0]-P[0,:,0]+Z[:,0],2)+np.power(P[0,i,1]-Z[i,1]-P[0,:,1]+Z[:,1],2))[J]

        phEi+=np.sum(phiEest(ssq,beta,phLB))

    phEi*=M/2

    dB=np.zeros(T)

    #이렇게 계속 append 하는 방식 말고 cum 미리 array 만들어 놓는게 빠른지 함 확인 해봐야 + 공간 어떤 방식이 더 많이 확보 가능한가
    #-> 속도는 큰 차이 없고 오히려 append하는게 더 빠른 것 같아서 당황

    Pcum=np.array([np.zeros((T,N,2))])
    Vcum=np.array([np.zeros((T,N,2))])
    nVcum=np.array([np.zeros((T,N,2))])

    Vnrmcum=np.array([np.zeros(T)])
    phEcum=np.array([np.zeros(T)])

    Xbnrmcum=np.array([np.zeros(T)])
    dBcum=np.array([np.zeros(T)])

    #지금 수정하기 귀찮다고 대충 했더니 cum ndarray들은 [0]=0 이고 [1]부터 가ㅄ이들어가는 상황 

    ## Main part

    if dataset==0:
        for trial in range (0,Trial):

            P=np.zeros((T,N,2))
            V=np.zeros((T,N,2))
            Vnrm=np.zeros(T)
            Xbnrm=np.zeros(T)
            dB=np.zeros(T)
            phE=np.zeros(T)
                    
            P[0]=np.array([Pinit])
            V[0]=np.array([Vinit])
            Vnrm[0]=Vnrmi
            Xbnrm[0]=Xbnrmi
            phE[0]=phEi

            ## Solving DE for each trial
            for t in range(1,T):
                # print ("start %dth loop" %t)
                
                Pnow=np.copy(P[t-1])
                Vnow=np.copy(V[t-1])

                ## Stochastic Runge Kutta.

                ## calculating P[t], V[t]
                ## P,V_t = P,V_t-1 + (K_1 /2 +K_2 /2)
                ## K_1 = h * csmpf (P,V_t-1)+ (dBt-1-S*sqrt(h)) * Br(P,V_t-1)
                ## K_2 = h * csmpf (P,V_t-1 + K_1)+ (dBt-1+S*sqrt(h)) * Br(P,V_t-1 + K_1)
                ## https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)

                dBt=dW(h)

                S=np.random.randint(0,1)*2-1

                K_1=np.array([Vnow,csmpf(Pnow,Vnow)])*h+np.array([np.zeros((N,2)),(dBt-S*np.sqrt(h))*brown(Vnow)])
                K_2=np.array([Vnow+K_1[1],csmpf(Pnow+K_1[0],Vnow+K_1[1])])*h+np.array([np.zeros((N,2)),(dBt+S*np.sqrt(h))*brown(Vnow+K_1[1])])

                Pnext=np.copy(Pnow)
                Vnext=np.copy(Vnow)

                Pnext+=(K_1[0]+K_2[0])/2
                Vnext+=(K_1[1]+K_2[1])/2

                Pnext=np.nan_to_num(Pnext)
                Vnext=np.nan_to_num(Vnext)

                ## appending P,V, dB
                P[t]=np.array([Pnext])
                V[t]=np.array([Vnext])
                dB[t]=dBt

                ## calculating and appending Vnrm, Xbnrm, phE
                s=np.power(Vnext[:,0],2)+np.power(Vnext[:,1],2)

                Vnrm[t]=np.sum(s)
                Xbnrm[t]=np.power(np.max(np.power(Pnext[:,0]-Z[:,0],2)+np.power(Pnext[:,1]-Z[:,1],2)),0.5)

                for i in range(N):
                    J = A_ph[i,:]==1
                    ssq=(np.power(P[t,i,0]-Z[i,0]-P[t,:,0]+Z[:,0],2)+np.power(P[t,i,1]-Z[i,1]-P[t,:,1]+Z[:,1],2))[J]

                    phE[t]+=np.sum(phiEest(ssq,beta,phLB))

                phE[t]*=M/2
        
            ##appending to cumulative data array
            Pcum=np.append(Pcum,np.array([P]),axis=0)
            Vcum=np.append(Vcum,np.array([V]),axis=0)
            Vnrmcum=np.append(Vnrmcum,np.array([Vnrm]),axis=0)
            Xbnrmcum=np.append(Xbnrmcum,np.array([Xbnrm]),axis=0)
            dBcum=np.append(dBcum,np.array([dB]),axis=0)
            phEcum=np.append(phEcum,np.array([phE]),axis=0)

            if trial % 10 ==0 :
                print("end %d" %trial)

    elif dataset==1:

        ## loading data
        ## Pcum, Vcum, Vnrmcum, Xbnrmcum, phEcum

        print(os.getcwd())
        Pcum_df=pd.read_csv('./Pcum.csv',delimiter='\t')
        Vcum_df=pd.read_csv('./Vcum.csv',delimiter='\t')
        Vnrmcum_df=pd.read_csv('./Vnrmcum.csv',delimiter='\t')
        Xbnrmcum_df=pd.read_csv('./Xbnrmcum.csv',delimiter='\t')
        phEcum_df=pd.read_csv('./phEcum.csv',delimiter='\t')

        Pcum_np=Pcum_df.to_numpy()
        Vcum_np=Vcum_df.to_numpy()
        
        Vnrmcum=Vnrmcum_df.to_numpy()
        Xbnrmcum=Xbnrmcum_df.to_numpy()
        phEnrmcum=phEcum_df.to_numpy()

        Pcum=Pcum_np.reshape(-1,T,N,2)
        Vcum=Vcum_np.reshape(-1,T,N,2)

        print(Pinit.shape)



    print ("End\n")

    medVnrm=np.median(Vnrmcum[1:],axis=0)
    stdVnrm=np.std(Vnrmcum[1:],axis=0)

    ##Outlier detection
    ##find appropriate constant 'cut'
    ## that the number of trials which is further than 'cut' * std from median
    ## is larger than 1% but not too large

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
    

    ##Plotting and saving plot
    # plotsave=(input("Do you want to save? Yes=else, No=0 "))
    plotsave=1    
    makeplot()
    
    #파일 저장은 어떻게 할지. 데이터 P,V,nV : Trial*T*N*2 / Vnrm, Xsnrm, dB : Trial*T
    #전자 : numpy를 썼을때 data모양 재조합하기 쉬운 방식으로 모양을 바꿔주면 될 듯.
    
    ##Saving data
    # if savemode==1 then save only chosen data(i.e. outliers),
    # elif savemode==2 then save all data 

    save_mode=2
    savedata(save_mode)

    print("DONE")



# 중간에 물결선 넣기
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html 물결선 넣기 -> 결국 그래프를 두개로 쪼개야함
# https://matplotlib.org/3.1.0/tutorials/intermediate/gridspec.html ->그래프를 격자 위에 임의로 배치할 수 있음. 조각보처럼
# https://stackoverflow.com/questions/53642861/broken-axis-slash-marks-inside-bar-chart-in-matplotlib 


# %%
