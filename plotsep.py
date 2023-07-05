import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo] [-i|--vmin] [-a|--vmax] [-p|--path]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov|-c     (optional): use scat_cov instead of scat')
    print('--out|-o     (optional): If not specified save in *_demo_*.')
    print('--map=jet|-m (optional): If not specified use cmap=jet')
    print('--geo|-g     (optional): If specified use cartview')
    print('--vmin|-i    (optional): specify the minimum value')
    print('--vmax|-a    (optional): specify the maximum value')
    print('--path|-p    (optional): Define the path where output file are written (default data)')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:gi:a:", ["nside", "cov","out","map","geo","vmin","vmax"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='viridis'
    docart=False
    vmin=-3
    vmax= 30
    outpath='data/'
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-g","--geo"):
            docart=True
        elif o in ("-m","--map"):
            cmap=a[1:]
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-o", "--out"):
            outname=a[1:]
        elif o in ("-i", "--vmin"):
            vmin=float(a[1:])
        elif o in ("-a", "--vmax"):
            vmax=float(a[1:])
        elif o in ("-p", "--path"):
            outpath=a[1:]
        else:
            print(o,a)
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>512:
        print('nside should be a pwer of 2 and in [2,...,512]')
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
    else:
        import foscat.scat as sc

    refX  = sc.read(outpath+'in_%s_%d'%(outname,nside))
    start = sc.read(outpath+'st_%s_%d'%(outname,nside))
    out   = sc.read(outpath+'out_%s_%d'%(outname,nside))

    log= np.load(outpath+'out_%s_log_%d.npy'%(outname,nside))
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(log.shape[0])+1,log,color='black')
    plt.xscale('log')
    plt.yscale('log')

    refX.plot(name='Model',lw=6)
    start.plot(name='Input',color='orange',hold=False)
    out.plot(name='Output',color='red',hold=False)
    #(refX-out).plot(name='Diff',color='purple',hold=False)

    #im = np.load(outpath+'in_%s_map_%d.npy'%(outname,nside))
    om = np.load(outpath+'out_%s_map_%d.npy'%(outname,nside))
    idx=hp.nest2ring(nside,np.arange(12*nside**2))
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    im=hp.ud_grade(np.load('/travail/jdelouis/CIB/857-1.npy'),nside)[idx]
    h1=hp.ud_grade(np.load('/travail/jdelouis/CIB/H1.npy'),nside)[idx]
    try:
        mm = np.load(outpath+'mm_%s_map_%d.npy'%(outname,nside))
    except:
        mm = np.ones([im.shape[0]])
    sm = np.load(outpath+'st_%s_map_%d.npy'%(outname,nside))

    idx=hp.ring2nest(nside,np.arange(12*nside**2))

    cib=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f857_ns512_rns32_IIcib_PR3_nhi_2p0.fits'),nside)[idx1]
    print(cib.min())
    a=np.polyfit(cib[cib>-1E20],(im-om)[cib>-1E20],1)
    print('CIB SCALLING ',a[0])
    a[0]=1.0
    plt.figure(figsize=(12,6))
    hp.gnomview(im[idx]-1.0,rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=-0.5,max=0.5,hold=False,sub=(2,3,1),nest=False,title='Input')
    hp.gnomview(om-1.0,rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=-0.5,max=0.5,hold=False,sub=(2,3,2),nest=True,title='Dust')
    hp.gnomview(im-om,rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,5),nest=True,title='CIB')
    hp.gnomview(a[0]*cib,rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,4),nest=True,title='CIB In')
    hp.gnomview(a[0]*cib-(im-om),rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,6),nest=True,title='CIB diff')
    hp.gnomview(h1,rot=[305,-60],xsize=256,reso=4,cmap=cmap,min=1,max=2.2,hold=False,sub=(2,3,3),nest=True,title='H1')

    plt.figure(figsize=(12,6))
    hp.gnomview(im[idx]-1.0,rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=-0.5,max=0.5,hold=False,sub=(2,3,1),nest=False,title='Input')
    hp.gnomview(om-1.0,rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=-0.5,max=0.5,hold=False,sub=(2,3,2),nest=True,title='Dust')
    hp.gnomview(im-om,rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,5),nest=True,title='CIB')
    hp.gnomview(a[0]*cib,rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,4),nest=True,title='CIB In')
    hp.gnomview(a[0]*cib-(im-om),rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=-0.25,max=0.25,hold=False,sub=(2,3,6),nest=True,title='CIB diff')
    hp.gnomview(h1,rot=[-10,-60],xsize=512,reso=4,cmap=cmap,min=1,max=2.2,hold=False,sub=(2,3,3),nest=True,title='H1')

    vmax=3
    plt.figure(figsize=(6,9))
    hp.mollview(im[idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,1,1),nest=False,title='Model',norm='hist')
    hp.mollview(om,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,1,2),nest=True,title='Output',norm='hist')
    hp.mollview(im-om,cmap=cmap,min=-0.3,max=0.3,hold=False,sub=(3,1,3),nest=True,title='CIB')
        
    
    val=np.argsort(im)
    nmask=5
    maskgal=np.zeros([nmask,12*nside**2],dtype='float32')
    plt.figure(figsize=(6,6))
    for i in range(4):
        maskgal[i]=np.expand_dims(hp.ud_grade(hp.smoothing(np.load('/travail/jdelouis/CIB/857-1.npy')<im[val[(i+1)*12*nside**2//nmask-1]],10/180.*np.pi),nside)[idx1],0)
        maskgal[i]/=maskgal[i].mean()

        cli=hp.anafast((maskgal[i]*(im-np.median(im)))[idx])
        clo=hp.anafast((maskgal[i]*(om-np.median(om)))[idx])
        cldiff=hp.anafast((maskgal[i]*(im-om-np.median(om)))[idx])

        plt.subplot(2,2,1+i)
        plt.plot(cli,color='blue',label=r'Model',lw=6)
        plt.plot(clo,color='orange',label=r'Output')
        plt.plot(cldiff,color='red',label=r'Diff')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel('Multipoles')
        plt.ylabel('C(l)')

    plt.show()

if __name__ == "__main__":
    main()
