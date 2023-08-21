import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask][-l|--lbfgs]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--path  (optional): Define the path where output file are written (default data)')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--k128  (optional): Work with 128 pixel kernel reproducing wignercomputation instead of a 3x3.')
    print('--data  (optional): If not specified use LSS_map_nside128.npy.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    print('--mask  (optional): if specified use a mask')
    print('--lbfgs (optional): If specified the minimisation DO NOT uses L-BFGS minimisation and work with ADAM')
    print('--nscale (optional): number of cutted scale')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xp:gkd:o:Kr:m:lz:", \
                                   ["nside", "cov","seed","steps","xstat","path","gauss","k5x5",
                                    "data","out","k128","orient","mask","lbfgs","nscale"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=300
    docross=False
    dogauss=False
    KERNELSZ=3
    dok128=False
    seed=1234
    outname='demo'
    data="data/LSS_map_nside128.npy"
    instep=16
    norient=4
    outpath='data/'
    imask=None
    dolbfgs=True
    nscale=4
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            print('Use SEED = ',seed)
        elif o in ("-z", "--nscale"):
            nscale=int(a[1:])
            print('Use NSCALE = ',nscale)
        elif o in ("-o", "--out"):
            outname=a[1:]
            print('Save data in ',outname)
        elif o in ("-d", "--data"):
            data=a[1:]
            print('Read data from ',data)
        elif o in ("-x", "--xstat"):
            docross=True
        elif o in ("-g", "--gauss"):
            dogauss=True
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-l", "--lbfgs"):
            dolbfgs=True
        elif o in ("-K", "--k64"):
            KERNELSZ=64
            instep=8
        elif o in ("-r", "--orient"):
            norient=int(a[1:])
            print('Use %d orientations'%(norient))
        elif o in ("-m", "--mask"):
            imask=np.load(a[1:])
            print('Use %s mask'%(a[1:]))
        elif o in ("-p", "--path"):
            outpath=a[1:]
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>512 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,512] or [2,...,%d] if -K|-k128 option has been choosen'%(2**instep))
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
        print('Work with ScatCov')
    else:
        import foscat.scat as sc
        print('Work with Scat')
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = 'data'

    #=================================================================================
    # Function to reduce the data used in the FoCUS algorithm 
    #=================================================================================
    def dodown(a,nside):
        nin=int(np.sqrt(a.shape[0]//12))
        if nin==nside:
            return(a)
        return(np.mean(a.reshape(12*nside*nside,(nin//nside)**2),1))

    lam=1.2
    if KERNELSZ==5:
        lam=1.0
    
    l_slope=1.0
    r_format=True
    if KERNELSZ==64:
        r_format=False
        l_slope=4.0
    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  #KERNELSZ,  # define the kernel size
                     OSTEP=nscale,           # get very large scale (nside=1)
                     LAMBDA=lam,
                     TEMPLATE_PATH=scratch_path,
                     slope=l_slope,
                     gpupos=2,
                     mask_norm=True,
                     mask_thres=0.7,
                     all_type='float32',
                     nstep_max=instep)
    
    #=================================================================================
    # Get patch data and convert it in healpix nested.
    #=================================================================================

    den=np.load('/travail/jdelouis/CIB/separation_method_low_dust_J_6_pbc_True.npy',allow_pickle=True)
    cib=den[3]
    try:
        idx=np.load('get_method_low_dust_J_6_pbc_idx.npy')
    except:
        idx=hp.gnomview(np.arange(12*512*512),xsize=256,ysize=256,reso=4,rot=(305,-60),return_projected_map=True)
        np.save('get_method_low_dust_J_6_pbc_idx.npy',idx.data.astype('int'))
        exit(0)

    him=np.bincount(idx.flatten(),minlength=12*512*512)
    im=np.bincount(idx.flatten(),weights=cib.flatten(),minlength=12*512*512)
    
    im[him>0]=im[him>0]/him[him>0]
    mask=him>0

    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    im=im[idx1]
    mask=np.expand_dims(mask[idx1],0)

    # older version using healpix data 
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f857_ns512_rns32_IIcib_PR3_nhi_2p0.fits'),nside)[idx1]
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f353_ns512_rns32_IIcib_PR3_nhi_6p0.fits'),nside)[idx1]
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/residual_map_f353_ns2048_cmb_subtracted_full.fits'),nside)[idx1]
    
    #mask=np.ones([1,im.shape[0]])
    #mask[0,:]=(im!=hp.UNSEEN)

    #=================================================================================
    # Compute and crop it to be used at the smallest scale
    #=================================================================================

    idx=hp.ring2nest(nside,np.arange(12*nside*nside))
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))

    amp=np.std(im[mask[0]==1])
    smask=hp.smoothing(mask[0,idx],np.pi/nside)[idx1]>0.7
    smask2=hp.smoothing(mask[0,idx],np.pi/32.0)[idx1]>0.
    
    """
    plt.figure()
    hp.gnomview(mask[0,:],rot=[305,-60],reso=4,xsize=260,cmap='jet',hold=False,sub=(1,2,1),nest=True)
    hp.gnomview(smask,rot=[305,-60],reso=4,xsize=260,cmap='jet',hold=False,sub=(1,2,2),nest=True)
    plt.show()
    exit(0)
    """

    mask[0]=smask/smask.mean()
    mask2=np.expand_dims(smask2/smask2.mean(),0)

    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
        
    def loss_diff(x,scat_operator,args):
        
        im  = scat_operator.to_R(args[0])
        mask = scat_operator.to_R(args[1][0])

        loss=scat_operator.backend.bk_reduce_mean(scat_operator.backend.bk_square(300*mask*(x-im)))      

        return(loss)

    def loss_scat(x,scat_operator,args):
        
        ref = args[0]
        refsig = args[1]

        learn=scat_operator.eval(x).iso_mean()

        loss=scat_operator.reduce_mean(scat_operator.square(refsig*(ref-learn)))      

        return(loss)

    def loss_scat_mask(x,scat_operator,args):
        
        ref = args[0]
        refsig = args[1]
        mask = args[2]

        learn=scat_operator.eval(x,mask=mask).iso_mean()

        loss=scat_operator.reduce_mean(scat_operator.square(refsig*(ref-learn)))      

        return(loss)

    refv=scat_op.eval(im,mask=mask)
    ref=refv.iso_mean()
    refsig=1/refv.iso_std()


    """
    print(refX.S1.numpy().flatten())
    print(refX.S2.numpy().flatten())
    exit(0)
    plt.subplot(2,2,1)
    plt.plot(refX.S1.numpy().flatten())
    plt.subplot(2,2,2)
    plt.plot(refX.S2.numpy().flatten())
    plt.show()
    exit(0)
    """
    loss1=synthe.Loss(loss_scat,scat_op,ref,refsig)
    loss2=synthe.Loss(loss_diff,scat_op,im.astype('float32'),mask.astype('float32'))
    loss3=synthe.Loss(loss_scat_mask,scat_op,ref,refsig,mask2)
        
    const_sy = synthe.Synthesis([loss1,loss2,loss3])
    blind_sy = synthe.Synthesis([loss1])

    #=================================================================================
    # RUN SYNTHESIS DRIVEN BY INPUT DATA
    #=================================================================================
    np.random.seed(seed)
    imap=np.random.randn(12*nside**2)*amp
    imap=(imap-hp.smoothing(imap,np.pi/(2**nscale)))[idx1]
    imap[mask[0]==1]=im[mask[0]==1]
    omap=const_sy.run(imap,NUM_EPOCHS = 3000,EVAL_FREQUENCY=10)
    
    np.save(outpath+'outD_%s_map_%d.npy'%(outname,nside),omap)
    np.save(outpath+'inD_%s_map_%d.npy'%(outname,nside),im)
    np.save(outpath+'maskD_%s_map_%d.npy'%(outname,nside),mask)

    #=================================================================================
    # RUN RANDOM SYNTHESIS
    #=================================================================================

    for i in range(100):
        np.random.seed(seed+1+i)
        imap=np.random.randn(12*nside**2)*amp
        imap=(imap-hp.smoothing(imap,np.pi/(2**nscale)))[idx1]

        omap=blind_sy.run(imap,NUM_EPOCHS = nstep,EVAL_FREQUENCY=10)

        np.save(outpath+'out_%s%d_map_%d.npy'%(outname,i,nside),omap)

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
