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
                     all_type='float32',
                     nstep_max=instep)
    
    #=================================================================================
    # Get data
    #=================================================================================
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f857_ns512_rns32_IIcib_PR3_nhi_2p0.fits'),nside)[idx1]
    im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f353_ns512_rns32_IIcib_PR3_nhi_6p0.fits'),nside)[idx1]
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/residual_map_f353_ns2048_cmb_subtracted_full.fits'),nside)[idx1]
    
    mask=np.ones([1,im.shape[0]])
    mask[0,:]=(im!=hp.UNSEEN)

    im[im==hp.UNSEEN]=0.0

    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    idx=hp.ring2nest(nside,np.arange(12*nside*nside))
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))

    amp=np.std(im[mask[0]==1])
    smask=((hp.smoothing(mask[0,idx],np.pi/(2**nscale))[idx1])>0.6)*mask[0]
    """
    hp.gnomview(mask[0,:],rot=[0,-80],reso=10,xsize=512,cmap='jet',hold=False,sub=(1,2,1),nest=True)
    hp.gnomview(smask,rot=[0,-80],reso=10,xsize=512,cmap='jet',hold=False,sub=(1,2,2),nest=True)
    plt.show()
    exit(0)
    """
    mask[0]=smask/smask.mean()

    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================


    def iso(res):
        idx=np.array([0,5,10,15, \
                      1,6,11,12, \
                      2,7,8,13,
                      3,4,9,14],dtype='int')

        res.S1  = res.backend.bk_reduce_mean(res.S1,2)
        res.P00 = res.backend.bk_reduce_mean(res.P00,2)
        shape=list(res.S2.shape)
        res.S2=res.backend.bk_reshape(res.backend.bk_gather(res.backend.bk_reshape(res.S2,[shape[0],shape[1],4*4]),idx,2),[shape[0],shape[1],4,4])
        res.S2L=res.backend.bk_reshape(res.backend.bk_gather(res.backend.bk_reshape(res.S2L,[shape[0],shape[1],4*4]),idx,2),[shape[0],shape[1],4,4])

        res.S2  =res.backend.bk_reduce_mean(res.S2,3)
        res.S2L=res.backend.bk_reduce_mean(res.S2L,3)
        return res
        
    def lossX(x,scat_operator,args):
        
        ref = args[0]
        im  = args[1]
        mask = args[2]

        learn=iso(scat_operator.eval(x))

        loss=scat_operator.reduce_mean(scat_operator.square(ref-learn))      

        return(loss)

    refX=iso(scat_op.eval(im,mask=mask))
    
    loss1=synthe.Loss(lossX,scat_op,refX,im,mask)
        
    sy = synthe.Synthesis([loss1])
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================

    for i in range(100):
        np.random.seed(seed+i)
        imap=np.random.randn(12*nside**2)*amp
        imap=(imap-hp.smoothing(imap,np.pi/(2**nscale)))[idx1]

        omap=sy.run(imap,NUM_EPOCHS = nstep)

        np.save(outpath+'out_%s%d_map_%d.npy'%(outname,i,nside),omap)

    mask=np.ones([1,12*nside**2])
    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    if docross:
        start=scat_op.eval(im,image2=imap,mask=mask)
        out =scat_op.eval(im,image2=omap,mask=mask)
    else:
        start=scat_op.eval(imap,mask=mask)
        out =scat_op.eval(omap,mask=mask)
    
    np.save(outpath+'in_%s_map_%d.npy'%(outname,nside),im)
    np.save(outpath+'mm_%s_map_%d.npy'%(outname,nside),mask[0])
    np.save(outpath+'st_%s_map_%d.npy'%(outname,nside),imap)
    np.save(outpath+'out_%s_map_%d.npy'%(outname,nside),omap)
    np.save(outpath+'out_%s_log_%d.npy'%(outname,nside),sy.get_history())

    refX.save( outpath+'in_%s_%d'%(outname,nside))
    start.save(outpath+'st_%s_%d'%(outname,nside))
    out.save(  outpath+'out_%s_%d'%(outname,nside))

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
