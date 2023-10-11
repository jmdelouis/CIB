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
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask][-l|--lbfgs][-C|--cib][-D|input_map]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--path  (optional): Define the path where output file are written (default data)')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--k128  (optional): Work with 128 pixel kernel reproducing wignercomputation instead of a 3x3.')
    print('--data  (optional): If not specified use /travail/jdelouis/CIB/857-1.npy.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    print('--mask  (optional): if specified use a mask')
    print('--lbfgs (optional): If specified the minimisation DO NOT uses L-BFGS minimisation and work with ADAM')
    print('--nscale (optional): number of cutted scale')
    print('--cib (optional): scale factor of the CIB from 353 to 857')
    
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xp:gkd:o:Kr:m:lz:C:D:t", \
                                   ["nside", "cov","seed","steps","xstat","path","gauss","k5x5",
                                    "data","out","k128","orient","mask","lbfgs","nscale","cib","input_map","simu_test"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    sim=False
    cov=False
    nside=-1
    nstep=300
    docross=False
    dogauss=False
    KERNELSZ=3
    dok128=False
    seed=1234
    outname='demo'
    data='/travail/jdelouis/CIB/857-1.npy'
    instep=16
    norient=4
    outpath='data/'
    imask=None
    dolbfgs=True
    nscale=4
    cib_scale=1.0
    atype='float32'
    input_map=None

    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-t", "--simu_test"):
            sim=True
        elif o in ("-C", "--cib"):
            cib_scale=float(a[1:])
        elif o in ("-D", "--input_map"):
            input_map=a[1:]
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

    print('Work with cib_scale=%lf'%(cib_scale))
    cib_scale=(49/10.5)*cib_scale

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
                     use_R_format=False,
                     slope=l_slope,
                     mask_norm=True,
                     mask_thres=0.7,
                     gpupos=0,
                     all_type=atype,
                     nstep_max=instep)
    
    #=================================================================================
    # Get data
    #=================================================================================
    den=np.load('/travail/jdelouis/CIB/separation_method_low_dust_J_6_pbc_True.npy',allow_pickle=True)
    cib=den[3]
    try:
        idx=np.load('get_method_low_dust_J_6_pbc_idx.npy')
    except:
        idx=hp.gnomview(np.arange(12*512*512),xsize=256,ysize=256,reso=4,rot=(305,-60),return_projected_map=True)
        np.save('get_method_low_dust_J_6_pbc_idx.npy',idx.data.astype('int'))
        exit(0)

    if nside!=512:
        th,ph=hp.pix2ang(512,idx)
        idx=hp.ang2pix(nside,th,ph)

    him=np.bincount(idx.flatten(),minlength=12*nside**2)
    im=np.bincount(idx.flatten(),weights=cib.flatten(),minlength=12*nside**2)
    
    im[him>0]=im[him>0]/him[him>0]
    mask=him>0

    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    cib_scale=(49/10.5)
    im=cib_scale*im[idx1]
    mask=np.expand_dims(mask[idx1],0)

    im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f353_ns512_rns32_IIcib_PR3_nhi_6p0.fits'),nside)[idx1]
    if sim:
        im=hp.ud_grade(np.load('data/CIB_SIMU.npy'),nside)[idx1]
    mask=np.ones([1,im.shape[0]])
    mask[0,:]=(im!=hp.UNSEEN)

    im[im==hp.UNSEEN]=0.0
    if not sim:
        im=cib_scale*im

    """
    hp.mollview(im,rot=[305,-60],nest=True)
    hp.mollview(mask[0],rot=[305,-60],nest=True)
    plt.show()
    cib_scale=(49/10.5)
    np.save('data/tuhin_wph.npy',cib_scale*im)
    exit(0)
    """

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
    smask2=hp.smoothing(mask[0,idx],np.pi/32.0)[idx1]

    mask[0]=smask/smask.mean()
    mask2=0.01*np.expand_dims((smask2/smask2.mean())**2*mask[0],0)

    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    #im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f857_ns512_rns32_IIcib_PR3_nhi_2p0.fits'),nside)[idx1]
    i857=hp.ud_grade(np.load(data),nside)

    h1=hp.ud_grade(np.load('/travail/jdelouis/CIB/H1.npy'),nside)
    if sim:
        h1=hp.ud_grade(np.load('/travail/jdelouis/CIB/DUST_SIMU.npy'),nside)
    s857=i857-hp.smoothing(i857,np.pi/(2**nscale))
    sh1=h1-hp.smoothing(h1,np.pi/(2**nscale))
    val=np.median(i857)
    lmask=hp.smoothing(i857<val/2,10/180.*np.pi)
    a=np.polyfit(sh1[i857<val/2],s857[i857<val/2],1)
    h1=a[0]*h1+a[1]
    hin=lmask*(hp.smoothing(i857,np.pi/(2**(nscale+1)))+h1-hp.smoothing(h1,np.pi/(2**(nscale+1))))+(1-lmask)*i857

    hin=hin[idx1].astype(atype)
    i857=i857[idx1].astype(atype)
    h1=h1[idx1].astype(atype)

    """
    hp.mollview(i857,cmap='jet',norm='hist',nest=True)
    hp.mollview(h1,cmap='jet',norm='hist',nest=True)
    hp.mollview(i857-hin,cmap='jet',norm='hist',nest=True)
    plt.show()
    exit(0)
    """
    
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    idx=hp.ring2nest(nside,np.arange(12*nside*nside))
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
        
    np.random.seed(seed)
    imap=hin
    smask=((hp.smoothing(mask[0,idx],np.pi/(2**nscale))[idx1])>0.6)*mask[0]

    #mask[0]=smask/smask.mean()

    val=np.argsort(i857)
    nmask=5
    maskgal=np.zeros([nmask,12*nside**2],dtype=atype)
    for i in range(nmask):
        maskgal[i]=np.expand_dims(hp.ud_grade(hp.smoothing(hp.ud_grade(np.load(data),nside)<i857[val[(i+1)*12*nside**2//(nmask+1)]],10/180.*np.pi),nside)[idx1],0)
        print(maskgal[i].mean())
        maskgal[i]/=maskgal[i].mean()
        #maskgal[i]/=(2**(i))
        
    # do multiscale h1 adjustment
    nadjust=32

    mat=np.zeros([2,2,12*(nside//nadjust)**2])
    vec=np.zeros([2,12*(nside//nadjust)**2])
    mat[0,0]=np.mean(maskgal[-2].reshape(12*(nside//nadjust)**2,nadjust*nadjust),1)
    mat[1,0]=np.mean((maskgal[-2]*i857).reshape(12*(nside//nadjust)**2,nadjust*nadjust),1)
    mat[1,1]=np.mean((maskgal[-2]*i857*i857).reshape(12*(nside//nadjust)**2,nadjust*nadjust),1)
    mat[0,1]=mat[1,0]
    vec[0]=np.mean((maskgal[-2]*h1).reshape(12*(nside//nadjust)**2,nadjust*nadjust),1)
    vec[1]=np.mean((maskgal[-2]*h1*i857).reshape(12*(nside//nadjust)**2,nadjust*nadjust),1)
    det = 1/(mat[0,0]*mat[1,1]-mat[1,0]*mat[0,1])
    imat=mat.copy()
    imat[0,0]=det*mat[1,1]
    imat[1,0]=-det*mat[1,0]
    imat[0,1]=-det*mat[0,1]
    imat[1,1]=det*mat[0,0]
    del mat

    for i in range(12*(nside//nadjust)**2):
        vec[:,i]=np.dot(imat[:,:,i],vec[:,i])

    b=scat_op.up_grade(vec[0,:].astype(atype),nside).numpy()
    a=scat_op.up_grade(vec[1,:].astype(atype),nside).numpy()
    
    hin=h1*a+b
    shin=hin.copy()
    s857=i857.copy()
    for i in range(nscale):
        shin=scat_op.ud_grade_2(shin)
        s857=scat_op.ud_grade_2(s857)
    
    hin=hin+scat_op.up_grade(s857,nside).numpy()-scat_op.up_grade(shin,nside).numpy()

    
    #=================================================================================
    # DEFINE WAHT IS CONSTANT FOR OPTIMAL TENSORFLOW RUNNING
    #=================================================================================
    i857=scat_op.backend.constant(i857)
    mask=scat_op.backend.constant(mask)
    maskgal=scat_op.backend.constant(maskgal)
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    def losscib(x,scat_operator,args,return_all=False):
        
        ref = args[0]
        i857 = args[1]
        sig  = args[2]
        avv  = args[3]
        mask = args[4]

        #learn=(scat_operator.eval(i857-x)/avv).iso_mean()
        learn=(scat_operator.eval(i857-x,image2=i857-x))#.iso_mean()

        loss=scat_operator.reduce_mean(scat_operator.square(sig*(ref-learn)))      

        return(loss)

    def lossX(x,scat_operator,args,return_all=False):
        
        ref = args[0]
        h1  = args[1]
        mg  = args[2]
        i857  = args[3]
        sig  = args[4]
        avv  = args[5]
        
        learn=(scat_operator.eval(x,image2=h1,mask=mg))

        loss=scat_operator.reduce_mean(scat_operator.square(sig*(ref-learn)))      

        return(loss)

    def lossD(x,scat_operator,args,return_all=False):
        
        i857  = args[0]
        mg  = args[1]
        sig  = args[2]
        avv  = args[3]

        learn1=(scat_operator.eval(i857,image2=x,mask=mg))
        learn2=(scat_operator.eval(x,image2=x,mask=mg))

        loss=scat_operator.reduce_mean(scat_operator.square(sig*(learn1-learn2-avv)))      

        return(loss)

    def lossH(x,scat_operator,args,return_all=False):
        
        mg  = args[0]
        i857  = args[1]
        sig  = args[2]

        learn=scat_operator.eval(i857-x,image2=h1,mask=mg)

        loss=scat_operator.reduce_mean(scat_operator.square(sig*learn))      

        return(loss)

    
    # compute the bias from cib 
    if sim:
        ncib=66
    else:
        ncib=15

    cib=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/data_resmap_f857_ns512_rns32_IIcib_PR3_nhi_2p0.fits'),nside)[idx1]
    if input_map is None:
        imap=np.random.randn(12*nside*nside)*np.std(im)
        imap=(imap-hp.smoothing(imap,np.pi/(2**nscale)))[idx1]+i857.numpy().copy()
        imap=i857.numpy().copy()
    else:
        print('================\n\n')
        print('Input:',input_map)
        print('\n\n================')
        imap=np.load(input_map)
    """
    hp.mollview(h1,nest=True,cmap='jet',norm='hist')
    hp.mollview((i857-h1).numpy(),nest=True,cmap='jet',norm='hist')
    plt.show()
    exit(0)

    
    refn=scat_op.eval(np.random.randn(im.shape[0]),mask=mask)
    for i in range(99):
        refn=refn+scat_op.eval(np.random.randn(im.shape[0]),mask=mask)
    refn=refn/100.0
    """

    for itt in range(5):
        if itt>0:
            imap=omap

        biasX=scat_op.eval(imap,image2=h1,mask=maskgal)
        bias=scat_op.eval(imap,image2=imap,mask=maskgal)

        for k in range(ncib):
            if sim:
                tmp=dodown(np.load('data/out_cibSIM%d_map_512.npy'%(k)),nside)
            else:
                if cov:
                    tmp=cib_scale*dodown(np.load('data/out_cibcNR%d_map_512.npy'%(k)),nside)
                else:
                    tmp=cib_scale*dodown(np.load('data/out_cibNR%d_map_512.npy'%(k)),nside)

            sc0=scat_op.eval_fast(tmp,image2=tmp,mask=mask)
            sc1=scat_op.eval_fast(imap+tmp,image2=imap,mask=maskgal)-bias
            sc2=scat_op.eval_fast(imap+tmp,image2=h1,mask=maskgal)-biasX

            if k==0:
                avv0=sc0
                avv02=sc0*sc0

                avv=sc1
                avvX=sc2

                avv2=sc1*sc1
                avvX2=sc2*sc2
            else:
                avv0=avv0+sc0
                avv02=avv02+sc0*sc0

                avv=avv+sc1
                avvX=avvX+sc2

                avv2=avv2+sc1*sc1
                avvX2=avvX2+sc2*sc2

        avv0=avv0/ncib
        avv02=avv02/ncib

        avv=avv/ncib
        avvX=avvX/ncib

        avv2=avv2/ncib
        avvX2=avvX2/ncib

        ref=scat_op.eval_fast(im,image2=im,mask=mask)
        """
        ref.save('refcib')
        refn.save('refncib')
        (scat_op.eval_fast(i857-imap,mask=mask)/refn).save('irefcib')
        exit(0)
        """
        #ref=refn
        #refsig=1/ref.iso_mean()
        #refsig=1/ref.iso_std()
        refcib=ref
        refsig=1/ref.iso_mean(repeat=True)
        #refsig=1/scat_op.sqrt(avv02-avv0*avv0)

        refX=scat_op.eval_fast(i857,image2=h1,mask=maskgal)
        #refXsig=1/refX.iso_std(repeat=True)
        refXsig=1/scat_op.sqrt(avvX2-avvX*avvX)

        refD=scat_op.eval_fast(i857,image2=imap,mask=maskgal)
        #refDsig=1/refD.iso_std(repeat=True)
        refDsig=1/scat_op.sqrt(avv2-avv*avv)

        """
        loss1=synthe.Loss(losscib,scat_op,refcib,scat_op.to_R(i857),refsig,avv0,maskgal)
        loss2=synthe.Loss(lossX,scat_op,refX-avvX/avv,scat_op.to_R(h1),maskgal,scat_op.to_R(i857),refXsig,avv)
        """
        loss1=synthe.Loss(losscib,scat_op,
                          refcib.constant(),
                          scat_op.backend.constant(i857),
                          refsig.constant(),
                          None, #refn.constant(),
                          scat_op.backend.constant(maskgal))

        loss2=synthe.Loss(lossX,scat_op,
                          (refX-avvX).constant(),
                          scat_op.backend.constant(h1),
                          scat_op.backend.constant(maskgal),
                          scat_op.backend.constant(i857),
                          refXsig.constant(),
                          avv.constant())

        loss3=synthe.Loss(lossD,scat_op,
                          scat_op.backend.constant(i857),
                          scat_op.backend.constant(maskgal),
                          refDsig.constant(),
                          avv.constant())

        sy = synthe.Synthesis([loss1,loss2,loss3])
        #=================================================================================
        # RUN ON SYNTHESIS
        #=================================================================================

        omap=sy.run(imap,NUM_EPOCHS = nstep,EVAL_FREQUENCY = 10)

        np.save(outpath+'out_%s_map_%d_%d.npy'%(outname,nside,itt),omap)

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


    
