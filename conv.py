import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

nside=512
im=hp.read_map('/travail/jdelouis/CIB/PR3_nocmb_3000GHz_full_16p2arcm_ns512.fits')
idx=hp.ring2nest(nside,np.arange(12*nside**2))
im=   np.load('data/out_sepcib-1_map_512.npy')[idx]
im=im+np.load('data/out_sepcib-2_map_512.npy')[idx]
im=im+np.load('data/out_sepcib-3_map_512.npy')[idx]
im=im+np.load('data/out_sepcib-4_map_512.npy')[idx]
im=im/4

ref=np.load('/travail/jdelouis/CIB/857-1.npy')
a=np.polyfit(im,ref,1)
cib_scale=49/10.5
cib=cib_scale*np.load('data/out_cibiso%d_map_512.npy'%(0))
idx=hp.ring2nest(512,np.arange(12*nside*nside))
cib=cib[idx]
test=a[0]*im+a[1]+cib

np.save('/travail/jdelouis/CIB/IRIS-SIMU.npy',test)
np.save('/travail/jdelouis/CIB/IRIS-DUST-SIMU.npy',a[0]*im+a[1])
np.save('/travail/jdelouis/CIB/IRIS-CIB-SIMU.npy',cib)
hp.gnomview(cib,cmap='jet',)
hp.mollview(im,cmap='jet',norm='hist')
plt.show()
exit(0)
print(a)



im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/NHI_HPX.fits',4),512)
im2=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/GASS_512_ring_local.fits'),512)

mask=(hp.smoothing(im2>0,10/180.*np.pi)>0.8)

a=np.polyfit(im[mask==1],im2[mask==1],1)
im=im*a[0]+a[1]
im[mask==1]=im2[mask==1]

hp.mollview(im,cmap='jet',norm='hist')

np.save('/travail/jdelouis/CIB/H1.npy',im)
plt.show()
exit(0)

for i in range(4):
    im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/SRoll30_SkyMap_857-%d_full.fits'%(i+1)),512)
    im=hp.smoothing(im,16.0/(60*180.)*np.pi)
    np.save('/travail/jdelouis/CIB/857-%d.npy'%(i+1),im)

hp.mollview(im,cmap='jet',norm='hist')
plt.show()
