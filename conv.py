import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

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
