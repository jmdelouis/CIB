import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/NHI_HPX.fits',4),512)
im2=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/GASS_512_ring_local.fits'),512)

mask=(im2>0) #hp.smoothing(im2>0,10/180.*np.pi)

a=np.polyfit(im[mask>0.8],im2[mask>0.8],1)
im=im*a[0]+a[1]
im[mask>0]=im2[mask>0]

np.save('/travail/jdelouis/CIB/H1.npy',im)

hp.mollview(im,cmap='jet',norm='hist')
plt.show()

im=hp.ud_grade(hp.read_map('/travail/jdelouis/CIB/SRoll30_SkyMap_857-1_full.fits'),512)
im=hp.smoothing(im,16.0/(60*180.)*np.pi)
np.save('/travail/jdelouis/CIB/857-1.npy',im)
hp.mollview(im,cmap='jet',norm='hist')
plt.show()
