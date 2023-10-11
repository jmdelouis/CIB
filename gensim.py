import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

nside=512

dust=32.289*hp.ud_grade(np.load('/travail/jdelouis/CIB/DUST_SIMU.npy'),nside)
idx=hp.ring2nest(nside,np.arange(12*nside*nside))
cib=np.load('data/outD_cibSIM_map_512.npy')[idx]

ref=dust+cib

np.save('/travail/jdelouis/CIB/DUST+CIB_SIMU.npy',ref)

hp.mollview(ref,cmap='jet',norm='hist')
hp.mollview(ref-cib,cmap='jet',norm='hist')
hp.mollview(cib,cmap='jet')
hp.gnomview(cib,cmap='jet',rot=[0,90],reso=6,xsize=512)
plt.show()
