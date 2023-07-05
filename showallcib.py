import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

ncib=7

imap=np.load('data/in_cib%d_map_512.npy'%(1))
rot=[305,-80]
amp=0.04
for i in range(ncib):
    im=np.load('data/out_cib%d_map_512.npy'%(i))
    hp.mollview(im,cmap='jet',hold=False,sub=(2,4,1+i),nest=True) 
    #hp.gnomview(im,rot=rot,reso=10,xsize=512,cmap='jet',hold=False,sub=(2,4,1+i),nest=True,min=-amp,max=amp) 
hp.mollview(imap,cmap='jet',hold=False,sub=(2,4,1+ncib),nest=True)
#hp.gnomview(imap,rot=rot,reso=10,xsize=512,cmap='jet',hold=False,sub=(2,4,1+ncib),nest=True,min=-amp,max=amp)

plt.show()
