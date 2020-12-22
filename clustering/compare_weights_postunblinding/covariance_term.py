import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 

wtheta_file = {
	#'fid_pca_108':'wtheta_redmagic_y3_data_bs0.05_fid_pca_108.fits',
	#'enet_std_107':'wtheta_redmagic_y3_data_bs0.05_enet_std_107.fits',

	'fid_pca_107':'wtheta_redmagic_y3_data_bs0.1_fid_pca_107nside4096.fits',
	'fid_pca_108':'wtheta_redmagic_y3_data_bs0.1_fid_pca_108nside4096.fits',
	'enet_std_107':'wtheta_redmagic_y3_data_bs0.1_enet_std_107nside4096.fits',
}

#extra_label = 'nside512'
extra_label = 'nside4096'

label1 = 'fid_pca_108'
#label1 = 'fid_pca_107'
label2 = 'enet_std_107'
labels = [label1, label2]

wdict = {}
for label in labels:
	w = twopoint.TwoPointFile.from_fits(wtheta_file[label],covmat_name=None).get_spectrum('wtheta')
	wdict[label] = w

diff = wdict[label1].value - wdict[label2].value
sigmaA = 1
diff_term_cov = sigmaA**2.*np.array( (np.matrix(diff).T*np.matrix(diff)) )

outfile = 'cov_diff_term_wtheta_{0}_{1}_{2}_sigmaA{3}.txt'.format(label1, label2, extra_label, sigmaA)

np.savetxt(outfile, diff_term_cov)


plt.figure()
plt.imshow(np.log10(np.abs(diff_term_cov)))
plt.colorbar(label=r'log$|\Delta w \Delta w^T|$)')
plt.title(outfile)
plt.savefig('cov_diff_{0}_{1}_{2}.png'.format(label1, label2, extra_label))
plt.close()


#remove off-diagonal blocks
diff_term_cov_5p = np.copy(diff_term_cov)
for ibin in range(5):
	for jbin in range(5):
		if ibin != jbin:
			diff_term_cov_5p[ibin*20:(ibin+1)*20,jbin*20:(jbin+1)*20] = 0.

np.savetxt(outfile[:-4]+'_5p.txt', diff_term_cov_5p)

plt.figure()
plt.imshow(np.log10(np.abs(diff_term_cov_5p)))
plt.colorbar(label=r'log$|\Delta w \Delta w^T|$)')
plt.title(outfile)
plt.savefig('cov_diff_{0}_{1}_{2}.png'.format(label1, label2, extra_label))
plt.close()
