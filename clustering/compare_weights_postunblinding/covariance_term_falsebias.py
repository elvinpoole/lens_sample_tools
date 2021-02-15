import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 

wbias_dir = '/Users/jackelvinpoole/DES/lss_sys/scripts/y3_enet_contaminated_tests/'
wbias_files = {
	'false_contam-enet_correct-fidpca108': wbias_dir+'false_bias_LNmock_enet_contam.txt',
	'false_contam-enet_correct-fidpca50': wbias_dir+'false_bias_LNmock_enet_contam_pca_50.txt',
}

extra_label = ''

label = 'false_contam-enet_correct-fidpca50'

wbias = np.loadtxt(wbias_files[label])
diff = wbias #the delta data vector iis just teh false correction bias (overcorrection)

#remove scales we couldn't measure (shoudl be 0 anyway but just makes sure)
mintheta = 10.
edges = np.logspace(np.log10(2.5), np.log10(250.), 21)
cen = (edges[1:]+edges[:-1])/2.
angle = np.hstack([cen for i in range(5)])
diff[angle < mintheta] = 0.0 

sigmaA = 1
diff_term_cov = sigmaA**2.*np.array( (np.matrix(diff).T*np.matrix(diff)) )

outfile = 'cov_diff_term_wtheta_falsebias_{0}_{1}_sigmaA{2}.txt'.format(label, extra_label, sigmaA)

np.savetxt(outfile, diff_term_cov)


plt.figure()
plt.imshow(np.log10(np.abs(diff_term_cov)))
plt.colorbar(label=r'log$|\Delta w \Delta w^T|$)')
plt.title(outfile)
plt.savefig('cov_diff_{0}_{1}.png'.format(label, extra_label))
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
plt.savefig('cov_diff_{0}_{1}.png'.format(label, extra_label+'_5p'))
plt.close()
