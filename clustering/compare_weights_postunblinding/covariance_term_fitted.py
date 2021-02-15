import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 
import scipy.optimize


tpfile = '/Users/jackelvinpoole/DES/y3-3x2pt/data/des-y3/2pt_NG_final_2ptunblind_11_13_20_wnz.fits'
tp = twopoint.TwoPointFile.from_fits(tpfile)
wcov = tp.covmat[-100:,-100:]
err = np.sqrt(wcov.diagonal())

wtheta_file = {
	#'fid_pca_108':'wtheta_redmagic_y3_data_bs0.05_fid_pca_108.fits',
	#'enet_std_107':'wtheta_redmagic_y3_data_bs0.05_enet_std_107.fits',

	'fid_pca_107':'wtheta_redmagic_y3_data_bs0.1_fid_pca_107nside4096.fits',
	'fid_pca_108':'wtheta_redmagic_y3_data_bs0.1_fid_pca_108nside4096.fits',
	'enet_std_107':'wtheta_redmagic_y3_data_bs0.1_enet_std_107nside4096.fits',

	'fid_pca_50':'wtheta_redmagic_y3_data_bs0.1_fid_pca_50nside4096.fits',
	'enet_pca_50':'wtheta_redmagic_y3_data_bs0.1_enet_pca_50nside4096.fits',
	'fid_gb':'gb_wtheta/corr_noFGCM_noDEPTH_noAIRMASS_noEBV.fits',
}

#extra_label = 'nside512'
extra_label = 'nside4096_broadbandfit_p3_m3'

#label1 = 'fid_pca_108'
label1 = 'fid_pca_50'
label2 = 'enet_pca_50'
labels = [label1, label2]

wdict = {}
for label in labels:
	w = twopoint.TwoPointFile.from_fits(wtheta_file[label],covmat_name=None).get_spectrum('wtheta')
	wdict[label] = w

sc = np.array([39.22610651, 24.75, 19.65962381, 15.61619428, 12.40438403] )

fit_xi_dict = {}
for label in labels:
	fit_xi = np.array([])
	for ibin in range(w.nbin()):
		theta, xi = wdict[label].get_pair(ibin+1, ibin+1)
		wcov_bin = wcov[(ibin)*20:(ibin+1)*20,(ibin)*20:(ibin+1)*20]
		err_bin = np.sqrt(wcov_bin.diagonal())

		#lg_theta = np.log(theta)
		#lg_xi = np.log(xi)

		def broad_band(theta, A0, A1, A2, A3, Am1, Am2, Am3):
			return A0 + \
			A1*theta**1. + \
			A2*theta**2. + \
			A3*theta**3. + \
			Am1*theta**-1. + \
			Am2*theta**-2. + \
			Am3*theta**-3.

		coeff, coeff_cov = scipy.optimize.curve_fit(broad_band, theta, xi, sigma=wcov_bin )
		fit_xi_bin = broad_band(theta, *coeff)

		fit_xi = np.append(fit_xi, fit_xi_bin)
		chi2 = lsssys.calc_chi2(xi, wcov_bin, fit_xi_bin, v=False)
		print('polynomial chi2 bin{0} {1}= {2}/{3}'.format(ibin+1,label,chi2,len(xi)))
	fit_xi_dict[label] = fit_xi

	sc_mask =  wdict[label].angle > sc[wdict[label].bin1-1]
	chi2_all = lsssys.calc_chi2(wdict[label].value, wcov, fit_xi_dict[label], v=False, mask=sc_mask)
	chi2_label = 'chi2={0}/{1}'.format(np.round(chi2_all,2), sum(sc_mask.astype('int')) )
	index = np.arange(len(wdict[label].value))
	plt.axhline(0,color='g')
	plt.errorbar(index, wdict[label].value-fit_xi_dict[label], err, color='k', fmt='.', label=None)
	plt.errorbar(index[sc_mask], (wdict[label].value-fit_xi_dict[label])[sc_mask], err[sc_mask], color='k', fmt='o', label=chi2_label)
	plt.ylabel('w_data - w_fit')
	plt.legend()
	plt.savefig( 'poly_fit_diff_{0}_{1}.png'.format(label, extra_label) )
	plt.close()


diff = fit_xi_dict[label1] - fit_xi_dict[label2]
sigmaA = 1
diff_term_cov = sigmaA**2.*np.array( (np.matrix(diff).T*np.matrix(diff)) )

outfile = 'cov_diff_term_wtheta_{0}_{1}_{2}_sigmaA{3}.txt'.format(label1, label2, extra_label, sigmaA)

np.savetxt(outfile, diff_term_cov)


plt.figure()
plt.imshow(np.log10(np.abs(diff_term_cov)))
plt.colorbar(label=r'log($|\Delta w \Delta w^T|$')
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

