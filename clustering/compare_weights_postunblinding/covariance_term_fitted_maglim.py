import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 
import scipy.optimize


#for maglim covariance 
tpfile = '/Users/jackelvinpoole/DES/y3-3x2pt-methods/cosmosis/data_vectors/fiducial_maglim_source_sompz_lens_DNFPDF_nu.fits' 
tp = twopoint.TwoPointFile.from_fits(tpfile)
wcov = tp.covmat[-120:,-120:]
err = np.sqrt(wcov.diagonal())

wtheta_file = {

	'fid_pca_50':'wtheta_maglim_y3_data_bs0.1_fid_pca_50nside4096.fits',

	'enet_pca_50':'wtheta_maglim_y3_data_bs0.1_enet_pca_50nside4096.fits',
}

#extra_label = 'nside512'
extra_label = 'maglim_nside4096_broadbandfit_p3_m3'

#label1 = 'fid_pca_108'
label1 = 'fid_pca_50'
label2 = 'enet_pca_50'
labels = [label1, label2]

label_names = {
	label1:'ISD',
	label2:'ENET',
}

wdict = {}
for label in labels:
	w = twopoint.TwoPointFile.from_fits(wtheta_file[label],covmat_name=None).get_spectrum('wtheta')
	wdict[label] = w

sc = np.array([33.8758448, 24.34675, 17.40771522, 14.48788252, 12.87933950, 12.05672751] )

sample_name = 'MagLim'

fit_xi_dict = {}
fig, axs = plt.subplots(2,1,figsize=(8,6))
axs = axs.flatten()
axs[0].set_title(sample_name)
fig1, axs1 = plt.subplots(1,1,figsize=(8,4))
axs1.set_title(sample_name)
fig2, axs2 = plt.subplots(2,1,figsize=(8,6))
axs2 = axs2.flatten()
axs2[0].set_title(sample_name, fontsize=16)
fig3, axs3 = plt.subplots(3,1,figsize=(10,6))
axs3 = axs3.flatten()
axs3[0].set_title(sample_name)
for ilabel, label in enumerate(labels):
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
		np.savetxt('coeff_B_maglim_bin{0}_{1}.txt'.format(ibin, label), coeff)

		fit_xi = np.append(fit_xi, fit_xi_bin)
		chi2 = lsssys.calc_chi2(xi, wcov_bin, fit_xi_bin, v=False)
		print('polynomial chi2 bin{0} {1}= {2}/{3}'.format(ibin+1,label,chi2,len(xi)))
	fit_xi_dict[label] = fit_xi

	sc_mask =  wdict[label].angle > sc[wdict[label].bin1-1]
	chi2_all = lsssys.calc_chi2(wdict[label].value, wcov, fit_xi_dict[label], v=False, mask=sc_mask)
	chi2_label = r'$\chi^{2}$'+'={0}/{1}'.format(np.round(chi2_all,2), sum(sc_mask.astype('int')) )
	index = np.arange(len(wdict[label].value))
	plt.figure()
	plt.axhline(0,color='g')
	plt.errorbar(index, wdict[label].value-fit_xi_dict[label], err, color='k', fmt='.', label=None)
	plt.errorbar(index[sc_mask], (wdict[label].value-fit_xi_dict[label])[sc_mask], err[sc_mask], color='k', fmt='o', label=chi2_label)
	plt.ylabel('w_data - w_fit')
	plt.legend()
	plt.savefig( 'poly_fit_diff_{0}_{1}.png'.format(label, extra_label) )
	plt.close()

	axs[ilabel].axhline(0,color='g')
	axs[ilabel].errorbar(index, wdict[label].value-fit_xi_dict[label], err, color='k', fmt='.', label=None)
	axs[ilabel].errorbar(index[sc_mask], (wdict[label].value-fit_xi_dict[label])[sc_mask], err[sc_mask], color='k', fmt='o', label=chi2_label)
	axs[ilabel].set_ylabel(r'$w(\theta)_{\rm data} - w(\theta)_{\rm poly-fit}$',fontsize=14)
	axs[ilabel].legend()

	axs3[ilabel].axhline(0,color='g')
	axs3[ilabel].errorbar(index, wdict[label].value-fit_xi_dict[label], err, color='k', fmt='.', label=None)
	axs3[ilabel].errorbar(index[sc_mask], (wdict[label].value-fit_xi_dict[label])[sc_mask], err[sc_mask], color='k', fmt='o', label=chi2_label)
	axs3[ilabel].set_ylabel(r'$w(\theta)_{\rm %s} - w(\theta)_{\rm poly-fit}$' % label_names[label],fontsize=12)
	axs3[ilabel].legend()

	if label == label1:
		axs2[ilabel].axhline(0,color='g')
		axs2[ilabel].errorbar(index, wdict[label].value-fit_xi_dict[label], err, color='k', fmt='.', label=None)
		axs2[ilabel].errorbar(index[sc_mask], (wdict[label].value-fit_xi_dict[label])[sc_mask], err[sc_mask], color='k', fmt='o', label=chi2_label)
		axs2[ilabel].set_ylabel(r'$w(\theta)_{\rm %s} - w(\theta)_{\rm poly-fit}$' % label_names[label],fontsize=16)
		axs2[ilabel].legend(fontsize=14)

############
axs1.axhline(0,color='g')
for ibin in range(w.nbin()):
	if ibin == 0:
		fit_label = 'fitted polynomial difference'
	else:
		fit_label = None
	axs1.plot(
		index[ibin*20:(ibin+1)*20],  
		(fit_xi_dict[label1]-fit_xi_dict[label2])[ibin*20:(ibin+1)*20],
		color='red', label = fit_label, lw=2.)
axs1.plot(index[~sc_mask], 
	(wdict[label1].value-wdict[label2].value)[~sc_mask], 
	'.', color='k', label=None)
axs1.plot(index[sc_mask], 
	(wdict[label1].value-wdict[label2].value)[sc_mask], 
	'o', color='k', label=None)
axs1.set_ylabel(r'$w(\theta)_{\rm ISD} - w(\theta)_{\rm ENET}$',fontsize=14)
axs1.legend()

############
axs2[1].axhline(0,color='g')
for ibin in range(w.nbin()):
	if ibin == 0:
		fit_label = 'fitted polynomial difference'
	else:
		fit_label = None
	axs2[1].plot(
		index[ibin*20:(ibin+1)*20],  
		(fit_xi_dict[label1]-fit_xi_dict[label2])[ibin*20:(ibin+1)*20],
		color='red', label = fit_label, lw=2.)
axs2[1].plot(index[~sc_mask], 
	(wdict[label1].value-wdict[label2].value)[~sc_mask], 
	'.', color='k', label=None)
axs2[1].plot(index[sc_mask], 
	(wdict[label1].value-wdict[label2].value)[sc_mask], 
	'o', color='k', label=None)
axs2[1].set_ylabel(r'$w(\theta)_{\rm ISD} - w(\theta)_{\rm ENET}$',fontsize=16)
axs2[1].legend(fontsize=14)

axs2[1].set_xlabel('INDEX',fontsize=14)
#tick size
for tick in axs2[0].xaxis.get_major_ticks():
	tick.label.set_fontsize(14) 
for tick in axs2[0].yaxis.get_major_ticks():
	tick.label.set_fontsize(14) 
for tick in axs2[1].xaxis.get_major_ticks():
	tick.label.set_fontsize(14) 
for tick in axs2[1].yaxis.get_major_ticks():
	tick.label.set_fontsize(14) 

############
axs3[2].axhline(0,color='g')
for ibin in range(w.nbin()):
	if ibin == 0:
		fit_label = 'fitted polynomial difference'
	else:
		fit_label = None
	axs3[2].plot(
		index[ibin*20:(ibin+1)*20],  
		(fit_xi_dict[label1]-fit_xi_dict[label2])[ibin*20:(ibin+1)*20],
		color='red', label = fit_label, lw=2.)
axs3[2].plot(index[~sc_mask], 
	(wdict[label1].value-wdict[label2].value)[~sc_mask], 
	'.', color='k', label=None)
axs3[2].plot(index[sc_mask], 
	(wdict[label1].value-wdict[label2].value)[sc_mask], 
	'o', color='k', label=None)
axs3[2].set_ylabel(r'$w(\theta)_{\rm ISD} - w(\theta)_{\rm ENET}$',fontsize=12)
axs3[2].legend()

fig.tight_layout()
fig.savefig('poly_fit_joint_{0}_{1}.png'.format('_'.join(labels), extra_label))

fig1.tight_layout()
fig1.savefig('poly_fit_diff_joint_{0}_{1}.png'.format('_'.join(labels), extra_label))

fig2.tight_layout()
fig2.savefig('poly_fit_label1_diff_joint_{0}_{1}.png'.format('_'.join(labels), extra_label))

fig3.tight_layout()
fig3.savefig('poly_fit_label1_label2_diff_joint_{0}_{1}.png'.format('_'.join(labels), extra_label))


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
nbins = 6
for ibin in range(nbins):
	for jbin in range(nbins):
		if ibin != jbin:
			diff_term_cov_5p[ibin*20:(ibin+1)*20,jbin*20:(jbin+1)*20] = 0.

np.savetxt(outfile[:-4]+'_6p.txt', diff_term_cov_5p)

plt.figure()
plt.imshow(np.log10(np.abs(diff_term_cov_5p)))
plt.colorbar(label=r'log$|\Delta w \Delta w^T|$)')
plt.title(outfile)
plt.savefig('cov_diff_{0}_{1}_{2}.png'.format(label1, label2, extra_label+'_6p'))
plt.close()

