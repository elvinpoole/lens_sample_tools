import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 

#for maglim covariance 
tpfile = '/Users/jackelvinpoole/DES/y3-3x2pt-methods/cosmosis/data_vectors/fiducial_maglim_source_sompz_lens_DNFPDF_nu.fits' 

diff_term_file1 = 'cov_diff_term_wtheta_fid_pca_50_enet_pca_50_maglim_nside4096_broadbandfit_p3_m3_sigmaA1_6p.txt'
diff_term_file2 = 'cov_diff_term_wtheta_falsebias_false_contam-enet_correct-fidpca50_maglim_sigmaA1_6p.txt'

extra_label = 'maglim_4096_nodiffterm'
#extra_label = 'maglim_4096_broadbandfit_p3_m3_5param_term1term3'
#extra_label = 'datacov_fidpca107'
#extra_label = 'broadbandfit_p3_m3'
#extra_label = 'broadbandfit_p3_m3_5params'

tp = twopoint.TwoPointFile.from_fits(tpfile)
wcov = tp.covmat[-120:,-120:]

if diff_term_file1 is None:
	diff_term_cov1 = np.zeros((len(wcov),len(wcov)))
else:
	diff_term_cov1 = np.loadtxt(diff_term_file1)

if diff_term_file2 is None:
	diff_term_cov2 = np.zeros((len(wcov),len(wcov)))
else:
	diff_term_cov2 = np.loadtxt(diff_term_file2)

diff_term_cov = diff_term_cov1 + diff_term_cov2 

wtheta_file = {

	'fid_pca_50':'wtheta_maglim_y3_data_bs0.1_fid_pca_50nside4096.fits',

	'enet_pca_50':'wtheta_maglim_y3_data_bs0.1_enet_pca_50nside4096.fits',

}
colors =  {
	'fid_std':'r',
	'fid_pca_108':'k',
	'fid_pca_107':'c',
	'enet_pca_108':'b',
	'fid_pca_50':'b',
	'enet_pca_108_hii99_cd99':'c',
	'enet_std_107':'y',
	'enet_std_108':'g',
	'enet_pca_50':'y',
	'fid_gb':'m',
}
plot_label_dict = {
	'fid_std':'fid STD',
	'fid_pca_108':'fid PCA108',
	'fid_pca_107':'fid PCA107',
	'fid_pca_50':'fid PC<50',
	'fid_gb':'NN',
	'enet_pca_108':'ENET PCA108',
	'enet_pca_108_hii99_cd99':'ENET PCA108 no-high-pixels',
	'enet_std_107':'ENET STD107',
	'enet_std_108':'ENET STD108',
	'enet_pca_50':'ENET PC<50',
}

#labels = [ 'fid_pca_108', 'fid_std', 'enet_pca_108', 'enet_pca_108_hii99_cd99', 'enet_std_107', ]
#labels = [ 'fid_std', 'fid_pca_107', 'fid_pca_108', 'enet_pca_108', 'enet_std_108', 'enet_std_107', ]
#labels = [ 'fid_std', 'fid_pca_107', 'fid_pca_108', 'enet_pca_108', 'enet_std_107', ]
#labels = [ 'fid_pca_108', 'fid_std', 'enet_pca_108', 'enet_std_108', ]
#labels = [ 'fid_pca_108', 'fid_std', 'enet_pca_108', 'enet_std_107','enet_std_108', ]
#labels = [ 'fid_pca_50', 'fid_std', 'fid_pca_107', 'enet_pca_50', ]
labels = [ 'fid_pca_50', 'enet_pca_50', ]
plot_label = '-'.join(labels)

compare_label = 'fid_pca_50'
plot_label += '-'+'compare'+compare_label + extra_label

sc = np.array([33.8758448, 24.34675, 17.40771522, 14.48788252, 12.87933950, 12.05672751] )

nbins = 6
ntheta = 20

#scale = 0.8
scale = 0.7
if nbins > 3:
	length = nbins-1
else:
	length = nbins
fig1, axs1 = plt.subplots(2,nbins/2,figsize=(scale*6.4*length/2., 2.*scale*4.8), sharey=True, squeeze=False)
fig2, axs2 = plt.subplots(2,nbins/2,figsize=(scale*6.4*length/2., 2.*scale*4.8), sharey=True, squeeze=False)
axs1 = axs1.flatten()
axs2 = axs2.flatten()

angle_min = hp.nside2resol(1024,arcmin=True)*2

#load all the wthetas
wdict = {}
for label in labels:
	w = twopoint.TwoPointFile.from_fits(wtheta_file[label],covmat_name=None).get_spectrum('wtheta')
	wdict[label] = w


for ibin in range(nbins):
	wcov_bin1 = wcov[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
	wcov_v2_bin1 = (wcov+diff_term_cov)[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
	err_bin1 = np.sqrt(wcov_bin1.diagonal())
	err_v2_bin1 = np.sqrt(wcov_v2_bin1.diagonal())
	theta0, wtheta0_bin1 = wdict[compare_label].get_pair(ibin+1,ibin+1)
	select = (theta0 > angle_min)
	axs1[ibin].errorbar(theta0[select], (theta0*wtheta0_bin1)[select], (theta0*err_v2_bin1)[select], fmt='.', color='b',capsize=2.)
	axs1[ibin].errorbar(theta0[select], (theta0*wtheta0_bin1)[select], (theta0*err_bin1)[select], fmt='.', color='k',capsize=2.)
	axs2[ibin].errorbar(theta0[select], (wtheta0_bin1-wtheta0_bin1)[select], err_v2_bin1[select], fmt='.', color='b',capsize=2.)
	axs2[ibin].errorbar(theta0[select], (wtheta0_bin1-wtheta0_bin1)[select], err_bin1[select], fmt='.', color='k',capsize=2.)
	axs1[ibin].grid()
	axs2[ibin].grid()


for label in labels:

	mask_list = []
	for ibin in range(nbins):
		wcov_bin1 = wcov[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
		wcov_v2_bin1 = (wcov+diff_term_cov)[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
		err_bin1 = np.sqrt(wcov_bin1.diagonal())

		theta,  wtheta_bin1 = wdict[label].get_pair(ibin+1,ibin+1)
		theta0, wtheta0_bin1 = wdict[compare_label].get_pair(ibin+1,ibin+1)

		select = (theta > angle_min)

		axs1[ibin].axvline(sc[ibin], color='k', ls='--')
		axs1[ibin].plot(theta[select], (theta*wtheta_bin1)[select], '-', color=colors[label], label=plot_label_dict[label] )
		#axs1[ibin].errorbar(theta[select], (theta*wmean0_bin1)[select], (theta*err_bin1)[select], fmt='.', color='k')
		axs1[ibin].set_title('bin {0}'.format(ibin+1) )
		axs1[ibin].set_xlabel(r'$\theta$', fontsize=14)
		if ibin == 0:
			axs1[ibin].set_ylabel(r'$\theta w(\theta)$', fontsize=14)
		axs1[ibin].legend()
		axs1[ibin].semilogx()


		mask = theta > sc[ibin]
		mask_list.append(mask)
		chi2 = lsssys.calc_chi2(wtheta0_bin1, wcov_v2_bin1, wtheta_bin1 , mask=mask, v=False )
		axs2[ibin].axvline(sc[ibin], color='k', ls='--')
		axs2[ibin].plot(theta[select], (wtheta_bin1 - wtheta0_bin1)[select],'-', color=colors[label], 
			lw=3., 
			label=plot_label_dict[label] + r' $\chi^2=$'+str(np.round(chi2,1)))
		#axs2[ibin].errorbar(theta[select], (wmean0_bin1-wmean0_bin1)[select], err_bin1[select], fmt='.', color='k', label='truth, cosmolike errors')
		axs2[ibin].set_title('bin {0}'.format(ibin+1) )
		axs2[ibin].set_xlabel(r'$\theta$', fontsize=14)
		if ibin == 0:
			axs2[ibin].set_ylabel(r'$\Delta w(\theta)$', fontsize=14)
		axs2[ibin].legend(loc='upper right')
		axs2[ibin].semilogx()

	#mask_all = np.hstack([np.append(np.zeros(6).astype('bool'), m1) for m1 in mask_list])
	mask_all = np.hstack(mask_list)


	#total chi2 impact
	chi2_wcov      = lsssys.calc_chi2(wdict[compare_label].value, wcov,               wdict[label].value , mask=mask_all, v=False)
	chi2_wcov_diff = lsssys.calc_chi2(wdict[compare_label].value, wcov+diff_term_cov, wdict[label].value , mask=mask_all, v=False)
	print('{compare_label} vs {label} chi2 all bins = {chi2_wcov} (wcov),  {chi2_wcov_diff} (wcov+diff)'.format(
		compare_label=plot_label_dict[compare_label],
		label=plot_label_dict[label], 
		chi2_wcov=np.round(chi2_wcov,3), 
		chi2_wcov_diff=np.round(chi2_wcov_diff,3), 
		))

fig1.tight_layout()
fig1.savefig('wtheta_allbins_{0}.png'.format(plot_label) )	
fig1.clear()

fig2.tight_layout()
axs2[0].set_ylim([-0.003, 0.003])
fig2.savefig('deltawtheta_allbins_{0}.png'.format(plot_label))	
fig2.clear()
