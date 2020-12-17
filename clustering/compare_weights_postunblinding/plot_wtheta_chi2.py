import numpy as np 
import pylab as plt 

import twopoint 
import lsssys
import healpy as hp 

tpfile = '/Users/jackelvinpoole/DES/y3-3x2pt/data/des-y3/2pt_NG_final_2ptunblind_11_13_20_wnz.fits'
tp = twopoint.TwoPointFile.from_fits(tpfile)
wcov = tp.covmat[-100:,-100:]

wtheta_file = {
	'fid_std':'wtheta_redmagic_y3_data_bs0.05_fid_std.fits',
	'fid_pca_108':'wtheta_redmagic_y3_data_bs0.05_fid_pca_108.fits',
	'enet_pca_108':'wtheta_redmagic_y3_data_bs0.05_enet_pca_108.fits',
	'enet_pca_108_hii99_cd99':'wtheta_redmagic_y3_data_bs0.05_enet_pca_108_hii99_cd99.fits',
	'enet_std_107':'wtheta_redmagic_y3_data_bs0.05_enet_std_107.fits',
}
colors =  {
	'fid_std':'r',
	'fid_pca_108':'k',
	'enet_pca_108':'b',
	'enet_pca_108_hii99_cd99':'c',
	'enet_std_107':'y',
}
plot_label_dict = {
	'fid_std':'fid STD',
	'fid_pca_108':'fid PCA108',
	'enet_pca_108':'ENET PCA108',
	'enet_pca_108_hii99_cd99':'ENET PCA108 no-high-pixels',
	'enet_std_107':'ENET STD107',
}

#labels = [ 'fid_pca_108', 'fid_std', 'enet_pca_108', 'enet_pca_108_hii99_cd99', 'enet_std_107', ]
labels = [ 'fid_pca_108', 'fid_std', 'enet_pca_108', 'enet_std_107', ]

compare_label = 'fid_pca_108'

sc = np.array([39.22610651, 24.75, 19.65962381, 15.61619428, 12.40438403] )

nbins = 5
ntheta = 20

scale = 0.8
if nbins > 3:
	length = nbins-1
else:
	length = nbins
fig1, axs1 = plt.subplots(1,nbins,figsize=(scale*6.4*length, scale*4.8), squeeze=False)
fig2, axs2 = plt.subplots(1,nbins,figsize=(scale*6.4*length, scale*4.8), sharey=True, squeeze=False)

angle_min = hp.nside2resol(1024,arcmin=True)*2

w0 = twopoint.TwoPointFile.from_fits(wtheta_file[compare_label],covmat_name=None).get_spectrum('wtheta')
for ibin in range(nbins):
	wcov_bin1 = wcov[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
	err_bin1 = np.sqrt(wcov_bin1.diagonal())
	theta0, wtheta0_bin1 = w0.get_pair(ibin+1,ibin+1)
	select = (theta0 > angle_min)
	axs1[0][ibin].errorbar(theta0[select], (theta0*wtheta0_bin1)[select], (theta0*err_bin1)[select], fmt='.', color='k')
	axs2[0][ibin].errorbar(theta0[select], (wtheta0_bin1-wtheta0_bin1)[select], err_bin1[select], fmt='.', color='k',)
	axs1[0][ibin].grid()
	axs2[0][ibin].grid()

for label in labels:
	w = twopoint.TwoPointFile.from_fits(wtheta_file[label],covmat_name=None).get_spectrum('wtheta')

	mask_list = []
	for ibin in range(nbins):
		wcov_bin1 = wcov[(ibin+1)*20-ntheta:(ibin+1)*20, (ibin+1)*20-ntheta:(ibin+1)*20]
		err_bin1 = np.sqrt(wcov_bin1.diagonal())

		theta,  wtheta_bin1 = w.get_pair(ibin+1,ibin+1)
		theta0, wtheta0_bin1 = w0.get_pair(ibin+1,ibin+1)

		select = (theta > angle_min)

		axs1[0][ibin].axvline(sc[ibin], color='k', ls='--')
		axs1[0][ibin].plot(theta[select], (theta*wtheta_bin1)[select], '-', color=colors[label], label=plot_label_dict[label] )
		#axs1[0][ibin].errorbar(theta[select], (theta*wmean0_bin1)[select], (theta*err_bin1)[select], fmt='.', color='k')
		axs1[0][ibin].set_title('bin {0}'.format(ibin+1) )
		axs1[0][ibin].set_xlabel(r'$\theta$', fontsize=14)
		if ibin == 0:
			axs1[0][ibin].set_ylabel(r'$\theta w(\theta)$', fontsize=14)
		axs1[0][ibin].legend()
		axs1[0][ibin].semilogx()


		mask = theta > sc[ibin]
		mask_list.append(mask)
		chi2 = lsssys.calc_chi2(wtheta0_bin1, wcov_bin1, wtheta_bin1 , mask=mask, v=False )
		axs2[0][ibin].axvline(sc[ibin], color='k', ls='--')
		axs2[0][ibin].plot(theta[select], (wtheta_bin1 - wtheta0_bin1)[select],'-', color=colors[label], lw=2., label=plot_label_dict[label] )
		#axs2[0][ibin].errorbar(theta[select], (wmean0_bin1-wmean0_bin1)[select], err_bin1[select], fmt='.', color='k', label='truth, cosmolike errors')
		axs2[0][ibin].set_title('bin {0}'.format(ibin+1) )
		axs2[0][ibin].set_xlabel(r'$\theta$', fontsize=14)
		if ibin == 0:
			axs2[0][ibin].set_ylabel(r'$\Delta w(\theta)$', fontsize=14)
		axs2[0][ibin].legend()
		axs2[0][ibin].semilogx()

	#mask_all = np.hstack([np.append(np.zeros(6).astype('bool'), m1) for m1 in mask_list])
	mask_all = np.hstack(mask_list)


	#total chi2 impact
	chi2 = lsssys.calc_chi2(w0.value, wcov, w.value , mask=mask_all, v=False)
	print('{compare_label} vs {label} chi2 all bins = {chi2}'.format(compare_label=plot_label_dict[compare_label],label=plot_label_dict[label], chi2=chi2))

fig1.tight_layout()
fig1.savefig('wtheta_allbins.png' )	
fig1.clear()

fig2.tight_layout()
fig2.savefig('deltawtheta_allbins.png')	
fig2.clear()



