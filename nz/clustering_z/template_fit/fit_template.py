"""
uses the gaussian sum n(z) as a template, manipulates it in some way, then fits to the xcorr n(z)
"""
import numpy as np
import pylab as plt

import autograd 
import scipy.interpolate as interp
import misc
import scipy.optimize
import scipy.stats
import h5py

outdir = misc.dir('./nov19_plots/')
plotdir = misc.dir(outdir + './plots/')

#rmdir = '/Users/jackelvinpoole/DES/cats/y3/redmagic/combined_sample_6.4.22/'
rm_nz_table = np.loadtxt( 'nz_combined_overlap.txt', unpack=True )

xcorr_file_list = [	'../eboss_xcorr_redshifts_mar19/y3_6.4.22_hidens_bin0_eboss5_mar19/BNz_Schmidt___biasr_AC_R_D___jackknife_pairs_100.h5',
					'../eboss_xcorr_redshifts_mar19/y3_6.4.22_hidens_bin1_eboss5_mar19/BNz_Schmidt___biasr_AC_R_D___jackknife_pairs_100.h5',
					'../eboss_xcorr_redshifts_mar19/y3_6.4.22_hidens_bin2_eboss5_mar19/BNz_Schmidt___biasr_AC_R_D___jackknife_pairs_100.h5',
					'../eboss_xcorr_redshifts_mar19/y3_6.4.22_hilum_bin3_eboss5_mar19/BNz_Schmidt___biasr_AC_R_D___jackknife_pairs_100.h5',
					'../eboss_xcorr_redshifts_mar19/y3_6.4.22_hilum_bin4_eboss5_mar19/BNz_Schmidt___biasr_AC_R_D___jackknife_pairs_100.h5', ]

gamma_correct_list = [
	#(gamma, gamma_var),
	(0.,1.5),
	(0.,1.5),
	(0.,1.5),
	(0.,2.),
	(0.,2.),
]

apply_gamma_error = False
use_jacobian = True

fitted_nz_table = []
fitted_nz_table.append(rm_nz_table[0])
coeff_mean_list = []
coeff_width_list = []
coeff_mean_width_list = []
delta_z_y1_method_list = []
for ibin in xrange(5):
	print 'bin', ibin+1

	#load the redmagic n(z) as a 'theory' prediction
	z_theory = rm_nz_table[0]
	nz_theory = rm_nz_table[ibin+1]

	#load the xcorr n(z) as a data prediction
	xcorr_file =h5py.File(xcorr_file_list[ibin])
	z_edges_data = np.transpose(xcorr_file['z_edges/block0_values'].value)[0]
	z_data = np.transpose(xcorr_file['z/block0_values'].value)[0]
	nz_data = np.transpose(xcorr_file['results/block0_values'].value)[0]
	cov = np.transpose(xcorr_file['cov/block0_values'].value)
	err = np.sqrt(np.diagonal(cov))
	dz_data = z_edges_data[1]-z_edges_data[0]

	##### apply bias correction #####
	def apply_gamma(nz_and_gamma):
		gammacorrection = (1. + z_data)**nz_and_gamma[-1]
		norm_gamma = sum(nz_and_gamma[:-1])/sum(nz_and_gamma[:-1]*gammacorrection)
		return nz_and_gamma[:-1] * gammacorrection * norm_gamma

	gamma = gamma_correct_list[ibin][0]
	gamma_var = gamma_correct_list[ibin][1]
	nz_and_gamma = np.append(nz_data, gamma)
	nz_data = apply_gamma(nz_and_gamma)

	#propagate the gamma variance into the n(z) covariance with a jacobian
	if apply_gamma_error==True and use_jacobian == True:
		print 'propagating gamma error in n(z) using the jacobian'
		j = autograd.jacobian( apply_gamma )
		covmat = np.zeros((len(nz_data)+1,len(nz_data)+1))
		covmat[:-1,:-1] = cov
		covmat[-1,-1] = gamma_var 
		cov_corrected = np.matrix(j(nz_and_gamma))*np.matrix(covmat)*np.matrix(j(nz_and_gamma)).T
		cov_corrected = np.array(cov_corrected)
		cov = cov_corrected
		err = np.sqrt(np.diagonal(cov))
		np.savetxt(outdir + 'cov_corrected_bin{0}_jacobian_method.txt'.format(ibin+1), cov_corrected)

	#propagate the gamma variance into the n(z) covariance with random draws
	elif apply_gamma_error==True and use_jacobian == False :
		print 'propagating gamma error in n(z) using random draws from the data'
		ndraws = 100000
		nz_draws = np.random.multivariate_normal(nz_data, cov, ndraws)
		gamma_draws = np.random.multivariate_normal([0], [[gamma_var]], ndraws)
		nz_and_gamma_draws = np.hstack((nz_draws,gamma_draws)) 
		nz_corrected_draws =  np.array([apply_gamma(nz_and_gamma_draw) for nz_and_gamma_draw in nz_and_gamma_draws])
		cov_corrected = np.cov(nz_corrected_draws, rowvar=False)
		cov = cov_corrected
		err = np.sqrt(np.diagonal(cov))
		np.savetxt(outdir + 'cov_corrected_bin{0}_random_draws_method.txt'.format(ibin+1), cov_corrected)
	else:
		print 'not applying the gamma error'

	select_range = (z_theory > z_edges_data.min())*(z_theory < z_edges_data.max())
	norm = dz_data
	nz_theory = norm*nz_theory

	def shift_1(z_eval,deltaz):
		"""
		n(z) = n_fid(z + deltaz)
		"""
		return interp.interp1d(z_theory+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)

	def stretch_1(z_eval,stretch):
		"""
		n(z) = n_fid(stretch*(z - zmean) + zmean)
		"""
		zmean = np.average(z_theory,weights=nz_theory)
		return interp.interp1d(stretch*(z_theory-zmean)+zmean, nz_theory,fill_value=0.,bounds_error=False)(z_eval)/stretch

	def stretch_shift_1(z_eval,stretch,deltaz):
		"""
		n(z) = n_fid(stretch*(z - zmean) + zmean + deltaz) I think
		"""
		zmean = np.average(z_theory,weights=nz_theory)
		return interp.interp1d(stretch*(z_theory-zmean)+zmean+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)/stretch


	delta_z_y1_method = np.trapz(z_data*nz_data,x=z_data)/np.trapz(nz_data,x=z_data)-np.trapz(z_theory*nz_theory,x=z_theory)/np.trapz(nz_theory,x=z_theory)
	delta_z_y1_method_list.append(delta_z_y1_method)

	coeff, coeff_cov = scipy.optimize.curve_fit(shift_1, z_data, nz_data, p0=[0.], sigma=cov,  absolute_sigma=True)
	coeff_mean_list.append(coeff)
	np.savetxt(outdir + 'nz_mean_cov_bin{0}.txt'.format(ibin+1), coeff_cov )

	coeff, coeff_cov = scipy.optimize.curve_fit(stretch_1, z_data, nz_data, p0=[1.], sigma=cov,  absolute_sigma=True)
	coeff_width_list.append(coeff)
	np.savetxt(outdir + 'nz_width_cov_bin{0}.txt'.format(ibin+1), coeff_cov )

	coeff, coeff_cov = scipy.optimize.curve_fit(stretch_shift_1, z_data, nz_data, p0=[1.,0.], sigma=cov,  absolute_sigma=True)
	coeff_mean_width_list.append(coeff)
	np.savetxt(outdir + 'nz_width_mean_cov_bin{0}.txt'.format(ibin+1), coeff_cov )

	ngrid = 100
	delta_array = np.linspace(coeff[1]-3.*np.sqrt(coeff_cov[1,1]),coeff[1]+3.*np.sqrt(coeff_cov[1,1]),ngrid)
	stretch_array = np.linspace(coeff[0]-3.*np.sqrt(coeff_cov[0,0]),coeff[0]+3.*np.sqrt(coeff_cov[0,0]),ngrid)

	X, Y = np.meshgrid(stretch_array, delta_array)
	from scipy.stats import multivariate_normal
	gauss = multivariate_normal(coeff, coeff_cov)
	Z = gauss.pdf(np.transpose([X.flatten(),Y.flatten()]))
	Z = Z.reshape(100,100)

	plt.figure()
	plt.axhline(0.0, color='k',ls='--')
	plt.axvline(1.0, color='k',ls='--')
	frac_levels = [0.9545,0.6827]
	Zsort = np.sort(Z.flatten())
	levels = [Zsort[np.abs(np.cumsum(Zsort)-(1-frac_level)*np.sum(Z)) == np.abs(np.cumsum(Zsort)-(1-frac_level)*np.sum(Z)).min()][0] for frac_level in frac_levels]
	plt.contour(X,Y,Z,levels=levels,colors='r',label='bin {0}'.format(ibin+1))
	plt.title('bin {0}'.format(ibin+1))
	plt.xlabel('stretch')
	plt.ylabel(r'$\Delta z$')
	plt.legend()
	plt.tight_layout()
	plt.savefig(plotdir + '2d_contours_deltaz_stretch_bin{0}.png'.format(ibin+1))
	plt.close()

	redmagic_chi2 = misc.calc_chi2(nz_data, cov, interp.interp1d(z_theory, nz_theory)(z_data))
	fitted_chi2   = misc.calc_chi2(nz_data, cov, interp.interp1d(z_theory, stretch_shift_1(z_theory,coeff[0],coeff[1]))(z_data))
	plt.plot(z_theory[select_range], nz_theory[select_range], 'b-', label='redmagic, chi2={0}/{1}'.format(np.round(redmagic_chi2,decimals=1), len(nz_data)))
	plt.plot(z_theory[select_range], stretch_shift_1(z_theory[select_range],coeff[0],coeff[1]), 'r-', label='fit to xcorr ' + r'$\Delta z, \ s$' + ' chi2={0}/{1}'.format(np.round(fitted_chi2,decimals=1),len(nz_data)))
	plt.errorbar(z_data, nz_data, err, fmt='.', label='xcorr')
	plt.xlabel('z')
	plt.legend()
	plt.savefig(plotdir + 'nz_bin{0}.png'.format(ibin+1))
	plt.close()


	fitted_nz_table.append(stretch_shift_1(z_theory,coeff[0],coeff[1]))

np.savetxt(outdir + 'nz_mean_delta_z_y1_method.txt', delta_z_y1_method_list , header='deltaz')
np.savetxt(outdir + 'nz_mean_coeff.txt', coeff_mean_list, header='deltaz')
np.savetxt(outdir + 'nz_width_coeff.txt', coeff_width_list, header='stretch')
np.savetxt(outdir + 'nz_width_mean_coeff.txt', coeff_mean_width_list, header='stretch\tdeltaz')

np.savetxt(outdir + 'nz_combined_fitted_mean_width.txt',  np.transpose(fitted_nz_table) )




