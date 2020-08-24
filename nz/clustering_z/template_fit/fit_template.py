"""
uses the gaussian sum n(z) as a template, manipulates it in some way, then fits to the xcorr n(z)
"""
import numpy as np
import pylab as plt

import autograd 
import scipy.interpolate as interp
import scipy.optimize
import scipy.stats
import h5py
import os

outdir = './output_Apr23_2020/'
plotdir = './plots_Apr23_2020/'

#do you want to propagate the gamma uncertainty into the n(z) covariance?
#'N' to ignore gamma uncertainty, 'J' to use the jacobian, 'R' for random draws
#apply_gamma_error = 'N'
apply_gamma_error = 'J'
#apply_gamma_error = 'R'

ndraws = 100000 #used if apply_gamma_error = R


#april 23 2020 results
samples = [
	{
		'label':'rm_0.5.1_wide_0.9binning_{apply_gamma_error}'.format(apply_gamma_error=apply_gamma_error),
		'wz_data_dir':'../data/y3_redmagic_clusteringz_Apr23_2020/',
		'nz_overlap_file':'output_Nov20_2019/nz_combined_overlap_rm_0.5.1_wide_0.9binning.txt', #should be same as nov result
		'gammaarray':[0.4,0.2,0.0,0.3,1.0],
		'gammaunc':[0.4,0.4,0.6,0.8,3.0],
		'nbins':5,
	}, 
]
tmp = [
	{
		'label':'maglim_v22_{apply_gamma_error}'.format(apply_gamma_error=apply_gamma_error),
		'wz_data_dir':'../data/y3_maglim_clusteringz_Apr23_2020/',
		'nz_overlap_file':'<PAU+VIPERS NZ FILE>',  #need to update this for PAU+VIPERS
		'gammaarray':[-0.6,-0.5,1.0,0.5,-1.3,0.2],
		'gammaunc':[0.3,0.7,0.9,0.8,1.8,1.8],
		'nbins':6,
	},
]

"""
samples_nov20_2019 = [
	{
		'label':'rm_0.5.1_wide_0.9binning_{apply_gamma_error}'.format(apply_gamma_error=apply_gamma_error),
		'wz_data_dir':'../data/y3_redmagic_clusteringz_nov20_2019/',
		'nz_overlap_file':'output/nz_combined_overlap_rm_0.5.1_wide_0.9binning.txt',
		'gammaarray':[-0.1,0.7,-0.1,2.4,-0.3],
		'gammaunc':[0.5,0.5,1.25,1.25,2.0],
		'nbins':5,
	}, 
	{
		'label':'maglim_v22_{apply_gamma_error}'.format(apply_gamma_error=apply_gamma_error),
		'wz_data_dir':'../data/y3_maglim_clusteringz_nov20_2019/',
		'nz_overlap_file':'output/nz_combined_overlap_maglim_v22.txt',
		'gammaarray':[-1.1,-3.1,0.7,3.7,1.9,-2.4],
		'gammaunc':[1.5,2.0,1.5,2.5,3.0,2.5],
		'nbins':6,
	},
]
"""

def calc_chi2( y, err, yfit ):
	if err.shape == (len(y),len(y)): #use full covariance
		inv_cov = np.linalg.inv( np.matrix(err) )
		chi2 = 0
		for i in xrange(len(y)):
			for j in xrange(len(y)):
				chi2 = chi2 + (y[i]-yfit[i])*inv_cov[i,j]*(y[j]-yfit[j])
		return chi2
	elif err.shape == (len(y),): #use sqrt(diagonal)
		return sum(((y-yfit)**2.)/(err**2.))
	else:
		raise IOError('error in err or cov_mat input shape')

for sample in samples:
	nz_table = np.loadtxt( sample['nz_overlap_file'], unpack=True )
	xcorr_file_list = np.sort([sample['wz_data_dir']+f for f in os.listdir(sample['wz_data_dir']) if '.h5' in f])

	#check you have the correct number of bins
	assert len(sample['gammaarray']) == sample['nbins']
	assert len(sample['gammaunc']) == sample['nbins']
	assert len(nz_table) == sample['nbins'] + 1
	assert len(xcorr_file_list) == sample['nbins']

	fitted_nz_table = []
	fitted_nz_table.append(nz_table[0])
	coeff_mean_list = []
	coeff_mean_cov_list = []
	coeff_width_list = []
	coeff_mean_width_list = []
	coeff_mean_width_cov_list = []
	delta_z_y1_method_list = []
	####
	# set up joint plots
	nx = int(np.floor(np.sqrt(sample['nbins'])))
	ny = int(np.ceil(np.sqrt(sample['nbins'])))
	fig1, axs1 = plt.subplots(nx, ny, figsize=(3*ny,3*nx))
	fig2, axs2 = plt.subplots(nx, ny, figsize=(3*ny,3*nx))
	####
	for ibin in xrange(sample['nbins']):
		print 'bin', ibin+1
		ix = np.repeat(np.arange(ny), nx)[ibin]
		iy =  np.tile(np.arange(nx), ny)[ibin]

		#load the photo-z n(z) as a 'theory' prediction
		z_theory = nz_table[0]
		nz_theory = nz_table[ibin+1]
		dz_theory = z_theory[1]-z_theory[0]

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

		#gamma = gamma_correct_list[ibin][0]
		#gamma_var = gamma_correct_list[ibin][1]
		gamma = sample['gammaarray'][ibin]
		gamma_var = sample['gammaunc'][ibin]
		nz_and_gamma = np.append(nz_data, gamma)
		nz_data = apply_gamma(nz_and_gamma)

		if apply_gamma_error == 'N':
			print('not applying the gamma error')

		#propagate the gamma variance into the n(z) covariance with a jacobian
		elif apply_gamma_error == 'J':
			print('propagating gamma error in n(z) using the jacobian')
			j = autograd.jacobian( apply_gamma )
			covmat = np.zeros((len(nz_data)+1,len(nz_data)+1))
			covmat[:-1,:-1] = cov
			covmat[-1,-1] = gamma_var 
			cov_corrected = np.matrix(j(nz_and_gamma))*np.matrix(covmat)*np.matrix(j(nz_and_gamma)).T
			cov_corrected = np.array(cov_corrected)
			cov = cov_corrected
			err = np.sqrt(np.diagonal(cov))
			np.savetxt(outdir + '{label}_cov_corrected_bin{ibin}_jacobian_method.txt'.format(ibin=ibin+1, label=sample['label']), cov_corrected)

		#propagate the gamma variance into the n(z) covariance with random draws
		elif apply_gamma_error == 'R':
			print('propagating gamma error in n(z) using random draws from the data')
			nz_draws = np.random.multivariate_normal(nz_data, cov, ndraws)
			gamma_draws = np.random.multivariate_normal([0], [[gamma_var]], ndraws)
			nz_and_gamma_draws = np.hstack((nz_draws,gamma_draws)) 
			nz_corrected_draws =  np.array([apply_gamma(nz_and_gamma_draw) for nz_and_gamma_draw in nz_and_gamma_draws])
			cov_corrected = np.cov(nz_corrected_draws, rowvar=False)
			cov = cov_corrected
			err = np.sqrt(np.diagonal(cov))
			np.savetxt(outdir + '{label}_cov_corrected_bin{ibin}_random_draws_method.txt'.format(ibin=ibin+1, label=sample['label']), cov_corrected)
		else:
			raise IOError("apply_gamma_error should be 'N', 'J' or 'R' ")

		select_range = (z_theory > z_edges_data.min())*(z_theory < z_edges_data.max())
		
		#normalise the photoz nz to match the clustering-z 
		#norm = dz_data
		norm = (sum(nz_data)*dz_data)/(sum(nz_theory)*dz_theory)
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


		#run the template fits (and a y1 style difference in means for comparison)
		delta_z_y1_method = np.trapz(z_data*nz_data,x=z_data)/np.trapz(nz_data,x=z_data)-np.trapz(z_theory*nz_theory,x=z_theory)/np.trapz(nz_theory,x=z_theory)
		delta_z_y1_method_list.append(delta_z_y1_method)

		coeff, coeff_cov = scipy.optimize.curve_fit(shift_1, z_data, nz_data, p0=[0.], sigma=cov,  absolute_sigma=True)
		coeff_mean_list.append(coeff)
		coeff_mean_cov_list.append(np.sqrt(coeff_cov)) #error
		np.savetxt(outdir + '{label}_nz_mean_cov_bin{ibin}_.txt'.format(ibin=ibin+1, label=sample['label']), coeff_cov )

		coeff, coeff_cov = scipy.optimize.curve_fit(stretch_1, z_data, nz_data, p0=[1.], sigma=cov,  absolute_sigma=True)
		coeff_width_list.append(coeff)
		np.savetxt(outdir + '{label}_nz_width_cov_bin{ibin}.txt'.format(ibin=ibin+1, label=sample['label']), coeff_cov )

		coeff, coeff_cov = scipy.optimize.curve_fit(stretch_shift_1, z_data, nz_data, p0=[1.,0.], sigma=cov,  absolute_sigma=True)
		coeff_mean_width_list.append(coeff)
		coeff_mean_width_cov_list.append(coeff_cov)
		np.savetxt(outdir + '{label}_nz_width_mean_cov_bin{ibin}.txt'.format(ibin=ibin+1, label=sample['label']), coeff_cov )

		ngrid = 100
		delta_array = np.linspace(coeff[1]-3.*np.sqrt(coeff_cov[1,1]),coeff[1]+3.*np.sqrt(coeff_cov[1,1]),ngrid)
		stretch_array = np.linspace(coeff[0]-3.*np.sqrt(coeff_cov[0,0]),coeff[0]+3.*np.sqrt(coeff_cov[0,0]),ngrid)

		########### start plot: 2d contours of the stretch+shift fit ############
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
		plt.savefig(plotdir + '{label}_2d_contours_deltaz_stretch_bin{ibin}.png'.format(ibin=ibin+1, label=sample['label']))
		plt.close()
		axs1[iy,ix].axhline(0.0, color='k',ls='--')
		axs1[iy,ix].axvline(1.0, color='k',ls='--')
		axs1[iy,ix].contour(X,Y,Z,levels=levels,colors='r',label='bin {0}'.format(ibin+1))
		axs1[iy,ix].set_title('bin {0}'.format(ibin+1))
		axs1[iy,ix].set_xlabel('stretch')
		axs1[iy,ix].set_ylabel(r'$\Delta z$')
		axs1[iy,ix].legend()
		########### end plot ############

		########### start plot: plot best fit n(z) ############
		photoz_chi2 = calc_chi2(nz_data, cov, interp.interp1d(z_theory, nz_theory)(z_data))
		fitted_chi2   = calc_chi2(nz_data, cov, interp.interp1d(z_theory, stretch_shift_1(z_theory,coeff[0],coeff[1]))(z_data))
		plt.figure()
		plt.plot(z_theory[select_range], nz_theory[select_range], 'b-', label='photoz, chi2={0}/{1}'.format(np.round(photoz_chi2,decimals=1), len(nz_data)))
		plt.plot(z_theory[select_range], stretch_shift_1(z_theory[select_range],coeff[0],coeff[1]), 'r-', label='fit to xcorr ' + r'$\Delta z, \ s$' + ' chi2={0}/{1}'.format(np.round(fitted_chi2,decimals=1),len(nz_data)))
		plt.errorbar(z_data, nz_data, err, fmt='.', label='xcorr')
		plt.xlabel('z')
		plt.legend()
		plt.savefig(plotdir + '{label}_nz_bin{ibin}.png'.format(ibin=ibin+1, label=sample['label']))
		plt.close()
		axs2[iy,ix].plot(z_theory[select_range], nz_theory[select_range], 'b-', label='photoz, chi2={0}/{1}'.format(np.round(photoz_chi2,decimals=1), len(nz_data)))
		axs2[iy,ix].plot(z_theory[select_range], stretch_shift_1(z_theory[select_range],coeff[0],coeff[1]), 'r-', label='fit to xcorr ' + r'$\Delta z, \ s$' + ' chi2={0}/{1}'.format(np.round(fitted_chi2,decimals=1),len(nz_data)))
		axs2[iy,ix].errorbar(z_data, nz_data, err, fmt='.', label='xcorr')
		axs2[iy,ix].set_xlabel('z')
		#axs2[iy,ix].legend(fontsize=8)
		########### end plot ############

		fitted_nz_table.append(stretch_shift_1(z_theory,coeff[0],coeff[1]))


	#save figs 
	fig1.tight_layout()
	fig1.savefig(plotdir + '{label}_2d_contours_deltaz_stretch_allbins.png'.format(label=sample['label']))
	fig1.clear()
	fig2.tight_layout()
	fig2.savefig(plotdir + '{label}_nz_allbins.png'.format( label=sample['label']) )
	fig2.clear()

	#save the results
	np.savetxt(outdir + '{label}_nz_mean_delta_z_y1_method.txt'.format(label=sample['label']), delta_z_y1_method_list , header='deltaz')
	np.savetxt(outdir + '{label}_nz_mean_coeff.txt'.format(label=sample['label']), coeff_mean_list, header='deltaz')
	np.savetxt(outdir + '{label}_nz_width_coeff.txt'.format(label=sample['label']), coeff_width_list, header='stretch')
	np.savetxt(outdir + '{label}_nz_width_mean_coeff.txt'.format(label=sample['label']), coeff_mean_width_list, header='stretch\tdeltaz')
	np.savetxt(outdir + '{label}_nz_combined_fitted_mean_width.txt'.format(label=sample['label']),  np.transpose(fitted_nz_table) )

	#save in cosmosis format
	cosmosis_style_mean = '[lens_photoz_errors]\n'+'\n'.join( ['bias_{0} = gaussian {1} {2}'.format( i+1, coeff_mean_list[i][0], np.sqrt(coeff_mean_cov_list[i][0][0])) for i in xrange(sample['nbins'])] )
	f = open(outdir + '{label}_nz_mean_cosmosis_style.txt'.format(label=sample['label']), 'w')
	f.write(cosmosis_style_mean)
	f.close()

	cosmosis_style_width_mean = "var_bias           =  {0}\n".format( ' '.join(["%.10f" % coeff_mean_width_cov_list[i][1][1] for i in xrange(sample['nbins'])]) ) + \
								"var_width          =  {0}\n".format( ' '.join(["%.10f" % coeff_mean_width_cov_list[i][0][0] for i in xrange(sample['nbins'])]) ) + \
								"covar_bias_width   =  {0}\n".format( ' '.join(["%.10f" % coeff_mean_width_cov_list[i][0][1] for i in xrange(sample['nbins'])]) ) + \
								"mean_bias          =  {0}\n".format( ' '.join(["%.10f" % coeff_mean_width_list[i][1] for i in xrange(sample['nbins'])]) ) + \
								"mean_width         =  {0}\n".format( ' '.join(["%.10f" % coeff_mean_width_list[i][0] for i in xrange(sample['nbins'])]) ) 
	f = open(outdir + '{label}_nz_width_mean_cosmosis_style.txt'.format(label=sample['label']), 'w')
	f.write(cosmosis_style_width_mean)
	f.close()
