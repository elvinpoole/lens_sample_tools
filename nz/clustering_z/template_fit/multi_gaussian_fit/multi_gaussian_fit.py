"""
my attempts to get a smoothed or splined WZ 
"""

import numpy as np
import pylab as plt 

import scipy.optimize
import sys
sys.path.append('../')
#import template_fit
from template_fit import *
import os
import misc
import scipy.stats

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def n_gaussians(x, n, *params):
	#assert len(params) == n*3+1, (len(params), n*3+1)
	assert len(params) == n*3, (len(params), n*3)
	output = np.zeros(len(x))
	for i in range(n):
		output += gaussian(x, params[3*i], params[3*i+1], params[3*i+2], offset=0)
	#return output + params[-1]
	return output

Ngauss_list = [1,2,3,4,5]
chi2red_threshold = 1.85
#chi2red_threshold = 1.3

if __name__ == "__main__":

	#os.chdir('../')
	#inifile = 'maglim_y3_dnf_overlap_PDF_training-def-p-v_octWZ_narrower5range.yaml'
	config = Config( sys.argv[1] )
	outdir = config.outdir
	plotdir = config.plotdir
	label =  config.label + '_multi_gaussian_fit_chi2red{0}'.format(chi2red_threshold)

	nz_data_fid = NZ_data( config.wz_data_dir, config )
	nz_data_fid.apply_gamma(config.gamma_array, config.gamma_var_array, config.apply_gamma_error, ndraws=config.ndraws)

	nz_data_full_range = NZ_data( config.wz_data_dir, None )
	nz_data_full_range.apply_gamma(config.gamma_array, config.gamma_var_array, config.apply_gamma_error, ndraws=config.ndraws)

	nbins = config.nbins

	zedges = np.arange(0,2.0+0.00001,0.02)
	zhigh = zedges[1:]
	zlow = zedges[:-1]
	zmid = (zlow+zhigh)/2.
	nzs_2_save = []


	fig, axs = plt.subplots(3,2, figsize=(8,10) )
	axs = axs.flatten()
	for ibin in range(nbins):
		print('bin', ibin+1)
		nz = nz_data_fid.nz_dict[ibin]
		nz_fr = nz_data_full_range.nz_dict[ibin]

		plt.figure('single')
		plt.errorbar(
			nz_fr.z, 
			nz_fr.nz, 
			nz_fr.err,
			fmt='b.')
		plt.errorbar(nz.z, nz.nz, nz.err,fmt='bo')

		axs[ibin].errorbar(
			nz_fr.z, 
			nz_fr.nz, 
			nz_fr.err,
			fmt='b.')
		axs[ibin].errorbar(nz.z, nz.nz, nz.err,fmt='bo')

		fid_height = nz.nz.max()
		fid_mean = np.average(nz.z, weights=nz.nz)
		fid_std  = np.sqrt( np.average(nz.z**2., weights=nz.nz)-np.average(nz.z, weights=nz.nz)**2. )

		for Ngauss in Ngauss_list:
			print('Ngauss {0}'.format(Ngauss))
			p0 = []
			for i in range(Ngauss):
				p0 += [fid_height/Ngauss, np.random.normal(fid_mean,0.1*fid_std), fid_std]
			#p0 += [0.]
			p0 = np.array(p0)

			#bounds = ([0.05*fid_height, nz.z.min(), nz.dz]*Ngauss+[-inf], [2.*fid_height, nz.z.max(), 5*fid_std]*Ngauss+[inf])
			#bounds = ([0.05*fid_height, nz.z.min(), nz.dz]*Ngauss, [2.*fid_height, nz.z.max(), 5*fid_std]*Ngauss)
			bounds = ([0., nz.z.min(), 0.]*Ngauss, [np.inf, nz.z.max(), np.inf]*Ngauss)

			def n_gaussians_4_optimize(x, *params):
				return n_gaussians(x, Ngauss, *params)
			coeff, cov = scipy.optimize.curve_fit(n_gaussians_4_optimize, nz.z, nz.nz, p0=p0, sigma=nz.cov, maxfev=100000 ) 
			chi2 = misc.calc_chi2(nz.nz, nz.cov, n_gaussians_4_optimize(nz.z, *coeff), v=False )
			ndof = len(nz.nz)-(3.*Ngauss+1)
			chi2red = chi2/ndof
			pvalue = 1.-scipy.stats.chi2.cdf(chi2,ndof)
			print( 'chi2_red = {0}/{1} = {2}'.format(np.round(chi2,1), ndof, np.round(chi2red, 2)) )
			print( 'p = {0}'.format(pvalue) )

			#plt.plot(nz.z, n_gaussians_4_optimize(nz.z, *coeff), label='Ngauss={0}'.format(Ngauss))
			plt.figure('single')
			plt.plot(nz_fr.z, n_gaussians_4_optimize(nz_fr.z, *coeff), label='Ngauss={0}'.format(Ngauss))

			axs[ibin].plot(nz_fr.z, n_gaussians_4_optimize(nz_fr.z, *coeff), label='Ngauss={0}'.format(Ngauss))

			if chi2red < chi2red_threshold:
				break
			#if pvalue > 0.05:
			#	break
		nz_2_save = n_gaussians_4_optimize(zmid, *coeff)
		nzs_2_save.append(nz_2_save)

		plt.figure('single')
		plt.legend()
		plt.savefig( plotdir+'/nz_{0}_bin{1}.png'.format(label, ibin+1 ))
		plt.close()

		axs[ibin].legend()
	fig.tight_layout()
	fig.savefig( plotdir+'/nz_{0}_allbins.png'.format(label) )
	fig.clear()

	import twopoint 
	k = twopoint.NumberDensity('nz_lens', zlow, zmid, zhigh, nzs_2_save )
	tp = twopoint.TwoPointFile([],[k],[],None)
	tp.to_fits(outdir+'/wz_{label}_nz_best_fit_template.fits'.format(label=label),
		overwrite=True)


