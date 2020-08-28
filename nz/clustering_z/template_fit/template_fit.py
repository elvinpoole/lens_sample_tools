"""
A set of tools to fit a template n(z) (e.g. redmagic) to a meausred set of points 
with errorbars (e.g. clustering-z)
"""

import numpy as np
import pylab as plt
import sys
import autograd 
import scipy.interpolate as interp
import scipy.optimize
import scipy.stats
import h5py
import os
import twopoint
import yaml

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

class NZ_theory:
	"""
	class for the template n(z) to be used in the fit
	"""
	def normalize(self, nz_data):
		assert self.nbins == nz_data.nbins
		for ibin in range(self.nbins):
			nz_data_bin = nz_data.nz_dict[ibin]
			norm = (sum( nz_data_bin.nz )*nz_data_bin.dz)/(sum(self.nzs[ibin])*self.dz)
			self.nzs[ibin] = norm*self.nzs[ibin]
		return

class NZ_redmagic_txt(NZ_theory):
	"""
	class for the template n(z) in txt format
	"""
	def __init__(self, filename):
		nz_table = np.loadtxt( filename, unpack=True )
		self.z = nz_table[0]
		self.nzs = nz_table[1:]
		self.dz = self.z[1]-self.z[0]
		self.nbins = len(self.nzs)
		return

class NZ_redmagic_tp(NZ_theory):
	"""
	class for the template n(z) in twopoint format
	"""
	def __init__(self, filename):
		tp = twopoint.TwoPointFile.from_fits(filename, covmat_name=None)
		self.z = tp.get_kernel('nz_lens').z
		self.nzs = tp.get_kernel('nz_lens').nzs
		self.dz = self.z[1]-self.z[0]
		self.nbins = len(self.nzs)
		return

class NZ_data:
	"""
	class for WZ redshift distributions
	loads all redshift bins
	"""
	def __init__(self, xcorr_dir, config):
		xcorr_file_list = np.sort([xcorr_dir+f for f in os.listdir(xcorr_dir) if '.h5' in f])
		nz_dict = {}
		for ibin, xcorr_file in enumerate(xcorr_file_list):
			nz_data = NZ_data_bin(xcorr_file, config.cutzarray_min[ibin], config.cutzarray_max[ibin])
			nz_dict[ibin] = nz_data
		self.nz_dict = nz_dict
		self.nbins = len(xcorr_file_list)
		return

	def apply_gamma(self, gamma_array, gamma_var_array, apply_gamma_error, ndraws=None):
		"""
		apply the bias z-dependence correction
		"""

		for ibin in range(self.nbins):
			self.nz_dict[ibin].apply_gamma(
				gamma_array[ibin], 
				gamma_var_array[ibin], 
				apply_gamma_error, 
				ndraws
				)
		return

	def save_cov(self, filename_template):
		for ibin in range(self.nbins):
			self.nz_dict[ibin].save_cov( filename_template.format(ibin=ibin+1) )
		return

class NZ_data_bin:
	"""
	class for WZ redhsift distributions
	loads a single redshift bin
	"""	
	def __init__(self, filename, cutz_min=None, cutz_max=None):

		xcorr_file = h5py.File(filename)
		z_edges_data = np.transpose(xcorr_file['z_edges/block0_values'].value)[0]
		z_data = np.transpose(xcorr_file['z/block0_values'].value)[0]
		
		if (cutz_min is not None) and (cutz_max is not None):
			select_edges = (z_edges_data >= cutz_min) * (z_edges_data <= cutz_max )
			z_edges_data = z_edges_data[select_edges]
			select = (z_data > cutz_min ) * (z_data < cutz_max )
			z_data = z_data[select]
		else:
			select = np.ones(len(z_data)).astype('bool')

		self.z_edges = z_edges_data
		self.z = z_data
		self.nz = np.transpose(xcorr_file['results/block0_values'].value)[0][select]
		cov = np.transpose(xcorr_file['cov/block0_values'].value)
		self.cov = np.array([line[select] for line in cov[select]])
		self.err = np.sqrt(np.diagonal(cov))
		self.dz = z_edges_data[1]-z_edges_data[0]
		return

	def apply_gamma(self, gamma, gamma_var, apply_gamma_error, ndraws=None):
		"""
		apply the bias z-dependence correction
		"""

		#Define function like this to make propagating errors easier later
		def apply_gamma_func(nz_and_gamma):
			gammacorrection = (1. + self.z)**nz_and_gamma[-1]
			norm_gamma = sum(nz_and_gamma[:-1])/sum(nz_and_gamma[:-1]*gammacorrection)
			return nz_and_gamma[:-1] * gammacorrection * norm_gamma

		#### Apply gamma correction to nz
		nz_and_gamma = np.append( self.nz, gamma )
		self.nz = apply_gamma_func(nz_and_gamma)


		#### Apply gamma correction to covariance
		if apply_gamma_error == 'N':
			print('not applying the gamma error')

		elif apply_gamma_error == 'J':
			print('propagating gamma error in n(z) using the jacobian')
			j = autograd.jacobian( apply_gamma_func )
			covmat = np.zeros((len(self.nz)+1,len(self.nz)+1))
			covmat[:-1,:-1] = self.cov
			covmat[-1,-1] = gamma_var 
			cov_corrected = np.matrix(j(nz_and_gamma))*np.matrix(covmat)*np.matrix(j(nz_and_gamma)).T
			cov_corrected = np.array(cov_corrected)
			self.cov = cov_corrected
			self.err = np.sqrt(np.diagonal(self.cov))

		elif apply_gamma_error == 'R':
			print('propagating gamma error in n(z) using random draws from the data')
			nz_draws = np.random.multivariate_normal(self.nz, self.cov, ndraws)
			gamma_draws = np.random.multivariate_normal([0], [[gamma_var]], ndraws)
			nz_and_gamma_draws = np.hstack((nz_draws, gamma_draws)) 
			nz_corrected_draws =  np.array([apply_gamma_func(nz_and_gamma_draw) for nz_and_gamma_draw in nz_and_gamma_draws])
			cov_corrected = np.cov(nz_corrected_draws, rowvar=False)
			self.cov = cov_corrected
			self.err = np.sqrt(np.diagonal(self.cov))
		else:
			raise IOError("apply_gamma_error should be 'N', 'J' or 'R' ")
		return 

	def save_cov(self, filename):
		np.savetxt(filename, self.cov )
		return


class Config:
	def __init__(self, yaml_file ):

		with open(yaml_file) as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		
		self.outdir = config['outdir']
		self.plotdir = config['plotdir']
		#
		self.label = config['label']
		self.wz_data_dir = config['wz_data_dir']
		self.nz_overlap_file = config['nz_overlap_file']
		self.nz_full_file = config['nz_full_file']
		self.nbins = config['nbins']
		#
		self.apply_gamma_error = config['apply_gamma_error']
		self.ndraws = config['ndraws']
		self.gamma_array = np.array(config['gammaarray'].split()).astype('float') 
		self.gamma_var_array = np.array(config['gammaunc'].split()).astype('float') 
		#
		self.apply_cut_array = config['apply_cut_array']
		self.cutzarray_min = np.array(config['cutzarray_min'].split()).astype('float') 
		self.cutzarray_max = np.array(config['cutzarray_max'].split()).astype('float') 
		#
		self.absolute_sigma = config['absolute_sigma']

		self.label = self.label + '_' + self.apply_gamma_error

		if os.path.exists(self.outdir) == False:
			os.mkdir(self.outdir)
		if os.path.exists(self.plotdir) == False:
			os.mkdir(self.plotdir)


		assert len(self.gamma_array) == self.nbins
		assert len(self.gamma_var_array) == self.nbins
		return

def apply_shift(z_eval,deltaz,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(z + deltaz)
	"""
	return A*interp.interp1d(z_theory+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)

def apply_stretch(z_eval,stretch,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(stretch*(z - zmean) + zmean)
	"""
	zmean = np.average(z_theory,weights=nz_theory)
	return A*interp.interp1d(stretch*(z_theory-zmean)+zmean, nz_theory,fill_value=0.,bounds_error=False)(z_eval)/stretch

def apply_stretch_shift(z_eval,stretch,deltaz,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A n_fid(stretch*(z - zmean) + zmean + deltaz) I think
	"""
	zmean = np.average(z_theory,weights=nz_theory)
	return A*interp.interp1d(stretch*(z_theory-zmean)+zmean+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)/stretch

class TemplateFit:
	"""
	class for running the fits to data
	"""
	def __init__(self, config, nz_theory, nz_data):
		self.config = config
		self.nz_theory = nz_theory
		self.nz_data = nz_data

		assert len(self.nz_theory.nzs) == config.nbins
		assert self.nz_data.nbins == config.nbins
		self.nbins = config.nbins
		return

	def fit_mean_diff(self):
		"""
		Y1 style difference in means (based on Ross, Marco and Chris code )
		"""
		raise RuntimeError
		delta_z = 0
		return delta_z

	def fit_shift(self, save=False):
		coeff_list = []
		cov_list = []
		for ibin in range(self.nbins):
			
			#set the template in the shift function to this redshift bin
			def apply_shift_bin(z_eval, deltaz, A):
				return apply_shift(z_eval,deltaz,A, z_theory=self.nz_theory.z, nz_theory=self.nz_theory.nzs[ibin])

			coeff1, coeff_cov = scipy.optimize.curve_fit(
				apply_shift_bin, 
				self.nz_data.nz_dict[ibin].z, 
				self.nz_data.nz_dict[ibin].nz,
				p0=[0.,1.], 
				sigma=nz_data.nz_dict[ibin].cov,  
				absolute_sigma=self.config.absolute_sigma )
			coeff_list.append(coeff1)
			cov_list.append(coeff_cov)
		self.shift_coeff = np.array(coeff_list)
		self.shift_cov = np.array(cov_list)
		self.shift_err = np.array([np.sqrt(cov.diagonal()) for cov in self.shift_cov ])
		if save==True:
			data = np.hstack( (self.shift_coeff, self.shift_err) )
			header = 'deltaz\tA\tdeltaz_err\tA_err'
			np.savetxt(
				self.config.outdir + '{label}_shift_coeff.txt'.format(label=self.config.label), 
				data, header=header)

			for ibin in range(self.nbins):
				np.savetxt(
					self.config.outdir + '{label}_shift_cov_bin{ibin}.txt'.format(ibin=ibin+1, label=self.config.label), 
					self.shift_cov[ibin] )

		return

	def fit_stretch(self, save=False):
		coeff_list = []
		cov_list = []
		for ibin in range(self.nbins):

			#set the template in the stretch function to this redshift bin
			def apply_stretch_bin(z_eval,stretch,A):
				return apply_stretch(z_eval,stretch,A, z_theory=self.nz_theory.z, nz_theory=self.nz_theory.nzs[ibin])

			coeff1, coeff_cov = scipy.optimize.curve_fit(
				apply_stretch_bin, 
				self.nz_data.nz_dict[ibin].z, 
				self.nz_data.nz_dict[ibin].nz,
				p0=[1.,1.], 
				sigma=nz_data.nz_dict[ibin].cov,  
				absolute_sigma=self.config.absolute_sigma )
			coeff_list.append(coeff1)
			cov_list.append(coeff_cov)
		self.stretch_coeff = np.array(coeff_list)
		self.stretch_cov = np.array(cov_list)
		self.stretch_err = np.array([np.sqrt(cov.diagonal()) for cov in self.stretch_cov ])
		if save==True:
			data = np.hstack( (self.stretch_coeff, self.stretch_err) )
			header = 's\tA\ts_err\tA_err'
			np.savetxt(
				self.config.outdir + '{label}_stretch_coeff.txt'.format(label=self.config.label), 
				data, header=header)

			for ibin in range(self.nbins):
				np.savetxt(
					self.config.outdir + '{label}_stretch_cov_bin{ibin}.txt'.format(ibin=ibin+1, label=self.config.label), 
					self.stretch_cov[ibin] )
		return

	def fit_stretch_shift(self, save=False):
		coeff_list = []
		cov_list = []
		for ibin in range(self.nbins):

			#set the template in the stretch shift function to this redshift bin
			def apply_stretch_shift_bin(z_eval,stretch,deltaz,A):
				return apply_stretch_shift(z_eval,stretch,deltaz,A, z_theory=self.nz_theory.z, nz_theory=self.nz_theory.nzs[ibin])

			coeff1, coeff_cov = scipy.optimize.curve_fit(
				apply_stretch_shift_bin, 
				self.nz_data.nz_dict[ibin].z, 
				self.nz_data.nz_dict[ibin].nz,
				p0=[1.,0.,1.], 
				sigma=nz_data.nz_dict[ibin].cov,  
				absolute_sigma=self.config.absolute_sigma )
			coeff_list.append(coeff1)
			cov_list.append(coeff_cov)
		self.stretch_shift_coeff = np.array(coeff_list)
		self.stretch_shift_cov = np.array(cov_list)
		self.stretch_shift_err = np.array([np.sqrt(cov.diagonal()) for cov in self.stretch_shift_cov ])
		if save==True:
			data = np.hstack( (self.stretch_shift_coeff, self.stretch_shift_err) )
			header = 's\tdeltaz\tA\ts_err\tdeltaz_err\tA_err'
			np.savetxt(
				self.config.outdir + '{label}_stretch_shift_coeff.txt'.format(label=self.config.label), 
				data, header=header)

			for ibin in range(self.nbins):
				np.savetxt(
					self.config.outdir + '{label}_stretch_shift_cov_bin{ibin}.txt'.format(ibin=ibin+1, label=self.config.label), 
					self.stretch_shift_cov[ibin] )
		return

	def plot_2d(self, ngrid=100):
		from scipy.stats import multivariate_normal

		nx = int(np.floor(np.sqrt(self.nbins)))
		ny = int(np.ceil(np.sqrt(self.nbins)))
		fig1, axs1 = plt.subplots(nx, ny, figsize=(3*ny,3*nx))

		for ibin in range(self.nbins):
			ix =  np.repeat(np.arange(ny), nx)[ibin]
			iy =  np.tile(np.arange(nx), ny)[ibin]

			coeff_2d = self.stretch_shift_coeff[ibin][:-1]
			coeff_cov_2d = self.stretch_shift_cov[ibin][:-1,:-1]

			delta_array = np.linspace(coeff_2d[1]-3.*np.sqrt(coeff_cov_2d[1,1]),coeff_2d[1]+3.*np.sqrt(coeff_cov_2d[1,1]), ngrid)
			stretch_array = np.linspace(coeff_2d[0]-3.*np.sqrt(coeff_cov_2d[0,0]),coeff_2d[0]+3.*np.sqrt(coeff_cov_2d[0,0]), ngrid)
			X, Y = np.meshgrid(stretch_array, delta_array)

			gauss = multivariate_normal(coeff_2d, coeff_cov_2d)
			Z = gauss.pdf(np.transpose([X.flatten(),Y.flatten()]))
			Z = Z.reshape(100,100)

			frac_levels = [0.9545,0.6827]
			Zsort = np.sort(Z.flatten())
			levels = [Zsort[np.abs(np.cumsum(Zsort)-(1-frac_level)*np.sum(Z)) == np.abs(np.cumsum(Zsort)-(1-frac_level)*np.sum(Z)).min()][0] for frac_level in frac_levels]

			axs1[iy,ix].axhline(0.0, color='k',ls='--')
			axs1[iy,ix].axvline(1.0, color='k',ls='--')
			axs1[iy,ix].contour(X,Y,Z,levels=levels,colors='r',label='bin {0}'.format(ibin+1))
			axs1[iy,ix].set_title('bin {0}'.format(ibin+1))
			axs1[iy,ix].set_xlabel('stretch')
			axs1[iy,ix].set_ylabel(r'$\Delta z$')
			axs1[iy,ix].legend()	
		fig1.tight_layout()
		fig1.savefig(self.config.plotdir + '{label}_2d_contours_stretch_shift_allbins.png'.format(label=self.config.label ))	 
		fig1.clear()
		return 

	def plot_fitted_nz(self,func=None, coeff=None, extra_label=None):
		nx = int(np.floor(np.sqrt(self.nbins)))
		ny = int(np.ceil(np.sqrt(self.nbins)))
		fig2, axs2 = plt.subplots(nx, ny, figsize=(3*ny,3*nx))

		for ibin in range(self.nbins):
			ix =  np.repeat(np.arange(ny), nx)[ibin]
			iy =  np.tile(np.arange(nx), ny)[ibin]

			z_data  = self.nz_data.nz_dict[ibin].z
			nz_data = self.nz_data.nz_dict[ibin].nz
			cov     = self.nz_data.nz_dict[ibin].cov
			err     = self.nz_data.nz_dict[ibin].err
			z_theory = self.nz_theory.z
			nz_theory = self.nz_theory.nzs[ibin]

			nz_theory_raw = interp.interp1d(z_theory, nz_theory )(z_data)
			fitted_nz = func(z_theory, *coeff[ibin], z_theory=z_theory, nz_theory=nz_theory )
			nz_theory_fit = interp.interp1d(z_theory, fitted_nz )(z_data)

			photoz_chi2   = calc_chi2(nz_data, cov, nz_theory_raw )
			fitted_chi2   = calc_chi2(nz_data, cov, nz_theory_fit )

			select_range = (z_theory > self.nz_data.nz_dict[ibin].z_edges.min())*(z_theory < self.nz_data.nz_dict[ibin].z_edges.max())

			axs2[iy,ix].plot(z_theory[select_range], nz_theory[select_range], 'b-', label='photoz, chi2={0}/{1}'.format(np.round(photoz_chi2,decimals=1), len(nz_data)))
			axs2[iy,ix].plot(z_theory[select_range], func(z_theory[select_range], *coeff[ibin], z_theory=z_theory, nz_theory=nz_theory ), 'r-', label='fit to xcorr chi2={0}/{1}'.format(np.round(fitted_chi2,decimals=1),len(nz_data)))
			axs2[iy,ix].errorbar(z_data, nz_data, err, fmt='.', label='xcorr')
			axs2[iy,ix].legend(loc='lower right')

		fig2.tight_layout()
		fig2.savefig(self.config.plotdir + '{label}_nz_{extra_label}_allbins.png'.format( label=self.config.label, extra_label=extra_label) )
		fig2.clear()

		return


	def save_cosmosis_style(self,):

		cosmosis_style_mean = '[lens_photoz_errors]\n'+'\n'.join( ['bias_{0} = gaussian {1} {2}'.format( i+1, self.shift_coeff[i][0], self.shift_err[i][0] ) for i in range( self.nbins )] )
		f = open(self.config.outdir + '{label}_nz_mean_cosmosis_style.txt'.format(label= self.config.label ), 'w')
		f.write(cosmosis_style_mean)
		f.close()

		cosmosis_style_width_mean = "var_bias           =  {0}\n".format( ' '.join(["%.10f" % self.stretch_shift_cov[i][1][1] for i in range(self.nbins)]) ) + \
									"var_width          =  {0}\n".format( ' '.join(["%.10f" % self.stretch_shift_cov[i][0][0] for i in range(self.nbins)]) ) + \
									"covar_bias_width   =  {0}\n".format( ' '.join(["%.10f" % self.stretch_shift_cov[i][0][1] for i in range(self.nbins)]) ) + \
									"mean_bias          =  {0}\n".format( ' '.join(["%.10f" % self.stretch_shift_coeff[i][1] for i in range(self.nbins)]) ) + \
									"mean_width         =  {0}\n".format( ' '.join(["%.10f" % self.stretch_shift_coeff[i][0] for i in range(self.nbins)]) ) 
		f = open(self.config.outdir + '{label}_nz_width_mean_cosmosis_style.txt'.format(label=self.config.label), 'w')
		f.write(cosmosis_style_width_mean)
		f.close()
		return


#####################################
#####################################
#####################################

#load things
config = Config(sys.argv[1])
nz_overlap = NZ_redmagic_txt( config.nz_overlap_file )
nz_full = NZ_redmagic_tp( config.nz_full_file )
nz_data = NZ_data( config.wz_data_dir, config )
nz_data.apply_gamma(config.gamma_array, config.gamma_var_array, config.apply_gamma_error, ndraws=config.ndraws)
nz_data.save_cov( config.outdir + '{label}_cov_corrected'.format(label=config.label ) + '_bin{ibin}.txt' )

nz_overlap.normalize(nz_data)
nz_full.normalize(nz_data)

#run fits
template_fit = TemplateFit(config, nz_overlap, nz_data)
template_fit.fit_shift(save=True)
template_fit.fit_stretch(save=True)
template_fit.fit_stretch_shift(save=True)
template_fit.save_cosmosis_style()

#make plots
template_fit.plot_2d()
template_fit.plot_fitted_nz(
	func=apply_shift, 
	coeff=template_fit.shift_coeff,
	extra_label='shift')
template_fit.plot_fitted_nz(
	func=apply_stretch_shift, 
	coeff=template_fit.stretch_shift_coeff,
	extra_label='stretch_shift')


