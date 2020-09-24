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

def nz_mean(z,nz):
	#return np.trapz( z*nz, x=z)/np.trapz(nz,x=z)
	return np.sum(z*nz)/np.sum(nz)

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
		self.zlow = nz_table[0]
		self.nzs = nz_table[1:]
		self.dz = self.zlow[1]-self.zlow[0]
		self.zhigh = self.zlow + self.dz
		self.nbins = len(self.nzs)

		self.z = self.zlow + self.dz/2.
		return

class NZ_redmagic_tp(NZ_theory):
	"""
	class for the template n(z) in twopoint format
	"""
	def __init__(self, filename):
		tp = twopoint.TwoPointFile.from_fits(filename, covmat_name=None)
		self.z = tp.get_kernel('nz_lens').z
		self.zlow = tp.get_kernel('nz_lens').zlow
		self.zhigh = tp.get_kernel('nz_lens').zhigh
		self.nzs = tp.get_kernel('nz_lens').nzs
		self.dz = self.z[1]-self.z[0]
		self.nbins = len(self.nzs)
		return

class NZ_data:
	"""
	class for WZ redshift distributions
	loads all redshift bins
	"""
	def __init__(self, xcorr_dir, config=None):
		xcorr_file_list = np.sort([xcorr_dir+f for f in os.listdir(xcorr_dir) if '.h5' in f])
		nz_dict = {}
		for ibin, xcorr_file in enumerate(xcorr_file_list):
			if config is not None:
				if config.apply_amp_err == True:
					nz_data = NZ_data_bin(xcorr_file, config.cutzarray_min[ibin], config.cutzarray_max[ibin], config.amp_err[ibin])
				else:
					nz_data = NZ_data_bin(xcorr_file, config.cutzarray_min[ibin], config.cutzarray_max[ibin], None )
			else:
				nz_data = NZ_data_bin(xcorr_file, None, None, None)
			nz_dict[ibin] = nz_data
		self.nz_dict = nz_dict
		self.nbins = len(xcorr_file_list)
		return

	def apply_gamma(self, gamma_array, gamma_var_array, apply_gamma_error, ndraws=None, v=True):
		"""
		apply the bias z-dependence correction
		"""

		for ibin in range(self.nbins):
			self.nz_dict[ibin].apply_gamma(
				gamma_array[ibin], 
				gamma_var_array[ibin], 
				apply_gamma_error, 
				ndraws,
				v=v
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
	def __init__(self, filename, cutz_min=None, cutz_max=None, amp_err=None):

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
		if amp_err is not None:
			self.apply_amp_err(amp_err)
		return

	def apply_gamma(self, gamma, gamma_var, apply_gamma_error, ndraws=None, v=True):
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
			if v == True:
				print('not applying the gamma error')

		elif apply_gamma_error == 'J':
			if v == True:
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
			if v == True:
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

	def apply_amp_err(self, amp_err):
		"""
		apply an additional variance to the amplitude
		assume it is uncorrelated with n(z) errors (might be wrong....?)
		"""

		def apply_amp_to_joint_array(nz_and_a):
			return nz_and_a[:-1] * nz_and_a[-1]

		nz_and_a = np.append( self.nz, 1.0 )

		j = autograd.jacobian( apply_amp_to_joint_array )
		covmat = np.zeros((len(self.nz)+1,len(self.nz)+1))
		covmat[:-1,:-1] = self.cov
		covmat[-1,-1] = amp_err**2.
		cov_corrected = np.matrix(j(nz_and_a))*np.matrix(covmat)*np.matrix(j(nz_and_a)).T
		cov_corrected = np.array(cov_corrected)
		self.cov = cov_corrected
		self.err = np.sqrt(np.diagonal(self.cov)) 

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
		self.gamma_unc_array = np.array(config['gammaunc'].split()).astype('float') 
		self.gamma_var_array = self.gamma_unc_array**2.
		#
		self.apply_cut_array = config['apply_cut_array']
		self.cutzarray_min = np.array(config['cutzarray_min'].split()).astype('float') 
		self.cutzarray_max = np.array(config['cutzarray_max'].split()).astype('float') 
		#
		self.absolute_sigma = config['absolute_sigma']
		#
		try:
			self.apply_amp_err = config['apply_amp_err']
			if self.apply_amp_err == True:
				self.amp_err = np.array(config['amp_err'].split()).astype('float') 
			else:
				self.amp_err = None
		except KeyError:
			self.apply_amp_err = False
			self.amp_err = None

		self.label = self.label + '_' + self.apply_gamma_error

		if os.path.exists(self.outdir) == False:
			os.mkdir(self.outdir)
		if os.path.exists(self.plotdir) == False:
			os.mkdir(self.plotdir)


		assert len(self.gamma_array) == self.nbins
		assert len(self.gamma_var_array) == self.nbins
		return

def apply_amp(z_eval,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(z)
	"""
	return A*interp.interp1d(z_theory, nz_theory, fill_value=0.,bounds_error=False)(z_eval)

def apply_power_law(z_eval,A,M,C,alpha,z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(z)
	"""
	return A*interp.interp1d(M*(z_theory**alpha) + C, nz_theory, fill_value=0.,bounds_error=False)(z_eval)

def apply_shift(z_eval,deltaz, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(z + deltaz)
	"""
	return interp.interp1d(z_theory+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)
def apply_shiftA(z_eval,deltaz,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(z + deltaz)
	"""
	return A*interp.interp1d(z_theory+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)

def apply_stretch(z_eval,stretch,z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(stretch*(z - zmean) + zmean)
	"""
	zmean = np.average(z_theory,weights=nz_theory)
	return interp.interp1d(stretch*(z_theory-zmean)+zmean, nz_theory, fill_value=0.,bounds_error=False)(z_eval)/stretch

def apply_stretchA(z_eval,stretch,A, z_theory=None, nz_theory=None):
	"""
	n(z) = A * n_fid(stretch*(z - zmean) + zmean)
	"""
	zmean = np.average(z_theory,weights=nz_theory)
	return A*interp.interp1d(stretch*(z_theory-zmean)+zmean, nz_theory, fill_value=0.,bounds_error=False)(z_eval)/stretch

def apply_stretch_shift(z_eval,stretch,deltaz, z_theory=None, nz_theory=None):
	"""
	n(z) = A n_fid(stretch*(z - zmean) + zmean + deltaz) I think
	"""
	zmean = np.average(z_theory,weights=nz_theory)
	return interp.interp1d(stretch*(z_theory-zmean)+zmean+deltaz, nz_theory, fill_value=0.,bounds_error=False)(z_eval)/stretch

def apply_stretch_shiftA(z_eval,stretch,deltaz,A, z_theory=None, nz_theory=None):
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

		if config is not None:
			assert len(self.nz_theory.nzs) == config.nbins
			assert self.nz_data.nbins == config.nbins
			self.nbins = config.nbins

		self.coeff     = {}
		self.coeff_cov = {}
		self.coeff_err = {}
		return

	def fit_mean_diff(self, save=False):
		"""
		Y1 style difference in means (based on Ross, Marco and Chris code )
		"""
		#delta_z_array = np.linspace(-0.05, 0.05, 201)
		delta_z_array = np.linspace(-0.051,0.051,103).round(15)

		coeff_list = []
		for ibin in range(self.nbins):
			select_data_zrange =  (self.nz_theory.z > self.nz_data.nz_dict[ibin].z_edges.min())
			select_data_zrange *= (self.nz_theory.z < self.nz_data.nz_dict[ibin].z_edges.max())
			z_theory = self.nz_theory.z[select_data_zrange]

			shifted_nz_array = [ apply_shiftA(z_theory, dz,1.0, z_theory=self.nz_theory.z, nz_theory=self.nz_theory.nzs[ibin]) for dz in delta_z_array ]
			shifted_means = np.array([nz_mean(z_theory, nz) for nz in shifted_nz_array])

			data_mean = nz_mean(self.nz_data.nz_dict[ibin].z, self.nz_data.nz_dict[ibin].nz)

			#best_fit_deltaz = interp.interp1d(shifted_means, delta_z_array)([data_mean])[0]
			abs_diff = np.abs(shifted_means-data_mean)
			best_fit_deltaz = delta_z_array[abs_diff == abs_diff.min()][0]

			coeff_list.append(best_fit_deltaz)
		self.meandiff_coeff = np.array(coeff_list)
		if save==True:
			np.savetxt(
				self.config.outdir + '{label}_mean_diff_y1style_coeff.txt'.format(label=self.config.label), 
				self.meandiff_coeff, header='deltaz')

		return

	def fit_func(self, func, label, p0, save=False):
		"""
		fit a function of the theory n(z) to the data n(z)
		
		func: 	Function in the form
				func(z_eval,param1,param2,...,z_theory=None, nz_theory=None)
		label: 	Label for the function (for save files and name in coeff dict)
		p0: 	Yhe start point of the fit, len() = nparams
		"""
		coeff_list = []
		cov_list = []
		nzs_bf = []
		for ibin in range(self.nbins):
			
			#set the template in the amp function to this redshift bin
			def func_bin(z_eval, *coeff):
				return func(z_eval,*coeff,z_theory=self.nz_theory.z, nz_theory=self.nz_theory.nzs[ibin])

			coeff1, coeff_cov = scipy.optimize.curve_fit(
				func_bin, 
				self.nz_data.nz_dict[ibin].z, 
				self.nz_data.nz_dict[ibin].nz,
				p0=p0, 
				sigma=nz_data.nz_dict[ibin].cov,  
				absolute_sigma=self.config.absolute_sigma )
			coeff_list.append(coeff1)
			cov_list.append(coeff_cov)
			nz_bf = func_bin(self.nz_theory.z, *coeff1)
			nzs_bf.append(nz_bf)
		self.coeff[label] = np.array(coeff_list)
		self.coeff_cov[label] = np.array(cov_list)
		self.coeff_err[label] = np.array([np.sqrt(cov.diagonal()) for cov in self.coeff_cov[label] ])
		if save==True:
			#save coeff
			data = np.hstack( (self.coeff[label], self.coeff_err[label]) )
			header = '\t'.join(['coeff{0}'.format(i+1) for i in range(len(self.coeff[label][0])) ])
			header += '\t'
			header += '\t'.join(['coeff{0}_err'.format(i+1) for i in range(len(self.coeff[label][0])) ])
			np.savetxt(
				self.config.outdir + '{label1}_{label2}_coeff.txt'.format(label1=self.config.label,label2=label), 
				data, header=header)

			#save coeff covariance
			for ibin in range(self.nbins):
				np.savetxt(
					self.config.outdir + '{label1}_{label2}_cov_bin{ibin}.txt'.format(ibin=ibin+1, label1=self.config.label, label2=label), 
					self.coeff_cov[label][ibin] )

			#save best fit n(z)
			k_bf = twopoint.NumberDensity('nz_lens', self.nz_theory.zlow, self.nz_theory.z, self.nz_theory.zhigh, nzs_bf )
			tp = twopoint.TwoPointFile([],[k_bf],[],None)

			tp.to_fits(self.config.outdir + '{label1}_{label2}_nz_best_fit_template.fits'.format(label1=self.config.label, label2=label),
				overwrite=True)
		return


	def plot_2d(self, label, ngrid=100):
		assert label == 'stretch_shift' or label == 'stretch_shiftA'
		from scipy.stats import multivariate_normal

		nx = int(np.floor(np.sqrt(self.nbins)))
		ny = int(np.ceil(np.sqrt(self.nbins)))
		fig1, axs1 = plt.subplots(nx, ny, figsize=(3*ny,3*nx))

		for ibin in range(self.nbins):
			ix =  np.repeat(np.arange(ny), nx)[ibin]
			iy =  np.tile(np.arange(nx), ny)[ibin]

			coeff_2d = self.coeff[label][ibin][:2]
			coeff_cov_2d = self.coeff_cov[label][ibin][:2,:2]

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
		fig1.savefig(self.config.plotdir + '{label1}_2d_contours_{label2}_allbins.png'.format(label1=self.config.label, label2=label ))	 
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

			axs2[iy,ix].plot(z_theory[select_range], nz_theory[select_range], 'b-', label='fid, chi2={0}/{1}'.format(np.round(photoz_chi2,decimals=1), len(nz_data)))
			axs2[iy,ix].plot(z_theory[select_range], func(z_theory[select_range], *coeff[ibin], z_theory=z_theory, nz_theory=nz_theory ), 'r-', label='fit chi2={0}/{1}'.format(np.round(fitted_chi2,decimals=1),len(nz_data)))
			#axs2[iy,ix].errorbar(z_data, nz_data, err, fmt='.', label='xcorr')
			axs2[iy,ix].errorbar(z_data, nz_data, err, fmt='.' )
			axs2[iy,ix].legend(loc='lower left')

		axs2[0,1].set_title(extra_label)
		fig2.tight_layout()
		fig2.savefig(self.config.plotdir + '{label}_nz_{extra_label}_allbins.png'.format( label=self.config.label, extra_label=extra_label) )
		fig2.clear()

		return


	def save_cosmosis_style(self,):

		cosmosis_style_mean = '[lens_photoz_errors]\n'+'\n'.join( ['bias_{0} = gaussian {1} {2}'.format( i+1, self.coeff['shift'][i][0], self.coeff_err['shift'][i][0] ) for i in range( self.nbins )] )
		f = open(self.config.outdir + '{label}_nz_mean_cosmosis_style.txt'.format(label= self.config.label ), 'w')
		f.write(cosmosis_style_mean)
		f.close()

		cosmosis_style_width_mean = "var_bias           =  {0}\n".format( ' '.join(["%.10f" % self.coeff_cov['stretch_shift'][i][1][1] for i in range(self.nbins)]) ) + \
									"var_width          =  {0}\n".format( ' '.join(["%.10f" % self.coeff_cov['stretch_shift'][i][0][0] for i in range(self.nbins)]) ) + \
									"covar_bias_width   =  {0}\n".format( ' '.join(["%.10f" % self.coeff_cov['stretch_shift'][i][0][1] for i in range(self.nbins)]) ) + \
									"mean_bias          =  {0}\n".format( ' '.join(["%.10f" % self.coeff['stretch_shift'][i][1] for i in range(self.nbins)]) ) + \
									"mean_width         =  {0}\n".format( ' '.join(["%.10f" % self.coeff['stretch_shift'][i][0] for i in range(self.nbins)]) ) 
		f = open(self.config.outdir + '{label}_nz_width_mean_cosmosis_style.txt'.format(label=self.config.label), 'w')
		f.write(cosmosis_style_width_mean)
		f.close()
		return


#####################################
#####################################
#####################################

if __name__ == "__main__":
	config = Config(sys.argv[1])
	nz_overlap = NZ_redmagic_txt( config.nz_overlap_file ) #redmagic n(z) in overlap region
	nz_full = NZ_redmagic_tp( config.nz_full_file ) #redmagic n(z) in full footpritn (for testing)
	nz_data = NZ_data( config.wz_data_dir, config ) #the clustering-z n(z)
	nz_data.apply_gamma(config.gamma_array, config.gamma_var_array, config.apply_gamma_error, ndraws=config.ndraws)
	nz_data.save_cov( config.outdir + '{label}_cov_corrected'.format(label=config.label ) + '_bin{ibin}.txt' )

	nz_overlap.normalize(nz_data)
	nz_full.normalize(nz_data)

	#run fits
	template_fit = TemplateFit(config, nz_overlap, nz_data)
	#config.label = config.label+'_total'
	#template_fit = TemplateFit(config, nz_full, nz_data)
	
	template_fit.fit_mean_diff(save=True) #Ross method
	#					  Function to be fit 	Label				Starting values for input params
	template_fit.fit_func(apply_amp, 			'amp', 				[1.], 			save=True)
	template_fit.fit_func(apply_shift, 			'shift', 			[0.], 			save=True)
	template_fit.fit_func(apply_shiftA, 		'shiftA', 			[0.,1.], 		save=True)
	template_fit.fit_func(apply_stretch, 		'stretch', 			[1.], 			save=True)
	template_fit.fit_func(apply_stretchA, 		'stretchA', 		[1.,1.0], 		save=True)
	template_fit.fit_func(apply_stretch_shift, 	'stretch_shift', 	[1.,0.], 		save=True)
	template_fit.fit_func(apply_stretch_shiftA, 'stretch_shiftA', 	[1.,0.,1.0], 	save=True)
	template_fit.fit_func(apply_power_law, 		'power_law', 		[1.,1.,0.,1.], 	save=True )

	template_fit.save_cosmosis_style()


	#make plots
	template_fit.plot_2d('stretch_shift')
	template_fit.plot_2d('stretch_shiftA')

	template_fit.plot_fitted_nz(
		func=apply_amp, 
		coeff=template_fit.coeff['amp'],
		extra_label='amp')
	template_fit.plot_fitted_nz(
		func=apply_shift, 
		coeff=template_fit.coeff['shift'],
		extra_label='shift')
	template_fit.plot_fitted_nz(
		func=apply_shiftA, 
		coeff=template_fit.coeff['shiftA'],
		extra_label='shiftA')
	template_fit.plot_fitted_nz(
		func=apply_stretch, 
		coeff=template_fit.coeff['stretch'],
		extra_label='stretch')
	template_fit.plot_fitted_nz(
		func=apply_stretchA, 
		coeff=template_fit.coeff['stretchA'],
		extra_label='stretchA')
	template_fit.plot_fitted_nz(
		func=apply_stretch_shift, 
		coeff=template_fit.coeff['stretch_shift'],
		extra_label='stretch_shift')
	template_fit.plot_fitted_nz(
		func=apply_stretch_shiftA, 
		coeff=template_fit.coeff['stretch_shiftA'],
		extra_label='stretch_shiftA')
	template_fit.plot_fitted_nz(
		func=apply_power_law, 
		coeff=template_fit.coeff['power_law'],
		extra_label='power_law')

