import numpy as np 

import sys
import matplotlib
matplotlib.use('Agg')
import lsssys
import twopoint

randoms_file = '/fs/project/PCON0008/des_y3/y3_redmagic/y3_redmagic_combined_sample_fid_x40_1_randoms.fits.gz'
rand = lsssys.Randoms(randoms_file, load_weights=False,)

cat_dir = '/users/PCON0008/osu10688/DES/cats/y3/redmagic/combined_sample_0.5.1_wide_0.9binning_zmax0.95/weighted_enet/'
cat_file_temp = cat_dir+'/y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_combined_hd3_hl2_sample_weighted{0}.fits.gz'
labels = ['enet','enet2','enet_kptpl']

num_threads=40
bin_slop = 0.01

if len(sys.argv) > 1:
	if sys.argv[1] == 'test':
		num_threads = 1
		bin_slop = 1.0
	if sys.argv[1] == 'quick':
		num_threads = 1
		bin_slop = 0.05
		rand.thin(0.3)

binedges = [ 0.15, 0.35, 0.50, 0.65, 0.80, 0.90 ]
nthetabins = 20
thetamax = 250./60.
thetamin = 2.5/60.


for imethod, label in enumerate(labels):
	cat_file = cat_file_temp.format(label)

	cat = lsssys.Redmagic(cat_file)

	w_dict, corr_dict = lsssys.random2ptcalc(cat,rand,binedges,nthetabins,thetamax,thetamin, 
	                    autoonly=True, bin_slop = bin_slop, bin_type='Log', num_threads=num_threads, 
	                    use_weights=True, comm=None, saveinfo=True, returncorr=True, split_random_z=True)

	np.save(w_dict,'w_dict_bs{bin_slop}_{label}.npy'.format(bin_slop = bin_slop, label=label ))
	np.save(corr_dict,'corr_dictbs{bin_slop}_{label}.npy'.format(bin_slop = bin_slop, label=label ))

	angle_edges = np.logspace(np.log10(thetamin*60.),np.log10(thetamin*60.), 21)
	w_dict['angle_min'] = angle_edges[:-1]
	w_dict['angle_max'] = angle_edges[1:]

	spectrum = lsssys.corrdict_2_spectrumtype(wdict, autoonly=False, name='wtheta', 
	                        kernel1='nz_lens', kernel2='nz_lens',)

	tp = twopoint.TwoPointFile([spectrum], kernels=None, windows={}, covmat_info=None)
	tp.to_fits('wtheta_redmagic_y3_data_bs{bin_slop}_{label}_UNBLIND.fits'.format(
	    bin_slop = bin_slop,
	    label=label ))


	    
