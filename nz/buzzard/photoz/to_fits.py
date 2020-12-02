import numpy as np 
import twopoint 

filename_list = [
	'maglim_3_bins/wz_sim_sample2_zmc_hist.txt',
	'maglim_3_bins/wz_sim_sample4_zmc_hist.txt',
	'maglim_3_bins/wz_sim_sample6_zmc_hist.txt',
	]

outfile = 'maglim_3_bins/wz_sim_sample246_zmc_hist.fits'

nzs = []
for ifile, filename in enumerate(filename_list):
	zmid1, nz = np.loadtxt(filename)
	if ifile == 0:
		zmid = zmid1
	assert (zmid == zmid1).all()

	nzs.append(nz)

dz = zmid[1]-zmid[0]
zlow = zmid-dz/2.
zhigh = zmid+dz/2.

k = twopoint.NumberDensity('nz_lens', zlow, zmid, zhigh, nzs )
tp = twopoint.TwoPointFile( [], [k], [], None)
tp.to_fits(outfile)

###################################


filename_list = [
	'redmagic_3_bins/wz_sim_sample1_zmc_hist.txt',
	'redmagic_3_bins/wz_sim_sample3_zmc_hist.txt',
	'redmagic_3_bins/wz_sim_sample5_zmc_hist.txt',
	]

outfile = 'redmagic_3_bins/wz_sim_sample135_zmc_hist.fits'

nzs = []
for ifile, filename in enumerate(filename_list):
	zmid1, nz = np.loadtxt(filename)
	if ifile == 0:
		zmid = zmid1
	assert (zmid == zmid1).all()

	nzs.append(nz)

dz = zmid[1]-zmid[0]
zlow = zmid-dz/2.
zhigh = zmid+dz/2.

k = twopoint.NumberDensity('nz_lens', zlow, zmid, zhigh, nzs )
tp = twopoint.TwoPointFile( [], [k], [], None)
tp.to_fits(outfile)