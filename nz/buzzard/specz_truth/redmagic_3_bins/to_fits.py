import numpy as np 
import twopoint 

filename_list = [
	'wz_sim_sample1_specz_hist.txt',
	'wz_sim_sample3_specz_hist.txt',
	'wz_sim_sample5_specz_hist.txt',
	]

outfile = 'wz_sim_sample135_redmagic_specz_hist.fits'

match_wz_binning = True
wz_binning = np.arange(0,2.0+0.0001, 0.02)

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


