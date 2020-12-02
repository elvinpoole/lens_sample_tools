import numpy as np 
import twopoint 

filename_list = [
	'wz_sim_sample2_specz_hist.txt',
	'wz_sim_sample4_specz_hist.txt',
	'wz_sim_sample6_specz_hist.txt',
	]

outfile = 'wz_sim_sample246_maglim_specz_hist.fits'

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


