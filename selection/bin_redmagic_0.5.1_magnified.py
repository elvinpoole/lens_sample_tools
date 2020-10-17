import numpy as np
import pylab as plt 
import lsssys
import twopoint

########### config ###########

redmagic_dir = 	'/Users/jackelvinpoole/DES/cats/y3/redmagic/redmagic_0.5.1_wide_magnified/'
highdens_file   = redmagic_dir + 'y3_gold_2.2.1_wide-kappa01_sofcol_run_redmagic_highdens.fit'
highlum_file    = redmagic_dir + 'y3_gold_2.2.1_wide-kappa01_sofcol_run_redmagic_highlum.fit'
higherlum_file  = None
#higherlum_file  = redmagic_dir + 'y3_gold_2.2.1_wide_sofcol_run_redmagic_higherlum.fit'

outdir = 	'/Users/jackelvinpoole/DES/cats/y3/redmagic/combined_sample_0.5.1_wide_magnified/'

label = 'y3_redmagic_v0.5.1_wide-kappa01_gold_2.2.1_combined_hd3_hl2'

zlum_cut = 4. 
binedges = [0.15, 0.35, 0.5, 0.65, 0.80, 0.90] #redshift bin edges
catlist = ['hidens', 'hidens','hidens','hilum', 'hilum'] 

make_randoms = False
make_nz = True

mask_dir = '/Users/jackelvinpoole/DES/cats/y3//masks/lens_sample_joint_mask/'
mask_file = mask_dir+'y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz'

randfiles = {
	'hidens': None,
	'hilum':  None,
	}

########### make catalog ###########

print ('highdens')
highdens = lsssys.Redmagic(highdens_file, )
highdens_zmin = binedges[np.where(np.array(catlist) == 'hidens')[0].min() ]
highdens_zmax = binedges[np.where(np.array(catlist) == 'hidens')[0].max() + 1]
highdens.cutz(highdens_zmin,highdens_zmax)
highdens.cut(highdens.zlum < zlum_cut)

print ('highlum')
highlum  = lsssys.Redmagic(highlum_file,  )
highlum_zmin = binedges[np.where(np.array(catlist) == 'hilum')[0].min() ]
highlum_zmax = binedges[np.where(np.array(catlist) == 'hilum')[0].max() + 1]
highlum.cutz(highlum_zmin, highlum_zmax )
highlum.cut(highlum.zlum < zlum_cut)

if 'higherlum' in catlist:
	print ('higherlum')
	higherlum  = lsssys.Redmagic(higherlum_file,  )
	higherlum_zmin = binedges[np.where(np.array(catlist) == 'higherlum')[0].min() ]
	higherlum_zmax = binedges[np.where(np.array(catlist) == 'higherlum')[0].max() + 1]
	higherlum.cutz(higherlum_zmin, higherlum_zmax )
	higherlum.cut(higherlum.zlum < zlum_cut)

highdens.addcat(highlum)
if 'higherlum' in catlist:
	highdens.addcat(higherlum)

mask = lsssys.Mask(mask_file)
highdens.maskcat(maskobj=mask)

outfile = outdir + '{0}_sample.fits.gz'.format(label)
highdens.savecat(outfile,clobber=True,)

########### make randoms ###########
if make_randoms == True:

	rand = lsssys.randoms(None, empty=True)
	rand_hidens = lsssys.randoms(randfiles['hidens'])
	rand_hidens.cutz(0.15, 0.65)
	rand_hilum  = lsssys.randoms(randfiles['hilum'])
	rand_hilum.cutz(0.65, 0.90)
	rand.addcat(rand_hidens)
	rand.addcat(rand_hilum)
	rand.maskcat(mask)
	rand.save(outdir + '{0}_randoms.fits.gz'.format(label), clobber=True)


########### save num_dens, nz and info ###########

n_obj = np.array([len(highdens.eqinbin(binedges[i],binedges[i+1])[0]) for i in range(len(binedges)-1)])
mask_area = mask.area(units='arcmin')
num_dens = n_obj/mask_area
np.savetxt(outdir + 'num_dens.txt', num_dens)

bin_masks = [highdens.eqinbin(binedges[i],binedges[i+1], returnmask=True)[-1] for i in range(len(binedges)-1) ]
sig_z = [np.mean(highdens.z_err[amask]) for amask in bin_masks]
np.savetxt(outdir + 'sig_z.txt', sig_z)

np.savetxt(outdir + 'binedges.txt', binedges)

info_temp = """
{sample}	{zlow}	{zhigh}
num_obj = {n_obj}
area = {area}
num_dens = {num_dens}
sig_z = {sig_z}
"""
info_string = ''.join([info_temp.format(
	sample = catlist[i],
	zlow = binedges[i],
	zhigh = binedges[i+1],
	n_obj = n_obj[i],
	area = mask.area(), 
	num_dens = num_dens[i],
	sig_z = sig_z[i],
	)
	for i in range(len(binedges)-1)])
info_file = open(outdir + 'info.txt','w')
info_file.write(info_string)
info_file.close()

#n(z)
if make_nz == True:
	print ('making n(z)')
	n_dz_bins = 200
	z_edges = np.linspace(0., 1.5, n_dz_bins+1)
	z_low = z_edges[:-1]
	z_high = z_edges[1:]
	z_mid = (z_low + z_high)/2.
	z_range = (z_low[0], z_high[-1])
	nz_table = highdens.makenz_table(binedges, bins=200, useweights=True, sample=False, normed=True, range=z_range, usecol='z_samp')
	assert (nz_table[0] == z_low).all()
	num_dens = twopoint.NumberDensity('nz_lens', zlow=z_low, z=z_mid, zhigh=z_high, nzs=nz_table[1:], ngal=num_dens )
	fits_hdu = num_dens.to_fits()
	fits_hdu.writeto(outdir + 'nz_{0}_z_samp.fits'.format(label), clobber=True)

