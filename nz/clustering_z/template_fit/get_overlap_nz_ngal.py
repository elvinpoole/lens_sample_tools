"""
gets the DES galaxy density in the DES-eboss overlap region
"""

import numpy as np
import pylab as plt 
import lsssys.catalog

rmdir = '/Users/jackelvinpoole/DES/cats/y3/redmagic/combined_sample_0.5.1_wide_0.9binning_zmax0.95/'
rmfile = rmdir + 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_combined_hd3_hl2_sample.fits.gz'

maglimdir = '/Users/jackelvinpoole/DES/cats/y3/mag_limit_lens/v2.2/'
maglimfile = maglimdir + 'mag_lim_lens_sample_v2p2.fits'

outdir  = 'output_overlap/'
plotdir = 'plots_overlap/'

samples = [
	{
		'label':'rm_0.5.1_wide_0.9binning_large-dz',
		'file':rmfile,
		'binedges':[0.15, 0.35, 0.50, 0.65, 0.80, 0.90],
		'catalog_class':lsssys.catalog.Redmagic,
		'usecol':'z_samp',
	},
]
smaple_tmp = [
	{
		'label':'maglim_v22',
		'file':maglimfile,
		'binedges': [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],
		'catalog_class':lsssys.catalog.Maglim,
		'usecol':'z_mc',
	},
]

zbins = 70
zlims = (0.1,1.5)

for sample in samples:
	label = sample['label']
	sample_file = sample['file']
	cat = sample['catalog_class'](sample_file)
	binedges = sample['binedges']

	select_overlap = (cat.dec > -10.6)*(~(  ((cat.ra<60)*(cat.dec<-7.5))*((cat.dec<(0.083333*cat.ra-12.766666))|((cat.ra<25.5)*(cat.dec<-8.6)))   ))

	#get number densities
	n_gal_total_list = []
	n_gal_overlap_list = []
	for ibin in xrange(len(binedges)-1):
		zmin = binedges[ibin]
		zmax = binedges[ibin+1]

		n_gal_total = np.sum(((cat.z >= zmin)*(cat.z < zmax)).astype('int'))
		n_gal_overlap = np.sum(((cat.z[select_overlap] >= zmin)*(cat.z[select_overlap] < zmax)).astype('int'))

		n_gal_total_list.append( n_gal_total )
		n_gal_overlap_list.append( n_gal_overlap )

		print 'total,   bin {0}, {1}'.format(ibin+1, n_gal_total)
		print 'overlap, bin {0}, {1}'.format(ibin+1, n_gal_overlap)
		print ''

	np.savetxt(outdir+'n_gal_total_{label}.txt'.format(label=label), n_gal_total_list )
	np.savetxt(outdir+'n_gal_overlap_{label}.txt'.format(label=label), n_gal_overlap_list )


	#make n(z)
	nz_table_total = cat.makenz_table(binedges, bins=zbins, range=zlims, useweights=True, sample=False, normed=False, usecol=sample['usecol'])
	select = (cat.dec > -10.6)*(~(  ((cat.ra<60)*(cat.dec<-7.5))*((cat.dec<(0.083333*cat.ra-12.766666))|((cat.ra<25.5)*(cat.dec<-8.6)))   ))
	cat.cut(select) 
	nz_table_overlap = cat.makenz_table(binedges, bins=200, useweights=True, sample=False, normed=False, usecol=sample['usecol'])

	cat.savecat('catalog_overlap_{label}.fits.gz'.format(label=label) )

	np.savetxt( outdir+'nz_combined_total_{label}.txt'.format(label=label) , np.transpose(nz_table_total))
	np.savetxt( outdir+'nz_combined_overlap_{label}.txt'.format(label=label) , np.transpose(nz_table_overlap))

	[plt.plot(nz_table_overlap[0],nz_table_overlap[i+1]/np.trapz(nz_table_overlap[i+1],nz_table_overlap[0]),'b-') for i in xrange(5)] 
	[plt.plot(nz_table_total[0],nz_table_total[i+1]/np.trapz(nz_table_total[i+1],nz_table_total[0]),'r--') for i in xrange(5)]
	plt.xlim([0.1,1.1])
	plt.savefig(plotdir+'nz_redmagic_overlap_{label}.png'.format(label=label))
	plt.close()



