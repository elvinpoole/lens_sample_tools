import numpy as np
import fitsio as fio
import pylab as plt
import pandas as pd
import astropy.table

#Y3 balrog detection file
balrog_detection_file = "/global/cfs/cdirs/des/severett/Balrog/run2a/stacked_catalogs/1.4/sof/balrog_detection_catalog_sof_run2a_v1.4.fits"

#Y3 balrog maglim sample file
balrog_maglim_file = "/global/cfs/cdirs/des/jelvinpo/cats/y3/balrog/run2/lens_samples_run2_v1.4/maglim_lens_v2p2_flux_balrog_run2_v1.4_masked.fits.gz"

#Y3 deep field magnitudes
deep_field_file = "/pscratch/sd/j/jelvinpo/DES/cats/y3/deepfields/Y3_DEEP_FIELDS_PHOTOM_ang_mag.fits"

#Y3 deep field star galaxy separator
deep_field_sg_file = "/pscratch/sd/j/jelvinpo/DES/cats/y3/deepfields/Y3_DEEP_FIELDS_PHOTOM_ang_mag_sg.fits"

#Y3 deep field matched to WISE forced photometry of Legacy Survey (from Noah)
deep_field_wise_file = "/global/cfs/projectdirs/des/nweaverd/y3_deepfields/dr10.0/south/matched/merged_ls-dr10.0-Y3_DEEP_FIELDS_PHOTOM_ang_mag.fits"


###############################
###### load the tables in pandas DataFrame
###############################

balrog_detection_table = astropy.table.Table.read(balrog_detection_file)
data = np.array([
    balrog_detection_table['true_id'],
    balrog_detection_table["true_bdf_mag_deredden"][:,0],
    balrog_detection_table["true_bdf_mag_deredden"][:,1],
    balrog_detection_table["true_bdf_mag_deredden"][:,2],
    balrog_detection_table["true_bdf_mag_deredden"][:,3],
    balrog_detection_table['detected'],
])
columns = [
    'true_id', 
    'true_bdf_mag_deredden_g',
    'true_bdf_mag_deredden_r',
    'true_bdf_mag_deredden_i',
    'true_bdf_mag_deredden_z',
    'detected',
]
balrog_detection = pd.DataFrame(data=data.T, columns=columns)

deep_field = astropy.table.Table.read(deep_field_file).to_pandas()
deep_field_sg = astropy.table.Table.read(deep_field_sg_file).to_pandas()
deep_field_wise_table = astropy.table.Table.read(deep_field_wise_file)
names = [name for name in deep_field_wise_table.colnames if len(deep_field_wise_table[name].shape) <= 1]
deep_field_wise = deep_field_wise_table[names].to_pandas()

balrog_maglim_fits = fio.read(balrog_maglim_file)
balrog_maglim = astropy.table.Table(data=[
    balrog_maglim_fits["true_id"],
    balrog_maglim_fits['meas_cm_mag_deredden'][:,1],
    balrog_maglim_fits['meas_cm_mag_deredden'][:,3],
    balrog_maglim_fits['Z_MEAN'],
    balrog_maglim_fits['true_bdf_T'],
    balrog_maglim_fits['true_bdf_T_err'],
    ],
    names=[
        'INJ_ID',
        'meas_cm_mag_deredden_r',
        'meas_cm_mag_deredden_z',
        'Z_MEAN',
        'true_bdf_T',
        'true_bdf_T_err',
    ],
    ).to_pandas()

###############################
###### match the catalogs 
###############################

n_deep = len(deep_field)
n_deep_sg = len(deep_field)
n_deep_wise = len(deep_field_wise)
assert n_deep == n_deep_sg

deep_field = deep_field.merge(
    deep_field_sg, 
    how='inner', 
    on='ID', 
)
assert len(deep_field) == n_deep
del deep_field_sg

deep_field_wise = deep_field_wise.merge(
    deep_field, 
    how='left', 
    on='ID', 
)
assert len(deep_field_wise) == n_deep_wise

balrog_detection_matched_deep = balrog_detection.merge(
    deep_field, 
    how='left', 
    left_on='true_id', 
    right_on='ID')
balrog_detection_matched_deep_wise = balrog_detection.merge(
    deep_field_wise, 
    how='left', 
    left_on='true_id', 
    right_on='ID')
is_match_det = ~np.isnan(np.array(balrog_detection_matched_deep_wise['ID']))

balrog_maglim_matched_deep = balrog_maglim.merge(
    deep_field, 
    how='left', 
    left_on='INJ_ID', 
    right_on='ID')
balrog_maglim_matched_deep_wise = balrog_maglim.merge(
    deep_field_wise, 
    how='left', 
    left_on='INJ_ID', 
    right_on='ID')
is_match_maglim = ~np.isnan(np.array(balrog_maglim_matched_deep_wise['ID']))

###############################
###### Count numbers
###############################

print("Fraction of balrog injections with deep field match (should be 1.0) =", 
      len(balrog_detection_matched_deep)/len(balrog_detection),
     )
print("Fraction of balrog injections with WISE match =", 
      len(balrog_detection_matched_deep_wise[is_match_det])/len(balrog_detection),
     )
print("Fraction of balrog detections with WISE match =", 
      np.sum(balrog_detection_matched_deep_wise['detected'][is_match_det])/np.sum(balrog_detection['detected']),
     )
print("Fraction of balrog-maglim with WISE match =", 
      len(balrog_maglim_matched_deep_wise[is_match_maglim])/len(balrog_maglim_matched_deep),
     )


###############################
###### Plot deep field ra/dec and matches
###############################

fig, axs = plt.subplots(2,2,figsize=(5,5))
axs = axs.flatten()
in_wise = np.in1d(deep_field['ID'],deep_field_wise['ID'])

ra  = np.array(deep_field['RA_x'])
dec = np.array(deep_field['DEC_x'])
select1 = (ra < 20)
select2 = (ra < 60)*(dec > -10)
select3 = (ra > 40)*(dec < -20)
select4 = (ra > 120)
select = [select1, select2, select3, select4,]
for i in range(4):
    axs[i].plot(ra[select[i]], dec[select[i]], ',', color='blue')
    axs[i].plot(ra[in_wise*select[i]], dec[in_wise*select[i]], ',', color='orange')

fig.savefig('deep_ra_dec.png')


###############################
###### get fractions objects in each KNN catagory
###############################
zminusw1 = np.array(balrog_maglim_matched_deep_wise["meas_cm_mag_deredden_z"])-np.array(balrog_maglim_matched_deep_wise["W1_MAG_CORR"])
rminusz  = np.array(balrog_maglim_matched_deep_wise["meas_cm_mag_deredden_r"])-np.array(balrog_maglim_matched_deep_wise["meas_cm_mag_deredden_z"])                                                                                         
z = np.array(balrog_maglim_matched_deep_wise["Z_MEAN"])
knn = np.array(balrog_maglim_matched_deep_wise['KNN_CLASS'])

cuts_y6 = [
    (zminusw1 < 0.7*(rminusz-0.75)-0.5)*(zminusw1 < 0.8*(rminusz-0.75)-0.5), 
    (zminusw1 < 0.7*(rminusz-0.75)-0.5)*(zminusw1 < 2.5*(rminusz-0.75)-0.5), 
    (zminusw1 < 0.7*(rminusz-0.75)-0.4)*(zminusw1 < 2.5*(rminusz-0.75)-0.4), 
    (zminusw1 < 0.7*(rminusz-1.2)+0  )*(zminusw1 < 3*(rminusz-1.2)+0), 
    (zminusw1 < 0.7*(rminusz-1.4)+0.1)*(zminusw1 < 6*(rminusz-1.4)+0.1), 
    (zminusw1 < 0.7*(rminusz-1.6)+0.1)*(zminusw1 < 6*(rminusz-1.6)+0.1), 
]
cuts_y3 = [
    (zminusw1 < 0.7*(rminusz-0.75)-0.5)*(zminusw1 < 0.8*(rminusz-0.75)-0.5), 
    (zminusw1 < 0.7*(rminusz-0.75)-0.5)*(zminusw1 < 0.8*(rminusz-0.75)-0.5), 
    (zminusw1 < 0.7*(rminusz-0.75)-0.4)*(zminusw1 < 2.5*(rminusz-0.75)-0.4), 
    (zminusw1 < 0.7*(rminusz-1.1)+0  )*(zminusw1 < 15*(rminusz-1.1)+0), 
    (zminusw1 < 0.7*(rminusz-1.35)+0.1)*(zminusw1 < 15*(rminusz-1.35)+0.1), 
    (zminusw1 < 0.7*(rminusz-1.6)+0.1)*(zminusw1 < 6*(rminusz-1.6)+0.1), 
]
cuts = cuts_y3

binedges = [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05]

print('Fraction of Y3 balrog-maglim objects removed by WISE cut, binned by KNN Star/gal classifier')
print('')
print ("bin", "No_Class", "Galaxy", "Star", "Ambiguous")
for ibin in range(len(binedges)-1):
    selectz = (z > binedges[ibin])*(z < binedges[ibin+1])

    knn_bin_before = knn[selectz]
    knn_bin_after = knn[selectz*~cuts[ibin]]

    nc_before   = np.sum((knn_bin_before == 0).astype(int))
    gal_before  = np.sum((knn_bin_before == 1).astype(int))
    star_before = np.sum((knn_bin_before == 2).astype(int))
    amb_before  = np.sum((knn_bin_before == 3).astype(int))
    if amb_before == 0:
        amb_before = 1

    nc_after   = np.sum((knn_bin_after == 0).astype(int))
    gal_after  = np.sum((knn_bin_after == 1).astype(int))
    star_after = np.sum((knn_bin_after == 2).astype(int))
    amb_after  = np.sum((knn_bin_after == 3).astype(int))
    
    print(
        str(ibin+1).ljust(3), 
        str(np.round(1-nc_after/nc_before,3)).ljust(8), 
        str(np.round(1-gal_after/gal_before,3)).ljust(6), 
        str(np.round(1-star_after/star_before,3)).ljust(4),
        str(np.round(1-amb_after/amb_before,3)).ljust(9),
        )
    
    
    
    
    