import os
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from aorta_pass_quality_control import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--pressure_csv', metavar='csv_name', default='', required=True)
    parser.add_argument('--output_csv', metavar='csv_name', default='', required=True)
    args = parser.parse_args()

    # Read the spreadsheet for blood pressure information
    # Use central blood pressure provided by the Vicorder software
    # [1] Steffen E. Petersen et al. UK Biobank’s cardiovascular magneticresonance protocol. JCMR, 2016.
    # Aortic distensibility represents the relative change in area of the aorta per unit pressure,
    # taken here as the "central pulse pressure".
    #
    # The Vicorder software calculates values for central blood pressure by applying a previously described
    # brachial-to-aortic transfer function. What I observed from the data and Figure 5 in the SOP pdf
    # (https://biobank.ctsu.ox.ac.uk/crystal/docs/vicorder_in_cmri.pdf) is that after the transfer, the DBP
    # keeps the same as the brachial DBP, but the SBP is different.
    df_info = pd.read_csv(args.pressure_csv, header=[0, 1], index_col=0)
    central_pp = df_info['Central pulse pressure during PWA'][['12678-2.0', '12678-2.1', '12678-2.2', '12678-2.3', '12678-2.4', '12678-3.0', '12678-3.1', '12678-3.2', '12678-3.3', '12678-3.4']].mean(axis=1)

    # Discard central blood pressure < 10 mmHg
    central_pp[central_pp < 10] = np.nan

    data_path = args.data_dir
    data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in data_list:
        data_dir = os.path.join(data_path, data)
        image_name = os.path.join(data_dir, 'ao.nii.gz')
        seg_name = os.path.join(data_dir, 'seg_ao.nii.gz')

        if os.path.exists(image_name) and os.path.exists(seg_name):
            print(data)

            # Read image
            nim = nib.load(image_name)
            dx, dy = nim.header['pixdim'][1:3]
            area_per_pixel = dx * dy
            image = nim.get_fdata()

            # Read segmentation
            nim = nib.load(seg_name)
            seg = nim.get_fdata()

            if not aorta_pass_quality_control(image, seg):
                continue

            # Measure the maximal and minimal area for the ascending aorta and descending aorta
            val = {}
            for l_name, l in [('AAo', 1), ('DAo', 2)]:
                val[l_name] = {}
                A = np.sum(seg == l, axis=(0, 1, 2)) * area_per_pixel
                val[l_name]['max area'] = A.max()
                val[l_name]['min area'] = A.min()
                val[l_name]['distensibility'] = (A.max() - A.min()) / (A.min() * central_pp.loc[int(data)]) * 1e3

            line = [val['AAo']['max area'], val['AAo']['min area'], val['AAo']['distensibility'],
                    val['DAo']['max area'], val['DAo']['min area'], val['DAo']['distensibility']]
            table += [line]
            processed_list += [data]

    # Save the spreadsheet for the measures
    df = pd.DataFrame(table, index=processed_list,
                      columns=['AAo max area (mm2)', 'AAo min area (mm2)', 'AAo distensibility (10-3 mmHg-1)',
                               'DAo max area (mm2)', 'DAo min area (mm2)', 'DAo distensibility (10-3 mmHg-1)'])
    df.to_csv(args.output_csv)
