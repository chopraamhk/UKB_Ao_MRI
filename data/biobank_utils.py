"""
    The UK Biobank cardiac image converting module.

    This module reads short-axis and long-axis DICOM files for a UK Biobank subject,
    looks for the correct series (sometimes there are more than one series for one slice),
    stack the slices into a 3D-t volume and save as a nifti image. ##edited for aortic_dist images

    pydicom is used for reading DICOM images. However, I have found that very rarely it could
    fail in reading certain DICOM images, perhaps due to the DICOM format, which has no standard
    and vary between manufacturers and machines.
"""
import os
import re
import pickle
import cv2
import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import nibabel as nib


def repl(m):
    """ Function for reformatting the date """
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))


def process_manifest(name, name2):
    """
        Read the lines in the manifest.csv file and check whether the date format contains
        a comma, which needs to be removed since it causes problems in parsing the file.
        """
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)


class BaseImage(object):
    """ Representation of an image by an array, an image-to-world affine matrix and a temporal spacing """
    volume = np.array([])
    affine = np.eye(4)
    dt = 1

    def WriteToNifti(self, filename):
        nim = nib.Nifti1Image(self.volume, self.affine)
        nim.header['pixdim'][4] = self.dt
        nim.header['sform_code'] = 1
        nib.save(nim, filename)


class Biobank_Dataset(object):
    """ Class for managing Biobank datasets """
    def __init__(self, input_dir, cvi42_dir=None):
        """
            Initialise data
            This is important, otherwise the dictionaries will not be cleaned between instances.
            """
        self.subdir = {}
        self.data = {}

        # Find and sort the DICOM sub directories
        subdirs = sorted(os.listdir(input_dir))
        ao_dir = []
       
        for s in subdirs:
            if re.match('CINE_segmented_Ao_dist$', s):
                ao_dir = os.path.join(input_dir, s)
                break

        if ao_dir:
            self.subdir['ao'] = [ao_dir]
        else:
            print("Warning: Ao subdirectory not found!")

        self.cvi42_dir = cvi42_dir

    def find_series(self, dir_name, T):
        """
            In a few cases, there are two or three time sequences or series within each folder.
            We need to find which series to convert.
            """
        files = sorted(os.listdir(dir_name))
        if len(files) > T:
            # Sort the files according to their series UIDs
            series = {}
            for f in files:
                d = dicom.read_file(os.path.join(dir_name, f))
                suid = d.SeriesInstanceUID
                if suid in series:
                    series[suid] += [f]
                else:
                    series[suid] = [f]

            # Find the series which has been annotated, otherwise use the last series.
            if self.cvi42_dir:
                find_series = False
                for suid, suid_files in series.items():
                    for f in suid_files:
                        contour_pickle = os.path.join(self.cvi42_dir, os.path.splitext(f)[0] + '.pickle')
                        if os.path.exists(contour_pickle):
                            find_series = True
                            choose_suid = suid
                            break
                if not find_series:
                    choose_suid = sorted(series.keys())[-1]
            else:
                choose_suid = sorted(series.keys())[-1]
            print('There are multiple series. Use series {0}.'.format(choose_suid))
            files = sorted(series[choose_suid])

        if len(files) < T:
            print('Warning: {0}: Number of files < CardiacNumberOfImages! '
                  'We will fill the missing files using duplicate slices.'.format(dir_name))
        return(files)

    def read_dicom_images(self):
        """ Read dicom images and store them in a 3D-t volume. """
        for name, dir in sorted(self.subdir.items()):
            # Read the image volume
            # Number of slices
            Z = len(dir)

            # Read a dicom file at the first slice to get the temporal information
            # We need the number of images in a sequence to check whether multiple sequences are recorded
            d = dicom.read_file(os.path.join(dir[0], sorted(os.listdir(dir[0]))[0]))
            T = d.CardiacNumberOfImages

            # Read a dicom file from the correct series when there are multiple time sequences
            d = dicom.read_file(os.path.join(dir[0], self.find_series(dir[0], T)[0]))
            X = d.Columns
            Y = d.Rows
            T = d.CardiacNumberOfImages
            dx = float(d.PixelSpacing[1])
            dy = float(d.PixelSpacing[0])

            # DICOM coordinate (LPS)
            #  x: left
            #  y: posterior
            #  z: superior
            # Nifti coordinate (RAS)
            #  x: right
            #  y: anterior
            #  z: superior
            # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
            # Refer to
            # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
            # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

            # The coordinate of the upper-left voxel of the first and second slices
            pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
            pos_ul[:2] = -pos_ul[:2]

            # Image orientation
            axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
            axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
            axis_x[:2] = -axis_x[:2]
            axis_y[:2] = -axis_y[:2]

            if Z >= 2:
                # Read a dicom file at the second slice
                d2 = dicom.read_file(os.path.join(dir[1], sorted(os.listdir(dir[1]))[0]))
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
            else:
                axis_z = np.cross(axis_x, axis_y)

            # Determine the z spacing
            if hasattr(d, 'SpacingBetweenSlices'):
                dz = float(d.SpacingBetweenSlices)
            elif Z >= 2:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                      'Calculate from two successive slices.')
                dz = float(np.linalg.norm(pos_ul2 - pos_ul))
            else:
                print('Warning: can not find attribute SpacingBetweenSlices. '
                      'Use attribute SliceThickness instead.')
                dz = float(d.SliceThickness)

            # Affine matrix which converts the voxel coordinate to world coordinate
            affine = np.eye(4)
            affine[:3, 0] = axis_x * dx
            affine[:3, 1] = axis_y * dy
            affine[:3, 2] = axis_z * dz
            affine[:3, 3] = pos_ul

            # The 4D volume
            volume = np.zeros((X, Y, Z, T), dtype='float32')
            if self.cvi42_dir:
                # Save both label map in original resolution and upsampled label map.
                # The image annotation by defaults upsamples the image using cvi42 and then
                # annotate on the upsampled image.
                up = 4
                label = np.zeros((X, Y, Z, T), dtype='int16')
                label_up = np.zeros((X * up, Y * up, Z, T), dtype='int16')

            # Go through each slice
            for z in range(0, Z):
                # In a few cases, there are two or three time sequences or series within each folder.
                # We need to find which seires to convert.
                files = self.find_series(dir[z], T)

                # Now for this series, sort the files according to the trigger time.
                files_time = []
                for f in files:
                    d = dicom.read_file(os.path.join(dir[z], f))
                    t = d.TriggerTime
                    files_time += [[f, t]]
                files_time = sorted(files_time, key=lambda x: x[1])

                # Read the images
                for t in range(0, T):
                    # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
                    # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
                    # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
                    # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
                    # with nibabel's dimension.
                    try:
                        f = files_time[t][0]
                        d = dicom.read_file(os.path.join(dir[z], f))
                        volume[:, :, z, t] = d.pixel_array.transpose()
                    except IndexError:
                        print('Warning: dicom file missing for {0}: time point {1}. '
                              'Image will be copied from the previous time point.'.format(dir[z], t))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                    except (ValueError, TypeError):
                        print('Warning: failed to read pixel_array from file {0}. '
                              'Image will be copied from the previous time point.'.format(os.path.join(dir[z], f)))
                        volume[:, :, z, t] = volume[:, :, z, t - 1]
                    except NotImplementedError:
                        print('Warning: failed to read pixel_array from file {0}. '
                              'pydicom cannot handle compressed dicom files. '
                              'Switch to SimpleITK instead.'.format(os.path.join(dir[z], f)))
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(os.path.join(dir[z], f))
                        img = sitk.GetArrayFromImage(reader.Execute())
                        volume[:, :, z, t] = np.transpose(img[0], (1, 0))

                    if self.cvi42_dir:
                        # Check whether there is a corresponding cvi42 contour file for this dicom
                        contour_pickle = os.path.join(self.cvi42_dir, os.path.splitext(f)[0] + '.pickle')
                        if os.path.exists(contour_pickle):
                            with open(contour_pickle, 'rb') as f:
                                contours = pickle.load(f)

                                # Labels
                                lv_endo = 1
                                lv_epi = 2
                                rv_endo = 3
                                la_endo = 1
                                ra_endo = 2

                                # Fill the contours in order
                                # RV endocardium first, then LV epicardium,
                                # then LV endocardium, then RA and LA.
                                #
                                # Issue: there is a problem in very rare cases,
                                # e.g. eid 2485225, 2700750, 2862965, 2912168,
                                # where LV epicardial contour is not a closed contour. This problem
                                # can only be solved if we could have a better definition of contours.
                                # Thanks for Elena Lukaschuk and Stefan Piechnik for pointing this out.
                                ordered_contours = []
                                if 'sarvendocardialContour' in contours:
                                    ordered_contours += [(contours['sarvendocardialContour'], rv_endo)]

                                if 'saepicardialContour' in contours:
                                    ordered_contours += [(contours['saepicardialContour'], lv_epi)]
                                if 'saepicardialOpenContour' in contours:
                                    ordered_contours += [(contours['saepicardialOpenContour'], lv_epi)]

                                if 'saendocardialContour' in contours:
                                    ordered_contours += [(contours['saendocardialContour'], lv_endo)]
                                if 'saendocardialOpenContour' in contours:
                                    ordered_contours += [(contours['saendocardialOpenContour'], lv_endo)]

                                if 'laraContour' in contours:
                                    ordered_contours += [(contours['laraContour'], ra_endo)]

                                if 'lalaContour' in contours:
                                    ordered_contours += [(contours['lalaContour'], la_endo)]

                                # cv2.fillPoly requires the contour coordinates to be integers.
                                # However, the contour coordinates are floating point number since
                                # they are drawn on an upsampled image by 4 times.
                                # We multiply it by 4 to be an integer. Then we perform fillPoly on
                                # the upsampled image as cvi42 does. This leads to a consistent volume
                                # measurement as cvi2. If we perform fillPoly on the original image, the
                                # volumes are often over-estimated by 5~10%.
                                # We found that it also looks better to fill polygons it on the upsampled
                                # space and then downsample the label map than fill on the original image.
                                lab_up = np.zeros((Y * up, X * up))
                                for c, l in ordered_contours:
                                    coord = np.round(c * up).astype(np.int)
                                    cv2.fillPoly(lab_up, [coord], l)

                                label_up[:, :, z, t] = lab_up.transpose()
                                label[:, :, z, t] = lab_up[::up, ::up].transpose()

            # Temporal spacing
            dt = (files_time[1][1] - files_time[0][1]) * 1e-3

            # Store the image
            self.data[name] = BaseImage()
            self.data[name].volume = volume
            self.data[name].affine = affine
            self.data[name].dt = dt

            if self.cvi42_dir:
                # Only save the label map if it is non-zero
                if np.any(label):
                    self.data['label_' + name] = BaseImage()
                    self.data['label_' + name].volume = label
                    self.data['label_' + name].affine = affine
                    self.data['label_' + name].dt = dt

                if np.any(label_up):
                    self.data['label_up_' + name] = BaseImage()
                    self.data['label_up_' + name].volume = label_up
                    up_matrix = np.array([[1.0/up, 0, 0, 0], [0, 1.0/up, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    self.data['label_up_' + name].affine = np.dot(affine, up_matrix)
                    self.data['label_up_' + name].dt = dt

    def convert_dicom_to_nifti(self, output_dir):
        """ Save the image in nifti format. """
        for name, image in self.data.items():
            image.WriteToNifti(os.path.join(output_dir, '{0}.nii.gz'.format(name)))
