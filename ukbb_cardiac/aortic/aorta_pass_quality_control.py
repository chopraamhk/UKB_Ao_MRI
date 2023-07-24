import numpy as np
import skimage

def aorta_pass_quality_control(image, seg):
    """ Quality control for aortic segmentation """
    for l_name, l in [('AAo', 1), ('DAo', 2)]:
        # Criterion 1: the aorta does not disappear at some point.
        T = seg.shape[3]
        for t in range(T):
            seg_t = seg[:, :, :, t]
            area = np.sum(seg_t == l)
            if area == 0:
                print('The area of {0} is 0 at time frame {1}.'.format(l_name, t))
                return False

        # Criterion 2: no strong image noise, which affects the segmentation accuracy.
        image_ED = image[:, :, :, 0]
        seg_ED = seg[:, :, :, 0]
        mean_intensity_ED = image_ED[seg_ED == l].mean()
        ratio_thres = 3
        for t in range(T):
            image_t = image[:, :, :, t]
            seg_t = seg[:, :, :, t]
            max_intensity_t = np.max(image_t[seg_t == l])
            ratio = max_intensity_t / mean_intensity_ED
            if ratio >= ratio_thres:
                print('The image becomes very noisy at time frame {0}.'.format(t))
                return False

        # Criterion 3: no fragmented segmentation
        pixel_thres = 10
        for t in range(T):
            seg_t = seg[:, :, :, t]
            cc, n_cc = skimage.measure.label(seg_t == l, background=8, return_num=True)
            count_cc = 0
            for i in range(1, n_cc + 1):
                binary_cc = (cc == i)
                if np.sum(binary_cc) > pixel_thres:
                    # If this connected component has more than certain pixels, count it.
                    count_cc += 1
            if count_cc >= 2:
                print('The segmentation has at least two connected components with more than {0} pixels '
                      'at time frame {1}.'.format(pixel_thres, t))
                return False

        # Criterion 4: no abrupt change of area
        A = np.sum(seg == l, axis=(0, 1, 2))
        for t in range(T):
            ratio = A[t] / float(A[t-1])
            if ratio >= 2 or ratio <= 0.5:
                print('There is abrupt change of area at time frame {0}.'.format(t))
                return False
    return True

