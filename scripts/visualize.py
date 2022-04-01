import numpy as np
import scripts.metrics as m

def visualize_with_gt(gt_img, pred_img):
    '''
    this function is for tune map your output image with gt and ratio
    the two input image(output by network) is within the range of 0-255
    :param gt_img: your ground truth image with float32 numpy BGR
    :param pred_img: your hdr image generated from the model with float32 numpy BGR
    :return: a tone-mapped hdr image with uint8 BGR numpy form
    '''

    # gamma correction: none liner to liner image
    pred_img **= 2.24
    gt_img **= 2.24
    # calculate norm from gt
    norm_perc = np.percentile(gt_img, 99)
    # tone mapping
    pred_img = (m.tanh_norm_mu_tonemap(pred_img, norm_perc) * 255.).round().astype(np.uint8)
    return pred_img


def visualize(pred_img):
    '''
    this function is used for tone-mapping a hdr image with just gamma correction
    :param pred_img: your hdr image generated from the model with float32 numpy BGR
    :param ratio_path: the alignratio.npy file path
    :return: a tone-mapped hdr image with uint8 BGR numpy form
    '''
    # gamma correction: none liner to liner image
    pred_img **= 2.24
    return pred_img.round().astype(np.uint8)
