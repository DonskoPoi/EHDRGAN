import numpy as np
import scripts.metrics as m

def visualize_with_gt(gt_img, pred_img, ratio_path):
    '''
    this function is for tune map your output image with gt and ratio
    :param gt_img: your ground truth image with float32 numpy BGR
    :param pred_img: your hdr image generated from the model with float32 numpy BGR
    :param ratio_path: the alignratio.npy file path
    :return: a tone-mapped hdr image with uint8 BGR numpy form
    '''

    # load ratio
    ratio = np.load(ratio_path).astype(np.float32)
    # 65536->256
    pred_img /= ratio
    gt_img /= ratio
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
    # load ratio
    ratio = np.float32(65535/255)
    # 65536->256
    pred_img /= ratio
    # gamma correction: none liner to liner image
    pred_img **= 2.24
    return pred_img.round().astype(np.unit8)
