import numpy as np
import cv2, time
from scipy.misc import imresize

def cacf_cxy(pos, sz):

    total_pos = []

    top_left = np.array([pos[0]-sz[0], pos[1]-sz[1]])
    mid_left = np.array([pos[0]-sz[0], pos[1]])
    bottom_left = np.array([pos[0]-sz[0], pos[1]+sz[1]])

    top_mid = np.array([pos[0], pos[1]-sz[1]])
    bottom_mid = np.array([pos[0], pos[1]+sz[1]])

    top_right = np.array([pos[0]+sz[0], pos[1]-sz[1]])
    mid_right = np.array([pos[0]+sz[0], pos[1]])
    bottom_right = np.array([pos[0]+sz[0], pos[1]+sz[1]])


    total_pos.append(top_left)
    total_pos.append(mid_left)
    total_pos.append(bottom_left)
    total_pos.append(top_mid)
    total_pos.append(bottom_mid)
    total_pos.append(top_right)
    total_pos.append(mid_right)
    total_pos.append(bottom_right)

    return total_pos


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index

def cxy_wh_2_bbox_cacf(cxy, wh):
    total_bbox = []

    for i in range(len(cxy)):

        total_bbox.append(np.array([cxy[i][0]-wh[0]/2, cxy[i][1]-wh[1]/2, cxy[i][0]+wh[0]/2, cxy[i][1]+wh[1]/2]))
    return total_bbox


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw_cacf(image, bbox, out_sz, padding=(0, 0, 0)):
    total_crop = []

    for i in range(len(bbox)):
        a = (out_sz - 1) / (bbox[i][2] - bbox[i][0])
        b = (out_sz - 1) / (bbox[i][3] - bbox[i][1])
        c = -a * bbox[i][0]
        d = -b * bbox[i][1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)

        total_crop.append(np.transpose(crop, (2, 0, 1)))

    return total_crop

def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    #crop = crop[None,:,:,:]
    return np.transpose(crop, (2, 0, 1))


def overlap_ratio(bbox, gt): #(x,y,w,h)
    bbox = np.asarray(bbox)
    bbox = bbox[None,:]
    gt = np.asarray(gt)

    Left = np.maximum(gt[:,0], bbox[:,0])
    Right = np.minimum(gt[:,0] + gt[:,2], bbox[:,0] + bbox[:,2])
    Top = np.maximum(gt[:,1], bbox[:,1])
    Bottom = np.minimum(gt[:,1] + gt[:,3], bbox[:,1] + bbox[:,3])

    inter_area = np.maximum(0, Right-Left) * np.maximum(0, Bottom-Top)
    union_area = gt[:,2] * gt[:,3] + bbox[:,2] * bbox[:,3] - inter_area

    IOU = np.clip(inter_area/union_area, 0, 1)
    precision_IOU = np.mean(IOU)
    return precision_IOU, IOU

def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    _, iou = overlap_ratio2(result_bb, gt_bb)
    for i in range(len(thresholds_overlap)):
        SSi = sum(iou > thresholds_overlap[i])
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def overlap_ratio2(bbox, gt): #(x,y,w,h)
    bbox = np.asarray(bbox)
    gt = np.asarray(gt)

    Left = np.maximum(gt[:,0], bbox[:,0])
    Right = np.minimum(gt[:,0] + gt[:,2], bbox[:,0] + bbox[:,2])
    Top = np.maximum(gt[:,1], bbox[:,1])
    Bottom = np.minimum(gt[:,1] + gt[:,3], bbox[:,1] + bbox[:,3])

    inter_area = np.maximum(0, Right-Left) * np.maximum(0, Bottom-Top)
    union_area = gt[:,2] * gt[:,3] + bbox[:,2] * bbox[:,3] - inter_area

    IOU = np.clip(inter_area/union_area, 0, 1)
    precision_IOU = np.mean(IOU)
    return precision_IOU, IOU

def EAO_Success_overlap(iou):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(iou)
    success = np.zeros(len(thresholds_overlap))

    for i in range(len(thresholds_overlap)):
        #SSi = sum(iou > thresholds_overlap[i])
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)

    return success



def location_precision(bbox, gt):
    max_threshold = 20
    max_threshold2 = 50
    bbox = np.asarray(bbox, dtype=int)
    #bbox = bbox[None,:]
    gt = np.asarray(gt, dtype=int)

    bbox[:,0] = bbox[:,0] + bbox[:,2]/2
    bbox[:,1] = bbox[:,1] + bbox[:,3]/2

    gt[:,0] = gt[:,0] + gt[:,2]/2
    gt[:,1] = gt[:,1] + gt[:,3]/2

    precision_avg = np.zeros(max_threshold2, dtype=float)
    precision_dp = np.zeros(bbox.shape[0], dtype=float)

    if bbox.shape[0] != gt.shape[0]:
        row_num = min(bbox.shape[0], gt.shape[0])
        bbox = bbox[:row_num, :]
        gt = gt[:row_num, :]

    distance = np.sqrt((bbox[:,0]-gt[:,0])**2 + (bbox[:,1]-gt[:,1])**2)
    distance[np.isnan(distance)] = 0


    for iter in range(max_threshold2):
        precision_avg[iter] = len(np.where((distance+1) <= iter)[0])/distance.shape[0]


    for iter2 in range(precision_dp.shape[0]):
        if distance[iter2] <= max_threshold:
            precision_dp[iter2] = 1
        else:
            precision_dp[iter2] = 0

    precision_dp = np.count_nonzero(precision_dp)/distance.shape[0]

    return precision_avg, precision_dp


def crop_image(img, bbox, img_size, padding, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w/2, h/2
    cen_x, cen_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding + w/img_size
        pad_h = padding + h/img_size

        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(cen_x - half_w + 0.5)
    min_y = int(cen_y - half_h + 0.5)
    max_x = int(cen_x + half_w + 0.5)
    max_y = int(cen_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y : max_y_val - min_y, min_x_val - min_x : max_x_val - min_x, :] = img[min_y_val:max_y_val, min_x_val:max_x_val, :]


    scaled = imresize(cropped, (img_size, img_size))

    return scaled

if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
    print(a)