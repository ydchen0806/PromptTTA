import os
import cv2
import time
import h5py
import waterz
import mahotas
import argparse
import numpy as np
from PIL import Image
import evaluate as ev
from scipy import ndimage


def randomlabel(segmentation):
    segmentation = segmentation.astype(np.uint32)
    uid = np.unique(segmentation)
    mid = int(uid.max()) + 1
    mapping = np.zeros(mid, dtype=segmentation.dtype)
    mapping[uid] = np.random.choice(len(uid), len(uid), replace=False).astype(
        segmentation.dtype)  # (len(uid), dtype=segmentation.dtype)
    out = mapping[segmentation]
    out[segmentation == 0] = 0
    return out


def watershed(affs, seed_method, use_mahotas_watershed=True):
    affs_xy = 1.0 - 0.5 * (affs[1] + affs[2])
    depth = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0 * affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds
    return fragments


def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x * num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y, num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds == next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary < 0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds == next_id] = 0

    return seeds, num_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', type=str, default=' ', help='path to config file')
    parser.add_argument('-id', '--model_id', type=int, default=51000)
    parser.add_argument('-m', '--mode', type=str, default='isbi')
    parser.add_argument('-mk', '--mask_fragment', type=float, default=None)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-st', '--start_th', type=float, default=0.1)
    parser.add_argument('-et', '--end_th', type=float, default=0.9)
    parser.add_argument('-s', '--stride', type=float, default=0.1)
    args = parser.parse_args()

    trained_model = args.in_path
    out_path = os.path.join('../inference', trained_model, args.mode)
    img_folder = 'affs_' + str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    print('out_path: ' + out_affs)
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    # load affs
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'r')
    affs = f['main'][:]
    f.close()

    # load raw images
    if args.mode == 'isbi':
        print('isbi')
        f = h5py.File('../data/snemi3d/isbi_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
        test_label = test_label[80:]
    elif args.mode == 'isbi_test':
        print('isbi')
        f = h5py.File('../data/snemi3d/isbi_test_labels.h5', 'r')
        test_label = f['main'][:]
        f.close()
    else:
        raise NotImplementedError

    thresholds = np.arange(args.start_th, args.end_th + args.stride, args.stride)
    thresholds = list(thresholds)
    print('thresholds:', thresholds)

    fragments = watershed(affs, 'maxima_distance')
    ### mask
    if args.mask_fragment is not None:
        tt = args.mask_fragment
        print('add mask and threshold=' + str(tt))
        affs_xy = 0.5 * (affs[1] + affs[2])
        fragments[affs_xy < tt] = 0

    # sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
    # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>'
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'

    seg_gt = True
    if seg_gt and test_label is not None:
        gt = test_label.astype(np.uint32)
    else:
        gt = None
    # seg = waterz.agglomerate(affs,
    #                     thresholds,
    #                     gt=gt,
    #                     fragments=fragments,
    #                     scoring_function=sf,
    #                     discretize_queue=256)
    seg = waterz.agglomerate(affs,
                             thresholds,
                             gt=gt,
                             fragments=fragments,
                             discretize_queue=256)

    if 'isbi' in args.mode:
        best_arand = 1000
        best_idx = 0
        f_txt = open(os.path.join(out_affs, 'seg_scores.txt'), 'w')
        seg_results = []
        for idx, seg_metric in enumerate(seg):
            if seg_gt:
                segmentation = seg_metric[0].astype(np.int32)
                metrics = seg_metric[1]
                print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, rand_split=%.6f, rand_merge=%.6f' % \
                      (thresholds[idx], metrics['V_Info_split'], metrics['V_Info_merge'], metrics['V_Rand_split'],
                       metrics['V_Rand_merge']))
            else:
                segmentation = seg_metric.astype(np.int32)
            # segmentation = (segmentation * affs_xy).astype(np.int32)
            seg_results.append(segmentation)
            segmentation, _, _ = ev.relabel_from_one(segmentation)
            voi_merge, voi_split = ev.split_vi(segmentation, test_label)
            voi_sum = voi_split + voi_merge
            arand = ev.adapted_rand_error(segmentation, test_label)
            print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                  (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
            f_txt.write('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                        (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
            f_txt.write('\n')
            if arand < best_arand:
                best_arand = arand
                best_idx = idx
        f_txt.close()
        print('Best threshold=%.2f, Best arand=%.6f' % (thresholds[best_idx], best_arand))
        best_seg = randomlabel(seg_results[best_idx]).astype(np.uint16)
        # tifffile.imwrite(os.path.join(out_affs, 'seg.tif'), best_seg)
        f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
        f.create_dataset('main', data=best_seg, dtype=np.uint16, compression='gzip')
        f.close()
        # show 
        # if args.show:
        # best_seg[test_label == 0] = 0
        # draw_fragments_3d(seg_img_path, best_seg, test_label, test_data)
        # for i in range(best_seg.shape[0]):
        #     pred = best_seg[i]
        #     gt = test_label[i]
        #     raw = test_data[i]
        #     draw_fragments_noseeds(seg_img_path, i, pred, gt, raw)
    else:
        for idx, seg_metric in enumerate(seg):
            segmentation = seg_metric.astype(np.int32)
        best_seg = randomlabel(segmentation).astype(np.uint16)
        # tifffile.imwrite(os.path.join(out_affs, 'seg.tif'), best_seg)
        f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
        f.create_dataset('main', data=best_seg, dtype=np.uint16, compression='gzip')
        f.close()
        # if args.show:
        # draw_fragments_3d(seg_img_path, best_seg, test_label, test_data)
        # for i in range(best_seg.shape[0]):
        #     pred = best_seg[i]
        #     raw = test_data[i]
        #     draw_fragments_noseeds(seg_img_path, i, pred, None, raw)
    print('Done')
