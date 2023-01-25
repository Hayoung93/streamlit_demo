import sys
import os
sys.path.insert(0, os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2])))
# print(os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2])))
# print(sys.path)
import cv2
import torch
import numpy as np
import torch
from argparse import Namespace
from tqdm import tqdm
from copy import copy

from face_det.utils.face_detection import load_face_det_model, detect_face_all
from face_det.utils.face_extraction import get_single_face_box_region, get_scale_face_region
from face_det.utils.run_tracker import track_and_postprocess
from face_det.utils.face_from_video import find_closest_id


def load_face_model(cfg):
    if cfg.device == 'cpu':
        device = 'cpu'
    else: 
        device = 'cuda:' + str(cfg.device) if (torch.cuda.is_available() and cfg.device is not None) else 'cpu'
    return load_face_det_model(cfg.pretrained_model_path, device)

def get_face_coords_from_image(real_imgs, face_net, device, stqdm):
    minimum_face_size = 100
    max_batch_size = 16
    frames = torch.from_numpy(np.stack(real_imgs, axis=0)).float()
    total_frame, height, width, _ = frames.shape
    frames = frames.permute(0, 3, 1, 2)  # B, C, H, W
    batch_index = torch.arange(total_frame).split(max_batch_size)
    frame_idx = 0
    bboxes_stack = {}
    pbar = stqdm(range(len(batch_index)), desc="Extracting faces...")
    for bi in batch_index:
        bboxes, _ = detect_face_all(face_net, frames[bi], device)
        pbar.update()
        for box_list in bboxes:
            face_coords = {}
            face_idx = 0
            # for box in box_list:  # there may exist more than 1 bbox
            box = box_list[0]
            face_box, bsize = get_single_face_box_region(box)
            if bsize > minimum_face_size:  # minimum face size
                x, y, size = get_scale_face_region(face_box, width, height, 1.3)
                face_box = (x, y, size, size)
                face_coords[face_idx] = face_box
                face_idx += 1
            bboxes_stack[frame_idx] = face_coords
            frame_idx += 1

    assert len(bboxes_stack) > 0, "No bounding box found"
    return bboxes_stack


def get_face_coords_from_video(video_filename, sub_sampling_rate, face_net, device, stqdm, placeholder):
    print("BATCH COMPUTATION")
    frames = []
    if device != 'cpu':
        device = 'cuda:' + str(device) if (torch.cuda.is_available() and device is not None) else 'cpu'
    cap = cv2.VideoCapture(video_filename)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frame > 0:
        fids = list(range(sub_sampling_rate-1, num_frame, sub_sampling_rate))
        pbar = stqdm(range(len(fids)), desc="Reading frames...")
        frames, keyframe_ids = extract_keyframe_index(cap, fids, 4, 0.3, False, pbar, placeholder)
        cap.release()
        pbar.close()

        # process batch-wise
        face_coords = get_face_coords_from_image(frames, face_net, device, stqdm)
        return face_coords, frames, keyframe_ids
    else:
        raise Exception("There are no frames to read")


def get_face_coords_from_tracker(video_filename, sub_sampling_rate, face_net, device, rm_short_th, nms_len_th, nms_region_th, allow_missing):
    idwise_bboxes, vid_size = track_and_postprocess(videopath=video_filename, device=device, subsampling_rate=sub_sampling_rate,
                                            rm_short_th=rm_short_th, nms_len_th=nms_len_th, nms_region_th=nms_region_th,
                                            allow_missing=allow_missing)
    keys = [*idwise_bboxes.keys()]
    pad_ratio = 0.1
    pad = (vid_size[0] * pad_ratio, vid_size[1] * pad_ratio)
    cap = cv2.VideoCapture(video_filename)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    coords = {}
    if num_frame > 0:
        fids = list(range(sub_sampling_rate-1, num_frame, sub_sampling_rate))
        for fid in tqdm(fids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            _, frame = cap.read()
            if len(keys) > 0:
                for key in keys:
                    if fid in idwise_bboxes[key]:
                        # cut, detect face, assign id
                        x1, y1, x2, y2 = idwise_bboxes[key][fid]
                        y1_crop, y2_crop = int(max(0, y1 - pad[1])), int(y2 + pad[1])
                        x1_crop, x2_crop = int(max(0, x1 - pad[0])), int(x2 + pad[0])
                        frame_buff = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        if (frame_buff.shape[0] == 0) or (frame_buff.shape[1] == 0):
                            continue
                        frame_buff = cv2.cvtColor(frame_buff, cv2.COLOR_BGR2RGB)
                        face_coords = get_face_coords_from_image(frame_buff, face_net, device)
                        if not (fid in coords):
                            coords[fid] = {}
                        x, y, w, h = face_coords[0]
                        coords[fid][key] = (x + x1_crop, y + y1_crop, w, h)
        # Check if all expected frames are in out_face_images
        for fid in fids:
            # key arrangement
            if fid in coords:
                keys = [*coords[fid].keys()]
                new_coords = {}
                for i, key in enumerate(keys):
                    new_coords[i] = coords[fid][key]
                coords[fid] = new_coords
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                _, frame = cap.read()
                frame_buff = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_coords = get_face_coords_from_image(frame_buff, face_net, device)
                for face_coord in face_coords:
                    _id = find_closest_id(face_coord[:-1], idwise_bboxes, fid, sub_sampling_rate)
                    if _id >= 0:
                        if not (fid in coords):
                            coords[fid] = {}
                        coords[fid][_id] = face_coord
                    else:
                        if not (fid in coords):
                            coords[fid] = {}
                            coords[fid][0] = face_coord
                        else:
                            coords[fid][max([*coords[fid].keys()]) + 1] = face_coord
                keys = [*coords[fid].keys()]
                new_coords = {}
                for i, key in enumerate(keys):
                    new_coords[i] = coords[fid][key]
                coords[fid] = new_coords
    else:
        raise Exception("There are no frames to read")
    cap.release()
    return coords


def extract_keyframe_index(cap, fids, max_return_num=None, th=0.3, end_flag=False, pbar=None, placeholder=None):
    """
    input cap: cv2 videocapture instance
    input fids: frame ids to use to find keyframes
    input max_return_num: maximum number of keyframe to extract
    input th: threshold for making keyframe decision
    input end_flag: prevents infinite recursive function call

    output keyframe_index: list of keyframe's index
    """
    frames = []
    if max_return_num is not None:
        if len(fids) <= max_return_num:
            return fids  # No need to compute keyframe
    keyframe_index = []
    diff_values = []
    last_keyframe = None
    for fid in fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        _, frame = cap.read()
        if frame is None:
            print("NoneType frame was grabbed")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame = frame.astype('uint8')
        if last_keyframe is not None:
            difference = (last_keyframe - frame) / 255.0
            difference = difference.mean()
            if difference >= th:
                diff_values.append(difference)
                last_keyframe = frame
                keyframe_index.append(fid)
        else:
            diff_values.append((frame / 255.0).mean())
            last_keyframe = frame
            keyframe_index.append(fid)
        if pbar is not None:
            pbar.update(1)
        if placeholder is not None:
            placeholder.image(frame)
    if max_return_num is None:
        return frames, keyframe_index
    else:
        if len(keyframe_index) > max_return_num:
            diff_sorted = sorted(enumerate(diff_values), key=lambda x: x[1], reverse=True)[:max_return_num]
            max_indices = sorted(list(zip(*diff_sorted))[0])
            return frames, [keyframe_index[i] for i in max_indices]
        elif len(keyframe_index) < max_return_num:
            if not end_flag:
                frames, keyframe_index = extract_keyframe_index(cap, fids, max_return_num, th * 0.5, end_flag=True)
            else:
                keyframe_index = insert_absent_frame(keyframe_index, max_return_num, fids)
            return frames, keyframe_index
        else:
            return frames, keyframe_index


def insert_absent_frame(original_sequence, max_frame, fids):
    """Select more frames if keyframe number doesn't match required number"""
    if len(original_sequence) >= max_frame:
        return original_sequence
    filled_sequence = copy(original_sequence)
    insert_flag = 0
    # compute intervals of origninal sequence, length would be +1
    intervals = list(map(lambda x: x[0]-x[1], zip(original_sequence+[fids[-1]], [fids[0]]+original_sequence)))
    interval_sorted_index = (np.asarray(intervals) * -1).argsort().tolist()
    original_sequence = [fids[0]] + original_sequence + [fids[-1]]
    # from descending order (large interval), search for interval which contains fid and insert it
    for isi in interval_sorted_index:
        keyframe_candidates = np.asarray(fids) - (original_sequence[isi+1] + original_sequence[isi]) / 2
        min_ind = np.abs(keyframe_candidates).argmin()
        if keyframe_candidates[min_ind] < intervals[isi] / 2:
            filled_sequence.insert(isi, fids[min_ind])
            insert_flag = 1
            break
    if insert_flag:
        filled_sequence = insert_absent_frame(filled_sequence, max_frame, fids)
    return filled_sequence
