import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm


def deepfake_detection(frames, face_coords, net, device, checkpoint_path, is_parallel, stqdm):
    checkpoint = torch.load(checkpoint_path)["weight"]
    _checkpoint = {}
    for key in checkpoint.keys():
        _checkpoint[".".join(key.split(".")[:])] = checkpoint[key]
    net.load_state_dict(_checkpoint)
    net.to(device)
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300), 2), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    idwise_probs = {}
    with torch.no_grad():
        pbar = stqdm(range(len(face_coords)), desc="Detecting deepfake")
        for frame_id in tqdm(face_coords):
            frame = frames[frame_id]
            frame_id_2 = frame_id + 1 if frame_id < len(face_coords) - 1 else frame_id - 1  # try to get next frame, else previous frame
            frame_2 = frames[frame_id_2]
            # h, w, _ = frame.shape
            # ratio_h, ratio_w = 300 / h, 300 / w
            # frame = transform(torch.from_numpy(frame / 255).permute(2, 0, 1).float()).to(device)  # Only work for torchvision >= 0.8, which allows tensors as input
            for face_id in face_coords[frame_id]:
                face_coord = face_coords[frame_id][face_id]  # x, y, w, h
                face_coord_2 = face_coords[frame_id_2][face_id]
                if face_coord[2] * face_coord[2] == 0:
                    continue
                face_img = frame[round(face_coord[1]):round(face_coord[1] + face_coord[3]),
                                 round(face_coord[0]):round(face_coord[0] + face_coord[2]), :]
                face_img = transform(torch.from_numpy(face_img / 255).permute(2, 0, 1)).to(device)
                face_img_2 = frame_2[round(face_coord_2[1]):round(face_coord_2[1] + face_coord_2[3]),
                                 round(face_coord_2[0]):round(face_coord_2[0] + face_coord_2[2]), :]
                face_img_2 = transform(torch.from_numpy(face_img_2 / 255).permute(2, 0, 1)).to(device)
                # print(face_img_2.shape)
                # print(torch.cat([face_img, face_img_2], dim=1).shape)
                output = net(torch.stack([face_img, face_img_2], dim=0).unsqueeze(1).float())
                prob = nn.functional.softmax(output)[0, 1].item()  # assuming 2nd(index 1) element represents fake class
                if not (face_id in idwise_probs):
                    idwise_probs[face_id] = {}
                idwise_probs[face_id][frame_id] = prob
            pbar.update()
        pbar.close()
    return idwise_probs