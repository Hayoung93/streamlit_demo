import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import streamlit as st
from stqdm import stqdm
import cv2
import json
import matplotlib.pyplot as plt
from argparse import Namespace
from torchvision.transforms import functional as ttf

# import sys
# print(sys.path)

from demo_gui.face_extraction_for_gui import get_face_coords_from_video, load_face_model
# print(sys.path)
from demo_gui.deepfake_detection_for_gui import deepfake_detection
from demo_gui.efficientnet_pytorch.model import EfficientNet
from demo_gui.post_functions import post_vote
from demo_gui.cam import GradCAMPlusPlus, ClassifierOutputTarget, show_cam_on_image
from model_archs import eff_model_ver2, EffFc_ver2


def run(root, filename, placeholder):
    # face detection
    cfg_fp = "/workspace/Streamlit/deepfake_batch_efffc/face_det/config/config_det.json"
    cfg = json.loads(open(cfg_fp, "rt").read())
    cfg = Namespace(**cfg["det"])
    face_net = load_face_model(cfg)
    face_net.eval()
    bbox_coords, frames, keyframe_ids = get_face_coords_from_video(os.path.join(root, filename), 1, face_net, 0, stqdm, placeholder)

    # fake classification
    fake_net_cp = "/workspace/Streamlit/deepfake_batch_efffc/model_acc_best.pth"
    net = eff_model_ver2("efficientnet_b3", 3)
    feature_dim = net.model_1[0][-1][0].out_channels
    mlp_head = nn.ModuleList([])
    mlp_head.append(nn.Dropout(0.4))
    mlp_head.append(nn.Linear(feature_dim * 2, 2))
    fake_net = EffFc_ver2(net, mlp_head, "cuda")
    fake_net.eval()
    target_layers = fake_net.eff_model.model_1[-1][-1]
    idwise_prob = deepfake_detection(frames, bbox_coords, fake_net, 0, fake_net_cp, False, stqdm)
    is_fake_score = post_vote([*idwise_prob[0].values()])
    is_fake_string = "Fake" if is_fake_score >= 0.5 else "Real"
    is_fake_dict = dict(map(lambda x: (x[0], "Fake" if x[1] >= 0.5 else "Real"), idwise_prob[0].items()))

    # generate box-overlaid video
    out_name = filename.replace(".mp4", "") + "_result.mp4"
    video_fp = os.path.join(root, out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(video_fp.replace("result", "temp"), fourcc, 25.0, (frames[0].shape[1], frames[0].shape[0]))
    writer = cv2.VideoWriter(video_fp.replace("result", "temp"), fourcc, 25.0, (1280, 720))
    cvt_ratio = (1280 / frames[0].shape[1], 720 / frames[0].shape[0])
    key = [*bbox_coords[0].keys()][0]
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (1280, 720))
        bx, by, bw, bh = bbox_coords[i][key]
        bx, by, bw, bh = bx * cvt_ratio[0], by * cvt_ratio[1], bw * cvt_ratio[0], bh * cvt_ratio[1]
        # draw bbox, frame info, video info
        cv2.rectangle(frame, (round(bx), round(by)), (round(bx + bw), round(by + bh)), (0, 0, 255), 3)
        # cv2.rectangle(frame, (round(bx), round(by) - 60), (round(bx) + bw, round(by)), (255, 0, 0), -1)
        if i in is_fake_dict:
            cv2.putText(frame, f"{is_fake_dict[i]} frame", ((round(bx), round(by) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        # cv2.rectangle(frame, (0, 0), (350, 50), (255, 0, 0), -1)
        cv2.putText(frame, f"{is_fake_string} video", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.putText(frame, "Frame {:03d}".format(i), (1090, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    os.system("ffmpeg -i {} -vcodec libopenh264 {}".format(video_fp.replace("result", "temp"), video_fp))
    os.system("rm {}".format(video_fp.replace("result", "temp")))

    # draw score image
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 1.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("fake\nprobability")
    ax.set_xlabel("frame number")
    for face_id in idwise_prob:
        coords = idwise_prob[face_id]
        ax.plot([*coords.keys()], [*coords.values()])
    # TODO: resize image
    img_fp = os.path.join(root, filename.replace(".mp4", "") + "_result.png")
    plt.tight_layout()
    fig.savefig(img_fp, dpi=100)

    cam_dict = {"model": fake_net, "frames": frames, "target_layers": target_layers, "face_coords": bbox_coords, "keyframe_ids": keyframe_ids}

    return video_fp, img_fp, cam_dict


if __name__ == "__main__":
    global uploaded_file
    # placeholder = st.empty()
    # with torch.no_grad():
    #     video_fp, img_fp, cam_dict = run("/workspace/Streamlit/deepfake_batch_efffc/videos", "/workspace/Streamlit/deepfake_batch_efffc/videos/000.mp4", placeholder)
    # col1, col2, col3 = st.columns(3)
    # cams = []
    # video_file = "000.mp4"
    # with GradCAMPlusPlus(model=cam_dict["model"], target_layers=cam_dict["target_layers"]) as cam:
    #     if len(video_file.split("_")) == 3:
    #         targets = [ClassifierOutputTarget(1)]
    #     else:
    #         targets = [ClassifierOutputTarget(0)]
    #     for kid in cam_dict["keyframe_ids"]:
    #         frame = cam_dict["frames"][kid]
    #         face_coord = cam_dict["face_coords"][kid][0]
    #         face_img = frame[round(face_coord[1]):round(face_coord[1] + face_coord[3]), round(face_coord[0]):round(face_coord[0] + face_coord[2]), :]
    #         # cams.append(face_img)
    #         face_img = ttf.resize(torch.from_numpy(face_img / 255.0).permute(2, 0, 1), (300, 300))
    #         input_tensor = torch.stack([face_img] * 2, dim=0).unsqueeze(1).float()
    #         grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    #         cam_image = show_cam_on_image(face_img.permute(1, 2, 0).numpy(), grayscale_cams[0, :], use_rgb=True, colormap=cv2.COLORMAP_JET)
    #         cams.append(cam_image)
    # with col1:
    #     st.image(cams[0])
    # with col2:
    #     st.image(cams[1])
    # with col3:
    #     st.image(cams[2])

    uploaded_file = None
    input_file_root = None
    file_root = None
    # if "results" not in st.session_state:
    #     st.session_state.results = False

    ph_logo = st.container()
    with ph_logo:
        col1_logo, _, _ = st.columns(3)
    ph_input = st.container()
    with ph_input:
        ph_root = st.empty()
        # col1_file, col2_clear = st.columns([5, 1])
        # with col1_file:
        ph_file = st.empty()
        # with col2_clear:
        #     _ = st.text('')
        #     _ = st.text('')
        #     _ = st.text('')
        #     btn_clear = st.empty()
    ph_vid_img = st.container()
    with ph_vid_img:
        ph_vid = st.empty()
        ph_graph = st.empty()
    ph_cam = st.container()
    with ph_cam:
        ph_cam_info = st.empty()
        col1_cam, col2_cam, col3_cam = st.columns(3)

    # def clear_results():
    #     print("clear")
    #     ph_vid_img.empty()
    #     ph_vid.empty()
    #     ph_graph.empty()
    #     ph_cam.empty()
    #     ph_cam_info.empty()
    #     col1_cam.empty()
    #     col2_cam.empty()
    #     col3_cam.empty()
    #     uploaded_file = None


    # logo & base components ------------------------------------------------
    keti_logo_fp = "/workspace/Streamlit/deepfake_batch_efffc/KETI_CI_Korean.png"
    with col1_logo:
        st.image(keti_logo_fp)

    file_root = "/workspace/Streamlit/deepfake_batch_efffc/videos"
    input_file_root = ph_root.text_input('File root', file_root)
    if input_file_root is not None:
        file_root = input_file_root
    # is_clear = btn_clear.button(label="Clear results")
    uploaded_file = ph_file.file_uploader("Choose a file", type="mp4")
    # if is_clear:
    #     clear_results()
    # -----------------------------------------------------------------------

    if uploaded_file is not None:
        video_file = uploaded_file.name
        video_b = uploaded_file.getvalue()
        video_savefp = os.path.join(file_root, video_file)
        with open(video_savefp, "wb") as f:
            f.write(video_b)
        # os.system("ffmpeg -i {} -vf scale=1280:720 {}".format(os.path.join(file_root, video_file), video_savefp.replace(".mp4", "_resized.mp4")))
        # placeholder.video(os.path.join(file_root, video_file.replace(".mp4", "") + "_resized.mp4"))
        with torch.no_grad():
            video_fp, img_fp, cam_dict = run(file_root, video_file, ph_vid)
        # col1, col2 = st.columns([1, 11])
        ph_vid.empty()
        with open(video_fp, "rb") as f:
            ph_vid.video(f)
        ph_graph.image(os.path.join(file_root, img_fp))
        os.system("rm {}".format(video_fp))
        # os.system("rm {}".format(video_fp.replace("_result.mp4", "_resized.mp4")))
        os.system("rm {}".format(img_fp))
        uploaded_file = None
        input_file_root = None
        file_root = None

        cams = []
        with GradCAMPlusPlus(model=cam_dict["model"], target_layers=cam_dict["target_layers"]) as cam:
            if len(video_file.split("_")) == 3:
                targets = [ClassifierOutputTarget(1)]
            else:
                targets = [ClassifierOutputTarget(0)]
            for kk in range(3):
                kid = cam_dict["keyframe_ids"][kk]
                kid_next = cam_dict["keyframe_ids"][kk+1]
                frame = cam_dict["frames"][kid]
                frame_next = cam_dict["frames"][kid_next]
                face_coord = cam_dict["face_coords"][kid][0]
                face_coord_next = cam_dict["face_coords"][kid_next][0]
                face_img = frame[round(face_coord[1]):round(face_coord[1] + face_coord[3]), round(face_coord[0]):round(face_coord[0] + face_coord[2]), :]
                face_img_next = frame_next[round(face_coord_next[1]):round(face_coord_next[1] + face_coord_next[3]), round(face_coord_next[0]):round(face_coord_next[0] + face_coord_next[2]), :]
                # cams.append(face_img)
                face_img = ttf.resize(torch.from_numpy(face_img / 255.0).permute(2, 0, 1), (300, 300))
                face_img_next = ttf.resize(torch.from_numpy(face_img_next / 255.0).permute(2, 0, 1), (300, 300))
                input_tensor = torch.stack([face_img, face_img_next], dim=0).unsqueeze(1).float()
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_image = show_cam_on_image(face_img.permute(1, 2, 0).numpy(), grayscale_cams[0, :], use_rgb=True, colormap=cv2.COLORMAP_JET)
                cams.append(cam_image)
        
        # st.markdown("""---""")
        ph_cam_info.write("GradCAM++ results on 3 keyframes")
        with col1_cam:
            st.image(cams[0])
        with col2_cam:
            st.image(cams[1])
        with col3_cam:
            st.image(cams[2])
        st.session_state.results = True
