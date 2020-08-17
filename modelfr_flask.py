import urllib.request
import json
import time
import cv2
from facebank import load_facebank, prepare_facebank
from face_model import MobileFaceNet, l2_norm
from MTCNN import create_mtcnn_net
from utils.align_trans import *
from utils.util import *
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from flask_script import Manager, Server
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as trans
import torch
import argparse
import sys
import os
import io
import json
from torchvision import models
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detect_model = MobileFaceNet(512).to(device_0)
detect_model.load_state_dict(torch.load(
    'Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
detect_model.eval()
target, name = load_facebank(path='facebank')
parser = argparse.ArgumentParser()
parser.add_argument('--miniface', default=20, type=int)
parser.add_argument('--update', default=False, type=bool)
args = parser.parse_args()
if args.update:
    targets, names = prepare_facebank(
        detect_model, path='facebank')
    print('facebank updated')
else:
    targets, names = load_facebank(path='facebank')
    print('facebank loaded')
URL_fr = ''


def URL2Frame(URL):
    img_arr = np.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    return frame


def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(
        img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized


def MTCNN_NET(frame, scale, device, p_model_path, r_model_path, o_model_path):
    input = resize_image(frame, scale)
    bboxes, landmarks = create_mtcnn_net(
        input, args.miniface, device, p_model_path, r_model_path, o_model_path)

    if bboxes != []:
        bboxes = bboxes / scale
        landmarks = landmarks / scale

    return bboxes, landmarks


def get_bbox(URL, device, targets=target, names=name):
    student_list = []
    frame = URL2Frame(URL)
    try:
        bboxes, landmarks = MTCNN_NET(frame, 0.5, device, 'MTCNN/weights/pnet_Weights',
                                      'MTCNN/weights/rnet_Weights', 'MTCNN/weights/onet_Weights')

        faces = Face_alignment(
            frame, default_square=True, landmarks=landmarks)

        embs = []

        test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        for img in faces:
            embs.append(detect_model(
                test_transform(img).to(device).unsqueeze(0)))

        if embs != []:
            source_embs = torch.cat(embs)
            diff = source_embs.unsqueeze(-1) - \
                targets.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > ((75-156)/(-80))] = -1
            results = min_idx

            for i, b in enumerate(bboxes):
                if results[i] == -1:
                    continue
                box = dict()
                box['id'] = names[results[i] + 1]
                box['x1'] = b[0]
                box['y1'] = b[1]
                box['x2'] = b[2]
                box['y2'] = b[3]
                student_list.append(box)
        return student_list
    except:
        return []


app = Flask(__name__)
CORS(app)
api = Api(app)


class CustomServer(Server):
    def __call__(self, app, *args, **kwargs):
        return Server.__call__(self, app, *args, **kwargs)


manager = Manager(app)
manager.add_command('runserver', CustomServer(host='0.0.0.0'))


class HTTPRequest(Resource):
    def get(self):
        URL_fr = request.args.get('ip', '')
        print(URL_fr)
        student_list = get_bbox(URL_fr, device_0, target, name)
        return {'box': student_list}


api.add_resource(HTTPRequest, '/modelfr')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1219)
