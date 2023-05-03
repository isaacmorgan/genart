##
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import numpy as np

matplotlib.use('TkAgg')

##
def clip_xy(img: Image.Image, center: (int, int), size: (int, int)) -> Image.Image:
    box = (center[0] - size[0]/2, center[1] - size[1]/2, center[0] + size[0]/2, center[1] + size[1]/2)
    return img.crop(box)

def get_frames(vid: cv2.VideoCapture) -> list[Image.Image]:
    success = True
    images = []
    while success:
        success, image = vid.read()
        if success:
            # CV2 uses BGR instead of RGB jesus christ
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image, 'RGB')
            images.append(image)
    return images

def overlap_1d(b1, b2):
    w_sum = b1[1] - b1[0] + b2[1] - b2[0]
    w_span = max(b1[1], b2[1]) - min(b1[0], b1[0])
    is_overlap = w_sum > w_span
    if is_overlap:
        return w_sum - w_span
    else:
        return 0

def area(b):
    # (x, y, x, y)
    return (b[2] - b[0]) * (b[3] - b[1])

def overlap_2d(b1, b2):
    # xyxy format
    # Return (area, percent)
    w_x = overlap_1d((b1[0], b1[2]), (b2[0], b2[2]))
    w_y = overlap_1d((b1[1], b1[3]), (b2[1], b2[3]))
    area_out = w_x * w_y
    perc_out = area_out / (area(b1) + area(b2) - area_out)
    return area_out, perc_out

def get_player_box(model, img: Image.Image):
    res = model.predict(img, classes=[0, 38])
    # Collect boxes of people that overlap with tennis rackets
    cls = res[0].boxes.cls
    box = res[0].boxes.xyxy
    ind_people = [i for i, x in enumerate(cls) if x == 0]
    ind_rackets = [i for i, x in enumerate(cls) if x == 38]
    players = []
    for i in ind_people:
        for j in ind_rackets:
            area_out, perc_out = overlap_2d(box[i], box[j])
            if area_out > 0.0:
                players.append([box[i]])
    return players

def get_ball_box(model, img: Image.Image):
    res = model.predict(img, classes=[32], conf=0.05)
    return res[0].boxes.xyxy

def constrain_crop(crop, img_size):
    if crop[0] < 0:
        d = 0 - crop[0]
        crop = (crop[0] + d, crop[1], crop[2] + d, crop[3])
    if crop[1] < 0:
        d = 0 - crop[1]
        crop = (crop[0], crop[1] + d, crop[2], crop[3] + d)
    if crop[2] > img_size[0]:
        d = crop[2] - img_size[0]
        crop = (crop[0] - d, crop[1], crop[2] - d, crop[3])
    if crop[3] > img_size[1]:
        d = img_size[1] - crop[3]
        crop = (crop[0], crop[1] - d, crop[2], crop[3] - d)
    return crop

def images_to_video(images: [Image.Image], fname):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fname, fourcc, 30.0, images[0].size)
    for img in images:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

##
model = YOLO("yolov8s.pt")

##
vid = cv2.VideoCapture(r'./data/shots/one.mp4')
images = get_frames(vid)

##
# Get rid of frames that aren't full tennis court view
i0 = 150
# Initialize 'ball' to location of the serving tennis racket
ball = (0, 0)
img = images[i0]
players = get_player_box(model, img)
p = players[0][0]
ball = ((p[0].item() + p[2].item())/2, (p[1].item() + p[3].item())/2)
ball_ind = 0
ball_vel = (0., 0.)
# Record (frame number, ball position) every time the ball is detected
ball_pos = []
hw = 128

for i in range(i0, len(images)):
    img = images[i]
    # For each image, take ball location, crop image around there, find next ball location, repeat
    crop = (ball[0] + ball_vel[0] * (i - ball_ind) - hw, ball[1] + ball_vel[1] * (i - ball_ind) - hw, ball[0] + ball_vel[0] * (i - ball_ind) + hw, ball[1] + ball_vel[1] * (i - ball_ind) + hw)
    crop = constrain_crop(crop, images[0].size)
    img = img.crop(crop)

    balls = get_ball_box(model, img)
    if balls.shape[0] > 0:
        ball = (crop[0] + (balls[0][0] + balls[0][2])/2, crop[1] + (balls[0][1] + balls[0][3])/2)
        ball = (ball[0].item(), ball[1].item())
        ball_ind = i
        ball_pos.append((i, ball))
        if len(ball_pos) > 1:
            ball_vel = ((ball_pos[-1][1][0] - ball_pos[-2][1][0])/(ball_pos[-1][0] - ball_pos[-2][0]), (ball_pos[-1][1][1] - ball_pos[-2][1][1])/(ball_pos[-1][0] - ball_pos[-2][0]))

## Make a video that tracks the ball
images_out = []
# Start video at frame where ball is first visible
f = ball_pos[0][0]

for (b, bn) in zip(ball_pos[:-2], ball_pos[1:]):
    # Estimate velocity between subsequent ball locations
    df = ((bn[1][0] - b[1][0])/(bn[0] - b[0]), (bn[1][1] - b[1][1])/(bn[0] - b[0]))
    
    # For every frame in between recorded locations b and bn, interpolate the ball location and crop the frame to the cropped location
    for (i, j) in enumerate(range(b[0], bn[0])):
        crop = (b[1][0] + i*df[0] - hw, b[1][1] + i*df[1] - hw, b[1][0] + i*df[0] + hw, b[1][1] + i*df[1] + hw)
        crop = constrain_crop(crop, images[0].size)
        img = images[f].crop(crop)
        images_out.append(img)
        f = f + 1

images_to_video(images_out, 'video.mp4')
