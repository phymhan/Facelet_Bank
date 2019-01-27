# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def shape_to_file(shape, filename):
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    with open(filename, 'w') as f:
        for i in range(0, 68):
            f.write('%f   %f\n' % (shape.part(i).x, shape.part(i).y))


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--root', type=str, default='/media/ligong/Passport/Datasets/SCUT-FBP/images_renamed')
parser.add_argument('--input', type=str, default='/media/ligong/Passport/Datasets/SCUT-FBP/train_all_scut-fbp.txt')
parser.add_argument('--output', type=str, default='/media/ligong/Passport/Datasets/SCUT-FBP/landmarks_renamed')
parser.add_argument('-p', '--shape-predictor', default='shape_predictor_68_face_landmarks.dat', help='path to facial landmark predictor')
opt = parser.parse_args()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(opt.shape_predictor)

with open(opt.input, 'r') as f:
    image_list = [l.rstrip('\n') for l in f.readlines()]

for filename in image_list:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(os.path.join(opt.root, filename))
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    shape = predictor(gray, rects[0])
    shape_to_file(shape, os.path.join(opt.output, '%s.landmark' % filename))

    print('-> %s' % filename)
