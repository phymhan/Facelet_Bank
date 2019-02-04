# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def in_list(list_a, list_b):
    # find if any in A is in B
    for a in list_a:
        if a in list_b:
            return True
    return False


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input', type=str, default='A.txt')
parser.add_argument('--output', type=str, default='B.txt')
parser.add_argument('--error_file', type=str, default='non_face.txt')
opt = parser.parse_args()

with open(opt.input, 'r') as f:
    src_list = [l.rstrip('\n') for l in f.readlines()]

with open(opt.error_file, 'r') as f:
    err_list = [l.rstrip('\n') for l in f.readlines()]

with open(opt.output, 'w') as f:
    for l in src_list:
        l_ = l.split()
        if in_list(l_, err_list):
            print('-> %s' % l)
        else:
            f.write(l + '\n')
