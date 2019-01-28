# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from util import alignface


face_d, face_p = alignface.load_face_detector('models/shape_predictor_68_face_landmarks.dat')
imageA = imutils.resize(cv2.imread('images/A.png'), width=400)
template = alignface.detect_landmarks_from_image(imageA, face_d, face_p)
imageB = imutils.resize(cv2.imread('images/B.png'), width=400)
lmB = alignface.detect_landmarks_from_image(imageB, face_d, face_p)
print(template)
print(lmB)
M, _ = alignface.fit_face_landmarks(lmB, template, landmarks=list(range(68)))
imageB_ = alignface.warp_to_template(imageB, M, (400, 400))
cv2.imwrite('images/B_.png', imageB_)
cv2.imshow("Output", imageB_)
cv2.waitKey(0)
print('hello')
