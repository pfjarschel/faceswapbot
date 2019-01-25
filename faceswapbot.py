#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

"""
Slightly modified to support images from urls, and to take a random source
image from ShitPostBot, using the http://www.shitpostbot.com/api/randsource
url, and parsing the output to find the full link to the image.

For now, the script attempts several images, until two faces are detected.
This could (hopefully) change, so weirder results are possible.

Modifications by pfjarschel
"""


import cv2
import dlib
import numpy as np
import urllib
import json
import facebook

import sys
import os
import time

# Change these to modify the script behaviour
post = True
show_image = False
get_from_spb = True #No use changing this variable yet

# Definitions
script_dir = os.path.dirname(os.path.realpath(__file__))
PREDICTOR_PATH = script_dir + "/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def get_landmarks_no_exceptions(im):
    rects = detector(im, 1)

    if len(rects) == 0:
        #print("No faces")
        return False
    else:
        face = np.random.randint(len(rects))
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[face]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def read_im_from_url_and_landmarks(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))

    s = get_landmarks_no_exceptions(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def get_link_from_spb():
    url = "https://www.shitpostbot.com/api/randsource"
    rawjson = urllib.request.urlopen(url).read().decode()
    spbjson = json.loads(rawjson)
    imgurl = "http://www.shitpostbot.com/" + spbjson["sub"]["img"]["full"]
    return imgurl, spbjson

def ensure_faces():
    l1, j1 = get_link_from_spb()
    l2, j2 = get_link_from_spb()

    count1 = 0
    count2 = 0

    im1, lm1 = read_im_from_url_and_landmarks(l1)
    while type(lm1) == bool:
        count1 = count1 + 1
        print("1 is not a face, %.d tries" % count1)
        l1, j1 = get_link_from_spb()
        im1, lm1 = read_im_from_url_and_landmarks(l1)

    im2, lm2 = read_im_from_url_and_landmarks(l2)
    while type(lm2) == bool:
        count2 = count2 + 1
        print("2 is not a face, %.d tries" % count2)
        l2, j2 = get_link_from_spb()
        im2, lm2 = read_im_from_url_and_landmarks(l2)

    return im1, im2, lm1, lm2, j1, j2

def perform_swap(img1, img2, lm1, lm2):
    M = transformation_from_points(lm1[ALIGN_POINTS],
                                   lm2[ALIGN_POINTS])

    mask = get_face_mask(img2, lm2)
    warped_mask = warp_im(mask, M, img1.shape)
    combined_mask = np.max([get_face_mask(img1, lm1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(img2, M, img1.shape)
    warped_corrected_im2 = correct_colours(img1, warped_im2, lm1)

    output_im = img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im

def get_one_image(im1, im2):
    # Make images roughly the same height or width
    w1 = len(im1[0])
    h1 = len(im1)
    w2 = len(im2[0])
    h2 = len(im2)

    if h1*w1 > h2*w2:
        ratio = np.min([h1/h2, w1/w2])
        im2 = cv2.resize(im2,(int(w2*ratio),int(h2*ratio)))
        w2 = len(im2[0])
        h2 = len(im2)
    else:
        ratio = np.min([h2/h1, w2/w1])
        im1 = cv2.resize(im1,(int(w1*ratio),int(h1*ratio)))
        w1 = len(im1[0])
        h1 = len(im1)

    # Join the two of them, vertically if the larger is too wide, horizontally
    # if larger is too high
    padding = 0
    imf = np.zeros([128, 128, 3])
    if np.max([w1, w2]) >= np.max([h1, h2]):
        wt = np.max([w1, w2])
        ht = h1 + h2 + padding
        leftover1 = int(np.abs(wt - im1.shape[1])/2)
        leftover2 = int(np.abs(wt - im2.shape[1])/2)
        imf = np.zeros((ht, wt, 3), dtype=np.uint8)
        imf[:im1.shape[0], leftover1:(im1.shape[1] + leftover1), :] = im1
        imf[(im1.shape[0] + padding):, leftover2:(im2.shape[1] + leftover2), :] = im2
    else:
        wt = w1 + w2 + padding
        ht = np.max([h1, h2])
        leftover1 = int(np.abs(ht - im1.shape[0])/2)
        leftover2 = int(np.abs(ht - im2.shape[0])/2)
        imf = np.zeros((ht, wt, 3), dtype=np.uint8)
        imf[leftover1:(im1.shape[0] + leftover1), :im1.shape[1], :] = im1
        imf[leftover2:(im2.shape[0] + leftover2), (im1.shape[1] + padding):, :] = im2

    return imf

# Standard script usage, pointing two images from disk
# im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
# im2, landmarks2 = read_im_and_landmarks(sys.argv[2])

# Just get images from SPB that contain faces
im1, im2, landmarks1, landmarks2, json1, json2 = ensure_faces()

# Swap both ways
swap1 = perform_swap(im1, im2, landmarks1, landmarks2)
swap2 = perform_swap(im2, im1, landmarks2, landmarks1)

timestr = "%d" % int(time.time()*1000)

os.makedirs(script_dir + "/Images/Swaps", exist_ok=True)
base_filename = script_dir + "/Images/Swaps/" + timestr
cv2.imwrite(base_filename + "_A.jpg", swap1)
cv2.imwrite(base_filename + "_B.jpg", swap2)

with open(base_filename + "_A.json", 'w') as outfile:
    json.dump(json1, outfile)
with open(base_filename + "_B.json", 'w') as outfile:
    json.dump(json2, outfile)

# Reload images because reasons, should be looked at later.
# The swap outputs are all messed up, but are fine once saved to disk. Probably
# some data format/encoding issues.
swap1 = cv2.imread(base_filename + "_A.jpg", cv2.IMREAD_COLOR)
swap2 = cv2.imread(base_filename + "_B.jpg", cv2.IMREAD_COLOR)

finalim = get_one_image(swap1, swap2)
os.makedirs(script_dir + "/Images/Joined", exist_ok=True)
joined_filename = script_dir + "/Images/Joined/" + timestr + ".jpg"
cv2.imwrite(joined_filename, finalim)

# Show image
if show_image:
    hf = len(finalim)
    wf = len(finalim[0])
    window_h = 768
    window_w = int(wf*(window_h/hf))
    cv2.imshow('a', cv2.resize(finalim,(window_w, window_h)))

# End
sources_string = "Sources used: \n" + \
    "1. http://www.shitpostbot.com" + json1["sub"]["link"] + "\n" + \
    "2. http://www.shitpostbot.com" + json2["sub"]["link"]
print(sources_string)

# But not before posting
if post:
    token = ""
    with open("token.txt", 'r') as file:
        token = file.read().replace("\n", "")
    cfg = {
    "page_id"      : "faceswapbot",
    "access_token" : token
    }

    api = facebook.GraphAPI(cfg['access_token'])
    post = api.put_photo(image=open(joined_filename, 'rb'), message='')
    comment = api.put_comment(object_id=post['post_id'], message=sources_string)
