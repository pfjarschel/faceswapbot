#!/usr/bin/python

# Copyright (c) 2019 Paulo Jarschel
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
Faceswap Bot
"""
import sys
import os
import time
import urllib
import json
import cv2
import numpy as np
import facebook
import faceswaplib


class FaceswapBot:

    # Definitions
    timestr = ""
    script_dir = ""
    base_filename = ""
    fslib = faceswaplib.FaceSwapLib()
    im1 = np.zeros((300, 300, 1), np.uint8)
    im2 = np.zeros((300, 300, 1), np.uint8)
    swap1 = np.zeros((300, 300, 1), np.uint8)
    swap2 = np.zeros((300, 300, 1), np.uint8)
    finalim = np.zeros((300, 300, 1), np.uint8)
    json1 = json.dumps({})
    json2 = json.dumps({})

    def __init__(self):
        self.timestr = "%d" % int(time.time()*1000)
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(self.script_dir + "/Images/Swaps", exist_ok=True)
        os.makedirs(self.script_dir + "/Images/Joined", exist_ok=True)
        os.makedirs(self.script_dir + "/Images/Backlog", exist_ok=True)
        self.base_filename = self.script_dir + "/Images/Swaps/" + self.timestr

    def __del__(self):
        True

    def get_random_link_from_spb(self):
        url = "https://www.shitpostbot.com/api/randsource"
        rawjson = urllib.request.urlopen(url).read().decode()
        spbjson = json.loads(rawjson)
        imgurl = self.get_link_from_spb_json(spbjson)
        return imgurl, spbjson

    def get_link_from_spb_json(self, spbjson):
        imgurl = "http://www.shitpostbot.com/" + spbjson["sub"]["img"]["full"]
        return imgurl

    def ensure_faces_from_spb(self):
        l1, j1 = self.get_random_link_from_spb()
        l2, j2 = self.get_random_link_from_spb()

        count1 = 0
        count2 = 0

        im1 = self.fslib.read_im_from_url(l1)
        lm1 = self.fslib.get_landmarks(im1)
        while type(lm1) == bool:
            count1 = count1 + 1
            print("1 is not a face, %.d tries" % count1)
            l1, j1 = self.get_random_link_from_spb()
            im1 = self.fslib.read_im_from_url(l1)
            lm1 = self.fslib.get_landmarks(im1)

        im2 = self.fslib.read_im_from_url(l2)
        lm2 = self.fslib.get_landmarks(im2)
        while type(lm2) == bool:
            count2 = count2 + 1
            print("2 is not a face, %.d tries" % count2)
            l2, j2 = self.get_random_link_from_spb()
            im2 = self.fslib.read_im_from_url(l2)
            lm2 = self.fslib.get_landmarks(im2)

        return im1, im2, lm1, lm2, j1, j2

    def get_imgs_from_backlog(self):
        bl_dir = self.script_dir + "/Images/Backlog"

        files = os.listdir(bl_dir)
        dummy_files = files.copy()

        if len(files) > 0:
            for filename in dummy_files:
                if ".json" not in filename:
                    files.remove(filename)
                    os.remove(bl_dir + "/" + filename)
            del dummy_files

            timestamp = files[0][:-7]
            j1 = json.dumps({})
            j2 = json.dumps({})
            if os.path.isfile(bl_dir + "/%s_A.json" % timestamp) and \
                    os.path.isfile(bl_dir + "/%s_B.json" % timestamp):
                with open(bl_dir + "/%s_A.json" % timestamp) as f:
                    j1 = json.load(f)
                with open(bl_dir + "/%s_B.json" % timestamp) as f:
                    j2 = json.load(f)

                l1 = self.get_link_from_spb_json(j1)
                l2 = self.get_link_from_spb_json(j2)

                im1 = self.fslib.read_im_from_url(l1)
                im2 = self.fslib.read_im_from_url(l2)

                lm1 = self.fslib.get_landmarks(im1)
                if type(lm1) == bool:
                    return [False]
                lm2 = self.fslib.get_landmarks(im2)
                if type(lm2) == bool:
                    return [False]

                os.remove(bl_dir + "/%s_A.json" % timestamp)
                os.remove(bl_dir + "/%s_B.json" % timestamp)

                return [True, im1, im2, lm1, lm2, j1, j2]

            else:
                return [False]
        else:
            return [False]

    def manual_imgs(self, fnam1, fname2):
        # TO-DO
        True

    def save_jsons(self, json1, json2):
        with open(self.base_filename + "_A.json", 'w') as outfile:
            json.dump(json1, outfile)
        with open(self.base_filename + "_B.json", 'w') as outfile:
            json.dump(json2, outfile)

    def save_swaps(self, sw1, sw2):
        cv2.imwrite(self.base_filename + "_A.jpg", sw1)
        cv2.imwrite(self.base_filename + "_B.jpg", sw2)

    def reload_swaps(self):
        # Reload images because reasons, should be looked at later.
        # The swap outputs are all messed up, but are fine once saved to disk. Probably
        # some data format/encoding issues.
        sw1 = cv2.imread(self.base_filename + "_A.jpg", cv2.IMREAD_COLOR)
        sw2 = cv2.imread(self.base_filename + "_B.jpg", cv2.IMREAD_COLOR)

        return sw1, sw2

    def save_final_img(self, im):
        joined_filename = self.script_dir + "/Images/Joined/" + self.timestr + ".jpg"
        cv2.imwrite(joined_filename, im)

        return joined_filename

    def create_comment(self, json1, json2):
        comment = "Sources used: \n" + \
            "1. http://www.shitpostbot.com" + json1["sub"]["link"] + "\n" + \
            "2. http://www.shitpostbot.com" + json2["sub"]["link"]
        print(comment)

        return comment

    def show_image(self, im):
        hf = len(im)
        wf = len(im[0])
        window_h = 768
        window_w = int(wf*(window_h/hf))
        cv2.imshow('a', cv2.resize(im, (window_w, window_h)))

    def post_to_fb(self, impath, com_str):
        token = ""
        with open(self.script_dir + "/token.txt", 'r') as file:
            token = file.read().replace("\n", "")
        cfg = {
            "page_id": "faceswapbot",
            "access_token": token
        }

        api = facebook.GraphAPI(cfg['access_token'])
        post = api.put_photo(image=open(impath, 'rb'), message='')
        api.put_comment(object_id=post['post_id'], message=com_str)

    def run_bot(self, post=True, show=False, backlog=True):
        if backlog:
            res = self.get_imgs_from_backlog()
            if res[0] and len(res) > 6:
                self.im1, self.im2, lm1, lm2, self.json1, self.json2 = \
                    res[1], res[2], res[3], res[4], res[5], res[6]
            else:
                self.im1, self.im2, lm1, lm2, self.json1, self.json2 = self.ensure_faces_from_spb()
        else:
            self.im1, self.im2, lm1, lm2, self.json1, self.json2 = self.ensure_faces_from_spb()
        self.save_jsons(self.json1, self.json2)
        self.swap1 = self.fslib.swap(self.im1, self.im2, lm1, lm2)
        self.swap2 = self.fslib.swap(self.im2, self.im1, lm2, lm1)
        self.save_swaps(self.swap1, self.swap2)
        self.swap1, self.swap2 = self.reload_swaps()
        self.finalim = self.fslib.get_one_image(self.swap1, self.swap2)
        fimpath = self.save_final_img(self.finalim)
        comment = self.create_comment(self.json1, self.json2)

        if post:
            self.post_to_fb(fimpath, comment)
        if show:
            self.show_image(self.finalim)


bot = FaceswapBot()
bot.run_bot(True, False, True)
