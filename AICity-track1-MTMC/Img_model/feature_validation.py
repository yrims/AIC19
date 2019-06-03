import os
import sys
import time
import math
import numpy as np
from PIL import Image
from random import sample
from scipy.spatial.distance import cdist
import argparse


TRACKLET = 'GroundTruth'
parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="D:/users/linjian/workspace/AICity19/dataset_track1/%s/bbox_img_resize"%TRACKLET,
                    help='path of dataset')

parser.add_argument('--feature_save_path',
                    default="D:/users/linjian/workspace/AICity19/dataset_track1/%s/bbox_img_feature"%TRACKLET,
                    help='path to feature saving')

parser.add_argument('--scene',
                    default='S1',
                    help='which scene')

opt = parser.parse_args()


class Feature_validation():
    
    def __init__(self):
        self.path = os.path.join(opt.feature_save_path, opt.scene+'_res152')
        self.feature_dict = {}
        self.f_len = len(os.listdir(self.path))
        self.select_n = 3
        self.key_len = 0

    def classify_feature(self):
        all_feature_file = os.listdir(self.path)
        for i in all_feature_file:
            key = '%s_%s'%(i[0:5],i[24:29]) # camid, sctid
            # key = '%s'%(i[24:29]) # only sctid for GroundTruth
            if key not in self.feature_dict.keys():
                self.feature_dict[key] = [i]
            else:
                self.feature_dict[key].append(i)
        self.key_len = len(self.feature_dict.keys())
        
    def feature_select(self): 
        for key in self.feature_dict.keys():
            if len(self.feature_dict[key]) < self.select_n:
                sp = sample(self.feature_dict[key], 1) * self.select_n
            else:
                sp = sample(self.feature_dict[key], (self.select_n))
            print('key:',key,'values:',sp)
            self.feature_dict[key] = sp

    def cal_diff_id_feature_dist(self, fs): # fs is feature selected
        d = []
        c = 0
        iters = 0
        for key_a in fs.keys():

            for n in range(self.select_n):
                p = os.path.join(self.path, fs[key_a][n])
                if n == 0:
                    fa = np.reshape(np.load(p), (1, 2048))                  
                else:
                    f_ = np.reshape(np.load(p), (1, 2048))
                    fa = np.append(fa, f_, axis=0)

            for key_b in fs.keys():

                if key_b != key_a:

                    for n in range(self.select_n):
                        p = os.path.join(self.path, fs[key_b][n])
                        if n == 0:
                            fb = np.reshape(np.load(p), (1, 2048))
                        else:
                            f_ = np.reshape(np.load(p), (1, 2048))
                            fb = np.append(fb, f_, axis=0)
                    for i in range(self.select_n):
                        Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'braycurtis') 
                        d.append(Y)
                        c += 1
            iters += 1
            loss = np.mean(d)
            std = np.std(d)
            print('\riter: %d loss(diff): %f std: %f'%(iters, loss, std), end='')
        
    def cal_diff_id_feature_dist_GT(self, fs): # fs is feature selected
        d = []
        c = 0
        iters = 0
        for key_a in fs.keys():
            
            for n in range(self.select_n):
                p = os.path.join(self.path, fs[key_a][n])
                if n == 0:
                    fa = np.reshape(np.load(p), (1, 2048))                  
                else:
                    f_ = np.reshape(np.load(p), (1, 2048))
                    fa = np.append(fa, f_, axis=0)

            for key_b in fs.keys():

                if (key_a[0:5] != key_b[0:5]) and (key_a[6:11] != key_b[6:11]):

                    for n in range(self.select_n):
                        p = os.path.join(self.path, fs[key_b][n])
                        if n == 0:
                            fb = np.reshape(np.load(p), (1, 2048))
                        else:
                            f_ = np.reshape(np.load(p), (1, 2048))
                            fb = np.append(fb, f_, axis=0)
                    for i in range(self.select_n):
                        Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'cosine') 
                        d.append(Y)
                        c += 1
            iters += 1
            loss = np.mean(d)
            std = np.std(d)
            print('\riter: %d loss(diff): %f std: %f'%(iters, loss, std), end='')

    def cal_same_id_feature_dist(self, fsa, fsb):
        d = []
        c = 0
        iters = 0
        for key_a in fsa.keys():
            for n in range(self.select_n):
                pa = os.path.join(self.path, fsa[key_a][n])
                pb = os.path.join(self.path, fsb[key_a][n])
                # print('pa:',pa)
                # print('pb:',pb)
                if n == 0:
                    fa = np.reshape(np.load(pa), (1,2048))
                    fb = np.reshape(np.load(pb), (1,2048))
                else:
                    fa_ = np.reshape(np.load(pa), (1,2048))
                    fa = np.append(fa, fa_, axis=0)
                    fb_ = np.reshape(np.load(pb), (1,2048))
                    fb = np.append(fb, fb_, axis=0)

            for i in range(self.select_n):
                # print('fa.shape', fa[np.newaxis,i].shape)
                # print('fb.shape', fb[np.newaxis,i].shape)
                
                Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'cosine') 
                d.append(Y)
                c += 1
            iters += 1
            loss = np.mean(d)
            std = np.std(d)
            print('\riter: %d loss(same): %f std: %f'%(iters, loss, std), end='')
        print('')
    
    def cal_same_id_feature_dist_GT(self, fsa, fsb):
        d = []
        c = 0
        iters = 0
        for key_a in fsa.keys():
            for key_b in fsb.keys():
                if (key_a[0:5] != key_b[0:5]) and (key_a[6:11] == key_b[6:11]):               
                    for n in range(self.select_n):
                        pa = os.path.join(self.path, fsa[key_a][n])
                        pb = os.path.join(self.path, fsb[key_b][n])
                        # print('pa:',pa)
                        # print('pb:',pb)
                        if n == 0:
                            fa = np.reshape(np.load(pa), (1,2048))
                            fb = np.reshape(np.load(pb), (1,2048))
                        else:
                            fa_ = np.reshape(np.load(pa), (1,2048))
                            fa = np.append(fa, fa_, axis=0)
                            fb_ = np.reshape(np.load(pb), (1,2048))
                            fb = np.append(fb, fb_, axis=0)

                    for i in range(self.select_n):
                        # print('fa.shape', fa[np.newaxis,i].shape)
                        # print('fb.shape', fb[np.newaxis,i].shape)
                        Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'cosine') 
                        d.append(Y)
                        c += 1
                    iters += 1
                    loss = np.mean(d)
                    std = np.std(d)
            print('\riter: %d loss(same): %f std: %f'%(iters, loss, std), end='')
        print('')

class Feature_validation_his():
    
    def __init__(self):
        self.path = os.path.join(opt.data_path, opt.scene)
        self.feature_dict = {}
        self.f_len = len(os.listdir(self.path))
        self.select_n = 3
        self.key_len = 0

    def classify_feature(self):
        all_feature_file = os.listdir(self.path)
        for i in all_feature_file:
            # key = '%s_%s'%(i[0:5],i[24:29]) # camid, sctid
            key = '%s'%(i[24:29]) # only sctid for GroundTruth
            if key not in self.feature_dict.keys():
                self.feature_dict[key] = [i]
            else:
                self.feature_dict[key].append(i)
        self.key_len = len(self.feature_dict.keys())
        print('key_len:', self.key_len)
        
    def feature_select(self): 
        for key in self.feature_dict.keys():
            if len(self.feature_dict[key]) < self.select_n:
                sp = sample(self.feature_dict[key], 1) * self.select_n
            else:
                sp = sample(self.feature_dict[key], (self.select_n))
            print('key:',key,'values:',sp)
            self.feature_dict[key] = sp
        
    def cal_diff_id_feature_dist_GT(self, fs): # fs is feature selected
        d = []
        c = 0
        iters = 0
        for key_a in fs.keys():
            
            for n in range(self.select_n):
                p = os.path.join(self.path, fs[key_a][n])
                if n == 0:
                    fa = Image.open(p)
                    fa = fa.histogram()
                    fa = np.reshape(np.array(fa), (1, 768))                  
                else:
                    f_ = Image.open(p)
                    f_ = f_.histogram()
                    f_ = np.reshape(np.array(f_), (1, 768))
                    fa = np.append(fa, f_, axis=0)

            for key_b in fs.keys():

                if (key_a[0:5] != key_b[0:5]) and (key_a[6:11] != key_b[6:11]):

                    for n in range(self.select_n):
                        p = os.path.join(self.path, fs[key_b][n])
                        if n == 0:
                            fb = Image.open(p)
                            fb = fb.histogram()
                            fb = np.reshape(np.array(fb), (1, 768)) 
                        else:
                            f_ = Image.open(p)
                            f_ = f_.histogram()
                            f_ = np.reshape(np.array(f_), (1, 768))
                            fb = np.append(fb, f_, axis=0)

                    for i in range(self.select_n):
                        Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'braycurtis') 
                        d.append(Y)
                        c += 1
            iters += 1
            loss = np.mean(d)
            std = np.std(d)
            print('iter: %d loss(diff): %f std: %f'%(iters, loss, std), end='\r')
    
    def cal_same_id_feature_dist_GT(self, fsa, fsb):
        d = []
        c = 0
        iters = 0
        for key_a in fsa.keys():
            for key_b in fsb.keys():
                if (key_a[0:5] != key_b[0:5]) and (key_a[6:11] == key_b[6:11]):               
                    for n in range(self.select_n):
                        pa = os.path.join(self.path, fsa[key_a][n])
                        pb = os.path.join(self.path, fsb[key_b][n])
                        # print('pa:',pa)
                        # print('pb:',pb)
                        if n == 0:
                            fa = Image.open(pa)
                            fa = fa.histogram()
                            fa = np.reshape(np.array(fa), (1, 768))
                            fb = Image.open(pb)
                            fb = fb.histogram()
                            fb = np.reshape(np.array(fb), (1, 768)) 
                            # fa = np.reshape(np.load(pa), (1,2048))
                            # fb = np.reshape(np.load(pb), (1,2048))
                        else:
                            f_ = Image.open(pa)
                            f_ = f_.histogram()
                            f_ = np.reshape(np.array(f_), (1, 768))
                            fa = np.append(fa, f_, axis=0)
                            f_ = Image.open(pb)
                            f_ = f_.histogram()
                            f_ = np.reshape(np.array(f_), (1, 768))
                            fb = np.append(fb, f_, axis=0)

                    for i in range(self.select_n):
                        # print('fa.shape', fa[np.newaxis,i].shape)
                        # print('fb.shape', fb[np.newaxis,i].shape)
                        Y = cdist(fa[np.newaxis,i], fb[np.newaxis,i], 'braycurtis') 
                        d.append(Y)
                        c += 1
                    iters += 1
                    loss = np.mean(d)
                    std = np.std(d)
            print('iter: %d loss(same): %f std: %f'%(iters, loss, std), end='\r')
        print('')
           

fv = Feature_validation()
fv_ = Feature_validation()
fv.classify_feature()
fv_.classify_feature()
fv.feature_select()
fv_.feature_select()
fv.cal_same_id_feature_dist_GT(fv.feature_dict, fv_.feature_dict)
fv.cal_diff_id_feature_dist_GT(fv.feature_dict)

'''
for j in fv.feature_dict.keys():
    print('key:',j)
    for k in fv.feature_dict[j]:
        print('value:',k)
'''
