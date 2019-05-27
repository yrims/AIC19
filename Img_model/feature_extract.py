import os 
import csv
import time
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
import keras_applications

from PIL import Image
from scipy import io
from random import sample, choices

from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from resnet_152 import resnet152_model
from custom_layers.scale_layer import Scale

# TRACKER = 'GroundTruth'
TRACKER = 'tc_tracklet'
# TRACKER = 'deep_sort'

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--classes', type=int, default=196, help='number of classes')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--gpu_device', default='1', help='gpu device')
parser.add_argument('--model_name', default='ResNet152', help='model choice')
parser.add_argument('--rowdata_path', default='dataset_track1/%s/bbox_img'%TRACKER, help='data have not be preprocessed')
parser.add_argument('--data_path', default='dataset_track1/%s/bbox_img_resize'%TRACKER, help='data have been preprocessed')
parser.add_argument('--data_path_padding', default='dataset_track1/%s/bbox_img_resize_padding'%TRACKER, help='data have been preprocessed with black padding')
parser.add_argument('--feature_path', default='dataset_track1/%s/bbox_img_feature'%TRACKER, help='path to save img names corresponding to img features.')
parser.add_argument('--feature_path_padding', default='dataset_track1/%s/bbox_img_feature_padding'%TRACKER, help='path to save img features with padding.')
parser.add_argument('--color_his_path', default='dataset_track1/%s/bbox_img_histogram'%TRACKER, help='path to save img names corresponding to img features.')
parser.add_argument('--weight_path', default='AICity19_Project/Tracking/MTMC/weights/VeRI_epoch5_v4.h5', help='path to weight.')
opt = parser.parse_args()
print(opt)

S1_SET = ['S1c01', 'S1c02', 'S1c03', 'S1c04', 'S1c05']
S2_SET = ['S2c06', 'S2c07', 'S2c08', 'S2c09']
S3_SET = ['S3c10', 'S3c11', 'S3c12', 'S3c13', 'S3c14', 'S3c15']
S4_SET = [
    'S4c16', 'S4c17', 'S4c18', 'S4c19', 'S4c20',
    'S4c21', 'S4c22', 'S4c23', 'S4c24', 'S4c25',
    'S4c26', 'S4c27', 'S4c28', 'S4c29', 'S4c30',
    'S4c31', 'S4c32', 'S4c33', 'S4c34', 'S4c35',
    'S4c36', 'S4c37', 'S4c38', 'S4c39', 'S4c40',
]
S5_SET = [
    'S5c10',
    'S5c16', 'S5c17', 'S5c18', 'S5c19', 'S5c20',
    'S5c21', 'S5c22', 'S5c23', 'S5c24', 'S5c25',
    'S5c26', 'S5c27', 'S5c28', 'S5c29',
    'S5c33', 'S5c34', 'S5c35', 'S5c36'
]
# ALL_SET = [S2_SET, S5_SET]
ALL_SET = [S1_SET, S3_SET, S4_SET]
# ALL_SET = [S2_SET, S5_SET]
# ALL_SET = [S5_SET]
# ALL_SET = [S1_SET]

class Tracklet_selection():

    def __init__(self, SET):
        self.path = os.path.join(opt.rowdata_path, SET[0][0:2])
        self.tracklet_dict = {}
        self.f_len = len(os.listdir(self.path))
        self.select_n = 3
        self.key_len = 0

    def classify_tracklet(self):
        all_img_file = os.listdir(self.path)
        for i in all_img_file:
            key = '%s_%s'%(i[0:5],i[24:29]) #camid, sctid
            if key not in self.tracklet_dict.keys():
                self.tracklet_dict[key] = [i]
            else:
                self.tracklet_dict[key].append(i)
        self.key_len = len(self.tracklet_dict.keys())
        print('key len:', self.key_len)
        
    def tracklet_select(self): 
        count = 1
        for key in self.tracklet_dict.keys():
            # print('value len:',len(self.tracklet_dict[key]))
            if len(self.tracklet_dict[key]) < self.select_n:
                sp = sample(self.tracklet_dict[key], 1) * self.select_n
            else:
                sp = sample(self.tracklet_dict[key], self.select_n)
            self.tracklet_dict[key] = sp
            print('number of sp: %d / %d'%(count, self.key_len), end='\r')
            # print('sample:',sp)
            count += 1
        return self.tracklet_dict

class Img_preprocess():

    def __init__(self, tracklet_dict, SET):
        self.SET = SET
        self.path = os.path.join(opt.rowdata_path, self.SET[0][0:2])
        self.tracklet_dict = tracklet_dict

    def resize_(self):
        n_tracklet = len(self.tracklet_dict.keys())
        count = 1
        for key in self.tracklet_dict.keys():
            print('number of tracklet resized: %d / %d'%(count, n_tracklet), end='\r')
            count += 1
            for value in self.tracklet_dict[key]:
                img = Image.open(self.path + '/' + value)
                img = img.convert('RGB')
                img = img.resize((opt.image_size, opt.image_size), Image.ANTIALIAS)
                img.save(os.path.join(opt.data_path, SET[0][0:2], value))

    def resize_padding(self):
        n_tracklet = len(self.tracklet_dict.keys())
        count = 1
        for key in self.tracklet_dict.keys():
            print('number of tracklet resized(padding): %d / %d'%(count, n_tracklet),end='\r')
            count += 1
            for value in self.tracklet_dict[key]:
                img = Image.open(self.path + '/' + value)
                new_size = max(img.size)
                new_im = Image.new("RGB", (new_size, new_size))
                new_im.paste(img, (int((new_size-img.size[0])/2), int((new_size-img.size[1])/2)))
                img = new_im.convert('RGB')
                img = img.resize((opt.image_size, opt.image_size), Image.ANTIALIAS)
                img.save(os.path.join(opt.data_path_padding, SET[0][0:2], value))

    def resize(self):
        n_tracklet = len(self.tracklet_dict.keys())
        count = 1
        for key in self.tracklet_dict.keys():
            print('number of tracklet resized: %d / %d'%(count, n_tracklet), end='\r')
            count += 1
            for value in self.tracklet_dict[key]:
                # resize without padding
                img = Image.open(self.path + '/' + value)
                img = img.convert('RGB')
                img = img.resize((opt.image_size, opt.image_size), Image.ANTIALIAS)
                img.save(os.path.join(opt.data_path, SET[0][0:2], value))
                
                # resize with padding
                img = Image.open(self.path + '/' + value)
                new_size = max(img.size)
                new_im = Image.new("RGB", (new_size, new_size))
                new_im.paste(img, (int((new_size-img.size[0])/2), int((new_size-img.size[1])/2)))
                img = new_im.convert('RGB')
                img = img.resize((opt.image_size, opt.image_size), Image.ANTIALIAS)
                img.save(os.path.join(opt.data_path_padding, SET[0][0:2], value))

class Extract_feature():
    
    def __init__(self, SET):
        self.SET = SET
        self.train_img = []
        self.img_name = []

    def load_data(self): 
        print('\nloading data ...')
        for img in os.listdir(opt.data_path + '/' + self.SET[0][0:2]):
            self.img_name.append(img)
            img = Image.open(opt.data_path + '/' + self.SET[0][0:2] + '/' + img)
            img = np.array(img)
            self.train_img.append(img)
        self.train_img = np.array(self.train_img)
        self.train_img = np.reshape(self.train_img, (-1, opt.image_size, opt.image_size, 3))  
        print('img_name len:', len(self.img_name))
        print('img shape:',self.train_img.shape)

        print('loading successfully.')  

    def get_Feature(self):
        GPU_setting(opt)
        print('\nloading model ...')
        if opt.model_name == 'ResNet152':
            # model_ = keras_applications.resnet.ResNet152(include_top=False, weights='imagenet', pooling='avg')
            model_ = resnet152_model(opt.image_size, opt.image_size, 3, opt.classes)
            model_ = Model(inputs=model_.input, outputs=model_.get_layer('avg_pool').output)
            model_.load_weights(opt.weight_path, by_name=True)
            # model_ = load_model(opt.weight_path, custom_objects = {'Scale': Scale})
            print('model loaded.')
        else:
            print('model name error.')

        for i in range(len(self.img_name)):
            pred = model_.predict(self.train_img[i][np.newaxis,], batch_size=1, verbose=2)
            pred = np.reshape(pred,(2048))
            # print('pred shape:', pred.shape)
            # np.save(opt.feature_path_padding + '/%s_res152/%s'%(SET[0][0:2],self.img_name[i]), pred)
            np.save(opt.feature_path + '/%s_res152_VeRI_color/%s'%(SET[0][0:2], self.img_name[i]), pred)
        # model_.summary()

    def get_color_his(self):
        for i in range(len(self.img_name)):
            img = Image.fromarray(self.train_img[i])
            color_his = img.histogram()
            color_his = np.array(color_his)
            np.save(opt.color_his_path + '/%s/%s' % (SET[0][0:2], self.img_name[i]), color_his)

    # def evaluate(self):

def GPU_setting(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_device
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)
    print('GPU setting successful.')

def eval(opt):
    GPU_setting(opt)
    img1 = Image.open(opt.data_path+'/S1/S1c03_frame_00103_SCTid_00010.jpg')
    img1 = np.array(img1)[np.newaxis,:]
    img2 = Image.open(opt.data_path+'/S1/S1c03_frame_00105_SCTid_00010.jpg')
    img2 = np.array(img2)[np.newaxis,:]
    if opt.model_name == 'ResNet152':
        # model_ = keras_applications.resnet.ResNet152(include_top=False, weights='imagenet', pooling='avg')
        model_ = resnet152_model(opt.image_size, opt.image_size, 3, opt.classes)
        model_.load_weights('./model.96-0.89.hdf5')
        model_extract_feature = Model(inputs=model_.input, outputs=model_.get_layer('avg_pool').output)

    else:
        print('model name error.')
    
    model_classify = Model(inputs=model_.input, outputs=model_.get_layer('fc8').output)
    model_classify.load_weights('./model.96-0.89.hdf5', by_name=True)
    class_1 = model_classify.predict(img1)
    class_2 = model_classify.predict(img2)
    print('class of img 1:', np.argmax(class_1))
    print('class of img 2:', np.argmax(class_2))
    pred_1 = model_extract_feature.predict(img1).reshape((2048))
    pred_2 = model_extract_feature.predict(img2).reshape((2048))
    norm_1 = np.linalg.norm(pred_1)
    norm_2 = np.linalg.norm(pred_2)
    pred_1 = pred_1.transpose()
    feature_dot = np.dot(pred_1, pred_2) 
    cos = feature_dot / (norm_1 * norm_2)
    print('cos: ',cos)


if __name__ == '__main__':
    #  eval(opt)
    GPU_setting(opt)
    for SET in ALL_SET:
        ts = Tracklet_selection(SET)
        ts.classify_tracklet()
        tracklet_dict = ts.tracklet_select()
        ip = Img_preprocess(tracklet_dict, SET)
        ip.resize()
        # ip.resize_padding()
        ef = Extract_feature(SET)
        ef.load_data()
        ef.get_Feature()
        # ef.get_color_his()

   