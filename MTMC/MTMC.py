'''
################################
#    multi-camera tracking     #
################################
'''

import os
import cv2
import math
import time
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.spatial.distance import cdist

np.set_printoptions(threshold=np.inf)

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
# ALL_SET = [S1_SET, S2_SET, S3_SET, S4_SET, S5_SET]
# ALL_SET = [S2_SET, S5_SET]
ALL_SET = [S1_SET, S3_SET, S4_SET]
TRACKLET = 'tc_tracklet'
# TRACKLET = 'GroundTruth'
# MODEL = 'res152'
MODEL = 'res152_VeRI_color'
# MODEL = 'mgn_veri'
EVALDIR = 'eval_0510_v2'

RES = 'res/%s'%TRACKLET
img_path = RES + '/bbox_img_resize_padding'
img_path_no_padding = RES + '/bbox_img_resize'
feature_path_padding = RES + '/bbox_img_feature_padding'
feature_path = RES + '/bbox_img_feature'
histogram_path = RES + '/bbox_img_histogram'
save_path = 'res/MCT'
VERBOSE = 0 # 0(only epoch), 1(normal), 2(detail print)

class MTMC():
    # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y , 
    # end_x, end_y, start_time, end_time, left, top, width, heigth
    def __init__(self, SET):
        self.SET = SET
        self.num_cam = len(self.SET)
        self.all_cam_num = 40
        self.PATH = img_path_no_padding
        # self.PATH = img_path
        # self.RESIZE_PATH = img_path_no_padding
        self.HISTOGRAM_PATH = histogram_path
        self.sct = None
        self.feature = []
        self.color_his = {}
        self.tracklet_dict = {}
        self.tracklet_coor_dict = {}
        self.trajectory_dict = {}
        self.trajectory_sample_dict = {}
        self.camera_id_list = []
        self.tracklet_num = 0
        self.trajectory_num = 0
        self.trajectory_selected_num = 100
        self.mtmc_id_table = None
        self.mtmc_loss_table = None
        self.mtmc_id = {}
        self.mtmc_file = []


        
    def load_sct(self):
        SCT_file = os.path.join(RES, '%s.txt'%self.SET[0][0:2])
        self.sct = np.loadtxt(SCT_file, delimiter=',')
        print('--------------------')
        print('|        %s        |'%self.SET[0][0:2])
        print('--------------------')
        print('SCT shape: ', self.sct.shape)
        
    def load_feature(self):
        # feature_dir = feature_path_padding + '/%s_%s'%(self.SET[0][0:2], MODEL)
        feature_dir = feature_path + '/%s_%s'%(self.SET[0][0:2], MODEL)
        feature_list = os.listdir(feature_dir)
        for fname in feature_list:    
            self.feature.append(np.load(feature_dir + '/' + fname))
        print('feature length: ',len(self.feature))

    def trajectory_classify(self):
        for i in range(self.sct.shape[0]):
            row = self.sct[i, :]
            frame, time = int(row[0]), row[1]
            cam = int(row[2])
            sct_id = int(row[3])
            lat, lon = row[4], row[5]
            left, top, width, height = int(row[12]), int(row[13]), int(row[14]), int(row[15])
            start_x, start_y, end_x, end_y = row[6], row[7], row[8], row[9]
            start_time, end_time = row[10], row[11]

            key = '%sc%02d_%05d'%(self.SET[0][0:2], cam, sct_id)
            # print('key:', key)
            if key not in self.trajectory_dict.keys():
                self.trajectory_dict[key] = [(lat, lon, time, frame, left, top, width, height, cam, 
                                              start_x, start_y, end_x, end_y, start_time, end_time )]
            else:
                self.trajectory_dict[key].append((lat, lon, time, frame, left, top, width, height, cam, 
                                            start_x, start_y, end_x, end_y, start_time, end_time ))

        self.trajectory_num = len(self.trajectory_dict.keys())
        print('total trajectory number: %d'%len(self.trajectory_dict.keys()))

    def trajectory_sampling(self):
        for _, key in enumerate(self.trajectory_dict.keys()):
            traj = self.trajectory_dict[key].copy()
            if len(traj) == 1:
                traj = traj*2
            # print('traj:', traj)
            sp = []
            
            # print('len(traj):',len(traj))
            if len(traj) >= self.trajectory_selected_num:
                interval = math.ceil(len(traj) / self.trajectory_selected_num)
                traj = traj[0::interval]
                #print('len sp :',len(sp))
                sp = traj
                while len(sp) < self.trajectory_selected_num:
                    
                    for j in range(1, len(sp)):
                        
                        if len(sp) < self.trajectory_selected_num:
                            sp_temp = [(sp[j-1][k] + sp[j][k]) / 2 for k in range(len(sp[j]))]
                            sp.append(sp[j-1])
                            sp.append(sp_temp)
                            sp.append(sp[j])
                        
                        else:
                            break
                sp = sp[0:self.trajectory_selected_num]
                # print('num of trajectory 1 %d: '%i, len(sp))
            else:
                sp = traj
                while len(sp) < self.trajectory_selected_num:
                    
                    for j in range(1, len(sp)):
                        
                        if len(sp) < self.trajectory_selected_num:
                            sp_temp = [(sp[j-1][k] + sp[j][k]) / 2 for k in range(len(sp[j]))]
                            sp.append(sp[j-1])
                            sp.append(sp_temp)
                            sp.append(sp[j])
                            
                        else:
                            break
                sp = sp[0:self.trajectory_selected_num]
                # print('num of trajectory 2 %d: '%i, len(sp))
            self.trajectory_sample_dict[key] = sp   
            # sp = np.array(sp, dtype=float)
            # np.savetxt('./dataset_track1/%s/traj_data/%s/%s.txt'%(TRACKLET, self.SET[0][0:2], key), sp[:, 0:3], fmt='%.10f,%.10f,%f,%f,%d,%d,%d,%d,%d', delimiter=",", newline='\n')
           
    def tracklet_classify(self):
        all_tracklet_img = os.listdir(self.PATH + '/%s'%self.SET[0][0:2])
        for i in all_tracklet_img:
            key = '%s_%s'%(i[0:5],i[24:29]) # camid, sctid
            # key = '%s'%(i[24:29]) # only sctid for GroundTruth
            if key not in self.tracklet_dict.keys():
                self.tracklet_dict[key] = [i]
            else:
                self.tracklet_dict[key].append(i)
        self.tracklet_num = len(self.tracklet_dict.keys())
        print('total tracklet number: %d'%len(self.tracklet_dict.keys()))

    def tracklet_coor(self):
        for i, key in enumerate(self.tracklet_dict.keys()):
            self.tracklet_coor_dict[key] = i

    def match(self):
        # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y , 
        # end_x, end_y, start_time, end_time, left, top, width, heigth
        lambda_img = 5
        lambda_traj = 1
        lambda_dir = 1
        lambda_tt = 4.5
        # thres = 0.3
      
        frame = self.sct[:,0]
        time = self.sct[:,1]
        cam_id = self.sct[:,2]
        car_id = self.sct[:,3]
        latitude = self.sct[:,4]
        longitude = self.sct[:,5]
        start_x = self.sct[:,6]
        start_y = self.sct[:,7]
        end_x = self.sct[:,8]
        end_y = self.sct[:,9]
        start_time = self.sct[:,10]
        end_time = self.sct[:,11]
        left, top, width, height = self.sct[:,12], self.sct[:,13], self.sct[:,14], self.sct[:,15]
        self.mtmc_id_table = np.chararray((self.tracklet_num, self.num_cam), itemsize=11, unicode=True)
        # self.mtmc_id_table[:] = ''
        self.mtmc_loss_table = np.ones((self.tracklet_num, self.num_cam))
        self.mtmc_loss_table[:] = 999999999
        print('id_table shape:',self.mtmc_id_table.shape)
        print('loss_table shape:',self.mtmc_loss_table.shape)
        # feature_dir = feature_path_padding + '/%s_%s'%(self.SET[0][0:2], MODEL)
        feature_dir = feature_path + '/%s_%s' % (self.SET[0][0:2], MODEL)
        histogram_dir = histogram_path + '/%s' % self.SET[0][0:2]
        center_idx = int(0.75 * self.trajectory_selected_num)

        # getting tracklet a
        for a, key_a in enumerate(self.tracklet_dict.keys()):
            # load feature a 
            # print('key a:',key_a)
            feature_a = np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_a][0]), 2048)
            np.append(feature_a, np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_a][1]), 2048))
            np.append(feature_a, np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_a][2]), 2048))
            # print('feature a shape:', feature_a.shape)
            # print('feature a :', feature_a)
            # load color histogram
            histogram_a = np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_a][0]), 768)
            np.append(histogram_a, np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_a][1]), 768))
            np.append(histogram_a, np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_a][2]), 768))
            
            # print('shape fa:', feature_a.shape)
            
            if VERBOSE == 0:
                print('epoch :%4d/%4d'%((a+1), self.tracklet_num), end='\r')
            
            # camera id
            cam_a = int(key_a[3:5]) - 1
            
            # center of trajectory in car a (lat, long, time)
            traj_a = self.trajectory_sample_dict[key_a]
            center_traj_a = traj_a[center_idx]

            # initialize variable
            count = 1
            min_loss_total = np.ones(self.num_cam)
            min_loss_img = np.ones(self.num_cam)
            min_loss_his = np.ones(self.num_cam)
            min_loss_traj = np.ones(self.num_cam)
            min_loss_dir = np.ones(self.num_cam)
            min_loss_tt = np.ones(self.num_cam)
            min_loss_total[:] = 999999999
            min_loss_his[:] = 999999999
            min_loss_img[:] = 999999999
            min_loss_traj[:] = 999999999
            min_loss_dir[:] = 999999999
            min_loss_tt[:] = 999999999
            min_id = np.chararray((self.num_cam), itemsize=11, unicode=True)
            
            # getting tracklet b
            for b, key_b in enumerate(self.tracklet_dict.keys()):
                
                # camera id
                cam_b = int(key_b[3:5]) - 1
                
                # center of trajectory in car b (lat, long, time)
                traj_b = self.trajectory_sample_dict[key_b]
                center_traj_b = traj_b[center_idx]

                # distance between center coordinate a and center coordinate b
                dis_ab = gp_dist_to_meter(center_traj_a[0], center_traj_a[1], center_traj_b[0], center_traj_b[1])
                #print('key b:',key_b)
                
                # time difference
                time_dif_ab = abs(center_traj_a[2] - center_traj_b[2])
                # camparing with only different camera and in 800 meters and in 3 minutes.
                # if cam_b != cam_a and dis_ab < 800 and time_dif_ab < 300: 
                if cam_b != cam_a and dis_ab < 400 and time_dif_ab < 30:     
                    # load feature a 
                    feature_b = np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_b][0]), 2048)
                    np.append(feature_b, np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_b][1]), 2048))
                    np.append(feature_b, np.reshape(np.load(feature_dir + '/%s.npy'%self.tracklet_dict[key_b][2]), 2048))
                    
                    # load color histogram
                    #histogram_b = np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_b][0]), 768)
                    #np.append(histogram_b, np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_b][1]), 768))
                    #np.append(histogram_b, np.reshape(np.load(histogram_dir + '/%s.npy'%self.tracklet_dict[key_b][2]), 768))

                    # print('key_%d:'%count, key_b)

                    #    0    1     2     3     4     5     6      7      8
                    # (lat, lon, frame, time, left, top, width, height, cam, 
                    #  start_x, start_y, end_x, end_y, start_time, end_time )
                    #     9        10     11     12        13         14
                    
                    # old dir arg
                    # cur_start_x, cur_start_y, cur_end_x, cur_end_y, cmp_start_x, cmp_start_y
                    
                    # new dir arg
                    # cur_start_x, cur_start_y, cur_end_x, cur_end_y, cmp_start_x, cmp_start_y, cmp_end_x, cmp_end_y

                    # travel time arg
                    # cur_start_x, cur_start_y, cur_end_x, cur_end_y, 
                    # cmp_start_x, cmp_start_y, start_time, end_time, 
                    # ori_time, time

                    idx_b = self.SET.index(key_b[0:5])

                    if (self.SET == S1_SET or self.SET == S2_SET):
                        if time_dif_ab < 15:
                            
                            
                            loss_img = lambda_img * loss_appearance(feature_a, feature_b)
                            #loss_his =loss_appearance_his(histogram_a, histogram_b)
                            # loss_traj = lambda_traj * loss_trajectory_smooth(traj_a, traj_b)
                            loss_dir = lambda_dir * loss_direction(self.trajectory_dict[key_a][0][9], self.trajectory_dict[key_a][0][10],
                                                    self.trajectory_dict[key_a][0][11], self.trajectory_dict[key_a][0][12],
                                                    self.trajectory_dict[key_b][0][9], self.trajectory_dict[key_b][0][10],
                                                    self.trajectory_dict[key_b][0][11], self.trajectory_dict[key_b][0][12])
                            '''
                            loss_tt = loss_travel_time(self.trajectory_dict[key_a][0][9], self.trajectory_dict[key_a][0][10],
                                                    self.trajectory_dict[key_a][0][11], self.trajectory_dict[key_a][0][12],
                                                    self.trajectory_dict[key_b][0][9], self.trajectory_dict[key_b][0][10],
                                                    self.trajectory_dict[key_a][0][13], self.trajectory_dict[key_a][0][14], 
                                                    self.trajectory_dict[key_a][0][3], self.trajectory_dict[key_b][0][3])
                            '''
                            loss_tt, velocity_a ,velocity_b = loss_travel_time(dis_ab, time_dif_ab, traj_a[int(0.75*self.trajectory_selected_num)], traj_a[int(0.95*self.trajectory_selected_num)],
                                                                                traj_b[int(0.75*self.trajectory_selected_num)], traj_b[int(0.95*self.trajectory_selected_num)])
                            # loss_tt = lambda_tt * abs(time_dif_ab)
                            loss_tt = lambda_tt * loss_tt
                            loss_total = loss_img + loss_dir +  loss_tt 
                            if VERBOSE == 2:
                                print('-------------------------------------------------------------------------')
                                print('a:%s with b:%s' % (key_a, key_b))
                                print('total: %.5f, img: %.5f, dir: %.5f, tt: %.5f, vel_a: %.5f, vel_b: %.5f' % (loss_total, loss_img, loss_dir, loss_tt, velocity_a, velocity_b))
                                print('-------------------------------------------------------------------------')
                            # replace the tracklet having min loss
                            if loss_total < min_loss_total[idx_b] and loss_img < 1.5 and loss_dir < 0.3 and loss_tt < 0.36:
                                min_loss_total[idx_b] = loss_total
                                # min_loss_total[idx_b] = loss_img
                                # min_loss_traj[idx_b] = loss_traj
                                min_loss_img[idx_b] = loss_img
                                min_loss_dir[idx_b] = loss_dir
                                min_loss_tt[idx_b] = loss_tt
                                min_id[idx_b] = key_b
                                min_cam = int(min_id[idx_b][3:5]) - 1
                                # print('replace the tracklet having min loss!')
                            count += 1

                    else:
                        
                        # loss_img = loss_appearance(feature_a, feature_b) + loss_appearance(histogram_a, histogram_b)
                        loss_img = 3 * loss_appearance(feature_a, feature_b)
                        # loss_his = loss_appearance_his(histogram_a, histogram_b)
                        # loss_traj = loss_trajectory_smooth(traj_a, traj_b)
                        loss_dir = 0.5 * lambda_dir *loss_direction(self.trajectory_dict[key_a][0][9], self.trajectory_dict[key_a][0][10],
                                                    self.trajectory_dict[key_a][0][11], self.trajectory_dict[key_a][0][12],
                                                    self.trajectory_dict[key_b][0][9], self.trajectory_dict[key_b][0][10],
                                                    self.trajectory_dict[key_b][0][11], self.trajectory_dict[key_b][0][12])
                        loss_tt, velocity_a, velocity_b = loss_travel_time(dis_ab, time_dif_ab, traj_a[int(0.75*self.trajectory_selected_num)], traj_a[int(0.95*self.trajectory_selected_num)],
                                                                          traj_b[int(0.75*self.trajectory_selected_num)], traj_b[int(0.95*self.trajectory_selected_num)])
                        loss_tt = loss_tt * 0.1
                        loss_total = loss_img + loss_dir + loss_tt
                        if VERBOSE == 2:
                                print('-------------------------------------------------------------------------')
                                print('a:%s with b:%s' % (key_a, key_b))
                                print('total: %.5f, img: %.5f, dir: %.5f, tt: %.5f, vel_a: %.5f, vel_b: %.5f' % (loss_total, loss_img, loss_dir, loss_tt, velocity_a, velocity_b))
                                print('-------------------------------------------------------------------------')
                        
                        # print('img: %.5f, dir: %.5f, tt: %.5f' % (loss_img, loss_dir, loss_tt))
                        # replace the tracklet having min loss
                        if loss_total < min_loss_total[idx_b] and loss_img < 0.7 and loss_dir < 1.5 and loss_tt < 0.6:
                            min_loss_total[idx_b] = loss_total
                            # min_loss_total[idx_b] = loss_img
                            min_loss_img[idx_b] = loss_img
                            # min_loss_dir[idx_b] = loss_dir
                            min_loss_tt[idx_b] = loss_tt
                            min_id[idx_b] = key_b
                            min_cam = int(min_id[idx_b][3:5]) - 1
                            # print('replace the tracklet having min loss!')
                        count += 1
            if VERBOSE >= 1:
                print('######################################################################################')
                print('%s is associated to | ' %(key_a), end='')
            for c, _ in enumerate(min_loss_img):
                if self.SET == S1_SET or self.SET == S2_SET:
                    if min_loss_img[c] < self.mtmc_loss_table[a, c] :
                        self.mtmc_loss_table[a, c] = min_loss_img[c]
                        self.mtmc_id_table[a, c] = min_id[c]
                    # print('min loss:', min_loss[c])
                    if VERBOSE >= 1:
                        print(min_id[c], end=' ')
                    # print('min id:', min_id[c])
                else:
                    if min_loss_img[c] < self.mtmc_loss_table[a, c]:
                        self.mtmc_loss_table[a, c] = min_loss_img[c]
                        self.mtmc_id_table[a, c] = min_id[c]
                    if VERBOSE >= 1:
                        print(min_id[c], end=' ')
            if VERBOSE >= 1:
                print('\n######################################################################################')
            '''           
            for c, _ in enumerate(min_loss_total):
                if self.SET == S1_SET or self.SET == S2_SET:
                    if min_loss_total[c] < self.mtmc_loss_table[a, c] and min_loss_img[c] < 0.5 and min_loss_dir[c] < 1 and min_loss_tt < 10:
                        self.mtmc_loss_table[a, c] = min_loss_total[c]
                        self.mtmc_id_table[a, c] = min_id[c]
                    # print('min loss:', min_loss[c])
                    # print('min id:', min_id[c])
                else:
                    if min_loss_total[c] < self.mtmc_loss_table[a, c] and min_loss_img[c] < 0.5 and min_loss_dir[c] < 1.2 and min_loss_tt < 30:
                        self.mtmc_loss_table[a, c] = min_loss_total[c]
                        self.mtmc_id_table[a, c] = min_id[c]
            '''
    '''
    def table_preprocess(self):
        for key in self.tracklet_dict:
            self.mtmc_id[key] = 0
        for i, key in enumerate(self.tracklet_dict.keys()):
            
    '''        

    def id_assign(self):
        if (self.mtmc_id_table is None) and (self.mtmc_loss_table is None):
            self.mtmc_id_table = np.loadtxt('./mtmc_id_table.txt', dtype=str, delimiter=',')
            self.mtmc_loss_table = np.loadtxt('./mtmc_loss_table.txt', delimiter=',')
            print('load table.')
        #　self.mtmc_id = np.zeros(self.tracklet_num)
        for key in self.tracklet_dict:
            self.mtmc_id[key] = 0
        id_count = 1

        # id table preprocess
        for i, key in enumerate(self.tracklet_dict.keys()):
            idxs = []
            min_tracklet_loss = 999999999
            min_idx = None
            for a in range(self.tracklet_num):
                for b in range(self.num_cam):
                    if self.mtmc_id_table[a, b] == key:
                        idxs.append([a, b])
            if len(idxs) != 0:
                for idx in idxs:
                    if self.mtmc_loss_table[idx[0], idx[1]] < min_tracklet_loss:
                        min_tracklet_loss = self.mtmc_loss_table[idx[0], idx[1]]
                        min_idx = idx
            for idx in idxs:
                if idx != min_idx:
                    self.mtmc_loss_table[idx[0], idx[1]] = 999999999
                    self.mtmc_id_table[idx[0], idx[1]] = ''
            
            row = self.mtmc_loss_table[i]
            if self.mtmc_id[key] == 0 and np.count_nonzero(row == 999999999) < self.num_cam: # haven't assigned ID and have matched other tracklets.
                for j in range(self.num_cam):
                    if row[j] < 999999999:
                        t_name = self.mtmc_id_table[i, j]



        for i, key in enumerate(self.tracklet_dict.keys()):
            row = self.mtmc_loss_table[i]
            # print('row:', row)
            if self.mtmc_id[key] == 0 and np.count_nonzero(row == 999999999) < self.num_cam: # haven't assigned ID and have matched other tracklets.
                self.mtmc_id[key] = id_count
                for j in range(self.num_cam):
                    if row[j] < 999999999:
                        t_name = self.mtmc_id_table[i, j]
                        # print('t_name:',t_name)
                        self.mtmc_id[t_name] = id_count
                        # print('mtmc_id[%s] :'%t_name, self.mtmc_id[t_name])
                id_count += 1
            elif self.mtmc_id[key] > 0:
                row = self.mtmc_loss_table[i]
                for j in range(self.num_cam):
                    if row[j] < 999999999:
                        t_name = self.mtmc_id_table[i, j]
                        # print('t_name:',t_name)
                        self.mtmc_id[t_name] = self.mtmc_id[key]

        print(self.mtmc_id)

    def save_mtmc(self):
        # submission format:
        # <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>

        # trajectory format:
        # lat, lon, time, frame, left, top, width, height
        # lat, lon, time, frame, left, top, width, height, cam, start_x, start_y, end_x, end_y, start_time, end_time 
        for i, key in enumerate(self.tracklet_dict.keys()):
            if self.mtmc_id[key] > 0:
                for j, item in enumerate(self.trajectory_dict[key]):
                    row = [int(key[3:5]), int(self.mtmc_id[key]), int(self.trajectory_dict[key][j][3]), 
                           int(self.trajectory_dict[key][j][4]), int(self.trajectory_dict[key][j][5]),
                           int(self.trajectory_dict[key][j][6]), int(self.trajectory_dict[key][j][7]),
                           self.trajectory_dict[key][j][1], self.trajectory_dict[key][j][0] ]  
                    self.mtmc_file.append(row)
        self.mtmc_file = np.array(self.mtmc_file).reshape((-1, 9))
        print(self.mtmc_file.shape)
        self.mtmc_file = self.mtmc_file[self.mtmc_file[:,2].argsort()]
        np.savetxt(save_path + '/' + EVALDIR + '/' + 'mtmc_%s.txt'%self.SET[0][0:2], self.mtmc_file, fmt='%d,%d,%d,%d,%d,%d,%d,%f,%f', delimiter=',')


def gp_dist_to_meter (lat1, lon1, lat2, lon2):
    R = 6378.137 #Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c * 1000 #meters
    return d    

def loss_appearance(fa, fb):
    # Y = cdist(fa[np.newaxis, :], fb[np.newaxis, :], 'braycurtis') 
    loss = cdist(fa[np.newaxis, :], fb[np.newaxis, :], 'cosine') 
    return loss

def loss_trajectory_smooth(traj_a, traj_b):
    # traj (lat, long, time)
    traj_a = np.array(traj_a)
    traj_a = traj_a[:, 0:3]
    norm_a = np.linalg.norm(traj_a, ord=2, axis=0)
    traj_a_norm = traj_a / norm_a
   
    traj_b = np.array(traj_b)
    traj_b = traj_b[:, 0:3]
    
    norm_b = np.linalg.norm(traj_b, ord=2, axis=0)
    traj_b_norm = traj_b / norm_b
    
    loss = np.mean(np.square((traj_a - traj_b)))
    # print('traj_a shape:', traj_a.shape)
    # print('traj_b shape:', traj_b.shape)
    # loss = cdist(traj_a_norm, traj_b_norm, 'euclidean')
    # loss = np.mean(loss)
    # print('loss traj shape:', loss.shape)

    # loss = math.exp(0.00001*loss) - 1
    return loss

def loss_direction(cur_start_x, cur_start_y, cur_end_x, cur_end_y, cmp_start_x, cmp_start_y, cmp_end_x, cmp_end_y):
        loss = []      #　1              1           1          1        1           1
        eps = 0.000000000001
    
        cmp_start = np.array([cmp_start_x , cmp_start_y])  # (2,)
        cmp_end = np.array([cmp_end_x, cmp_end_y])
        ori_start = np.array([cur_start_x, cur_start_y])  # (2,)
        ori_end = np.array([cur_end_x, cur_end_y]) # (2,)
        cmp_direction = cmp_end - cmp_start # (2,)
        cur_direction =  ori_end - ori_start # (2,)
        loss = cdist(cmp_direction[np.newaxis, :], cur_direction[np.newaxis, :], 'cosine') 
        
        return loss

def loss_travel_time(dis, time_dif, traj_a_start, traj_a_end, traj_b_start, traj_b_end):
        eps = 0.001
        d_a = abs(gp_dist_to_meter(traj_a_start[0], traj_a_start[1], traj_a_end[0], traj_a_end[1]))
        t_a = abs(traj_a_end[2] - traj_a_start[2])
        v_a = d_a / (t_a + eps)
        d_b = abs(gp_dist_to_meter(traj_b_start[0], traj_b_start[1], traj_b_end[0], traj_b_end[1]))
        t_b = abs(traj_b_end[2] - traj_b_start[2])
        v_b = d_b / (t_b + eps)
        travel_t = dis / (v_a + eps)
        true_t = time_dif
        loss = abs(travel_t - true_t)
        if loss < 50:
            loss = math.exp(0.01*loss) - 1
        else:
            loss = math.log(loss)
        
        return loss, v_a, v_b

class Eval():
    def __init__(self, SET):
        self.SET = SET
        self.txt = None

    def load_txt(self):
        # c001_train
        self.txt = np.loadtxt(save_path + '/' + EVALDIR + '/' + 'mtmc_%s.txt'%self.SET[0][0:2], delimiter=',')
    
    def split_txt(self):
        for cam in self.SET:
            cam_txt = []
            print('cam:', int(cam[3:]))
            for i in range(self.txt.shape[0]):
                if self.txt[i][0] == int(cam[3:]):
                    tmp = self.txt[i][1]
                    self.txt[i][1] = self.txt[i][2]
                    self.txt[i][2] = tmp
                    cam_txt.append(self.txt[i][1:])
            np.savetxt(save_path + '/' + EVALDIR + '/' + 'c%03d_train.txt'%int(cam[3:]), cam_txt, fmt='%d,%d,%d,%d,%d,%d,%d,%d')

if __name__ == '__main__':
    tStart = time.time()
    for SET in ALL_SET:
        mtmc = MTMC(SET)
        mtmc.load_sct()
        mtmc.load_feature()
        mtmc.tracklet_classify()
        mtmc.tracklet_coor()
        mtmc.trajectory_classify()
        mtmc.trajectory_sampling()
        mtmc.match()
        mtmc.id_assign()
        mtmc.save_mtmc()
        EVAL = Eval(SET)
        EVAL.load_txt()
        EVAL.split_txt()
        
        # MTMC = MTMC(SET)
        #SCT = load_SCT(RES, SET)
        #feature = load_feature(feature_path, SET)
        # feature_coorespond = load_feature_coorespond_file(feature_path, SET)
        #cal_loss(SCT, feature, SET)
    tEnd = time.time()
    print("It cost %f sec" % (tEnd - tStart))