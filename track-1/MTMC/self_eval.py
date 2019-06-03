import os
import numpy as np

ROOTPATH = 'D:/users/linjian/workspace/AICity19/dataset_track1/'
TRACKLET = 'tc_tracklet'
EVALDIR = 'eval_0504'
MTMC_bbox_path = ROOTPATH + '/' + TRACKLET + '/MCT'  
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
ALL_SET = [S1_SET, S3_SET, S4_SET]
# ALL_SET = [S3_SET]
class Eval():
    def __init__(self, SET):
        self.SET = SET
        self.txt = None

    def load_txt(self):
        # c001_train
        self.txt = np.loadtxt(MTMC_bbox_path + '/' + EVALDIR + '/' + 'mtmc_%s.txt'%self.SET[0][0:2], delimiter=',')
    
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
            np.savetxt(MTMC_bbox_path + '/' + EVALDIR + '/' + 'c%03d_train.txt'%int(cam[3:]), cam_txt, fmt='%d,%d,%d,%d,%d,%d,%d,%d') 
    
if __name__ == '__main__':
    for SET in ALL_SET:
        EVAL = Eval(SET)
        EVAL.load_txt()
        EVAL.split_txt()