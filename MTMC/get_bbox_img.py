import os
import numpy as np
import cv2

#TRACKLET = 'tc'
TRACKLET = 'deep_sort'
#TRACKLET = 'GroundTruth'
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

# ALL_SET = [S3_SET, S4_SET]
# ALL_SET = [S5_SET]
ALL_SET = [S2_SET, S5_SET]



class Crop():

    def __init__(self, SET):
        self.SET = SET
        self.vdo_path = []
        self.bbox_path = []
        self.vdo_shape = []
        self.bbox_info = []
        self.bbox_img = []
        self.save_path = os.path.join('res', TRACKLET, 'bbox_img')

    def find_vdo_bbox_path(self):
        for dir in self.SET:
            self.vdo_path.append( os.path.join('dataset/data', dir, '%s.avi'%dir))
            self.bbox_path.append( os.path.join('res', TRACKLET, 'SCT', 'deep_sort_mrcnn_%s.txt'%dir))
            roi = cv2.imread( os.path.join('dataset/data', dir, 'roi.jpg'))
            self.vdo_shape.append(roi.shape)
            # self.bbox_path.append( os.path.join('dataset/data', dir, 'gt.txt'))
        print(self.bbox_path)

    def read_bbox(self):
        for (bbox_txt, height_width) in zip(self.bbox_path, self.vdo_shape):
            row = []
            img_h = height_width[0]
            img_w = height_width[1]
            txt = np.loadtxt(bbox_txt, delimiter=',', dtype=np.int16)
            for info in txt:
                info[info < 0] = 0
                frame = info[0]
                car_id = int(info[1])
                left = int(info[2])
                top = int(info[3])
                width = int(info[4])
                height = int(info[5])
                if top < 10:
                    top = 10
                if left < 10:
                    left = 10
                if width < 30:
                    width = 30
                if height < 30:
                    height = 30
                if left > img_w:
                    left = img_w - 100
                if top > img_h:
                    top = img_h - 100
                if left + width > img_w:
                    width = img_w - left
                if top + height > img_h:
                    height = img_h - top
                row.append([frame, car_id, left, top, width, height])
            row = np.array(row)
            row = np.reshape(row, (-1, 6))
            self.bbox_info.append(row)
        # bboxes = np.array(bboxes)
        # print('size:',len(bboxes))
        # np.save(os.path.join(ROOTPATH,'out','%d'%i),f)
        # print(f)

    def crop_img(self):
        print('\nScene: {}'.format(SET[0][0:2]))            
        for vdo_num, vdo in enumerate(self.vdo_path):
            print('\ncam:', vdo_num+1)
            cap = cv2.VideoCapture(vdo)
            bbox = self.bbox_info[vdo_num]
            bbox_max_num = bbox.shape[0]
            print('\nnum_of_crop_img: ', bbox_max_num)
            frame_num = 1
            bbox_num = 0
            while cap.isOpened():
                _, frame = cap.read() 
                while int(bbox[bbox_num][0]) == frame_num:
                    # frame, car_id, left, top, width, height
                    sct_id = int(bbox[bbox_num][1])
                    x = int(bbox[bbox_num][2])
                    y = int(bbox[bbox_num][3])
                    w = int(bbox[bbox_num][4])
                    h = int(bbox[bbox_num][5])
                    bbox_img = frame[y:y+h-1, x:x+w-1, :]
                    # print('%s_frame_%05d_SCTid_%05d.jpg'%(vdo[-9:-4], frame_num, SCT_id))
                    save_name = os.path.join(self.save_path, '%s/%s_frame_%05d_SCTid_%05d.jpg'%(self.SET[0][:2], vdo[-9:-4], frame_num, sct_id))
                    cv2.imwrite(save_name, bbox_img)
                    if bbox_num+1 == bbox_max_num:
                        break
                    else:
                        bbox_num += 1
                if bbox_num+1 == bbox_max_num:
                    bbox_num = 0
                    break
                frame_num += 1    
            cap.release()


if __name__ == '__main__':

    for SET in ALL_SET:
        crop = Crop(SET)
        crop.find_vdo_bbox_path()
        crop.read_bbox()
        crop.crop_img()

