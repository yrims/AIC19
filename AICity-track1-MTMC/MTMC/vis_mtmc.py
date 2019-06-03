import os 
import cv2
import numpy as np

ROOTPATH = 'dataset'
#TRACKLET = 'tc_tracklet'
TRACKLET = 'deep_sort'
MTMC_bbox_path = 'res/' + TRACKLET + '/MCT'  


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
# ALL_SET = [S1_SET, S3_SET, S4_SET]
ALL_SET = [S2_SET]
def find_vdo_path(path, SET):
    vdo_path = []
    height_width_of_video = []
    data_path = os.path.join(path, 'data')
    
    for dir in SET:
        vdo_path.append( os.path.join(data_path, dir, dir+'.avi') )
        roi = cv2.imread( os.path.join(data_path, dir, 'roi.jpg'))
        height_width_of_video.append(roi.shape)
    
    return vdo_path, height_width_of_video

def read_bbox(MTMC_bbox_path, height_width_of_video, SET):
    bboxes = []
    f_path = os.path.join(MTMC_bbox_path, 'eval_0528/mtmc_%s.txt'%SET[0][0:2])
    txt = np.loadtxt(f_path, delimiter=',')
   
    for info in txt:
        cam_id = int(info[0])
        car_id = int(info[1])
        frame = int(info[2])
        # sct_id = int(bbox[bbox_num][3])
        left = int(info[3])
        top = int(info[4])
        width = int(info[5])
        height = int(info[6])
        bboxes.append([cam_id, car_id, frame, left, top, width, height])  

    bboxes = np.array(bboxes)
    bboxes = np.reshape(bboxes,(-1,7))
    # bboxes = np.array(bboxes)
    # print('size:',len(bboxes))

    return bboxes

def read_vdo(vdo_path, bbox_path, height_width_of_video, SET):
    
    bboxes = read_bbox(bbox_path, height_width_of_video, SET)
    print('\nScene: {}'.format(SET[0][0:2]))   

    for vdo_num, (vdo, height_width) in enumerate(zip(vdo_path, height_width_of_video)):
        print('\ncam:', int(SET[vdo_num][3:]))
        
        cap = cv2.VideoCapture(vdo)
        # 設定擷取影像的尺寸大小
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_width[0])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, height_width[1])
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        write_path = os.path.join(MTMC_bbox_path, 'eval_0528/%s_mtmc.avi'%SET[vdo_num])
        out = cv2.VideoWriter(write_path, fourcc, 10.0, (height_width[1], height_width[0]))

        bbox = bboxes[bboxes[:,0] == int(SET[vdo_num][3:])]
        bbox_max_num = bbox.shape[0]
        print('\nnum_of_crop_img: ',bbox_max_num)
        frame_num = 1
        bbox_num = 0
        
        while(cap.isOpened()):
            ret, frame = cap.read() 
            if ret == True:    
                want = bbox[ bbox[:,2] == frame_num]
                for want_box in want:
                    frame = cv2.rectangle(frame, (want_box[3],want_box[4]), (want_box[3]+want_box[5],want_box[4]+want_box[6]), (255,0,0), 5)
                    frame = cv2.putText(frame, str(want_box[1]), (want_box[3],want_box[4]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                '''
                    cam_id = int(bbox[bbox_num][0])
                    mtmc_id = int(bbox[bbox_num][1])
                    x = int(bbox[bbox_num][3])
                    y = int(bbox[bbox_num][4])
                    w = int(bbox[bbox_num][5])
                    h = int(bbox[bbox_num][6])
                '''
                out.write(frame) 
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break 
            else:
                break
            frame_num += 1        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

#def extract_bbox_img():

if __name__ == '__main__':
  
    for SET in ALL_SET:

        vdo_path, height_width_of_video = find_vdo_path(ROOTPATH, SET)
        # print(vdo_path)
        # print(bbox_path)

        # read_bbox(bbox_path)
        read_vdo(vdo_path, MTMC_bbox_path, height_width_of_video, SET)