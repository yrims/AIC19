import os
import cv2
import numpy as np
'''
CNN_feature
His_feature

Using Deepsort Mask RCNN
'''

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

# TRACKLET = 'tc'
TRACKLET = 'deep_sort'
# TRACKLET = 'GroundTruth'
ALL_SET = [S1_SET, S2_SET, S3_SET, S4_SET, S5_SET]
# ALL_SET = [S5_SET]
# SELECT_FILE = 'mtsc_deepsort_mask_rcnn.txt'
# SELECT_FILE = 'mtsc_tc_mask_rcnn.txt'
Homo_file = 'calibration.txt'

time = {}
# S1
time['S1c01'] = 0
time['S1c02'] = 1.640
time['S1c03'] = 2.049
time['S1c04'] = 2.177
time['S1c05'] = 2.235
# S2
time['S2c06'] = 0
time['S2c07'] = 0.061
time['S2c08'] = 0.421
time['S2c09'] = 0.660
# S3
time['S3c10'] = 8.715
time['S3c11'] = 8.457
time['S3c12'] = 5.879
time['S3c13'] = 0
time['S3c14'] = 5.042
time['S3c15'] = 8.492
# S4 key need add scene
time['S4c16'] = 0
time['S4c17'] = 14.318
time['S4c18'] = 29.955
time['S4c19'] = 26.979
time['S4c20'] = 25.905
time['S4c21'] = 39.973
time['S4c22'] = 49.422
time['S4c23'] = 45.716
time['S4c24'] = 50.853
time['S4c25'] = 50.263
time['S4c26'] = 70.450
time['S4c27'] = 85.097
time['S4c28'] = 100.110
time['S4c29'] = 125.788
time['S4c30'] = 124.319
time['S4c31'] = 125.033
time['S4c32'] = 125.199
time['S4c33'] = 150.893
time['S4c34'] = 140.218
time['S4c35'] = 165.568
time['S4c36'] = 170.797
time['S4c37'] = 170.567
time['S4c38'] = 175.426
time['S4c39'] = 175.644
time['S4c40'] = 175.838
#S5 key need add scene
time['S5c10'] = 0
time['S5c16'] = 0
time['S5c17'] = 0
time['S5c18'] = 0
time['S5c19'] = 0
time['S5c20'] = 0
time['S5c21'] = 0
time['S5c22'] = 0
time['S5c23'] = 0
time['S5c24'] = 0
time['S5c25'] = 0
time['S5c26'] = 0
time['S5c27'] = 0
time['S5c28'] = 0
time['S5c29'] = 0
time['S5c33'] = 0
time['S5c34'] = 0
time['S5c35'] = 0
time['S5c36'] = 0
#print(time[str])
VERBOSE = 0


def img_to_latitude_longitude(xi, yi, H):
    S1 = np.array([xi, yi, 1]).reshape(3, 1)
    H = np.array(H).reshape(3, 3)
    Hinv = np.linalg.inv(H)
    gp = np.dot(Hinv, S1)
    latitude = (gp[0] / gp[2]).item()
    longitude = (gp[1] / gp[2]).item()
    return latitude, longitude

def read_homography_file(homo_txt):
    if VERBOSE > 1:
        print('Reading %s' % (homo_txt))
    fp = open(homo_txt, 'r')
    line = fp.readline()
    line2 = fp.readline()
    fp.close
    s = line.replace(';', ' ')
    floats = [float(x) for x in s.split()]
    H = np.array(floats).reshape(3, 3)

    if len(line2):
        distort_params = [float(x) for x in line2.split()]
        D = np.array(distort_params)
        if VERBOSE > 1:
            print('Fisheye parameters:' + line2)
    else:
        D = None
    return H, D

for SET in ALL_SET:
    record = np.zeros((41,6000,6))
    out = []
    save_path = os.path.join('res', TRACKLET, '{}.txt'.format(SET[0][0:2]) )
    print('\nScene: {}'.format(SET[0][:2]))
    
    # save record array about start and end of a SCT car.
    for cam in SET:
        # cam_file = os.path.join('dataset/data', cam, SELECT_FILE)
        cam_file = os.path.join('res' +'/deep_sort/SCT', 'deep_sort_mrcnn_%s.txt'%cam)
        # cam_file = os.path.join('dataset/data', cam, 'gt.txt')
        # cam_file = os.path.join('dataset', TRACKLET, 'SCT', '%s.txt'%cam)
        roi = cv2.imread( os.path.join('dataset/data', cam, 'roi.jpg'))
        img_h, img_w = roi.shape[0], roi.shape[1]
        txt = np.loadtxt(cam_file, delimiter=',')
        homo_txt = os.path.join('dataset/data', cam, Homo_file)
        H, D = read_homography_file(homo_txt)
        cam_id = int(cam[3:])
        print('camid: {:0>2d}'.format(cam_id))
        
        print(record.shape)
        for info in txt:
            times = (info[0]/10) + time[cam]
            car_id = int(info[1])
            left = info[2]
            top = info[3]
            width = info[4]
            height = info[5]
            # print(car_id)
            x = int(info[2] + (width/2))
            y = int(info[3] + height)
            if x < 0.2 * img_w or x > 0.8 * img_w or y < 0.1 * img_h or y > 0.9 * img_h:
                continue
            if roi[y, x, 0] == 0:
                continue
            
            latitude, longitude = img_to_latitude_longitude(x, y, H)
            # frame ,time, cam_id ,SCT_id ,latitude ,longitude, (start_x,start_y ,end_x,end_y)
            if record[cam_id-1][car_id-1][0] == 0 and record[cam_id-1][car_id-1][1] == 0:
                record[cam_id-1][car_id-1][0] = latitude
                record[cam_id-1][car_id-1][1] = longitude
                record[cam_id-1][car_id-1][4] = times
                
            else:
                record[cam_id-1][car_id-1][2] = latitude
                record[cam_id-1][car_id-1][3] = longitude
                record[cam_id-1][car_id-1][5] = times
      
    for cam in SET:
        # cam_file = os.path.join('dataset/data', cam, SELECT_FILE)
        cam_file = os.path.join('res' +'/deep_sort/SCT', 'deep_sort_mrcnn_%s.txt'%cam)
        # cam_file = os.path.join('res', TRACKLET, 'SCT', '%s.txt'%cam)
        # cam_file = os.path.join('dataset/data', cam, 'gt.txt')
        roi = cv2.imread( os.path.join('dataset/data', cam, 'roi.jpg'))
        img_h, img_w = roi.shape[0], roi.shape[1]
        txt = np.loadtxt(cam_file, delimiter=',')
        homo_txt = os.path.join('dataset/data', cam, Homo_file)
        H, D = read_homography_file(homo_txt)
        cam_id = int(cam[3:])
        
        for info in txt:
            frame = info[0]
            times = (info[0]/10) + time[cam]
            car_id = int(info[1])
            left = info[2]
            top = info[3]
            width = info[4]
            height = info[5]
            
            x = int(info[2] + (width/2))
            y = int(info[3] + height)
            if x < 0.2 * img_w or x > 0.8 * img_w or y < 0.1 * img_h or y > 0.9 * img_h:
                continue
            if roi[y, x, 0] == 0:
                continue
            
            latitude, longitude = img_to_latitude_longitude(x, y, H)
            # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y , end_x, end_y, start_time, end_time, left, top, width, heigth, row
            
            out.append([frame, 
                        times, 
                        cam_id, 
                        car_id, 
                        latitude, 
                        longitude, 
                        record[cam_id-1][car_id-1][0], 
                        record[cam_id-1][car_id-1][1], 
                        record[cam_id-1][car_id-1][2], 
                        record[cam_id-1][car_id-1][3], 
                        record[cam_id-1][car_id-1][4], 
                        record[cam_id-1][car_id-1][5],
                        left, 
                        top, 
                        width, 
                        height
                        ])
            

    out = np.array(out)
    # sort by times
    out=out[out[:,1].argsort()]
    out = out.tolist()
    count = 0 
    for num_row in range(len(out)):
        out[num_row].append(count) 
        count += 1
    
    np.savetxt(save_path,
            out, fmt='%d,%.3f,%d,%d,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.3f,%.3f,%d,%d,%d,%d,%d',
            delimiter=",")
            

