import os 
import time
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
import keras_applications

from PIL import Image
from scipy import io

from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from resnet_152 import resnet152_model
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--optimizer', default='adam', help='which optimizer')
parser.add_argument('--test_interval', type=int, default=10, help='number of interval epoch of testing')
parser.add_argument('--savemodel_interval', type=int, default=5, help='number of interval epoch of saving')
parser.add_argument('--dense_size', type=int, default=224, help='size of the dense size')
parser.add_argument('--classes', type=int, default=196, help='number of classes')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--gpu_device', default='0', help='gpu device')
parser.add_argument('--model_name', default='ResNet152', help='model choice')
parser.add_argument('--datapath', default='D:/Users/linjian/Downloads/dataset/stanford_car/data', help='data path for training and testing')
parser.add_argument('--rowdata_path', default='D:/Users/linjian/Downloads/dataset/stanford_car', help='data have not be preprocessed')
parser.add_argument('--log_file_path', default='./logs/training.csv', help='path to log file')
opt = parser.parse_args()
print(opt)


def GPU_setting(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_device
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)
    print('GPU setting successful.')

def preprocess_data(opt):
    print("Processing train data...")
    
    train_path = os.path.join(opt.rowdata_path,'cars_train')
    # test_path = os.path.join(opt.datapath,'cars_test')
    # label_path = os.path.join(opt.datapath,'car_devkit/train_label.txt')
    
    '''
    ---------------
    | load bboxes |
    ---------------
    '''
    cars_annos = io.loadmat(os.path.join(opt.rowdata_path,'car_devkit/cars_train_annos'))
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    # fnames = []
    class_ids = []
    bboxes = []
    fnames = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        class_ids.append(class_id-1)
        fnames.append(fname)
    
    data_len = len(fnames)
    train_len = math.floor(data_len*0.8)
    valid_len = data_len - train_len
    print('data len:{}'.format(data_len))
    print('train len:{}'.format(train_len))
    print('valid len:{}'.format(valid_len))

    '''
    ---------------
    | load labels |
    ---------------
    '''
    train_ids = class_ids[:train_len]
    valid_ids = class_ids[train_len:]
    train_labels = np_utils.to_categorical(train_ids, num_classes=opt.classes)
    valid_labels = np_utils.to_categorical(valid_ids, num_classes=opt.classes)
    np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/train_labels',train_labels)
    np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/valid_labels',valid_labels)
    print('train label shape:', train_labels.shape)
    print('valid label shape:', valid_labels.shape)
    
    '''
    # mat = io.loadmat(label_path+'/cars_train_annos.mat')

    with open(label_path) as label_file:
        label = []
        label = label_file.read()
        label = label.split('\n')
        label = np.array(label)
        label = label.astype(int)
        label = label-1 
        label = np_utils.to_categorical(label, classes)
        np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/train_label',label)
        # print(label.shape)
    '''

    '''
    ---------------
    | load images |
    ---------------
    '''
    train_img = []
    valid_img = []
    #test_img = []
    for i, f in enumerate(os.listdir(train_path)):
        img = Image.open(os.path.join(train_path,f))
        img = img.convert('RGB')
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]

        # print('{}    image shape: {}'.format(i,img.shape))
      
        (x1, y1, x2, y2) = bboxes[i]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        crop_img = img[y1:y2, x1:x2 ]
        # print(crop_img.shape)
        
        crop_img = Image.fromarray(crop_img,'RGB')
        crop_img = crop_img.resize((opt.image_size,opt.image_size))
        
        if i < train_len:
            crop_img.save(os.path.join(opt.rowdata_path, 'cars_train_crop','%05d.jpg'%(i+1)))
            crop_img = np.array(crop_img)
            train_img.append(crop_img)
        else:
            crop_img.save(os.path.join(opt.rowdata_path, 'cars_valid_crop','%05d.jpg'%(i+1-train_len)))
            crop_img = np.array(crop_img)
            valid_img.append(crop_img)
        '''
        because we can't correct resize img after converting to ndarray, it's would be save image then append to train_img 
        '''
        if (i+1) % 1000 == 0:
            print('preprocessing images : {}'.format(i))
        
        
    train_img = np.array(train_img)
    valid_img = np.array(valid_img)
    # test_img = np.array(test_img)
    train_img = np.reshape(train_img, (-1,opt.image_size,opt.image_size,3))
    valid_img = np.reshape(valid_img, (-1,opt.image_size,opt.image_size,3))
    # test_img = np.reshape(test_img, (-1,opt.image_size,opt.image_size,3))
    # print(train_img.shape)
    # print(test_img.shape)

    np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/train_img_%d'%opt.image_size,train_img)
    np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/valid_img_%d'%opt.image_size,valid_img)
    # np.save('D:/Users/linjian/Downloads/dataset/stanford_car/data/test_img_%d'%opt.image_size,test_img)

    print('preprocessing data end.')
    

def load_data(opt):
    print('loading data ...')
    train_img = np.load(opt.datapath+'/train_img_{}.npy'.format(opt.image_size))
    train_label = np.load(opt.datapath+'/train_labels.npy')
    valid_img = np.load(opt.datapath+'/valid_img_{}.npy'.format(opt.image_size))
    valid_label = np.load(opt.datapath+'/valid_labels.npy')
    print('loading successful.')
    return train_img, train_label, valid_img, valid_label 


def train_(opt):
    GPU_setting(opt)
    train_img , train_label, valid_img, valid_label = load_data(opt)
    print('loading model ...')
    
    if opt.model_name == 'ResNet152V2':
        model_ = keras_applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet')
        # model_.load_weights('./resnet152_weights_tf.h5', by_name=True)
    elif opt.model_name == 'ResNet152':
        # model_ = keras_applications.resnet.ResNet152(include_top=False, weights='imagenet', pooling='avg')
        model_ = resnet152_model(opt.image_size, opt.image_size, 3, opt.classes)
        model_.load_weights('./model.96-0.89.hdf5')
        model_ = Model(inputs=model_.input, outputs=model_.output)

    elif opt.model_name == 'ResNet101V2':
        model_ = keras_applications.resnet_v2.ResNet101V2(include_top=True, weight='imagenet')

    elif opt.model_name == 'ResNet101':
        model_ = keras_applications.resnet.ResNet101(include_top=True, weights='imagenet')
        model_.load_weights('./ResNet101weight_epoch_243_val_acc_0.62983.hdf5', by_name=True)
    
    elif opt.model_name == 'ResNeXt101':
        model_ = keras_applications.resnext.ResNeXt101(include_top=True, weights='imagenet')

    elif opt.model_name == 'NASNetLarge':
        model_ = keras_applications.nasnet.NASNetLarge(include_top=True, weights='imagenet')
    
    elif opt.model_name == 'NASNetMobile':
        model_ = keras_applications.nasnet.NASNetMobile(include_top=True, weights='imagenet')

    elif opt.model_name == 'InceptionResNetV2':
        model_ = keras_applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet')

    elif opt.model_name == 'vgg19':
        model_ = applications.vgg19.VGG19(include_top=True, weights='imagenet')
    
    else:
        print('model name error.')

    

    # x = model_.output
    # x = Dropout(0.1)(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1000, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # predictions = Dense(opt.classes, activation='softmax')(x)
    
    # model = Model(inputs=model_.input, outputs=predictions)

    #for i, layer in enumerate(model_.layers):
     #   print(i, layer.name)

    if opt.optimizer == 'adam':
        adam = Adam(lr=opt.learning_rate)
        model_.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        sgd = SGD(lr=opt.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model_.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model_.summary()
    print('loading successful.')
    print('data_shape:')
    print('train_img:',train_img.shape)
    print('train_label:', train_label.shape)
    print('valid_img:', valid_img.shape)
    print('valid_label:', valid_label.shape)

    loss, acc = model_.evaluate(train_img, train_label, batch_size=opt.batch_size)
    print('loss:',loss,'\nacc:', acc)
    '''
    for epoch in range(opt.epochs):
    
        print('\nTrue Epoch:%d'%(epoch+1))
        model.fit(x=train_img, y=train_label, batch_size=opt.batch_size, epochs=1, validation_split=0.15, shuffle=True )
        
        if (epoch+1) % opt.savemodel_interval == 0:
            model.save('model/Xception_model_epoch_{}.h5'.format((epoch+1)))
            model.save_weights('weight/Xception_weights_epoch_{}.h5'.format((epoch+1)))
    '''
    '''
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)
    
    valid_datagen = ImageDataGenerator()

    train_datagen.fit(train_img)
    valid_datagen.fit(valid_img)

   
    

    # callbacks
    patience = 50
    tensor_board = TensorBoard(
        log_dir='./logs/new', 
        histogram_freq=0, 
        batch_size=opt.batch_size, 
        write_graph=True, 
        write_images=True)
    
    csv_logger = CSVLogger(opt.log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1, min_delta=0.0005)
    
    model_names = './models/' + opt.model_name + 'model_{epoch:02d}_val_acc_{val_acc:.5f}.hdf5'
    model_checkpoint = ModelCheckpoint(
        filepath=model_names, 
        monitor='val_acc',
        verbose=1, 
        save_best_only=True)

    weight_names = './weights/' + opt.model_name + 'weight_epoch_{epoch:02d}_val_acc_{val_acc:.5f}.hdf5'
    weight_checkpoint = ModelCheckpoint(
        filepath=weight_names, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True)

    callbacks = [tensor_board, model_checkpoint, weight_checkpoint, csv_logger, early_stop, reduce_lr]

    model.fit_generator(
        train_datagen.flow(train_img, train_label, batch_size=opt.batch_size), 
        steps_per_epoch=train_img.shape[0] / opt.batch_size,
        validation_data=valid_datagen.flow(valid_img, valid_label, batch_size=opt.batch_size), 
        validation_steps=valid_img.shape[0]/opt.batch_size,
        epochs=opt.epochs, 
        callbacks=callbacks)

    '''
    '''
    for epoch in range(opt.epochs):
        batches = 0
        for x_batch, y_batch in datagen.flow(train_img, train_label, batch_size=opt.batch_size):
            model.train_on_batch(x_batch, y_batch)
            batches += 1
            if batches >= train_img.shape[0] / N_batch:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        
        cost, acc = model.evaluate(x=train_img, y=train_label, batch_size=N_batch, verbose=False)
        print('-----------------')
        print('epochs: ', (e + 1), '/', epochs,
          '\ncost: ', cost,
          '\naccuracy: ', round(acc, 4))
     
        print('-----------------')
    '''
        
        

if __name__ == '__main__':
    # preprocess_data(opt)
    train_(opt)

