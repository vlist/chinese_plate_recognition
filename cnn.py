# -*- coding:utf-8 -*-
# author: DuanshengLiu
import tensorflow as tf
from tf import keras as K
from tf.keras import layers, losses, models
import numpy as np
import cv2
import os

logdir = "./logdir"
csv_log = os.path.join(logdir, "result.csv")
ckpt_path = "models/ckpt/"
model_path = "models/cnn_model.h5"
epochs = 5
batch_size = 16

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


def cnn_train():
    char_dict = {'皖': 0, '沪': 1, '津': 2, '渝': 3, '冀': 4, '晋': 5, '蒙': 6, '辽': 7, '吉': 8, '黑': 9, '苏': 10,
                 '浙': 11, '京': 12, '闽': 13, '赣': 14, '鲁': 15, '豫': 16, '鄂': 17, '湘': 18, '粤': 19, '桂': 20,
                 '琼': 21, '川': 22, '贵': 23, '云': 24, '藏': 25, '陕': 26, '甘': 27, '青': 28, '宁': 29, '新': 30,
                 '警': 31, '学': 32,
                 'A': 33, 'B': 34, 'C': 35, 'D': 36, 'E': 37, 'F': 38, 'G': 39, 'H': 40, 'J': 41, 'K': 42,
                 'L': 43, 'M': 44, 'N': 45, 'P': 46, 'Q': 47, 'R': 48, 'S': 49, 'T': 50, 'U': 51, 'V': 52,
                 'W': 53, 'X': 54, 'Y': 55, 'Z': 56, '0': 57, '1': 58, '2': 59, '3': 60, '4': 61, '5': 62,
                 '6': 63, '7': 64, '8': 65, '9': 66, 'O': 67}

    # 读取数据集
    train_path = 'plate_dataset/train/'  # 车牌号数据集路径(车牌图片宽240，高80)
    pic_name = sorted(os.listdir(train_path))
    n = len(pic_name)
    X_train, y_train = [], []
    for i in range(n):
        # cv2.imshow无法读取中文路径图片，改用此方式
        img = cv2.imdecode(np.fromfile(
            train_path + pic_name[i], dtype=np.uint8), -1)
        label = [char_dict[name] for name in pic_name[i][0:7]]  # 图片名前7位为车牌标签
        X_train.append(img)
        y_train.append(label)
    X_train = np.array(X_train)
    # y_train是长度为7的列表，其中每个都是shape为(n,)的ndarray，分别对应n张图片的第一个字符，第二个字符....第七个字符
    y_train = [np.array(y_train)[:, i] for i in range(7)]

    X_val, y_val = [], []
    val_path = 'plate_dataset/val/'
    pic_name = sorted(os.listdir(val_path))
    n = len(pic_name)
    for i in range(n):
        # cv2.imshow无法读取中文路径图片，改用此方式
        img = cv2.imdecode(np.fromfile(
            val_path + pic_name[i], dtype=np.uint8), -1)
        label = [char_dict[name] for name in pic_name[i][0:7]]  # 图片名前7位为车牌标签
        X_val.append(img)
        y_val.append(label)
    X_val = np.array(X_val)
    y_val = [np.array(y_val)[:, i] for i in range(7)]

    # cnn模型
    Input = layers.Input((80, 240, 3))  # 车牌图片shape(80,240,3)
    x = Input
    x = layers.Conv2D(filters=16, kernel_size=(
        3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
    for i in range(3):
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3),
                          padding='valid', activation='relu')(x)
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3),
                          padding='valid', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
        x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    Output = [layers.Dense(68, activation='softmax', name='c%d' % (
        i + 1))(x) for i in range(7)]  # 7个输出分别对应车牌7个字符，每个输出都为68个类别类概率
    model = models.Model(inputs=Input, outputs=Output)
    model.summary()
    model.compile(optimizer='adam',
                  # y_train未进行one-hot编码，所以loss选择sparse_categorical_crossentropy
                  loss='sparse_categorical_crossentropy',
                  metrics=[
                      'accuracy',
                  ])

    if os.path.exists(ckpt_path):
        latest = tf.train.latest_checkpoint(ckpt_path)
        if latest != None:
            model.load_weights(latest)

    # 模型训练
    cnn_callback = [
        K.callbacks.TensorBoard(logdir),
        K.callbacks.ModelCheckpoint(ckpt_path,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1),
        K.callbacks.CSVLogger(csv_log, append=True)
    ]
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
                        callbacks=cnn_callback,
                        validation_data=(X_val, y_val))  # 总loss为7个loss的和
    model.save(model_path)
    print('cnn model保存成功')


def cnn_predict(cnn, Lic_img):
    characters = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁',
                  '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    Lic_pred = []
    for lic in Lic_img:
        lic_pred = cnn.predict(lic.reshape(
            1, 80, 240, 3))  # 预测形状应为(1,80,240,3)
        lic_pred = np.array(lic_pred).reshape(7, 68)  # 列表转为ndarray，形状为(7,68)
        # 统计其中预测概率值大于80%以上的个数，大于等于4个以上认为识别率高，识别成功
        if len(lic_pred[lic_pred >= 0.8]) >= 4:
            chars = ''
            for arg in np.argmax(lic_pred, axis=1):  # 取每行中概率值最大的arg,将其转为字符
                chars += characters[arg]
            chars = chars[0:2] + '·' + chars[2:]
            Lic_pred.append((lic, chars))  # 将车牌和识别结果一并存入Lic_pred
    return Lic_pred
