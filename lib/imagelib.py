# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import hashlib
import cv2
import os


def cut_image_by_alpha(img):
    """
    对客户的图像进行审核，将多余的区域裁剪掉
    :param img: RGBA四通道图像
    :return:
    """
    index = np.nonzero(img[:, :, 3])  # 获取Alpha不为0的索引值

    y0 = index[0][0]  # index[0]为行索引，从小到大排列
    y = index[0][-1]

    x_index = sorted(index[1])  # index[1]为列索引，需先排序
    x0 = x_index[0]
    x = x_index[-1]

    return img[y0:y + 1, x0:x + 1, :]


def base64_to_rgba_ndarray(img_base64, format='png'):
    """
    将base64图片转成用于cv2操作的RGBA,ndarray图片
    :param img_base64: base64格式的图片
    :return: ndarray，RGBA格式
    """
    content = BytesIO(base64.b64decode(img_base64))
    img = Image.open(content)
    if format == 'png':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    img = np.array(img)

    return img


def ndarray_to_rgba_base64(img, format='png'):
    """
    将ndarray图片转成base64
    :param img: ndarray
    :return: base64图片
    """
    if format == 'png':
        img = Image.fromarray(img.astype('uint8')).convert('RGBA')
    elif format == 'jpeg':
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format)
    img_io = output_buffer.getvalue()
    img_base64 = base64.b64encode(img_io)

    return img_base64


def base64_to_bin(img_base64):
    """
    base64编码图片转成2进制格式
    :return:
    """
    return base64.b64decode(img_base64)


# def get_img(oss_address):
#     img_bin = oss_bucket.get_object(oss_address)
#     img_b64 = base64.encode(img_bin)
#     return img_b64


def md5(img_b64):
    """

    :param binary:
    :return:
    """
    md5 = hashlib.md5()
    md5.update(img_b64)
    md5_hex = md5.hexdigest()

    return md5_hex


# 叠加图像，对超出区域的部分自动进行裁剪，（img_1：图像，img_2：背景图，pos_1：图像在背景图上的位置）
def MixWithCut(img_1, img_2, pos_1):  # 将img1放置在img2上，pos1为img1在img2上的位置
    try:
        img1 = img_1.copy()
        img2 = img_2.copy()
        x0 = pos_1[0]
        y0 = pos_1[1]
        width = img1.shape[1]
        height = img1.shape[0]
        back_width = img2.shape[1]
        back_height = img2.shape[0]

        if x0 > back_width or y0 > back_height:
            return img2

        xe0 = int(-1 * min(x0, 0))
        ye0 = int(-1 * min(y0, 0))
        width = int(min(width, x0 + width, back_width - x0, img2.shape[1]))
        height = int(min(height, y0 + height, back_height - y0, img2.shape[0]))
        x0 = int(max(0, x0))
        y0 = int(max(0, y0))

        img2[y0:y0 + height, x0:x0 + width, :] = MisByAlpha(
            img1[ye0:ye0 + height, xe0:xe0 + width, :]
            , img2[y0:y0 + height, x0:x0 + width, :])
    except:
        pass

    return img2


# 叠加图像，（img_1：混合后在上层的图像，img_2：混合后再下层的图像）
def MisByAlpha(img_1, img_2):  # img-1覆盖在img2上，若img1的像素为0，则不覆盖
    img = img_1.copy()
    if img.shape[2] == 4:
        for i in range(3):
            img[:, :, i] = img[:, :, i] * (img[:, :, 3] / 255) + img_2[:, :, i] * (1 - img[:, :, 3] / 255)
        img[:, :, 3] = img[:, :, 3] + img_2[:, :, 3] * (img[:, :, 3] == 0)
    else:
        pass

    return img


def rgba2bgra(img):
    """
    将rgba图像转为bgra图像
    :param img:
    :return: bgra 图像
    """
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)


def bgr2rgb(img):
    """
    将bgr图像转为rgb图像
    :param img:
    :return: bgra 图像
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detector_importance_region(img):
    """
    检查人脸
    :param img:
    :return:
    """
    img = img.copy()

    detect_data_path = os.path.join(os.path.dirname(__file__), 'detectdata/haarcascade_frontalface_default.xml')
    region_cascade = cv2.CascadeClassifier(detect_data_path)

    # eye_cascade = cv2.CascadeClassifier('detectdata/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return region_cascade.detectMultiScale(gray, 1.3, 5)


def get_pixel(img, i, j):
    try:
        return int(img[i, j])
    except:
        return 0


def judge_endpoint(img, i, j, direction):
    if get_pixel(img, i, j) == 0:
        return False
    if direction == 'horizon':
        left = get_pixel(img, i - 1, j - 1) + get_pixel(img, i, j - 1) + get_pixel(img, i + 1, j - 1) > 0
        right = get_pixel(img, i - 1, j + 1) + get_pixel(img, i, j + 1) + get_pixel(img, i + 1, j + 1) > 0
        if left and right:
            return False
    elif direction == 'vertical':
        up = get_pixel(img, i - 1, j - 1) + get_pixel(img, i - 1, j) + get_pixel(img, i - 1, j + 1) > 0
        down = get_pixel(img, i + 1, j - 1) + get_pixel(img, i + 1, j) + get_pixel(img, i + 1, j + 1) > 0
        if up and down:
            return False
    # print('debug{}'.format([i, j]))
    return True


def is_cut(img):
    debug = False
    if debug:
        cv2.imshow('win1', img)
        cv2.waitKey(0)
    img = img.copy()
    img = img[:, :, 3]
    if debug:
        cv2.imshow('win2', img)
        cv2.waitKey(0)
    img = cv2.Canny(img, 0, 150)
    [up, down, left, right] = [False, False, False, False]
    if debug:
        cv2.imshow('win3', img)
        cv2.waitKey(0)

    xx1 = sum(img[0, :] > 0)
    xx2 = sum(img[-1, :] > 0)
    yy1 = sum(img[:, 0] > 0)
    yy2 = sum(img[:, -1] > 0)

    x1 = x2 = y1 = y2 = 0
    c = 0
    for i in range(len(img[0])):
        if img[0, i] != 0:
            c += 1
        else:
            x1 = max(x1, c)
            c = 0
    c = 0
    for i in range(len(img[0])):
        if img[len(img) - 1, i] != 0:
            c += 1
        else:
            x2 = max(x2, c)
            c = 0

    if len(img) > 1:
        xx1 += sum(img[1, :] > 0)
        xx2 += sum(img[-2, :] > 0)
        c = 0
        for i in range(len(img[0])):
            if img[1, i] != 0:
                c += 1
            else:
                x1 = max(x1, c)
                c = 0
        c = 0
        for i in range(len(img[0])):
            if img[len(img) - 2, i] != 0:
                c += 1
            else:
                x2 = max(x2, c)
                c = 0

    c = 0
    for i in range(len(img)):
        if img[i, 0] != 0:
            c += 1
        else:
            y1 = max(y1, c)
            c = 0
    c = 0
    for i in range(len(img)):
        if img[i, len(img[0]) - 2] != 0:
            c += 1
        else:
            y2 = max(y2, c)
            c = 0

    if len(img[0]) > 1:
        yy1 += sum(img[:, 1] > 0)
        yy2 += sum(img[:, -2] > 0)
        c = 0
        for i in range(len(img)):
            if img[i, 1] != 0:
                c += 1
            else:
                y1 = max(y1, c)
                c = 0
        c = 0
        for i in range(len(img)):
            if img[i, len(img[0]) - 1] != 0:
                c += 1
            else:
                y2 = max(y2, c)
                c = 0

    return [((x1 < min(img.shape[0] / 3, 20)) or (xx1 < img.shape[0] / 2)),
            ((x2 < min(img.shape[0] / 3, 20)) or (xx2 < img.shape[0] / 2)),
            ((y1 < min(img.shape[1] / 3, 20)) or (yy1 < img.shape[1] / 2)),
            ((y2 < min(img.shape[0] / 3, 20)) or (yy2 < img.shape[1] / 2))]
