import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse

def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def locate_and_correct(img_src, img_mask):
    """
    该函数通过cv2对get_mask进行边缘检测，获取车牌区域的边缘坐标(存储在contours中)和最小外接矩形4个端点坐标,
    再从车牌的边缘坐标中计算出和最小外接矩形4个端点最近的点即为平行四边形车牌的四个端点,从而实现车牌的定位和矫正
    :param img_src: 原始图片
    :param img_mask: 二值化图片，车牌区域呈现白色，背景区域为黑色
    :return: 定位且矫正后的车牌
    """
    try:
        contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:  # 防止opencv版本不一致报错
        ret, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # contours1长度为0说明未检测到车牌
        # print("未检测到车牌")
        return [], []
    else:
        Lic_img = []
        for _, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # 获取最小外接矩形
            img_cut_mask = img_mask[y:y + h, x:x + w]  # 将标签车牌区域截取出来
            # contours中除了车牌区域可能会有宽或高都是1或者2这样的小噪点，
            # 而待选车牌区域的均值应较高，且宽和高不会非常小，因此通过以下条件进行筛选
            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
                rect = cv2.minAreaRect(cont)  # 针对坐标点获取带方向角的最小外接矩形，中心点坐标，宽高，旋转角度
                box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标
                cont = cont.reshape(-1, 2).tolist()
                # 由于转换矩阵的两组坐标位置需要一一对应，因此需要将最小外接矩形的坐标进行排序，最终排序为[左上，左下，右上，右下]
                box = sorted(box, key=lambda xy: xy[0])  # 先按照左右进行排序，分为左侧的坐标和右侧的坐标
                box_left, box_right = box[:2], box[2:]  # 此时box的前2个是左侧的坐标，后2个是右侧的坐标
                box_left = sorted(box_left, key=lambda x: x[1])  # 再按照上下即y进行排序，此时box_left中为左上和左下两个端点坐标
                box_right = sorted(box_right, key=lambda x: x[1])  # 此时box_right中为右上和右下两个端点坐标
                box = np.array(box_left + box_right)  # [左上，左下，右上，右下]
                # print(box)
                x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]

                def point_to_line_distance(X, Y):
                    if x2 - x0:
                        k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
                        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                    else:  # 斜率无穷大
                        d_up = abs(X - x2)
                    if x1 - x3:
                        k_down = (y1 - y3) / (x1 - x3)  # 斜率不为无穷大
                        d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                    else:  # 斜率无穷大
                        d_down = abs(X - x1)
                    return d_up, d_down

                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
                for each in cont:  # 计算cont中的坐标与矩形四个坐标的距离以及到上下两条直线的距离，对距离和进行权重的添加，成功计算选出四边形的4个顶点坐标
                    x, y = each[0], each[1]
                    dis0 = (x - x0) ** 2 + (y - y0) ** 2
                    dis1 = (x - x1) ** 2 + (y - y1) ** 2
                    dis2 = (x - x2) ** 2 + (y - y2) ** 2
                    dis3 = (x - x3) ** 2 + (y - y3) ** 2
                    d_up, d_down = point_to_line_distance(x, y)
                    weight = 0.975
                    if weight * d_up + (1 - weight) * dis0 < d0:  # 小于则更新
                        d0 = weight * d_up + (1 - weight) * dis0
                        l0 = (x, y)
                    if weight * d_down + (1 - weight) * dis1 < d1:
                        d1 = weight * d_down + (1 - weight) * dis1
                        l1 = (x, y)
                    if weight * d_up + (1 - weight) * dis2 < d2:
                        d2 = weight * d_up + (1 - weight) * dis2
                        l2 = (x, y)
                    if weight * d_down + (1 - weight) * dis3 < d3:
                        d3 = weight * d_down + (1 - weight) * dis3
                        l3 = (x, y)
                p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角，右下角，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # 我们所需的长方形
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
                lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正
                Lic_img.append(lic)
    return Lic_img

def get_mask(img_src):
    # 高斯模糊
    img_Gas = cv2.GaussianBlur(img_src, (5, 5), 0)
    # RGB通道分离
    img_B = cv2.split(img_Gas)[0]
    img_G = cv2.split(img_Gas)[1]
    img_R = cv2.split(img_Gas)[2]
    # 读取灰度图和HSV空间图
    img_gray = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2HSV)

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            # 普通蓝色车牌，同时排除透明反光物质的干扰
            if ((img_HSV[:, :, 0][i, j]-115)**2 < 15**2) and (img_B[i, j] > 70) and (img_R[i, j] < 40):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    ori_gray = img_gray.copy()
    # 定义核
    kernel_small = np.ones((3, 3))
    kernel_big = np.ones((7, 7))

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯平滑
    img_dilate = cv2.dilate(img_gray, kernel_small, iterations=5)  # 腐蚀5次
    img_close = cv2.morphologyEx(img_dilate, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
    img_close = cv2.GaussianBlur(img_close, (5, 5), 0)  # 高斯平滑
    _, img_edge = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY)  # 二值化

    return img_edge, ori_gray

def run(model='models/cnn_model.h5', image='docs/images/plate.jpg'):
    image_src = cv2.imread(image)
    mask = get_mask(image_src)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='models/cnn_model.h5', help='model path(s)')
    parser.add_argument('--image', type=str, default='docs/images/plate.jpg', help='image file')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)