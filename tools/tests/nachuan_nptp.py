import numpy as np
import cv2 as cv
import os


def read_image(img_path):
    return cv.imread(img_path)

def show_result(img_path, label_path, out_path, save_path):
    images = os.listdir(img_path)
    labels = os.listdir(label_path)
    print(images)
    for image in images:
        # label.shape : (256, 256, 3) --> (256, 256, 1)
        # value in label: 0 and 255
        label_img = read_image(label_path + image)
        label_img = cv.cvtColor(label_img, cv.COLOR_BGR2GRAY)
        # out.shape : (256, 256, 3) --> (256, 256, 1)
        # value in out: 0 and 255
        out_img = read_image(out_path + image)
        out_img = cv.cvtColor(out_img, cv.COLOR_BGR2GRAY)

        threshold = 10
        ret, label_img = cv.threshold(label_img, threshold, 255, cv.THRESH_BINARY)
        ret, out_img = cv.threshold(out_img, threshold, 255, cv.THRESH_BINARY)


        # file = open('/remote-home/jywang/swin/result/DM/TJ/DB_test/color/test.txt', mode='w', encoding='utf-8')
        # file.write(out_img)
        # file.close()
        # np.savetxt('/remote-home/jywang/swin/result/DM/TJ/DB_test/color/test.txt', out_img)

        green_idx = np.argwhere((label_img == 255) & (out_img == 255))
        red_idx = np.argwhere((label_img == 255) & (out_img == 0))
        blue_idx = np.argwhere((label_img == 0) & (out_img == 255))


        # green_idx = np.argwhere(label_img == out_img)
        # red_idx = np.argwhere(label_img != out_img)
        # blue_idx = np.argwhere((label_img == 0) & (out_img == 255))

        input_img = read_image(img_path + image)
        for green_idx_single in green_idx:
            input_img[green_idx_single[0], green_idx_single[1]] = 0.5 * input_img[green_idx_single[0], green_idx_single[1]] + (0, 128, 0)
        for red_idx_single in red_idx:
            input_img[red_idx_single[0], red_idx_single[1]] = 0.5 * input_img[red_idx_single[0], red_idx_single[1]] + (0, 128, 128) # (0, 128, 128)
        for blue_idx_single in blue_idx:
            input_img[blue_idx_single[0], blue_idx_single[1]] = 0.5 * input_img[blue_idx_single[0], blue_idx_single[1]] + (128, 0, 0)
        cv.imwrite(save_path + image, input_img)

def post_process(img_root_path, img_save_path):
    images = os.listdir(img_root_path)
    for image in images:
        post_map = read_image(img_root_path + image)
        post_map = cv.cvtColor(post_map, cv.COLOR_BGR2GRAY)
        print(post_map.shape)

        _, post_map = cv.threshold(post_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imwrite(img_save_path + image, post_map)


if __name__ == '__main__':
    # img_root_path = '/home/nachuan/Crack_Seg/CGAN-Crack-test/161_results/161_200_AD_2/'
    # img_save_path = './outputs_all/'
    # post_process(img_root_path, img_save_path)
    img_path = '/remote-home/jywang/swin/tools/data/VOCdevkit/TJ/test_image/'
    label_path = '/remote-home/jywang/swin/tools/data/VOCdevkit/TJ/test_label_jpg/'
    out_path = '/remote-home/jywang/swin/result/DA/TJ/TJ/'
    save_path = '/remote-home/jywang/swin/result/DA/TJ/TJ/color/'
    show_result(img_path, label_path, out_path, save_path)
