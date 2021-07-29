import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/116.jpg')
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image: object, parsing: object, part: object = 17, color: object = [230, 50, 20]) -> object:
    b, g, r = color  # [10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image2 = np.interp(image, (0, 255), (50, 200))
    image2 = np.array(image2, dtype=np.uint8)

    image_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)  # imagen original a HSV
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)  # color target a HSV

    # image_hsv[:, :, 2] += 75
    cv2.imshow('1', cv2.resize(image_hsv, (512, 512)))

    # image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]  # aplico matiz (H), conservo saturación (S) y valor (V)
    # image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]  # aplico matiz (H) y saturación (S), conservo valor (V)
    # image_hsv[:, :, 0:3] = tar_hsv[:, :, 0:3]  # piso con el color target
    # TODO: el pelo negro (u oscuro) tiene V muy bajo y en HSV no importa mucho el H y S, así que la imagen es no se
    #  modifica mucho.

    alfa = 1
    beta = 0
    gama = 0
    image_hsv[:, :, 0:1] = image_hsv[:, :, 0:1] * (1-alfa) + tar_hsv[:, :, 0:1] * alfa
    image_hsv[:, :, 1:2] = image_hsv[:, :, 1:2] * (1-beta) + tar_hsv[:, :, 1:2] * beta
    image_hsv[:, :, 2:3] = image_hsv[:, :, 2:3] * (1-gama) + tar_hsv[:, :, 2:3] * gama

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)  # paso a RGB

    cv2.imshow('2', cv2.resize(image_hsv, (512, 512)))
    cv2.imshow('3', cv2.resize(tar_color, (512, 512)))
    cv2.imshow('4', cv2.resize(tar_hsv, (512, 512)))
    cv2.imshow('5', cv2.resize(changed, (512, 512)))

    # if part == 17:
    #     changed = sharpen(changed)

    cv2.imshow('6', cv2.resize(changed, (512, 512)))

    changed[parsing != part] = image[parsing != part]  # rescato lo que sea != de 17
    # filtro: matriz like parsing con 1 en colores similares al subconjunto "cara"
    # changed[(parsing == 17) && filtro] = image[(parsing == 17) && filtro]

    return changed


if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    args = parse_args()

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }
    parts = [table['hair'], table['upper_lip'], table['lower_lip']]
    colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]]

    table = {
        'hair': 17
    }
    parts = [table['hair']]
    colors = [[255, 0, 0]]  # B G R
    # colors = [[255, 150, 255]]  # B G R ROSADO

    image_path = args.img_path
    cp = 'cp/79999_iter.pth'

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    parsing = np.uint8(np.repeat(parsing[:, :, np.newaxis], 3, axis=2))
    print(np.unique(parsing))
    parsing = np.where(parsing == 1, 50, parsing)
    parsing = np.where(parsing == 11, 100, parsing)
    parsing = np.where(parsing == 12, 150, parsing)
    parsing = np.where(parsing == 13, 200, parsing)
    parsing = np.where(parsing == 17, 255, parsing)
    cv2.imshow('parsing', cv2.resize(parsing, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
