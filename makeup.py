import cv2
# import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/116.jpg')
    return parse.parse_args()


def sharpen(img, sigma=5, alpha=1.5):
    img = img * 1.0
    gauss_out = gaussian(img, sigma, multichannel=True)

    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(img: object, seccion: object, parte: object = 17, col: object = [230, 50, 20]) -> object:
    img = np.array(img, dtype=np.uint8)
    img_col = np.copy(img)
    img_out = np.copy(img)

    img_col[seccion == parte] = col

    # 0: variación en alpha de addWeithed
    # 1: fijar alpha, variación en sigma y alfa del filtro gaussiano
    # 2: fijar alpha y parámetros gaussianos, variación en borde
    test_tipo = 9

    alto, ancho = 2, 4  # cantidad de pruebas (primera y última no cuentan)

    a1, a2 = 0.1, 0.6  # para variación en alpha
    alpha = 0.4

    # f1, f2 = .5, 3  # para variación en alpha en sharpen
    alpha_gauss = 1
    f1, f2 = 3, 8  # para variación en alpha en sharpen
    sigma_gauss = 5

    fig, axes = plt.subplots(alto, ancho)

    if test_tipo == 0:
        for i, ax, x in zip(range(alto * ancho), axes.flatten(),
                            np.roll(np.append(np.linspace(a1, a2, alto * ancho - 2), [0, 0]), 1)):
            if i == 0:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif i == alto * ancho - 1:
                ax.imshow(cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB))
            else:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # imagen original a HSV
                hh, ss, vv = cv2.split(img_hsv)
                brillo = 50
                vv[vv > 255 - brillo] = 255
                vv[vv <= 255 - brillo] += brillo
                img_hsv = cv2.merge((hh, ss, vv))
                img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # imagen original a BGR

                cv2.addWeighted(img_col, x, img_bgr, 1 - x, 0, img_out)  # composición

                img_out = sharpen(img_out)  # filtro 1:

                img_out[seccion != parte] = img[seccion != parte]  # rescato de la original lo que no es cabello

                cv2.putText(img_out,
                            'Alpha: %.2f' % x,
                            (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            2)
                ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        plt.show()
    elif test_tipo == 1:
        for i, ax, x in zip(range(alto * ancho), axes.flatten(),
                            np.roll(np.append(np.linspace(f1, f2, alto * ancho - 2), [0, 0]), 1)):
            cv2.addWeighted(img_col, alpha, img, 1 - alpha, 0, img_out)  # composición
            if i == 0:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif i == alto * ancho - 1:
                ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            else:
                img_out = sharpen(img_out, x, 1)  # filtro gaussiano

                img_out[seccion != parte] = img[seccion != parte]  # rescato de la original lo que no es cabello

                cv2.putText(img_out,
                            'Filtro: %.2f' % x,
                            (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            2)
                ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        plt.show()
    elif test_tipo == 2:
        cv2.addWeighted(img_col, alpha, img, 1 - alpha, 0, img_out)  # composición
        img_out = sharpen(img_out, sigma_gauss, alpha_gauss)  # filtro gaussiano
        img_out[seccion != parte] = img[seccion != parte]  # rescato de la original lo que no es cabello

        img_sec = np.where(seccion != 17, 0, seccion)
        img_sec = np.where(img_sec != 0, 255, img_sec)
        img_sec = np.uint8(np.repeat(img_sec[:, :, np.newaxis], 3, axis=2))
        img_gry = cv2.cvtColor(img_sec, cv2.COLOR_BGR2GRAY)  # sección del pelo, blanco y negro

        for i, ax, x in zip(range(alto * ancho), axes.flatten(),
                            np.roll(np.append(np.linspace(10, 500, alto * ancho - 2), [0, 0]), 1)):
            if i == 0:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif i == alto * ancho - 1:
                ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            else:
                # tresh = 100
                tresh = x
                ancho_con = 20
                ksize = (61, 61)
                img_blr = cv2.GaussianBlur(img_out, ksize, 0)  # suavizado

                img_con = np.zeros(img_sec.shape)
                ret, img_tre = cv2.threshold(img_gry, tresh, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(img_tre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_con, contours, -1, (255, 255, 255), ancho_con)

                img_aux = np.where(img_con == np.array([255, 255, 255]), img_blr, img_out)

                cv2.putText(img_aux,
                            'Filtro: %.2f' % x,
                            (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            2)
                ax.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        plt.show()
    elif test_tipo == 9:
        cv2.addWeighted(img_col, alpha, img, 1 - alpha, 0, img_out)  # composición
        img_out = sharpen(img_out, sigma_gauss, alpha_gauss)  # filtro gaussiano
        img_out[seccion != parte] = img[seccion != parte]  # rescato de la original lo que no es cabello

        img_sec = np.where(seccion != 17, 0, seccion)
        img_sec = np.where(img_sec != 0, 255, img_sec)
        img_sec = np.uint8(np.repeat(img_sec[:, :, np.newaxis], 3, axis=2))
        img_gry = cv2.cvtColor(img_sec, cv2.COLOR_BGR2GRAY)  # sección del pelo, blanco y negro

        tresh = 100
        ancho_con = 15
        ksize = (31, 31)
        img_blr = cv2.GaussianBlur(img_out, ksize, 0)  # suavizado

        img_con = np.zeros(img_sec.shape)
        ret, img_tre = cv2.threshold(img_gry, tresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_tre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_con, contours, -1, (255, 255, 255), ancho_con)

        img_fil = np.where(img_con == np.array([255, 255, 255]), img_blr, img_out)

        cv2.imshow('1', cv2.resize(img, (512, 512)))
        cv2.imshow('2', cv2.resize(img_out, (512, 512)))
        cv2.imshow('3', cv2.resize(img_fil, (512, 512)))

    return img_out


if __name__ == '__main__':
    args = parse_args()

    part = 17
    color = [135, 20, 200]  # B G R
    # color = [255, 150, 255]  # B G R: rosado

    image_path = args.img_path
    cp = 'cp/79999_iter.pth'

    img_ori = cv2.imread(image_path)
    img_par = evaluate(image_path, cp)
    img_par = cv2.resize(img_par, img_ori.shape[0:2], interpolation=cv2.INTER_NEAREST)

    DEBUG = 0

    if DEBUG == 0:
        img_fin = hair(img_ori, img_par, part, color)
    elif DEBUG == 1:
        cv2.imshow('1', cv2.resize(img_ori, (512, 512)))
        img_aux = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)  # imagen original a HSV

        print(np.min(img_aux[:, :, 2:3]), np.max(img_aux[:, :, 2:3]))
        print(img_aux[0:2, 0:2, 2])

        h, s, v = cv2.split(img_aux)
        val = 50
        v[v > 255 - val] = 255
        v[v <= 255 - val] += val
        img_aux = cv2.merge((h, s, v))
        # img_aux[:, :, 2:3] = np.clip(img_aux[:, :, 2:3] + 50, 0, 255)

        print(np.min(img_aux[:, :, 2:3]), np.max(img_aux[:, :, 2:3]))
        print(img_aux[0:2, 0:2, 2])

        img_aux = cv2.cvtColor(img_aux, cv2.COLOR_HSV2BGR)  # imagen original a BGR
        cv2.imshow('2', cv2.resize(img_aux, (512, 512)))
    # elif DEBUG == 2:
        # cv2.imshow('1', cv2.resize(img_ori, (512, 512)))
        # img_par = np.where(img_par != 17, 0, img_par)
        # img_par = np.where(img_par != 0, 255, img_par)
        # img_par = np.uint8(np.repeat(img_par[:, :, np.newaxis], 3, axis=2))
        # cv2.imshow('2', cv2.resize(img_par, (512, 512)))
        #
        # tresh = 100
        # ancho_con = 6
        #
        # # img_par
        # blurred_img = cv2.GaussianBlur(img_ori, (21, 21), 0)
        # img_con = np.zeros(img_par.shape)
        #
        # img_grey = cv2.cvtColor(img_par, cv2.COLOR_BGR2GRAY)
        # ret, thresh_img = cv2.threshold(img_grey, tresh, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # cv2.drawContours(img_con, contours, -1, (255, 255, 255), ancho_con)
        # output = np.where(img_con == np.array([255, 255, 255]), blurred_img, img_ori)
        #
        # cv2.imshow('3', cv2.resize(img_con, (512, 512)))
        # cv2.imshow('4', cv2.resize(blurred_img, (512, 512)))
        # cv2.imshow('5', cv2.resize(output, (512, 512)))



    cv2.waitKey(0)
    cv2.destroyAllWindows()
