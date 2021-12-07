import cv2 as cv
import numpy as np
import math

def rescaleFrame(frame, scale):
    h = int(frame.shape[0] * scale)
    w = int(frame.shape[1] * scale)

    dimensi = (w, h)

    return cv.resize(frame, dimensi, interpolation=cv.INTER_AREA)

def get_gauss_kernel(size = 21, sigma = 6.8, tetha = -2.1, gamma = 0.5, lamda = 10):
    center = (int)(size/2)
    kernel = np.zeros((size, size))
    for i in range(-center, center+1, 1):
        for j in range (-center, center+1, 1):
            x1 = i * math.cos(tetha ) + j * math.sin(tetha)
            y1 = -1 * i * math.sin(tetha) + j * math.cos(tetha)
            kernel[i + 2, j + 2] = np.exp(-(x1**2 + gamma**2 * y1**2) / (2 * sigma**2)) * math.cos(2 * math.pi * x1**2 / lamda)
    return kernel

g_kernel = cv.getGaborKernel(ksize=(21, 21),
                                  sigma=10,
                                  theta=-1.8,
                                  lambd=9.9,
                                  gamma=0.5,
                                  psi=0,
                                  ktype=cv.CV_32F)


capture = cv.VideoCapture("ori.mp4") #untuk file
counter = 0
counterNormal = 0
counterNormalSebelum = 0
posisi = 0

while True:
    isTrue, frame = capture.read()

    # rescale
    frameScale = rescaleFrame(frame, 0.4)
    frameScaleRGB = frameScale.copy()
    frameScaleRGBHasil = frameScale.copy()
    frameScale = cv.cvtColor(frameScale, cv.COLOR_BGR2GRAY)

    # gabor filter
    kernel1 = get_gauss_kernel()
    hasil_gabor = cv.filter2D(frameScale, cv.CV_8UC3, g_kernel)

    # canny filter
    dst = cv.Canny(hasil_gabor, 50, 200, None, 3)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 20)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # if(l[3] > 290) and (abs(l[0] - l[2]) > 100) and l[3] < 710:
            cv.line(frameScaleRGB, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv.LINE_AA)

    b = int(frameScaleRGB[750, 130, 0])
    g = int(frameScaleRGB[762, 130, 1])
    r = int(frameScaleRGB[750, 130, 2])

    selisih = counterNormal - counterNormalSebelum
    # print("posisi ", posisi, " normal ", counterNormal, " sebelum ", counterNormalSebelum, " selisih ", selisih)

    if abs(g - 255) < 80:
        # print("hijau ", counter)
        if counter == 0 and selisih >= 20:
            counterNormalSebelum = counterNormal
            posisi = posisi + 1
        # print(" b = ", b, " g = ", g, " r = ", r)
        counter = counter + 1
        # counterNormal = counterNormal + 1
    else: 
        counter = 0

    counterNormal = counterNormal + 1
    
    # print("sekarang ", counterNormal, " sebelum ", counterNormalSebelum, " selisih ", selisih)
    # counterNormal = 0
    
    cv.putText(frameScaleRGBHasil, "Keramik Terlewat", (70, 300), cv.FONT_HERSHEY_TRIPLEX, 1,(255,0,0),2)
    cv.putText(frameScaleRGBHasil, str(posisi), (220, 350), cv.FONT_HERSHEY_TRIPLEX, 1,(255,0,0),2)

    # cv.putText(frameScaleRGB, str(posisi), (240, 768), cv.FONT_HERSHEY_TRIPLEX, 2,(255,0,0),2)

    # cv.imshow("jendala gabor", hasil_gabor)
    # cv.imshow("jendala canny", dst)
    cv.imshow("jendala video hasil", frameScaleRGBHasil)
    cv.imshow("jendala video", frameScaleRGB)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()