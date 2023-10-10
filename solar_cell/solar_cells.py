import cv2
from glob import glob
import numpy as np
files = glob('./solar_cell/*')
files = [file for file in files if not file.__contains__('.db')]
thres_value = 159





i = 0
while 1:
    img = cv2.imread(files[i], cv2.IMREAD_ANYCOLOR)
    img_red = img[:,:,2]
    
    
    ret, thres = cv2.threshold(img_red, thres_value, 255, cv2.THRESH_BINARY_INV)
    ret, thres0 = cv2.threshold(img_red, thres_value, 255, cv2.THRESH_BINARY)
    
    contours, h = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    zeros = np.zeros(thres.shape, dtype=np.uint8)
    zeros_hull = np.zeros(thres.shape, dtype=np.uint8)
    zeros_cont = np.zeros(thres.shape, dtype=np.uint8)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        (p1,p2),(w,h),r = cv2.minAreaRect(contours[c])
        ratio = 0
        if w == 0 or h == 0:
            continue
        if w > h:
            ratio = w/h
        else:
            ratio = h/w
        
        if area < 100:
            continue

        cond_area = not ((area > 4000 and area < 5000) or (area > 8000 and area < 10000))
        cond_ratio = (ratio > 8 or ratio < 5) and ratio > 3
        if  (cond_area and cond_ratio) or (not cond_ratio and (p1 < thres.shape[1]/4 or p1 > thres.shape[1]/4*3)):
            cv2.drawContours(zeros, contours, c, (255), -1)
        
        if area < 20000:
            cv2.drawContours(zeros_cont, contours, c, (255), -1)
            hull = cv2.convexHull(contours[c], clockwise=True)
            cv2.drawContours(zeros_hull, [hull], 0, (255), -1)

    se = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    zeros_cont = cv2.morphologyEx(zeros_cont,cv2.MORPH_DILATE, se, iterations=1)
    
    cv2.imshow("temp_c", zeros_cont)
    cv2.imshow("temp_h", zeros_hull)
    se = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    temp =cv2.morphologyEx(cv2.bitwise_xor(thres,(zeros_cont-zeros_hull)), cv2.MORPH_ERODE, se, iterations=1)
    temp = ~temp
    tcontours, h = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for tc in tcontours:
        area = cv2.contourArea(tc)
        (p1,p2),(w,h),r = cv2.minAreaRect(tc)
        x, y, w, h = cv2.boundingRect(tc)
        ratio = 0
        if w == 0 or h == 0:
            continue
        if w > h:
            ratio = w/h
        else:
            ratio = h/w
        if area > 11 and area < 17 and (ratio < 1.5 and ratio > 1):
            x, y, w, h = cv2.boundingRect(tc)
            (cx,cy) = (x+w/2, y+h/2)
            cv2.circle(img, (int(cx),int(cy)), 10, (0,255,255), 2)
    cv2.imshow("temp_ch",temp )
    se = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1,3))
    zeros = cv2.morphologyEx(zeros,cv2.MORPH_OPEN, se, iterations=10)
    
    contours, h = cv2.findContours(zeros, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x,y),r = cv2.minEnclosingCircle(c)
        (p1,p2),(w,h),r = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        (cx,cy) = (x+w/2, y+h/2)
        cv2.circle(img, (int(cx),int(cy)), 10, (0,0,255), 2)
        cv2.drawContours(img, c, -1, (0,255,255), -1)

    cv2.imshow('zeros', zeros)
    cv2.imshow('thres', thres)
    
    cv2.imshow('img', img)
    
    key =  cv2.waitKey(5) & 0xff
    if key == ord('q'):
        break
    if key == ord('d'):
        i = (i+1)%len(files)
    if key == ord('a'):
        i = (i-1+len(files))%len(files)
    if key == ord('s'):
        cv2.imwrite(f'result/0000{i}.png', img)
cv2.destroyAllWindows()
