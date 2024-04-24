import cv2
import numpy as np
import imutils
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'




minArg=4
maxArg=6

img=cv2.imread("Img//bien_xe.png")
# cv2.imshow("Image1",img)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# cv2.imshow("RGB2GRAY",gray)


# Structure element
rectKern=cv2.getStructuringElement(cv2.MORPH_RECT,(13,5))
squareKern=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# print(rectKern)

img_gaussian=cv2.GaussianBlur(gray,(5,5),0)
# cv2.imshow("gaussian",img_gaussian)
blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rectKern)
# cv2.imshow("blackhat",blackhat)
light  =cv2.morphologyEx(img_gaussian,cv2.MORPH_CLOSE,squareKern)
light=cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("light",light)

sobelx=cv2.Sobel(src=blackhat,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=-1)
sobelx=np.absolute(sobelx)
(minVal,maxVal)=(np.min(sobelx),np.max(sobelx))

sobelx=255*((sobelx-minVal)/(maxVal-minVal))
sobelx=sobelx.astype("uint8")
sobelx=cv2.GaussianBlur(sobelx,(5,5),0)
sobelx=cv2.morphologyEx(sobelx,cv2.MORPH_CLOSE,rectKern)
cv2.imshow("sobelx",sobelx)

thresh=cv2.threshold(sobelx,0,225,cv2.THRESH_BINARY |cv2.THRESH_OTSU)[1]
thresh=cv2.erode(thresh,None,iterations=2)
thresh=cv2.dilate(thresh,None,iterations=2)
thresh=cv2.bitwise_and(thresh,thresh,mask=light)
thresh=cv2.dilate(thresh,None,iterations=2)
thresh=cv2.erode(thresh,None,iterations=1)

# cv2.imshow("thresh",thresh)


cnts=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]


for i in cnts:
    (x,y,w,h)=cv2.boundingRect(i)
    # print(x,y,w,h)
    # img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    arr=w/float(h)
    if arr >= minArg and arr <= maxArg:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # print(x, y, w, h)
        lpCnt = i
        gray = gray[y:y + h, x:x + w]
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


        alphanumeric="-.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options="-c tessedit_char_whitelist={}".format(alphanumeric)
        psm=7

        options+=" --psm {}".format(psm)
        clearBorder=False
        IpText=None

        lpText=pytesseract.image_to_string(gray,config=options)

print(lpText)
file = open("recognized.txt", "a")
file.write(lpText)
file.write("\n")
# Close the file
file.close()
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()