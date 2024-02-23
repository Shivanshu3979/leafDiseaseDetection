import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from model import *
import numpy as np
import os
import tensorflow as tf

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    highlight=255;
    shadow=0;
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
            
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
        
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
            
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
    
color = "rgb"
bins = 16
resizeWidth = 0
# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
else:
    ax.set_title('Histogram (grayscale)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 6
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
    lr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#654321', lw=lw, alpha=alpha)
    lbr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#123456', lw=lw, alpha=alpha)
    lg, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#225522', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
plt.ion()
#plt.show()
bi=None
gi=None
ri=None

    
#harcascade file to detect Face
facec = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
#loading emaotion Detection Model
model = FacialExpressionModel("models/emotions_model.json", "models/emotions_dataset.h5")
g_model=GenderModel("models/gender_model.json","models/gender_prediction.h5")
leaf_model=LeafModel("models/leaf_Diseases_rgb.json","models/leaf_Diseases_rgb.h5") #for rgb image
#leaf_model=LeafModel("models/leaf_Diseases_gray.json","models/leaf_Diseases_gray.h5") #for gray image
font = cv2.FONT_HERSHEY_DUPLEX

process=input("[1] for image from files\n[2] camera\n choice: ")
if process=="1":
        flag=0;
        demoimg=storage=0;
        for img_ex in os.listdir("Try_images/"):
            image=cv2.imread("Try_images/"+img_ex)
            
            
            image=cv2.resize(image,(300,300))
            scale_factor=255/np.max(image)
            for i in range(len(image)):
                for j in range(len(image[1])):
                    image[i][j][0]=int(image[i][j][0]*scale_factor)
                    image[i][j][1]=int(image[i][j][1]*scale_factor)
                    image[i][j][2]=int(image[i][j][2]*scale_factor)
            numPixels = np.prod(image.shape[:2])
            demoimg= np.zeros(image.shape, dtype="uint8")
            
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
            image.flags.writeable = False
            #results = holistic.process(image)

            
            gray_fr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.2, 5)
            count=0;
            flag=1
            #img=apply_brightness_contrast(gray_fr, 64, 64)#for gray image
            img=apply_brightness_contrast(image, 64, 64)#for rgb image
            
            img=cv2.resize(img,(128,128))
            img=img[np.newaxis,:,:] 
            pred_leaf_list=leaf_model.predict_leaf(img);
            output="Plant:\t"+pred_leaf_list[0]+"\nStatus:\t"+pred_leaf_list[1];
            cv2.putText(image, "Plant: ", (10,20), font, 0.8, (255, 255, 110), 2)
            cv2.putText(image, pred_leaf_list[0], (120,20), font, 0.7, (255, 0, 110), 1);
            cv2.putText(image, "Status: ", (10,40), font, 0.8, (255, 255, 110), 2);
            cv2.putText(image, pred_leaf_list[1], (120,40), font, 0.7, (255, 0, 110), 1);
            cv2.putText(image, "Accuracy: ", (10,60), font, 0.8, (255, 255, 110), 2);
            cv2.putText(image, pred_leaf_list[2], (150,60), font, 0.6, (255, 0, 110), 1);
            cv2.putText(demoimg, "Face Detection Frame", (10,20), font, 1, (255, 0, 110), 3);
            for (x, y, w, h) in faces:
                    fc = gray_fr[y:y+h, x:x+w]

                    roi = cv2.resize(fc, (48, 48))
                    roi2 = cv2.resize(fc, (64, 64))
                    pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                    data=cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
                    
                    pred_g=g_model.predict_gender(roi2[np.newaxis, :, :, np.newaxis])
                    if flag:
                        cv2.putText(demoimg, "Expresion: "+pred, (x+10, y+w+10),
                                    font, 1*(abs(w)/240), (255, 255, 110), 1)
                        
                        cv2.putText(demoimg, "Gender: "+pred_g, (x+10, y+w+35),
                                    font, 1*(abs(w)/240), (255, 255, 110), 1)
                        
                        cv2.rectangle(demoimg,(x,y),(x+w+10,y+h+40),(255,0,0),2)
                        flag=0;
                    else:
                        cv2.putText(demoimg, "Expresion: "+pred, (x+10, y-35),
                                    font, 1*(abs(w-x)/240), (255, 255, 110), 1)
                        
                        cv2.putText(demoimg, "Gender: "+pred_g, (x+10, y-10),
                                    font, 1*(abs(w-x)/240), (255, 255, 110), 1)
                        
                        cv2.rectangle(demoimg,(x-10,y-20),(x+w+10,y+h+40),(255,0,0),1)
                        flag=1;
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                    #print( mp_holistic.FACEMESH_TESSELATION)
                    
                
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            
            

            
            #image=cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
            #demoimg=cv2.resize(demoimg,(image.shape[1]//2,image.shape[0]//2))
            (b, g, r) = cv2.split(image)
            
            if bi==0 or gi==0 or ri==0:
                bi=b
                gi=g
                ri=r
                histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
                lr.set_ydata(histogramRi)
                lg.set_ydata(histogramGi)
                lbr.set_ydata(histogramBi)
            else:
                histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
                histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
                lineR.set_ydata(histogramR)
                lineG.set_ydata(histogramG)
                lineB.set_ydata(histogramB)
                fig.canvas.draw()
                lr.set_ydata(histogramRi)
                lg.set_ydata(histogramGi)
                lbr.set_ydata(histogramBi)
            fig.canvas.draw()
            fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
            plotteddata=cv2.resize(cv2.imread("plot.jpg"),(image.shape[1],image.shape[0]))
            halfFrame=np.concatenate((demoimg,plotteddata),axis=0)
            halfFrame=cv2.resize(halfFrame,(halfFrame.shape[1]//2,halfFrame.shape[0]//2))
            #cv2.imshow("halfFrame",halfFrame)
            fullFrame=np.column_stack((image,halfFrame))
            cv2.imshow(img_ex,fullFrame);
            if cv2.waitKey(1) == ord('q'):
              cv2.destroyAllWindows()
else:
    flag=0;
    demoimg=storage=0;
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        numPixels = np.prod(image.shape[:2])
        demoimg= np.zeros(image.shape, dtype="uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        #results = holistic.process(image)

        
        gray_fr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.2, 5)
        count=0;
        flag=1
        img=apply_brightness_contrast(gray_fr, 64, 64)#for gray image
        #img=apply_brightness_contrast(image, 64, 64)#for rgb image
        if len(faces)==0:
            img=cv2.resize(img,(128,128))
            img=img[np.newaxis,:,:] 
            pred_leaf_list=leaf_model.predict_leaf(img);
            output="Plant:\t"+pred_leaf_list[0]+"\nStatus:\t"+pred_leaf_list[1];
            cv2.putText(image, "Plant: ", (10,20), font, 0.8, (255, 255, 110), 2)
            cv2.putText(image, pred_leaf_list[0], (120,20), font, 0.7, (255, 0, 110), 1);
            cv2.putText(image, "Status: ", (10,40), font, 0.8, (255, 255, 110), 2);
            cv2.putText(image, pred_leaf_list[1], (120,40), font, 0.7, (255, 0, 110), 1);
            cv2.putText(image, "Accuracy: ", (10,60), font, 0.8, (255, 255, 110), 2);
            cv2.putText(image, pred_leaf_list[2], (150,60), font, 0.6, (255, 0, 110), 1);
        cv2.putText(demoimg, "Face Detection Frame", (10,20), font, 1, (255, 0, 110), 3);
        for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]

                roi = cv2.resize(fc, (48, 48))
                roi2 = cv2.resize(fc, (64, 64))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                data=cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
                
                pred_g=g_model.predict_gender(roi2[np.newaxis, :, :, np.newaxis])
                if flag:
                    cv2.putText(demoimg, "Expresion: "+pred, (x+10, y+w+10),
                                font, 1*(abs(w)/240), (255, 255, 110), 1)
                    
                    cv2.putText(demoimg, "Gender: "+pred_g, (x+10, y+w+35),
                                font, 1*(abs(w)/240), (255, 255, 110), 1)
                    
                    cv2.rectangle(demoimg,(x,y),(x+w+10,y+h+40),(255,0,0),2)
                    flag=0;
                else:
                    cv2.putText(demoimg, "Expresion: "+pred, (x+10, y-35),
                                font, 1*(abs(w-x)/240), (255, 255, 110), 1)
                    
                    cv2.putText(demoimg, "Gender: "+pred_g, (x+10, y-10),
                                font, 1*(abs(w-x)/240), (255, 255, 110), 1)
                    
                    cv2.rectangle(demoimg,(x-10,y-20),(x+w+10,y+h+40),(255,0,0),1)
                    flag=1;
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                #print( mp_holistic.FACEMESH_TESSELATION)
                
            
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        
        

        
        #image=cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
        #demoimg=cv2.resize(demoimg,(image.shape[1]//2,image.shape[0]//2))
        (b, g, r) = cv2.split(image)
        
        if bi==0 or gi==0 or ri==0:
            bi=b
            gi=g
            ri=r
            histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
            lr.set_ydata(histogramRi)
            lg.set_ydata(histogramGi)
            lbr.set_ydata(histogramBi)
        else:
            histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
            histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
            lineR.set_ydata(histogramR)
            lineG.set_ydata(histogramG)
            lineB.set_ydata(histogramB)
            fig.canvas.draw()
            lr.set_ydata(histogramRi)
            lg.set_ydata(histogramGi)
            lbr.set_ydata(histogramBi)
        fig.canvas.draw()
        fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
        plotteddata=cv2.resize(cv2.imread("plot.jpg"),(image.shape[1],image.shape[0]))
        halfFrame=np.concatenate((demoimg,plotteddata),axis=0)
        halfFrame=cv2.resize(halfFrame,(halfFrame.shape[1]//2,halfFrame.shape[0]//2))
        #cv2.imshow("halfFrame",halfFrame)
        fullFrame=np.column_stack((image,halfFrame))
        cv2.imshow("Data Visualization",fullFrame);
        if cv2.waitKey(1) == ord('q'):
          break
    cap.release()
    cv2.destroyAllWindows()

