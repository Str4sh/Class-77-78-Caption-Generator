import cv2
from cvzone.FaceDetectionModule import FaceDetector

import numpy as np
from keras.models import load_model


detector=FaceDetector(minDetectionCon=0.8)

model=load_model("age_model.h5",compile=False)

video=cv2.VideoCapture(0)
while True:
    try:
        dummy,image=video.read()
        image=cv2.flip(image,1)

        image,bboxes=detector.findFaces(image,draw=False)


        if bboxes:
            for box in bboxes:
                x,y,w,h=box["bbox"]
                croppedImage=image[y:y+h,x:x+w]
                resizedImage=cv2.resize(croppedImage,(200,200))
                resizedImage_array=np.array([resizedImage])

                prediction=model.predict(resizedImage_array)
                print(prediction[0][0])


                image=cv2.putText(image,str(int(prediction[0][0])/100), (x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)               
                image=cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)




        cv2.imshow("Age Prediction",image)

        # cv2.imshow("Age Prediction", image)
 
        if cv2.waitKey(25)==32:
            break

    except Exception as e:
        print(e)

video.release()
cv2.destroyAllWindows()