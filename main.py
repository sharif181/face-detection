import cv2

faceCascadeObject = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
webCam = cv2.VideoCapture(0)
webCam.set(3,400)
webCam.set(4,400)

while True:
    success,img = webCam.read()
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascadeObject.detectMultiScale(grayImage,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,124,143),2)
        cv2.putText(img,"Face",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Faces",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
