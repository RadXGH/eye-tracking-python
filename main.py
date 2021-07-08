import cv2
import numpy as np

#define cascade path for face and eyes recognition
faceCascade = cv2.CascadeClassifier("D:/Coding/Python 3/Eye Tracking/classifier/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("D:/Coding/Python 3/Eye Tracking/classifier/haarcascade_eye.xml")
#blob init
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detectFace(img, cascade):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect the coords of faces in the gray image
    coords = cascade.detectMultiScale(
        grayImg,
        scaleFactor = 1.2,
        minNeighbors = 4,
        minSize = (30, 30)
    )
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    frame = None
    #Draw a rectangle around the faces
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

def detectEyes(img, cascade):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect the coords of eyes in the gray image
    coords = cascade.detectMultiScale(
        grayImg,
        scaleFactor = 1.2,
        minNeighbors = 4,
    )
    #find the height and width of the eyes in relation from the face
    height = np.size(img, 0)
    width = np.size(img , 1)

    leftEye = None
    rightEye = None
    for (x, y, w, h) in coords:
        if y > height / 2:
            pass
        eyecenter = x + w / 2
        if eyecenter < width * 0.5:
            leftEye = img[y:y + h, x:x + w]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            rightEye = img[y:y + h, x:x + w]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
    return leftEye, rightEye

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

def blob_process(img, threshold, detector):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(grayImg, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints

def nothing(x):
    pass

def main():
    #assign the captured video from webcam into cam
    cam = cv2.VideoCapture(0)
    #changing the resolution of the video from the webcam
    cam.set(3, 1024)
    cam.set(4, 576)

    cv2.namedWindow('eye tracking')
    cv2.createTrackbar('threshold', 'eye tracking', 0, 255, nothing)
    while True:
        ret, img = cam.read() #assigning the video into an image per frame
        
        faceFrame = detectFace(img, faceCascade)
        if faceFrame is not None:
            eyesFrame = detectEyes(img, eyesCascade)
            for eye in eyesFrame:
               if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'eye tracking')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #press ESC to quit the program
        cv2.imshow('eye tracking', img)
        if cv2.waitKey(5) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

main()