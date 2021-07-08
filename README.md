# Eye Tracking Using Python #

Program for eye tracking using a normal webcam (or any other video capture device).

### Short Explanation ###

The program uses OpenCV module and the haar cascade classifier for face recognition/tracking.

- The OpenCV module is used for getting the video from a capture device.
- Using functions provided from OpenCV, the captured video can be altered to help computer vision (making a colored video into a monoscale/gray video).
- The haar cascade classifiers are used to detect/recognize face and eyes in a frame/picture.
- Program uses blob on the frame to recognize eyes.

### Dependencies ###

- Python 3 (program made with ver 3.9.5)
- OpenCV module (version 4.5.2)
- NumPy module (version 1.20.3)
- Face and Eyes Haar cascade classifier (provided in the classifier folder)
- A video capture device (i.e. webcam)

### Program Set Up ###

- Install the required modules.
- Install the classifier file and include it in the program code with it's file path.
- Run main.py.

### Notes ###

```
#Get the video from a capture device
cam = cv2.VideoCapture(0)
```
- Change which video source to be captured with the program using the code above.
