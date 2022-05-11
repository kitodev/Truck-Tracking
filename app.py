import cv2

img_file = 'car2.png'
video = cv2.VideoCapture('car_video.mp4')

classifier = 'car_detect.xml'

car_track = cv2.CascadeClassifier(classifier)

""" while True:
    (read_succ, frame) = video.read()
    if read_succ:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_track.detectMultiScale(grayscale)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    
    cv2.imshow('Truck-Tracker', frame)

    cv2.waitKey(1) """

img = cv2.imread(img_file)

bgrtogray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_track = cv2.CascadeClassifier(classifier)

cars = car_track.detectMultiScale(bgrtogray, 1.3, 5)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Truck-Tracker', img)

cv2.waitKey()