'''Import the necessary packages'''
import numpy
import argparse
import cv2
import imutils
from imutils import contours as ct

'''Get the image'''
argparse = argparse.ArgumentParser()
argparse.add_argument("-i", "--image", required=True, help="Adds the image from the command line")
image = vars(argparse.parse_args())

'''Define the answer key that maps to the question number
The answers are
A, B, C, D, E, E, D, C, B, A
'''
ANSWER_KEY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}

'''Preprocess the input image'''
image = cv2.imread(image["image"])
#change the width and height of the original image
imageCopy = imutils.resize(image, height=500)
#Start main preprocessing
gray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (1, 1), 0)
edged = cv2.Canny(blur, 105, 150)

'''Display each image when it was processed'''
cv2.imshow("Original Image", imageCopy)
cv2.waitKey(0)

cv2.imshow("Grayed Image", gray)
cv2.waitKey(0)

cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)

cv2.imshow("Edged Image", edged)
cv2.waitKey(0)

'''Apply binary threshold to the image'''
binImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Binary Image", binImage)
cv2.waitKey(0)

#Find the contours in the image then the bubble contours
contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

#question bubble variable
questionBubble = []

#find the bubble contours
for item in contours:
  #find the bounding box of the contour and find the aspect ratio
  (x, y, w, h) = cv2.boundingRect(item)
  aspRatio = w / float(h)

  #append the bubble to the bubble variable
  if w >= 20 and h >= 20 and aspRatio >= 0.8 and aspRatio <= 1.5:
    questionBubble.append(item)

questionColoured = imageCopy.copy()
for item in questionBubble:
  cv2.drawContours(questionColoured, [item], -1, (240, 0, 159), 2)

cv2.imshow("Contours", questionColoured)
cv2.waitKey(0)

#sort the contours and arrange them from top to bottom
arrangedContours = ct.sort_contours(questionBubble, method="top-to-bottom")[0]
#draw different colours for each row
colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (50, 50, 50), (100, 100, 100)]
#loop through the contour and draw the colours
#increment count
count = 0
step = 0
colour = colours[step]
sortedQuestion = imageCopy.copy()
for item in arrangedContours:
  if count % 5 == 0:
    step = step + 1
    colour = colours[step]
    print("Colour {} is: {}".format(count, colour))

    if step == 4 or step > 4:
      step = 0

  count = count + 1
  cv2.drawContours(sortedQuestion, [item], -1, colour, 2)

cv2.imshow("Sorted Questions", sortedQuestion)
cv2.waitKey(0)
