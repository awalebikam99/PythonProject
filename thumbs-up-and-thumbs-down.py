import cv2
import math
import numpy as np

camera = cv2.VideoCapture(0)
while camera.isOpened():
    reb, img = camera.read()
    cv2.rectangle(img, (300, 300), (150, 150), (120, 212, 120), 0)
    anonymous_img = img[150:300, 150:300]

    grey = cv2.cvtColor(anonymous_img, cv2.COLOR_BGR2GRAY)
    Am = (15, 15)
    Bi = cv2.GaussianBlur(grey, Am, 0)
    _, thresh = cv2.threshold(Bi, 100, 100, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (version, _, _) = cv2.__version__.split('.')
    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh.copy(),
               cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,
               cv2.CHAIN_APPROX_TC89_KCOS)

    on = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(on)
    drawing = np.zeros(anonymous_img.shape, np.uint8)
    # we are drawing contours
    cv2.drawContours(drawing, [on], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    hull = cv2.convexHull(on, returnPoints = False)
    defects = cv2.convexityDefects(on, hull)
    countdefects = 0
    cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)
    # Through this rule we want to find angle of the thumb.
    for i in range(defects.shape[0]):
        a, b, c, d = defects[i, 0]
        start = tuple(on[a][0])
        end = tuple(on[b][0])
        far = tuple(on[c][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        # we used acos method to define arc cosine of our variables from radians.
        Ang = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        # we declared the angle of the thumb based on the thumb position.
        if Ang <= 90:
            countdefects = countdefects + 1
    if countdefects == 1:
        # In this section we declared rules based on thumbs conditions.
        cv2.putText(img, "Thumb Down Zoom out", (120, 310), cv2.FONT_HERSHEY_PLAIN, 2,412)
        pts1 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(500,500))
    else:
        cv2.putText(img, "Thumb Up Zoom in", (120, 310), cv2.FONT_HERSHEY_PLAIN, 2, 412)
        pts1 = np.float32([[120,120],[380,120],[120,380],[380,380]])
        pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(600,600))
    cv2.imshow('Thumb Recognition', dst)
    all_img = np.hstack((drawing, anonymous_img))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()