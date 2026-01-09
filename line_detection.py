import cv2
import numpy as np
from picamera2 import Picamera2

picam= Picamera2()
picam.configure(picam.create_video_configuration(main={"format": "RGB888", "size": (640,480)}))
picam.start()

while True:
	frame= picam.capture_array()

	hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_yellow= np.array([18, 80, 80])
	upper_yellow= np.array([35, 255, 255])

	mask= cv2.inRange(hsv, lower_yellow, upper_yellow)

	kernel= np.ones((5,5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	edges= cv2.Canny(mask, 50, 150, apertureSize=3)

	lines= cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=50, maxLineGap=10)

	if lines is not None:
		for line in lines:
			x1,y1,x2,y2= line[0]
			cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 3)

	
	cv2.imshow("Camera", frame)
	cv2.imshow("Mask", mask)
	cv2.imshow("Edges", edges)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break

cv2.destroyAllWindows()
