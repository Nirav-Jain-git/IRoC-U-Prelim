#!/usr/bin/env python3
from picamera2 import Picamera2
import numpy as np
import serial
import cv2 as cv

ser = serial.Serial(
    port='/dev/serial0',
    baudrate=115200,
    timeout=1
)            
picam2 = Picamera2()
count = 10
config = picam2.create_preview_configuration(
    main={"format":"RGB888","size":(640,480)}
    )
picam2.configure(config)
picam2.start()
# ================= USER-ADJUSTABLE PARAMETERS =================
fx = 549.89   # focal length in pixels (CHANGE LATER after calibration)
fy = 551.43
Z  = 0.3     # height above ground in meters (from lidar / fixed for now)
# =============================================================

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

class App:
    def __init__(self, video_src=0):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0

        self.cap = cv.VideoCapture(video_src)
        self.prev_gray = None

        # ----------- Position state (meters) -----------
        self.X = 0.0
        self.Y = 0.0
        #self.Z = 0.0
        
    def run(self):
        while True:
            frame = picam2.capture_array()
            #frame = cv.flip(frame, 1)
            #frame = cv.undistort(frame, K, dist)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            dx_list = []
            dy_list = []

            # ================= TRACKING =================
            if self.tracks and self.prev_gray is not None:
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                p1, _, _ = cv.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, p0, None, **lk_params
                )

                p0r, _, _ = cv.calcOpticalFlowPyrLK(
                    gray, self.prev_gray, p1, None, **lk_params
                )

                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                for tr, (x, y), (x0, y0), good_flag in zip(
                        self.tracks,
                        p1.reshape(-1, 2),
                        p0.reshape(-1, 2),
                        good):

                    if not good_flag:
                        continue

                    # Pixel displacement
                    dx_list.append(x - x0)
                    dy_list.append(y - y0)

                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        tr.pop(0)

                    new_tracks.append(tr)
                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks],
                             False, (0, 255, 0))

                
                # ========== POSITION UPDATE ==========
                if len(dx_list) > 10:  # ensure enough points
                    dx_med = np.median(dx_list)
                    dy_med = np.median(dy_list)

                    
                    # Pixel → meter conversion
                    dX = (dx_med * Z) / fx
                    dY = (dy_med * Z) / fy

                    self.X += dX
                    self.Y += dY

            # ================= FEATURE DETECTION =================
            if self.frame_idx % self.detect_interval == 0:
                mask = np.ones_like(gray) * 255
                for tr in self.tracks:
                    x, y = map(int, tr[-1])
                    cv.circle(mask, (x, y), 5, 0, -1)

                p = cv.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            # ================= DISPLAY =================
            cv.putText(vis, f"X: {self.X:.2f} m",
                       (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(vis, f"Y: {self.Y:.2f} m",
                       (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(vis, f"Z: {Z:.2f} m",
		       (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv.putText(vis, f"Features: {len(self.tracks)}",
             #          (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.prev_gray = gray
            self.frame_idx += 1

            cv.imshow("Optical Flow Position Estimation", vis)
            if cv.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    App(0).run()
