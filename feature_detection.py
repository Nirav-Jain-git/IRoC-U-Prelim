import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import models
from torch.nn import functional as F

# ============================================================
# SAFETY
# ============================================================
cv2.setNumThreads(1)
torch.set_num_threads(1)

DEVICE = "cpu"

# ============================================================
# PATHS
# ============================================================
SEED_PATHS = [
    "seed/image4.png",
    "seed/image3.png"
]

VIDEO_PATH = "vid/vid1.mp4"

# ============================================================
# PARAMETERS (FINAL TUNED)
# ============================================================
WINDOW_SIZES = [224, 320]
STRIDE_RATIO = 0.5

DETECT_EVERY = 12
MAX_TRACKS = 5
TRACK_HOLD = 20

IOU_THRESHOLD = 0.3
DIST_THRESH = 120

CONFIRM_FRAMES = 2
RELATIVE_MARGIN = 0.02

# ============================================================
# CNN EMBEDDING MODEL
# ============================================================
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)
model.classifier = torch.nn.Identity()
model.eval().to(DEVICE)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def embed(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        f = model(t)
    return F.normalize(f, dim=1)

# ============================================================
# GEOMETRY HELPERS
# ============================================================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def center(b):
    return ((b[0]+b[2])//2, (b[1]+b[3])//2)

def dist(a, b):
    c1, c2 = center(a), center(b)
    return np.hypot(c1[0]-c2[0], c1[1]-c2[1])

# ============================================================
# LOAD SEEDS & ADAPTIVE THRESHOLD
# ============================================================
seed_embs = []
for p in SEED_PATHS:
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError(f"Missing seed {p}")
    seed_embs.append(embed(img))

seed_embs = torch.cat(seed_embs, dim=0)
mean_seed = seed_embs.mean(dim=0, keepdim=True)

seed_sim = torch.matmul(seed_embs, seed_embs.T)
intra_sim = seed_sim[seed_sim < 0.999].mean().item()
SIM_THRESHOLD = intra_sim * 0.85

print(f"[INFO] Similarity threshold = {SIM_THRESHOLD:.3f}")

# ============================================================
# VIDEO
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video not found")

tracks = []
frame_id = 0
last_detect_frame = -999

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ---------------- FAST PATH ----------------
    if frame_id - last_detect_frame < DETECT_EVERY:
        for t in tracks:
            if t["conf"] < CONFIRM_FRAMES:
                continue
            x1,y1,x2,y2 = t["box"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("GENERIC SEARCH", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # ---------------- DETECTION ----------------
    last_detect_frame = frame_id
    h,w,_ = frame.shape
    raw_dets = []

    global_score = torch.sum(embed(frame)*mean_seed).item()
    if global_score >= SIM_THRESHOLD * 0.9:

        window_scores = []

        for win in WINDOW_SIZES:
            stride = int(win * STRIDE_RATIO)
            if win >= h or win >= w:
                continue

            for y in range(0, h-win, stride):
                for x in range(0, w-win, stride):
                    crop = frame[y:y+win, x:x+win]
                    score = torch.sum(embed(crop)*mean_seed).item()
                    window_scores.append(score)
                    if score >= SIM_THRESHOLD:
                        raw_dets.append((score,(x,y,x+win,y+win)))

        if len(window_scores):
            mean_score = np.mean(window_scores)
            raw_dets = [
                d for d in raw_dets
                if d[0] > mean_score + RELATIVE_MARGIN
            ]

    # ---------------- CLUSTER + AVERAGE ----------------
    raw_dets.sort(reverse=True, key=lambda x: x[0])
    clusters = []

    for score, box in raw_dets:
        assigned = False
        for c in clusters:
            if iou(box, c["rep"]) > IOU_THRESHOLD or dist(box, c["rep"]) < DIST_THRESH:
                c["boxes"].append(box)
                c["scores"].append(score)
                xs = [b[0] for b in c["boxes"]]
                ys = [b[1] for b in c["boxes"]]
                xe = [b[2] for b in c["boxes"]]
                ye = [b[3] for b in c["boxes"]]
                c["rep"] = (
                    int(np.mean(xs)),
                    int(np.mean(ys)),
                    int(np.mean(xe)),
                    int(np.mean(ye))
                )
                assigned = True
                break

        if not assigned:
            clusters.append({
                "boxes": [box],
                "scores": [score],
                "rep": box
            })

    detections = []
    for c in clusters[:MAX_TRACKS]:
        detections.append((np.mean(c["scores"]), c["rep"]))

    # ---------------- TRACK UPDATE ----------------
    for score, box in detections:
        matched = False
        for t in tracks:
            if iou(box, t["box"]) > IOU_THRESHOLD:
                t["box"] = box
                t["hold"] = TRACK_HOLD
                t["conf"] = min(t["conf"] + 2, 10)
                matched = True
                break

        if not matched:
            tracks.append({
                "box": box,
                "hold": TRACK_HOLD,
                "conf": 1
            })

    # ---------------- TRACK DECAY ----------------
    new_tracks = []
    for t in tracks:
        t["hold"] -= 1
        if t["hold"] > 0:
            new_tracks.append(t)
    tracks = new_tracks[:MAX_TRACKS]

    # ---------------- DRAW ----------------
    for t in tracks:
        if t["conf"] < CONFIRM_FRAMES:
            continue
        x1,y1,x2,y2 = t["box"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("GENERIC SEARCH", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
