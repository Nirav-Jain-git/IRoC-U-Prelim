import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from skimage.feature import local_binary_pattern


SEED_PATH = r"img1.jpg"
CAND_PATH = r"img2.jpg"


LBP_DIST_THRESH = 0.55
CNN_SIM_THRESH  = 0.50
GEOM_SCORE_THRESH = 0.10

device = torch.device("cpu")

seed = cv2.imread(SEED_PATH, cv2.IMREAD_GRAYSCALE)
cand = cv2.imread(CAND_PATH, cv2.IMREAD_GRAYSCALE)
assert seed is not None and cand is not None


mobilenet = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).features[:7]

mobilenet.eval().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

@torch.no_grad()
def embed(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x = transform(rgb).unsqueeze(0).to(device)
    fmap = mobilenet(x)
    mean = fmap.mean(dim=(2, 3))
    std = fmap.std(dim=(2, 3))
    feat = torch.cat([mean, std], dim=1)
    feat = feat.squeeze().cpu().numpy()
    return feat / (np.linalg.norm(feat) + 1e-6)


e1 = embed(seed)
e2 = embed(cand)
cnn_sim = float(np.dot(e1, e2))


def lbp_hist(img):
    lbp = local_binary_pattern(img, 16, 2, "uniform")
    h, _ = np.histogram(lbp, bins=int(lbp.max() + 1))
    return h / (h.sum() + 1e-6), lbp

h1, lbp1 = lbp_hist(seed)
h2, lbp2 = lbp_hist(cand)

lbp_dist = 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-6))


sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(seed, None)
kp2, des2 = sift.detectAndCompute(cand, None)

geom_score = 0.0
match_img = None

if des1 is not None and des2 is not None:
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, 2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) >= 8:
        geom_score = min(len(good) / 50.0, 1.0)
        match_img = cv2.drawMatches(
            seed, kp1, cand, kp2, good[:30], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )


orb = cv2.ORB_create(nfeatures=2000)

kp1_o, des1_o = orb.detectAndCompute(seed, None)
kp2_o, des2_o = orb.detectAndCompute(cand, None)

orb_match_img = None
orb_match_count = 0

if des1_o is not None and des2_o is not None:
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb_matches = bf_orb.match(des1_o, des2_o)

    orb_matches = sorted(orb_matches, key=lambda m: m.distance)
    orb_match_count = len(orb_matches)

    if orb_match_count > 10:
        orb_match_img = cv2.drawMatches(
            seed, kp1_o, cand, kp2_o,
            orb_matches[:30], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )


score = 0.0
if cnn_sim >= CNN_SIM_THRESH:
    score += 0.4
if lbp_dist <= LBP_DIST_THRESH:
    score += 0.4
if geom_score >= GEOM_SCORE_THRESH:
    score += 0.2

accepted = score >= 0.6

seed_vis = cv2.cvtColor(seed, cv2.COLOR_GRAY2BGR)
cand_vis = cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR)

def resize_h(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*s), h))

h = min(seed_vis.shape[0], cand_vis.shape[0])
seed_vis = resize_h(seed_vis, h)
cand_vis = resize_h(cand_vis, h)

canvas = np.hstack([seed_vis, cand_vis])

lines = [
    f"CNN similarity : {cnn_sim:.3f}",
    f"LBP distance   : {lbp_dist:.3f}",
    f"SIFT score    : {geom_score:.2f}",
    f"ORB matches    : {orb_match_count}",
    f"DECISION      : {'SAME TERRAIN' if accepted else 'DIFFERENT'}"
]

for i, t in enumerate(lines):
    cv2.putText(canvas, t, (20, 35 + 35*i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

cv2.imshow("Terrain Comparison", canvas)


lbp_seed_vis = cv2.normalize(lbp1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
lbp_cand_vis = cv2.normalize(lbp2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("LBP Seed", lbp_seed_vis)
cv2.imshow("LBP Candidate", lbp_cand_vis)

if match_img is not None:
    cv2.imshow("SIFT Matches (Weak Geometry)", match_img)

if orb_match_img is not None:
    cv2.imshow("ORB Matches (Optional Exact Geometry)", orb_match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
