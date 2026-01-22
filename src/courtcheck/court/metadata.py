"""Court keypoint metadata: names, flip map, skeleton, and court lines."""

# Define keypoint names
KEYPOINT_NAMES = [
    "BTL", "BTLI", "BTRI", "BTR", "BBR", "BBRI", "IBR", "NR", "NM", "ITL",
    "ITM", "ITR", "NL", "BBL", "IBL", "IBM", "BBLI"
]

# Define keypoint flip map for data augmentation
KEYPOINT_FLIP_MAP = [
    ("BTL", "BTR"), ("BTLI", "BTRI"), ("BBL", "BBR"), ("BBLI", "BBRI"), ("ITL", "ITR"),
    ("ITM", "ITM"), ("NL", "NR"), ("IBL", "IBR"), ("IBM", "IBM"), ("NM", "NM")
]

# Skeleton connections (empty for now, can be extended)
SKELETON = []

# Court lines connecting keypoints
COURT_LINES = [
    ("BTL", "BTLI"), ("BTLI", "BTRI"), ("BTL", "NL"), ("BTLI", "ITL"),
    ("BTRI", "BTR"), ("BTR", "NR"), ("BTRI", "ITR"),
    ("ITL", "ITM"), ("ITM", "ITR"), ("ITL", "IBL"), ("ITM", "NM"), ("ITR", "IBR"),
    ("NL", "NM"), ("NL", "BBL"), ("NM", "IBM"), ("NR", "BBR"), ("NM", "NR"),
    ("IBL", "IBM"), ("IBM", "IBR"), ("IBL", "BBLI"), ("IBR", "BBRI"),
    ("BBR", "BBRI"), ("BBRI", "BBLI"), ("BBL", "BBLI")
]

# Line colors (RGB tuples for visualization)
LINE_COLORS = [(0, 255, 0)] * len(COURT_LINES)

