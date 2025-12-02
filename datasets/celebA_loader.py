import os

def load_celeba_bboxes(root):
    """
    Returns dict: {image_name: [x1,y1,x2,y2]}
    """
    bbox_file = os.path.join(root, "Anno/list_bbox_celeba.txt")
    out = {}

    with open(bbox_file, "r") as f:
        lines = f.read().strip().split("\n")

    # Skip header lines
    lines = lines[2:]

    for line in lines:
        parts = line.split()
        image_id = parts[0]
        x, y, w, h = map(int, parts[1:])
        out[image_id] = [x, y, x+w, y+h]

    return out
