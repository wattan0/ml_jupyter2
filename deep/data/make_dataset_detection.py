import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import pickle


# PASCAL VOC from http://host.robots.ox.ac.uk/pascal/VOC/
data_dir = 'VOCdevkit/VOC2007'
targets = ('person', 'dog')  # 抽出するクラス
target_ids = []
n_images = 100  # 画像数
for target in targets:
    id_list_file = os.path.join(data_dir, 'ImageSets/Main/{}_train.txt'.format(target))

    with open(id_list_file, 'r') as f:
        lines = f.readlines()

    id_label = []
    for line in lines:
        id_, label = line.strip().split()
        id_label.append([id_, label])

    ids, labels = np.array(id_label).T
    target_ids.extend(ids[labels=='1'][:n_images])  # labels == 1ならその物体が画像に含まれている


dir = 'detection_data'
dst_dir = os.path.join(dir, 'JPEGImages')
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

dic = {}
idx = 0
for target_id in target_ids:
    src = os.path.join(data_dir, 'JPEGImages', target_id + '.jpg')
    shutil.copy(src, dst_dir)

    anno = ET.parse(os.path.join(data_dir, 'Annotations', target_id + '.xml'))
    bbox = []
    label = []
    for obj in anno.findall('object'):
        # when in not using difficult split, and the object is
        # difficult, skipt it.
        if int(obj.find('difficult').text) == 1:
            continue

        name = obj.find('name').text.lower().strip()
        if name not in targets:
            continue

        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
            int(bndbox_anno.find(tag).text) - 1
            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

        label.append(targets.index(name))

    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)

    dic[idx] = {'id': target_id,
                'bbox': bbox,
                'label': label}
    idx += 1

with open('detection_data/detection_data.pkl', 'wb') as f:
    pickle.dump(dic, f)
