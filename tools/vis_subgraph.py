import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

sns.set(font_scale=1.2)

graph_a = pickle.load(open('/home/cyan/code/mmdetection/tools/graph/new_COCO_graph_a.pkl', 'rb'))
graph_r = pickle.load(open('/home/cyan/code/mmdetection/tools/graph/new_COCO_graph_r.pkl', 'rb'))
graph_a = np.float32(graph_a)
graph_r = np.float32(graph_r)

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
#graph_r = 1 - graph_r
#graph_a = 1 - graph_a
start = 43
end = 53
f, (ax1,ax2) = plt.subplots(figsize=(10,5), ncols=2)

sns.heatmap(graph_r[start:end,start:end], cmap=cm.Blues, annot=True, annot_kws={'size':8}, cbar_kws={"shrink":0.2}, ax=ax1, linewidths = 0.02, xticklabels=CLASSES[start:end], yticklabels=CLASSES[start:end], square=True)
labelx = ax1.get_xticklabels()
plt.setp(labelx, rotation=30, horizontalalignment='right')
ax1.set_title("Visualization of Relation Subgraph")
sns.heatmap(graph_a[start:end,start:end], cmap=cm.Blues, annot=True, annot_kws={'size':8}, cbar_kws={"shrink":0.2}, ax=ax2, linewidths = 0.02, xticklabels=CLASSES[start:end], yticklabels=CLASSES[start:end], square=True)
labelx = ax2.get_xticklabels()
plt.setp(labelx, rotation=30, horizontalalignment='right')
ax2.set_title("Visualization of Attribute Subgraph")
plt.savefig('./work_dirs/vis/subgraph2.png')
plt.show()