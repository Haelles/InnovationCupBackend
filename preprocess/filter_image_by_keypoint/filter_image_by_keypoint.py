
import os
import fnmatch
import json

def get_points(path):
    data = json.load(open(path))
    peoples = data['people']
    point_list = peoples[0]['pose_keypoints_2d'] if len(peoples) > 0 else []

    return point_list

def get_all_paths(root):
    matches = []
    for root, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, '*.json'):
            matches.append(os.path.join(root, filename))
    return matches


filtered_file_set = set()
root = r'I:/datasets/fashion_editing/fashionE/fashionE_320_512_keypoint'
all_paths = get_all_paths(root)
for path in all_paths:
    point_list = get_points(path)
    point_num = (len(point_list) - point_list.count(0)) / 3
    if point_num >= 5:
        filtered_file_set.add(path.replace(root + '\\', '').replace('_keypoints.json', '.jpg').replace('\\', '/'))


file = open('./fashionE_list.txt', 'a')
for item in filtered_file_set:
    file.write(item + '\n')
file.close()

print(len(filtered_file_set))
print(len(all_paths))
