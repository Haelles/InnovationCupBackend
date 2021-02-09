
import os
import fnmatch

def get_all_paths(root):
    matches = []
    for root, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))
    return matches

root = r'I:/datasets/fashion_editing/fashionE/fashionE_320_512_image'
all_paths = get_all_paths(root)
file = open('./fashionE_allfile_list.txt', 'a')
for p in all_paths:
    filename = p.replace(root + '\\', '').replace('\\', '/') + '\n'
    file.write(filename)
file.close()

print(len(all_paths))