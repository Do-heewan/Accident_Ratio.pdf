import os.path as osp
import copy

multi_class = True
video_infos = []
fin = open('D:\\DLCode\\VideoRecongnition\\labelTest.txt', 'r')
# with open('D:\\DLCode\\VideoRecongnition\\labelTest.txt', 'r') as fin:
lines=fin.readlines()
print(len(lines))

line_split = lines[0].strip().split()
video = line_split[0]
label = line_split[2:]
label = [int(x) for x in label]
print(video)
print(label)

for line in fin:
    line_split = line.strip().split()
    print(line_split)
    video_info = {}
    idx = 0
    # idx for frame_dir
    frame_dir = line_split[idx]
    idx += 1
    # idx for total_frames
    video_info['total_frames'] = int(line_split[idx])
    idx += 1
    # idx for label[s]
    label = [int(x) for x in line_split[idx:]]
    assert label, f'missing label in line: {line}'
    if multi_class:
        video_info['label'] = label
    else:
        assert len(label) == 1
        video_info['label'] = label[0]
    video_infos.append(video_info)
        
print(video_infos)