import torch
import torch.nn as nn

# PATH = '/media/mingfan/DATASSD/Video-Swin-Transformer/name_changed_weight.pth'
PATH = '/media/mingfan/DATASSD/Video-Swin-Transformer/work_dirs/k400_swin_base_patch244_window877.py/latest.pth'
# PATH = '/home/mingfan/Downloads/swin_tiny_patch244_window877_kinetics400_1k.pth'
model_pretrained =torch.load(PATH)

for key in list(model_pretrained['state_dict']):
    newKeyName_piece = key.split('.')[1:]
    newKeyName = '.'.join(newKeyName_piece)
    model_pretrained['state_dict'][newKeyName] = model_pretrained['state_dict'].pop(key)


torch.save(model_pretrained, './name_changed_weight.pth')