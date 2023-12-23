import os

data_root = '/temp/data/PennFudanPed'
save_path = os.path.expanduser('~/VisionFinetuning/results')

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
