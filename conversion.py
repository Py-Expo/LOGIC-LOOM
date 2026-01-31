from os import path
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py
from PIL import Image
import re
import sys
from glob import glob
import numpy as np
import h5py

dir_path = path.dirname(path.abspath(__file__))
path_to_mat_files = path.join(dir_path, "*.mat")
found_files = glob(path_to_mat_files, recursive=True)
total_files = 0

f = h5py.File('1.mat','r')
data = f.get('data/variable1')
data = np.array(data) 

print (data)

def convert_to_png(file: str, number: int):
    global total_files
    if path.exists(file):
        print(file, "already exist\nSkipping...")
    else:
        h5_file = h5py.File(file, 'r')
        png = file[:-3] + "png"
        cjdata = h5_file['cjdata']
        image = np.array(cjdata.get('image')).astype(np.float64)
        label = cjdata.get('label')[0,0]
        PID = cjdata.get('PID')
        PID = ''.join(chr(c) for c in PID)
        tumorBorder = np.array(cjdata.get('tumorBorder'))[0]
        tumorMask = np.array(cjdata.get('tumorMask'))
        h5_file.close()
        hi = np.max(image)
        lo = np.min(image)
        image = (((image - lo)/(hi-lo))*255).astype(np.uint8)
        im = Image.fromarray(image)
        im.save(png)
        os.system(f"mv {png} {dir_path}\\png_images")#make sure folder png_images exist
        total_files += 1
        print("saving", png, "File No: ", number)
        
for file in found_files:
    if "cvind.mat" in file:
        continue
    convert_to_png(file, total_files)
print("Finished converting all files: ", total_files)