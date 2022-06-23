import sys
sys.path.append(sys.path[0] + "/..")
import os
from file_io import load_image
import swc_handler


if __name__ == '__main__':
    test_dir = 'D:/PengLab/200k_testdata'
    swc_dir = os.path.join(test_dir, 'Img_X_5922.13_Y_7785.15_Z_1800.97.swc')
    img_dir = os.path.join(test_dir, 'Img_X_5922.13_Y_7785.15_Z_1800.97.v3dpbd')
    tree = swc_handler.parse_swc(swc_dir)