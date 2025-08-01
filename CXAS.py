import os
os.chdir('../')
from PIL import Image
import cxas.visualize as cxas_vis
import argparse

def CXAS(input, label, output, target):
    out_dir = output

    # print('Set of files before storing:', os.listdir(out_dir))

    _ = cxas_vis.visualize_from_file(
        class_names = [target],
        img_path = input,
        label_path = label,
        img_size = 512,
        cat      = True,
        axis     = 1,
        do_store = True,
        out_dir  = output,
    )

    # print('Set of files after storing:',os.listdir(out_dir))

    Image.open(os.path.join(out_dir, os.listdir(out_dir)[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input')
    parser.add_argument('-label')
    parser.add_argument('-output')
    parser.add_argument('-target')
    args = parser.parse_args()

    CXAS(args.input,args.label,args.output,args.target)