"""
Predict flower name from an image with predict.py along with the probability of that name.
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top_K most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""

import argparse
import fun_lib as fn

parser = argparse.ArgumentParser(description="Predict Image Class")

parser.add_argument('--image_path', default='flowers/test/1/image_06752.jpg', nargs='*',
                    action="store", type=str, metavar='', help="Define the directory for the Image")
parser.add_argument('--checkpoint', default='checkpoint.pth', nargs='*',
                    action="store", type=str, metavar='', help="Define the directory to the PTH file")
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int,
                    metavar='', help="To show the top_k Prediction")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json',
                    metavar='', help="Define The category name")
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu", metavar='', help="To use gpu power")
args = parser.parse_args()


_, _, _, cat_to_name, _ = fn.load_data(args.data_dir)
fn.predict_class(cat_to_name, args.imade_path, args.top_k, args.gpu)
