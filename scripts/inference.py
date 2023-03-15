import cv2
import torch
import argparse
from pathlib import Path
import os

def find_images(path,file_type):
    if not os.path.isdir(path):
        raise Exception(f'Input directory "{args.input_dir}" does not exist. Check the dir path.')
    
    for path, subdirs, files in os.walk(path):
        imgs = [os.path.join(path,x) for x in files if file_type in x]

        if len(imgs) == 0:
            raise Exception(f'Input directory "{path}" does not contain any {file_type} files.')
        
        return imgs
    return None

def score_image(model,image):
    results = model([image])

    return results.pandas().xyxy[0]

def plot_boxes(results,image):
    for i in range(len(results.xmin)):
        x1, y1, x2, y2 = int(results.xmin[i]), int(results.ymin[i]), int(results.xmax[i]), int(results.ymax[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
        cv2.putText(image, f'{results.name[i]} {float(results.confidence[i]):.3f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return image

#ToDo solve default path to trained model
parser = argparse.ArgumentParser()
parser.add_argument('-c',"--confidence",required=False,default=0.0,type=float,help='Confidence threshold. Valid interval <0;1>.')
parser.add_argument('-i',"--input_dir",required=False,default='input',type=str,help='Path to input directory containing images to be processed.')
parser.add_argument('-m',"--model",required=False,default='model_path',type=str,help='Path to trained model.')
parser.add_argument('-f',"--file_type",required=False,default='png',type=str,help='Image file type, for example tif,png,bmp...')
parser.add_argument('-o',"--output_dir",required=False,default='results',type=str,help='Path to output directory.')
args = parser.parse_args()

try:
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo/runs/train/exp54/weights/best.pt')  # custom model
    """if not os.path.isfile(args.model):
        raise Exception(f'Trained model "{args.model} does not exist. Check the path and try again."')"""
    if not os.path.isdir(args.output_dir):
        print(f'Creating "{args.output_dir}" directory.')
        os.makedirs(args.output_dir)

    images = find_images(args.input_dir,args.file_type)
    if images is None:
        raise Exception(f'Invalid input directory.')

    model = torch.hub.load('ultralytics/yolov5','yolov5s')

    if args.confidence < 0 or args.confidence > 1:
        print('Invalid confidence threshold, skipping.')
    elif args.confidence != 0:
        model.conf = args.confidence

    print(f'Processing {len(images)} images...')
    for i,name in enumerate(images):
        print(f'{i+1}/{len(images)}... {name}')
        iput = cv2.imread(name)

        results = score_image(model,iput)
        oput = plot_boxes(results,iput)
        cv2.imwrite(os.path.join(args.output_dir,Path(name).stem+'.png'),oput)
    
    print('Done')

except Exception as exc:
    print(exc)