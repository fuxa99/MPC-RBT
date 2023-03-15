import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input_file',help="Path to the input video file to be processed.")
parser.add_argument('-f',"--frame_count",required=False,default=10,help="Set every which frame should be saved from the input video.",type=int)
parser.add_argument('-o',"--output_dir",required=False,default='output',help="Path to output directory where output images should be stored.")
args = parser.parse_args()

try:
    if not os.path.isfile(args.input_file):
        raise Exception(f'Input file "{args.input_file}" does not exist, check path to the file.')

    output_path = os.path.join(args.output_dir,Path(args.input_file).stem)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(args.input_file)

    counter = 0
    every_what = args.frame_count
    counter_glob = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            counter += 1
            if counter % every_what == 0:
                counter = 0
                name = os.path.join(output_path,f'{counter_glob:04}.jpg')
                cv2.imwrite(name,frame)
                counter_glob += 1
                print(f'Frame saved: {name}')
        else:
            break
    
    print(f'Successfully saved {counter_glob} frames.')
    cap.release()

except Exception as exc:
    print(exc)