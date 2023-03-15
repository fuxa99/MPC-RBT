from roboflow import Roboflow
import shutil
import os
import glob

def handle_train_dir():
    res_dir_count = len(glob.glob('runs/train/*'))
    print(f"Current number of result directories: {res_dir_count}")
    dir_name = f"results_{res_dir_count+1}"
    print(dir_name)

    return dir_name

if not os.path.exists('yolov5'):
    os.system('git clone https://github.com/ultralytics/yolov5.git')
    os.system('pip install -r yolov5/requirements.txt')

os.system('cd yolov5')

# download annotate dataset from roboflow
rf = Roboflow(api_key="oXfnSDIgD09GTBglk8V6")
project = rf.workspace("rbt").project("tmp-rrkjt")
dataset = project.version(1).download("yolov5")
#shutil.move('tmp-1','yolo/tmp-1')

#train on downloaded dataset
oput_dir = handle_train_dir()
os.system(f'python train.py --data tmp-1/data.yaml --weights yolov5s.pt --cache --name {oput_dir}')