import os
import shutil
import random
import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def make_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def main():
    #获取根目录
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    IMAGEPATH=ROOT/"dataset/cat12/images"
    LABELPATH=ROOT/"dataset/cat12/labels"

    #开关模块，调试用
    prepare=1
    train=1
    predict=1
    sort=1

    #预处理
    if prepare==1:
        for path in [IMAGEPATH/"cat12train",LABELPATH/"cat12train",IMAGEPATH/"cat12val",LABELPATH/"cat12val"]:
            make_path(path)
        with open(ROOT/'train_list.txt', 'r') as f1:
            line = f1.readline().rstrip("\n")
            imagecount = 0
            while line != "":
                data=line.split('\t')
                image=data[0][13:]
                label=data[1]

                #图片预处理
                img=Image.open(ROOT/"cat_12_train"/image)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img=img.resize((640,640))
                img.save(IMAGEPATH/"cat12train"/(str(imagecount)+'.jpg'), 'JPEG', quality=100)

                #生成yolo格式的标签
                path=ROOT/"dataset/cat12/labels/cat12train"
                with open(path/(str(imagecount)+'.txt'), 'w') as f:
                    f.write(label+" 0.50 0.50 0.99 0.99")
                imagecount += 1
                line = f1.readline().rstrip("\n")

            #产生验证集
            number = len(os.listdir(IMAGEPATH/"cat12train"))  #读取文件个数
            for i in range(500):
                j = random.randint(1,number)
                print(j,number)
                path1 = str(IMAGEPATH/"cat12train")
                file_name1 = str(j)+".jpg"
                path2 = str(IMAGEPATH/"cat12val")
                file_name2 = str(i)+".jpg"
                src_fileA = os.path.join(path1, file_name1)
                dst_folderA = os.path.join(path2, file_name2)
                shutil.copy(src_fileA, dst_folderA)
                path3 = str(LABELPATH/"cat12train")
                file_name3 = str(j)+".txt"
                path4 = str(LABELPATH/"cat12val")
                file_name4 = str(i)+".txt"
                src_fileB = os.path.join(path3, file_name3)
                dst_folderB = os.path.join(path4, file_name4)
                shutil.copy(src_fileB, dst_folderB)
        print("处理完成")

    #训练模型
    if train==1:
        model = YOLO(ROOT/'yolov8n.pt')      #读取预训练模型
        model.train(model=ROOT/'yolov8n.pt', 
                    data=ROOT/'data/cat12.yaml', 
                    project=ROOT/"runs/train", 
                    device=0, 
                    lr0=0.0001, 
                    lrf=0.01, 
                    warmup_bias_lr=0.001, 
                    epochs=200)                   #训练新模型

    #推理
    if predict==1:
        trains=sorted(os.listdir(ROOT/"runs/train"),key=len)
        model_new=YOLO(ROOT/"runs/train/{}/weights/best.pt".format(trains[-1]))
        model_new.predict(IMAGEPATH/"cat12test", project=ROOT/"runs/detect", save_txt=True)

    #整理结果
    if sort==1:
        respath=ROOT/"runs/detect/predict"
        folder = sorted(os.listdir(respath),key=len)
        labels = os.listdir(respath/folder[-1])
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(ROOT/"output/{}.csv".format(current_time), 'w') as resfile:
            for text in labels: 
                with open(respath/folder[-1]/text,"r") as textfile:
                    res=textfile.readline().split(' ')[0]
                    resfile.writelines("{},{}\n".format(text.replace(".txt",".jpg"),res))

if __name__ == '__main__':
    main()