#-*-coding:utf-8-*-
import os
import cv2
import numpy as np



def splitImage(ImgPath,save_dir):
    image_name=os.path.basename(ImgPath)
    lastdir_name=os.path.basename(os.path.dirname(ImgPath))
    image=cv2.imread(ImgPath,0)
    label_pixel_path=ImgPath.split('.')[0] + '_label.bmp'
    label_pixel=cv2.imread(label_pixel_path)

    #创建路径
    save_dir = save_dir+ "/"+lastdir_name+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #循环
    for index in range(3):
        sample = []
        col_begin=index*420
        col_end=(index+1)*420
        img_patch=image[col_begin:col_end, :]
        label_patch=label_pixel[col_begin:col_end, :]
        cls=1 if np.sum(label_patch)>50 else 0
        sample_name=image_name.split('.')[0]+"_"+str(index)
        cv2.imwrite(save_dir+str(sample_name)+".jpg", img_patch)
        cv2.imwrite(save_dir+str(sample_name)+'_label.bmp', label_patch)




def main(dir,save_dir):
    listdirs=os.listdir(dir)
    for child_dir in listdirs:
        images=os.listdir(dir+"/"+child_dir)
        for image_name in images:
            im=image_name.split('.')[-1]
            if im=='jpg':
                img_path=dir+"/"+str(child_dir)+'/'+image_name
                splitImage(img_path,save_dir)


if __name__=="__main__":
    dir=r"C:\Datasets\KolektorSDD"
    save_dir="./KolektorSDD"
    main(dir,save_dir)