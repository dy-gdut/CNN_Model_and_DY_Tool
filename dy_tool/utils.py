from PIL import Image
import numpy as np
import logging
import os
import time
import sys
import cv2
import random
#打印日志到控制台和log_path下的txt文件
def get_logger(log_path):
	log_path=log_path+"/Log"
	if not os.path.exists(log_path):
			os.makedirs(log_path)
	timer=time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
	logger = logging.getLogger(log_path)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(levelname)s]   %(asctime)s		%(message)s')
	consoleHandel=logging.StreamHandler()
	consoleHandel.setFormatter(formatter)
	logger.addHandler((consoleHandel))
	txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
	txthandle.setFormatter(formatter)
	logger.addHandler(txthandle)
	return logger

#将输入路径的上两级路径加入系统
def set_projectpath(current_path):
	curPath = os.path.abspath(current_path)
	#curPath = os.path.abspath(os.path.dirname(__file__))
	rootPath = os.path.split(curPath)[0]
	sys.path.append(rootPath)
	rootPath = os.path.split(rootPath)[0]
	sys.path.append(rootPath)

def listData(self,data_dir):
	"""# list the files  of  the currtent  floder of  'data_dir' 	,subfoders are not included.
	:param data_dir:
	:return:  list of files
	"""
	data_list=os.listdir(data_dir)
	data_list=[x[2] for x in os.walk(data_dir)][0]
	data_size=len(data_list)
	return data_list,data_size

def Ostu(array):
	array = np.array(array * 255, dtype=np.uint8)
	best_threshold, binary_output = cv2.threshold(array, 100, 1, cv2.THRESH_BINARY)  # cv2.THRESH_OTSU
	area = np.sum(np.array(binary_output))
	predict = 1 if area > 1 else 0
	return predict

def concatImage(images,mode="Adapt",scale=0.5):
	if not isinstance(images, list):
		raise Exception('images must be a  list  ')
	if mode not in ["Row" ,"Col","Adapt"]:
		raise Exception('mode must be "Row" ,"Adapt",or "Col"')
	for i in range(len(images)):
		images[i]=np.uint8(images[i])
	count = len(images)
	img_ex = Image.fromarray(images[0])
	size=img_ex.size
	if mode=="Adapt":
		mode= "Row" if size[0]<=size[1] else "Col"
	offset = int(np.floor(size[0] * 0.02))
	if mode=="Row":
		target = Image.new(img_ex.mode, (size[0] * count+offset*(count-1), size[1] * 1),100)
		for i  in  range(count):
			image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
			target.paste(image, (i*(size[0]+offset), 0, i*(size[0]+offset)+size[0], size[1]))
		return target
	if mode=="Col":
		target = Image.new(img_ex.mode, (size[0] , size[1]* count+offset*(count-1)),100)
		for i  in  range(count):
			image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
			target.paste(image, (0,i*(size[1]+offset),size[0],i*(size[1]+offset)+size[1]))
		return target

def visualization(list_batchs,filenames,save_dirs):
	"""
	:param list_batchs:  list{  array[b,h,w,1]   }
	:param filenames:		list{ filename }
	:param save_dir: 	
	:return:  
	"""
	#batch_num= filenames.shape(0)


	for i, filename in enumerate(filenames):
		if not isinstance(save_dirs, list):
			#save_dirs = [save_dirs for i in len(batch_num)]
			save_dir = save_dirs
		else:
			save_dir=save_dirs[i]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		if not isinstance(filename,str): filename=filename.decode("utf-8")
		filename =filename.replace("/","_")
		list_images=[]
		for batchs in list_batchs:
			image=np.array(batchs[i]).squeeze(2)
			list_images.append(image)
		img_visual=concatImage(list_images)
		visualization_path = os.path.join(save_dir,filename)
		try:
			img_visual.save(visualization_path)
		except:
			print("图片保存失败【[]】".format(visualization_path))

def save_image(image,save_dir,filename):
	image = Image.fromarray(np.uint8(image)) if not isinstance(image, Image.Image) else image
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	visualization_path = os.path.join(save_dir, filename)
	image.save(visualization_path)

def write_txt(W_dir,dataset_dict,):
	if not os.path.exists(W_dir):
		os.makedirs(W_dir)
	for key,image_list in  dataset_dict.items(): #每个类别
		W_path = os.path.join(W_dir,key)+'.txt'
		file = open(W_path, 'w')
		file.close()
		file = open(W_path, 'a')
		for image in  image_list:
			for item in image:
				file.write(str(item))
				file.write(" ")
			file.write('\n')
		file.close()


def read_txt(phases,data_dir):
	"""Read the content of the text file and store it into lists."""
	#phases=["training","validation","testing"]
	data_dict={}
	for phase in phases:
		data_list=[]
		txt_file=data_dir+"/"+phase+".txt"
		with open(txt_file, 'r') as f:
			lines = f.readlines()
			for line in lines:
				items = list(line.strip().split(' '))
				data_list.append(items)
		data_dict[phase]=data_list
	return data_dict


def transform(image, reverse=False):
	if not reverse:
		image=np.array(image).astype(np.float)
		image=image / 255.0
		return image
	else:
		image = np.array(image)
		image=(image)*255
		image=image.astype("uint8")
		return image

def statistics(label_batch,score_batch,threshold):
	label_batch=np.array(label_batch).flatten()
	score_batch=np.array(score_batch).flatten()
	#真正例
	count_TP = np.sum((label_batch == 1) & (score_batch >= threshold))
	#真反例
	count_TN = np.sum((label_batch == 0) & (score_batch < threshold))
	#假正例
	count_FP = np.sum((label_batch == 0) & (score_batch >= threshold))
	#假反例
	count_FN = np.sum((label_batch == 1) & (score_batch < threshold))
	return count_TP,count_TN,count_FP,count_FN

def calculate(count_TP,count_TN,count_FP,count_FN):
	count=count_TP+count_TN+count_FP+count_FN
	# 准确率
	accuracy = (count_TP + count_TN) / count
	# 查准率
	prescision = count_TP / (count_TP + count_FP) if (count_TP + count_FP) != 0 else -1
	# 查全率
	recall = count_TP / (count_TP + count_FN) if (count_TP + count_FN) != 0 else -1
	return count,accuracy,prescision,recall

def concatImage191115(images,mode="Adapt",scale=0.5):
	"""
	生成可视化的结果
	【512*9+8*512*0.02，1280】
	:param images:
	:param mode:
	:param scale:
	:return:
	"""
	if not isinstance(images, list):
		raise Exception('images must be a  list  ')
	if mode not in ["Row" ,"Col","Adapt"]:
		raise Exception('mode must be "Row" ,"Adapt",or "Col"')
	for i in range(len(images)):
		images[i]=np.uint8(images[i])
	count = len(images)
	img_ex = Image.fromarray(images[0])
	size=img_ex.size

	size=[512,1280]
	offset = int(np.floor(size[0] * 0.02))
	background=100
	target = Image.new(img_ex.mode, (size[0]*(count+2)+offset*(count-1), size[1] * 1),background)

	for i  in  range(count):
		if i ==0:
			image = Image.fromarray(images[i])
			target.paste(image, (0, 0, (i+2)*(size[0]+offset)+size[0], size[1]))
		elif i<2:
			image = Image.fromarray(images[i])
			image = image.crop( (2*(size[0]+offset), 0, 2*(size[0]+offset)+size[0], size[1]))
			target.paste(image, ((i+2)*(size[0]+offset), 0, (i+2)*(size[0]+offset)+size[0], size[1]))
		elif i==2:
			# image = Image.fromarray(images[i])
			# image = image.crop( (2*(size[0]+offset), 0, 2*(size[0]+offset)+size[0], size[1]))
			# image=np.uint8(np.zeros(image.size()))
			target.paste(0, ((i+2)*(size[0]+offset), 0, (i+2)*(size[0]+offset)+size[0], size[1]))
		else:
			image = Image.fromarray(images[i-1])
			image = image.crop( (2*(size[0]+offset), 0, 2*(size[0]+offset)+size[0], size[1]))
			target.paste(image, ((i+2)*(size[0]+offset), 0, (i+2)*(size[0]+offset)+size[0], size[1]))
	return target

#不同的地方是：内置sorted返回一个新的列表，而list.sort是对列表进行操作
def cat_and_save(root,child_dirs):
	def read_image(path):
		img = Image.open(path)
		return img

	for child_dir in child_dirs:
		dir=os.path.join(root,str(child_dir))
		images=os.listdir(dir)
		images.sort(key=lambda x: int(x.split("_")[0].split("-")[-1]))
		images=[os.path.join(dir,img) for img in  images]
		images=list(map(read_image,images))
		image_concat=concatImage191115(images)
		save_image(image_concat,root,str(child_dir)+".jpg")


# if __name__=="__main__":
# 	img=cv2.imread("Part4.jpg",0)
# 	img1=transform(img)
# 	img2=transform(img1,True)

