from pycocotools.coco import COCO
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from utils import join, mrange,  imshow, readImgById

coco = COCO(r"E:/LocalData/coco/annotations/instances_train2017.json")


# 참고, coco의 box는 x,y, width, height 순으로 되어있음. x, y는 왼쪽 위 기준
def get_center_boxes(image_number):  # yx  순으로 return
	"""

	:param image_number:	load image's boxes, it's shape is [x, y, w, h], top left base
	:return:	boxes, it's shape is [y, x, h, w], center base
	"""
	anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_number]))
	boxes = [ann["bbox"] for ann in anns]		# box = [x, y, w, h] 왼쪽 위 기준
	categories = [ann['category_id'] for ann in anns]

	def center(box):
		result = [None]*4
		result[1] = box[0] + box[2] / 2
		result[0] = box[1] + box[3] / 2
		result[2] = box[3]
		result[3] = box[2]
		return result			# result = [y, x, h, w] 중앙 기준

	boxes = [center(box) for box in boxes]
	return boxes, categories


def grid_image(image_number, yn, xn,  root=r'E:/LocalData/coco/train2017'):
	image = readImgById(image_number, coco,  root=root)
	boxes, categories = get_center_boxes(image_number)
	grid_dict = dict()
	grid_height = image.shape[0] / yn
	grid_width = image.shape[1] / xn
	for (i, box) in enumerate(boxes):
		y_idx = int(box[0] // grid_height)
		height_ratio = box[2] / image.shape[0]
		grid_y_ratio = (box[0] % grid_height) / grid_height

		x_idx = int(box[1] // grid_width)
		width_ratio = box[3] / image.shape[1]
		grid_x_ratio = (box[1] % grid_width) / grid_width

		grid_dict[y_idx, x_idx] = [1, grid_y_ratio, grid_x_ratio, height_ratio, width_ratio, categories[i]]
		if y_idx==7 or x_idx == 7:
			print("{}file's index number is wrong".format(image_number))
	return grid_dict


def onehot(index, num_classes):
	onehot_array = np.zeros(num_classes)
	onehot_array[index] = 1
	return onehot_array


def coco_onehot(category_id):
	cats = coco.loadCats(coco.getCatIds())
	category = coco.loadCats(category_id)[0]['supercategory']
	super_idx = sorted(list(set([cat['supercategory'] for cat in cats])))
	idx = super_idx.index(category)
	return onehot(idx, 12)

def coco_onehot_decode(category_onehot):
	cats = coco.loadCats(coco.getCatIds())
	super_idx = sorted(list(set([cat['supercategory'] for cat in cats])))
	category_number = torch.argmax(category_onehot)
	return super_idx[category_number]

def coco_to_yolo(image_number, yn=7, xn=7,  root=r'E:/LocalData/coco/train2017'):
	yolo_array = np.zeros([7, 7, 17], dtype=np.float32)
	grid_dict = grid_image(image_number, yn, xn,  root=root)
	for key in grid_dict.keys():
		box_info = np.array(grid_dict[key][0:5])
		onehot_array = coco_onehot(grid_dict[key][5])
		yolo_array[key] = np.concatenate([box_info, onehot_array])
	yolo_array = np.transpose(yolo_array, [2, 0, 1])
	return yolo_array


class ExpandTransform:
	def __call__(self, img:torch.Tensor):
		if img.shape[0] == 1:
			img = img.repeat(3, 1, 1)
		return img[:3, :, :]

transform = transforms.Compose([
	transforms.Resize(size=[448, 448]),
	transforms.ToTensor(),
	ExpandTransform()])

class COCODataset(torch.utils.data.Dataset):
	def __init__(self, root=r'E:/LocalData/coco/train2017'):
		self.file_list = coco.getImgIds()
		self.root = root

	def __getitem__(self, index):
		filenumber = self.file_list[index]
		filename = coco.loadImgs(filenumber)[0]["file_name"]
		image = Image.open(join(self.root, filename))
		image = transform(image)
		label = coco_to_yolo(filenumber, root=self.root)
		return image, label

	def __len__(self):
		return self.file_list.__len__()


def print_result(images, labels, path=None, threshhold=0.7):
	image = images[0]
	label = labels[0]
	grid_scale = 448/7
	boxes = []
	class_names = []
	scores = []
	for y, x in mrange(range(7), range(7)):
		if label[0, y, x] > threshhold:
			box_height = label[3, y, x]*448
			box_width = label[4, y, x]*448
			box_y = (y + label[1, y, x]) * grid_scale - box_height/2
			box_x = (x + label[2, y, x]) * grid_scale - box_width/2
			class_name = coco_onehot_decode(label[5:17, y, x])
			boxes.append([box_x, box_y, box_width, box_height])
			class_names.append(class_name)
			scores.append(label[0, y, x])
	image = np.array(image.cpu())
	image = np.transpose(image, [1,2,0])
	imshow(image, boxs=boxes, names=class_names, scores=scores, path=path)

