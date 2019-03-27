import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import skimage.io as io
import re

def join(*paths):
	paths = [str(path) for path in paths]
	return str(pathlib.Path(*paths))

def is_dir(path):
	assert exist(path)
	path = pathlib.Path(path)
	return path.is_dir()

def exist(path):
	return os.path.exists(str(path))

def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def parent(path):
	path = pathlib.Path(path)
	return join(*path.parts[:-1])

def path_local_data():
	return os.environ["PATH_LOCAL_DATA"]

def mrange(*iterables):
   if not iterables:
      yield []
   else:
      for item in iterables[0]:
         for rest_tuple in mrange(*iterables[1:]):
            yield [item] + rest_tuple

def ls(path, pattern=r".*", recursive=False, file_only=True):
	"""
	:param path: 		찾기 시작할 root 경로
	:param pattern: 	해당 패턴이 re.match 로 만족하는 모든 것을 찾음
	:param recursive: 	폴더를 타고 들어가서 다 찾을것인지 여부
	:param file_only: 	파일만 리턴할 것인지. False면 폴더도 결과에 포함됨
	:return: 			path str list
	"""
	path = pathlib.Path(path)

	# check: 해당 폴더 존재 여부
	if not exist(path) or not is_dir(path):
		# print("ioparser.ls: path ({}) not exist or not dir".format(path))
		print("ioparser.ls: path ({}) not exist or not dir".format(path))

		return []

	# search
	out = []
	if recursive: # 폴더를 타고 들어가면서 확인함
		for x in path.iterdir(): # x는 path가 포함된 이름임. join(path, x) 필요 없음
			if x.is_dir(): # 폴더인 경우
				if not file_only and re.match(pattern, str(x)): # file_only=False면 해당 dir도 등록시킴
					out.append(str(x))
				out += ls(str(x), pattern, recursive) # 폴더 타고 들어감. re.match 매치 안될때도 들어가야
														# 매칭되는 모든 파일을 찾을 수 있음.
			else: # 파일인 경우
				if re.match(pattern, str(x)): # 매치 되는 파일들만 등록함
					out.append(str(x))
	else: # 폴더를 타고들어가지 않음
		for x in path.iterdir():
			if re.match(pattern, str(x)): # 매치되는 경우
				if file_only: # file_only=True면 파일만 등록
					if not x.is_dir():
						out.append(str(x))
				else:
					out.append(str(x)) # file_only=False 면 폴더든 파일이든 모두 등록
	return out


def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def rm(path):
	path = str(path)
	if os.path.exists(path):
		os.remove(path)
	else:
		print("ioparser.rm: path({}) not exist".format(path))

def mrange(*iterables):
   if not iterables:
      yield []
   else:
      for item in iterables[0]:
         for rest_tuple in mrange(*iterables[1:]):
            yield [item] + rest_tuple


def apply_mask(img, mask, color, alpha=0.5):
	idx = (mask != 0)
	y,x = np.where(idx)
	img[y,x] = np.clip(img[y,x] * (1-alpha) + alpha * mask[y,x,None] * color, 0, 255).astype(np.uint8)

def imshow(img, boxs, masks=None, names=None, scores=None, colors=None, title=None, path=None):
	"""
	:param img:		yxc, uint8,
	:param boxs:	(x,y,w,h)
	:param masks:	yx, uint8 (0~1)
	:param names:	class 이름들
	:param scores:	score 들
	:param colors:	박스 색깔

	:note: boxs, masks, names, scroes, colors 는 None 이 아니면 갯수가 같아야함
	"""
	if boxs is None:
		print("no boxes")
		return
	if masks is not None: assert len(boxs) == len(masks)
	if names is not None: assert len(boxs) == len(names)
	if scores is not None: assert len(boxs) == len(scores)
	if colors is not None: assert len(boxs) == len(colors)

	if path is not None:
		plt.ioff()
		if not exist(parent(path)) or not is_dir(parent(path)):
			mkdir(parent(path))

	if title is None:
		fig = plt.figure()
	else:
		fig = plt.figure(num=title)

	ax = plt.gca()
	img = img.copy()
	plt.imshow(img)
	for i in range(len(boxs)):
		box = boxs[i]
		color = colors[i] if colors is not None else "red"
		mask = masks[i] if masks is not None else None
		name = names[i] if names is not None else ""
		score = scores[i] if scores is not None else None

		ax.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, alpha=1, edgecolor=color, facecolor="None"))
		if score is not None:
			caption = "{} ({:.2f})".format(name, score)
		else:
			caption = name
		ax.text(box[0] + 1, box[1] + 2, caption, backgroundcolor=color, fontsize=6, alpha=1, color="white", fontweight="semibold")
		if mask is not None:
			apply_mask(img, mask, np.clip(color * 255, 0, 255).astype(np.uint8), alpha=0.5)
	if path is None:
		plt.imshow(img)
		plt.pause(0.001)
		plt.show(block=False)
	else:
		plt.imshow(img)
		plt.savefig(path)
	plt.close(fig)


def readImgById(id, coco, root=r"E:/LocalData/coco/train2017"):
	path = join(root, coco.loadImgs(id)[0]["file_name"])
	img = io.imread(path)
	return img