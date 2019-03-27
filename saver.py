import re
import torch
import torch.nn as nn
from utils import exist, is_dir, mkdir, join, ls, rm


def ls_model(fdir, fname, fext="pth", pattern=None) -> list:
	"""
	:param fdir:		모델이 저장된 폴더
	:param fname:		iteration 번호를 제외한 model 이름
	:param ext:		확장자
	:param pattern:	model 이름 패턴. 기본적으로는 ".*{}-(\d+).{}$".format(join(fdir, fname, fext)) 형태
	:return:			모델 경로 리스트, 모델 번호 리스트
	"""
	# 파일들 리스트 가져옴
	pattern = pattern if pattern is not None else r".*{}-(\d+)\.{}$".format(fname, fext)
	files = ls(fdir, pattern=pattern)

	# idx 기준으로 정렬함
	def get_key(s):
		m = re.match(pattern=pattern, string=s)
		return int(m.group(1))

	files = sorted(files, key=get_key)
	idxs = list(map(get_key, files))

	return files, idxs


def latest_idx(fdir, fname, fext="pth", pattern=None):
	"""
	:param fdir:		모델이 저장된 폴더
	:param fname:		iteration 번호를 제외한 model 이름
	:param fext:		확장자
	:param pattern:	model 이름 패턴. 기본적으로는 ".*{}-(\d+).{}$".format(join(fdir, fname, fext)) 형태
	:return:			해당 파일이 있으면 iteration 인덱스를 줌, 없으면 None
	"""

	# 경로 없으면 None
	if not exist(fdir):
		return None

	# 폴더인지 여부
	if not is_dir(fdir):
		raise ValueError("pc6.util_pt.saver.latest_idx: path ({}) not dir".format(fdir))

	# 인덱스 찾아서 리턴
	files, idxs = ls_model(fdir, fname, fext, pattern=pattern)
	if len(idxs) == 0:
		return None
	else:
		return idxs[-1]


class Saver:
	def __init__(self, model: nn.Module, fdir, fname, fext="pth", max_to_keep=10):
		self.model = model
		self.fdir = fdir  # file 을 포함하는 폴더
		self.fname = fname  # file 이름부
		self.fext = fext  # 확장자
		self.max_to_keep = max_to_keep

	def save(self, i):
		# 우선 저장
		fpath = join(self.fdir, "{}-{}.{}".format(self.fname, i, self.fext))  # full path
		mkdir(self.fdir)
		torch.save(self.model.state_dict(), fpath)

		# 현재 폴더에 저장된 같은 종류의 모델 확인
		files, idxs = ls_model(self.fdir, self.fname, self.fext)

		# 저장이 max_to_keep 보다 많으면 가장 오래된것 삭제
		if self.max_to_keep < len(files):
			rm(files[0])

	def load(self, i=None):
		if i is None:
			files, idxs = ls_model(self.fdir, self.fname, self.fext)
			if len(files) > 0:  # 있으면 가장 최신을 로드
				self.model.load_state_dict(torch.load(files[-1]))
				print("util_pt.saver.Saver.load: loaded {}".format(files[-1]))
				return latest_idx(self.fdir, self.fname)
			else:  # 로드 안함
				basename = join(self.fdir, self.fname)
				print("util_pt.saver.Saver.load: {} not exist".format(basename))
				return 0
		else:
			fpath = join(self.fdir, r"{}-{}.{}".format(self.fname, i, self.fext))
			self.model.load_state_dict(torch.load(fpath))
			print("util_pt.saver.Saver.load: loaded {}".format(fpath))
			return i

	def ls_model(self):
		return ls_model(self.fdir, self.fname, self.fext)

	def latest_idx(self):
		return latest_idx(self.fdir, self.fname)




