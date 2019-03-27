import torch.nn as nn
import torch
from utils import mrange

class YOLO_loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, out: torch.Tensor, label):
		# out.reshape(-1, 17, 7, 7)
		loss = 0
		for b, y, x in mrange(range(out.shape[0]), range(7), range(7)):
			if label[b, 0, y, x] == 0:
				loss += 0.5 * torch.sum((label[b, 0, y, x] - out[b, 0, y, x]) ** 2)
			else:
				loc_loss = torch.sum((label[b, 1:3, y, x] - out[b, 1:3, y, x]) ** 2)
				size_loss = torch.sum((torch.sqrt(label[b, 3:5, y, x]) - torch.sqrt(out[b, 3:5, y, x])) ** 2)
				prob_loss = torch.sum((label[b, 0, y, x] - out[b, 0, y, x]) ** 2)
				class_loss = torch.sum((label[b, 5:17, y, x] -out[b, 5:17, y, x]) ** 2)
				#class_loss = torch.sum(-label[b, 5:17, y, x] * F.log_softmax(out[b, 5:17, y, x], -1), -1)

				loss += 5 * (loc_loss + size_loss) + prob_loss + class_loss
				# loss += 5 * (loc_loss + size_loss + prob_loss)
				if torch.any(torch.isnan(loss)):
					raise ValueError

		return loss / (out.shape[0]*49)