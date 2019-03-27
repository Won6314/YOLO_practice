import torch.nn as nn
import torch
from loss import YOLO_loss
from dataset import COCODataset, print_result
from saver import Saver
from network import YOLO
from utils import join

num_epochs = 10
learning_rate = 1e-6
data_root = r'E:/LocalData/coco/train2017'
save_root = r'C:\LocalData'

data_loader = torch.utils.data.DataLoader(dataset=COCODataset(root=data_root),
										   batch_size=20,
										   shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.to(device)

criterion = YOLO_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
yolo_saver = Saver(model, join(save_root,r'yolo',r'save'), "yolo", max_to_keep=20)
loaded_index = yolo_saver.load()

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(data_loader):
		images = images.cuda()
		labels = labels.cuda()

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i) % 100 == 0:
			print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
				  .format(epoch + 1, num_epochs, i, loss.item()))
			print_result(images, outputs, path=join(save_root, r'yolo_all', r'image', r'result_{}.jpg'.format(i)), threshhold=0.5)
			print_result(images, labels, path=join(save_root, r'yolo_all', r'image', r'label_{}.jpg'.format(i)), threshhold=0.5)

		if i % 4000 == 0:
			for param in optimizer.param_groups:
				param['lr'] *= 0.5

		if (i) % 1000 == 0:
			yolo_saver.save(loaded_index+i+epoch*data_loader.__len__())
			print("saved at iter_{}".format(i+epoch*data_loader.__len__()))

if __name__ == "__main__":
	del(model)
	torch.cuda.empty_cache()