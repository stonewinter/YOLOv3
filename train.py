from torch.utils.data import Dataset
from utils.build_model import *


cfg_path = "./model/yolov3.cfg"
#weight_path = "./model/yolov3.weights"
data_path = "./images/object_detection/COCO/5k.txt"
epochs = 3

# hyperparams, _ = parse_model_config(cfg_path)
# lr = float(hyperparams["learning_rate"])
# momentum = float(hyperparams["momentum"])
# decay = float(hyperparams["decay"])
# burn_in = int(hyperparams["burn_in"])


model = Darknet(cfg_path)
model.apply(weights_init_normal)



dataloader = torch.utils.data.DataLoader(
    YoloSet(data_path), batch_size=1, shuffle=False, num_workers=1
)


model = model.cuda()
model.train()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(epochs):
    for batch_i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.FloatTensor).cuda()
        labels.requires_grad = False

        optimizer.zero_grad()

        loss = model(imgs, labels)

        if(batch_i % 5 == 0):
            print("loss =", loss)

        loss.backward()
        optimizer.step()
