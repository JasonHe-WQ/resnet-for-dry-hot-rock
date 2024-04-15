import torchsummary

import model_training

model = model_training.ModifiedResNet50().cuda()
torchsummary.summary(model, (1, 10, 10))