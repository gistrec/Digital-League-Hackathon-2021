######## ML #########
from __future__ import print_function, division
from albumentations.pytorch import ToTensorV2
import pandas as pd
import math
import tqdm
import seaborn as sns
from IPython.display import display
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
import PIL
from PIL import Image
import random
import matplotlib.pyplot as plt
import torchvision
from torchvision import models as tvmodels
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import albumentations as A
import cv2
from sklearn.model_selection import GroupKFold, StratifiedKFold
from albumentations import Compose
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pathlib import Path
import timm
from pprint import pprint
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchsummary import summary
import torchvision.models as models
import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIMM_MODEL = 'res2next50'
import torch.nn.functional as F

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

NAME2CLASS = {'human':0, 'target_human':1, 'target_laser':2, 'target_gun':3, 'target_tank':4}
CLASS2NAME = {v: k for k, v in NAME2CLASS.items()}

def valid_transforms():
    return Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1)

def plot_images(loader, num_images=5, denorm = True):
    images, label = next(iter(loader))
    # convert to numpy and transpose as (Batch Size, Height, Width, Channel) as needed by matplotlib
    images = images#.numpy().transpose(0, 2, 3, 1)

    # Analysing images of a train batch
    num_cols = 5
    num_rows = 1
    if num_images > 5:
        num_cols = 5
        num_rows = math.ceil(num_images / 5)
    np.random.seed(100)
    indices = np.random.choice(range(len(label)), size=num_images, replace=False)
    width = 20
    height = 5*num_rows
    plt.figure(figsize=(width, height))
    for i, idx in enumerate(indices):
        plt.subplot(num_rows, num_cols, i + 1)
        if denorm:
            image = denormalize(images[idx])
        else:
            image = images[idx]
#         img = img      # unnormalize
#         npimg = img.numpy()
        plt.imshow(image.numpy().transpose(1, 2, 0));
        plt.title(f'class: {CLASS2NAME[label[idx].item()]}');
        plt.axis("off")
    plt.show()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)

class FirstNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = timm.create_model(TIMM_MODEL, pretrained=True, exportable=True)
        print(backbone)
        try:
            n_features = backbone.classifier.in_features
        except:
            n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-1]
        self.backbone = nn.Sequential(*self.backbone, nn.LeakyReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_features, 5)

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = feats.clone() #self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)

        return x
##################################################

class ShitPL(pl.LightningModule):

    def __init__(self, model_itself):
        super().__init__()
        self.model = model_itself.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        self.val_criterion = nn.CrossEntropyLoss().to(device)
        self.num_workers = 1
        self.last_val_loss = 0
        self.last_train_loss = 0
        self.last_val_acc = 0
        self.val_idxer = 0
        self.train_idxer = 0
        self.last_train_acc = 0

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        image, label = batch
        X, y = image.to(device).float(), label.to(device).long()
        rand = np.random.rand()
        outputs = self.model(X)
        loss = torch.mean(self.criterion(outputs, y))
        preds = F.softmax(outputs).argmax(axis=1)
        acc = accuracy_metric(preds, y)
        del X
        del y
        self.last_train_acc = acc
        self.last_train_loss = loss.item()
        self.log("train/loss",loss.item())
        self.log("train/acc",  acc)
        self.train_idxer += 1
        torch.cuda.empty_cache()
        return loss
    def validation_step(self, batch,batch_idx):
        image, label = batch
        X, y = image.to(device), label.to(device)
        outputs = self.model(X)
        l = self.val_criterion(outputs, y)
        preds = F.softmax(outputs).argmax(axis=1)
        loss = l 
        acc = accuracy_metric(preds, y)
        self.last_val_acc = acc
        self.last_val_loss = loss.item()
        del X
        del y
        torch.cuda.empty_cache()
        self.log("val/loss" , loss.item())
        self.log("val/acc", acc )
        self.val_idxer += 1
        return {"loss":loss, "acc": acc}

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return valid_loader

    def configure_optimizers(self):
        param_groups = [
          {'params': self.model.backbone.parameters(), 'lr': 1e-2},
          {'params': self.model.classifier.parameters()},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=4e-3,
                                weight_decay=1e-3, )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,20,40], 
                                                     gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]

from scipy.special import *
import onnxruntime

ort_session = onnxruntime.InferenceSession("resnet.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



VAL_TRF = valid_transforms()

def get_img_bgr_to_rgb(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

# compute ONNX Runtime output prediction
def predict_onnx(path, ort_session,  show=False):
    im = orig = get_img_bgr_to_rgb(path)
    im = VAL_TRF(image=im)['image']
    im = torch.unsqueeze(im, 0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(im)}
    ort_outs = ort_session.run(None, ort_inputs)

    preds = softmax(np.array(ort_outs)).argmax()

    if show:
      plt.figure(figsize=(15, 15))
      plt.imshow(orig);
      plt.title(f'class: {CLASS2NAME[preds.item()]}');
      plt.axis("off")
      plt.show()
    return CLASS2NAME[preds.item()]

###### Argument parser #######
import sys

if not sys.argv[1]:
    print("Первым параметром введите токен от бота телеграма")
    exit(1)

token = sys.argv[1]


###### Telegram ##########
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import string
import random
import time

def name_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def photo(update: Update, context: CallbackContext) -> int:
    try:
        """Stores the photo and asks for a location."""
        user = update.message.from_user
        photo_file = update.message.photo[-1].get_file()


        photo_name = name_generator()
        photo_file.download(photo_name)

        print("Photo of %s: %s" % (user.first_name, photo_name))

        start = time.time()
        class_str = predict_onnx(photo_name, ort_session, False)
        done = time.time()
        elapsed = done - start
        print(elapsed)

        update.message.reply_text(class_str)
    except Exception as e:
        update.message.reply_text("Ой, случилась какая-то ошибка :c")
        print(e)

def file(update: Update, context: CallbackContext) -> int:
    try:
        user = update.message.from_user
        document = update.message.document.get_file()

        update.message.reply_text("Скачиваю файл.")
        folder_name = name_generator()
        document_name = folder_name + ".zip"
        document.download(document_name)

        print("Name of %s: %s" % (user.first_name, document_name))
        update.message.reply_text("Начинаю обработку файла.")

        import zipfile
        with zipfile.ZipFile(document_name, 'r') as zip_ref:
            zip_ref.extractall(folder_name)

        data = {0: [], 1: []}

        from os import listdir
        from os.path import isfile, join
        for file in listdir(folder_name):
            file_name=os.path.join(folder_name, file)
            if not isfile(file_name):
                continue

            class_str = predict_onnx(file_name, ort_session, False)
            #data[file] = NAME2CLASS[class_str]

            data[0].append(NAME2CLASS[class_str])
            data[1].append(file)

        import pandas  as pd
        df = pd.DataFrame(data)
        df.to_csv(folder_name + ".csv", index=False)

        context.bot.sendDocument(chat_id=update.message.chat_id,
                                document=open(folder_name + ".csv", "r"))
    except Exception as e:
        update.message.reply_text("Ой, случилась какая-то ошибка :c")
        print(e)

# Define a few command handlers. These usually take the two arguments update and context.
def start(update: Update, context: CallbackContext) -> None:
    """"""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Привет, {user.mention_markdown_v2()}\! Отправь фотографию, чтобы получить её класс\.',
        reply_markup=ForceReply(selective=True),
    )


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(update.message.text)

updater = Updater(token)

# Get the dispatcher to register handlers
dispatcher = updater.dispatcher

# Оn different commands - answer in Telegram
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.photo, photo))
dispatcher.add_handler(MessageHandler(Filters.document.mime_type("application/zip"), file))

# Start the Bot
updater.start_polling()

# Run the bot until you press Ctrl-C or the process receives SIGINT,
# SIGTERM or SIGABRT. This should be used most of the time, since
# start_polling() is non-blocking and will stop the bot gracefully.


###### Fast API ########
from fastapi import FastAPI
import requests
import uvicorn

app = FastAPI()

@app.get("/predict/")
def root(image_url: str = ""):
    print(image_url)
    r = requests.get(image_url, allow_redirects=False)

    photo_name = name_generator()
    open(photo_name, 'wb').write(r.content)

    class_str = predict_onnx(photo_name, ort_session, False)
    return class_str

@app.get("/")
def root(image_url: str = ""):
    return "Ok"


uvicorn.run(app, host="46.17.97.44", port=3030)
