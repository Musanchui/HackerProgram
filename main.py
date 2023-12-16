import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from dataloader import MyDataset
from model import Net
import whisper
from record import record
from detect import detect,strip_optimizer

from pygame.locals import *
import pyaudio
import argparse
import time
from pathlib import Path

import cv2

import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image

import warnings

import pygame
import time
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




# 训练模型的函数
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()  # 将模型设置为训练模式
    total_loss = 0
    correct_predictions = 0

    for sentences, visions, labels in dataloader:
        optimizer.zero_grad()  # 清空上一步的梯度

        visions = visions.to(device)  # 将数据移动到设备上
        labels = labels.to(device)  # 将标签移动到设备上

        outputs = model(sentences, visions)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()  # 累加损失
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        correct_predictions += (predicted == labels).sum().item()  # 计算正确的预测数

    avg_loss = total_loss / len(dataloader)  # 计算平均损失
    accuracy = correct_predictions / len(dataloader.dataset)  # 计算准确率

    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    model.eval()  # 将模型设置为评估模式
    correct_predictions = 0

    with torch.no_grad():  # 在评估阶段不计算梯度
        for sentences, visions, labels in dataloader:
            visions = visions.to(device)
            labels = labels.to(device)

            outputs = model(sentences, visions)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / len(dataloader.dataset)
    return accuracy


# 主函数
def main():
    json_path = 'train.json'  # 替换为你的实际json文件路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集
    dataset = MyDataset(json_path)

    # 划分数据集为训练集和测试集
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 初始化模型
    model = Net(device=device).to(device)

    # 定义优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 用于跟踪最佳模型的变量
    best_accuracy = 0.0
    best_epoch = 0
    best_model_path = 'best_model.pth'

    # 运行训练循环
    epochs = 50
    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, train_dataloader, optimizer, criterion, device)
        test_accuracy = evaluate_model(model, test_dataloader, device)

        # 如果当前epoch的测试准确率是到目前为止最好的，则保存模型
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch + 1} with Test Accuracy: {test_accuracy:.4f}")

        print(
            f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    print(f'Best Test Accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}')


def load_model(model_path, device):
    # 初始化模型结构
    model = Net(device=device).to(device)

    # 加载模型状态
    model_state = torch.load(model_path, map_location=device)

    # 应用模型状态
    model.load_state_dict(model_state)

    # 将模型置于评估模式
    model.eval()

    return model


def predict(model, sentence,vision):
    # 将输入数据转换为适合模型的格式
    if (vision == "00"):
        vision_tensor = torch.tensor([0, 0])
    if (vision == "01"):
        vision_tensor = torch.tensor([0, 1])
    if (vision == "10"):
        vision_tensor = torch.tensor([1, 0])
    if (vision == "11"):
        vision_tensor = torch.tensor([1, 1])
    vision_tensor = vision_tensor.unsqueeze(0).to('cpu')
    if sentence[0].find("左")!=-1:
        if(vision == "00" or vision == "01"):
            return 1;
        else:
            return 0;
    elif sentence[0].find("右")!=-1:
        if (vision == "00" or vision == "10"):
            return 1;
        else:
            return 0;
    elif sentence[0].find("灯")!=-1 or sentence[0].find("燈")!=-1:
        if(vision == "00"):
            return 1;
        else:
            return 0;
    else:
        return 4


    # 确保不会计算梯度
    with torch.no_grad():
        # 进行预测
        outputs = model(sentence,vision_tensor)
    _, predicted = torch.max(outputs.data, 1)

    # 返回预测结果
    return predicted.data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():

        vision = detect(opt=opt)



    model1 = whisper.load_model("base")
    # record("record.wav", time=3)
    # result = model.transcribe("record.wav")
    # sentence=result["text"]
    # model=load_model('best_model.pth','cuda')
    # sentence=[sentence]
    # print(sentence)
    # predicted=predict(model,sentence,vision=vision)
    # act=4  #未知
    #
    # if sentence[0].find("左")!=-1:
    #     act = 1;  #左转
    # elif sentence[0].find("右")!=-1:
    #     act=2; #右转
    # elif sentence[0].find("灯")!=-1 or sentence[0].find("燈")!=-1:
    #     act=3;  #灯
    # if (predicted == 0):
    #     act = 0 # 非法
    # print(predicted)
    # main()

    # 游戏窗口尺寸
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1200

    # 赛车尺寸
    CAR_WIDTH = 50
    CAR_HEIGHT = 100

    # 列宽度和间隔
    COL_WIDTH = 100
    COL_GAP = 50

    # 背景图像移动速度
    BACKGROUND_SPEED = 2

    # 颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


    # 赛车类
    class Car(pygame.sprite.Sprite):
        def __init__(self, x, y, image_path):
            super().__init__()
            self.image = pygame.image.load(image_path)  # 加载车辆图像
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
            self.spawn_time = time.time()
            self.alpha = 255  # 初始透明度

        def move_left(self):
            if self.rect.x > 0:
                self.rect.x -= COL_WIDTH + COL_GAP

        def move_right(self):
            if self.rect.x < (COL_WIDTH + COL_GAP) * 3:
                self.rect.x += COL_WIDTH + COL_GAP

        def move_forward(self, offset):
            if self.rect.y < SCREEN_HEIGHT:
                self.rect.y += offset
                if self.rect.y >= SCREEN_HEIGHT - CAR_HEIGHT:
                    self.fade_out(5)  # 逐渐减小透明度

        def fade_out(self, fade_speed):
            self.alpha -= fade_speed
            if self.alpha < 0:
                self.alpha = 0


    recording = False

    # 游戏窗口尺寸
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1200

    # 赛车尺寸
    CAR_WIDTH = 50
    CAR_HEIGHT = 100

    # 列宽度和间隔
    COL_WIDTH = 100
    COL_GAP = 50

    # 背景图像移动速度
    BACKGROUND_SPEED = 2

    # 颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


    # 赛车类
    class Car(pygame.sprite.Sprite):
        def __init__(self, x, y, image_path):
            super().__init__()
            self.image = pygame.image.load(image_path)  # 加载车辆图像
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
            self.spawn_time = time.time()
            self.alpha = 255  # 初始透明度

        def move_left(self):
            if self.rect.x > 0:
                self.rect.x -= COL_WIDTH + COL_GAP

        def move_right(self):
            if self.rect.x < (COL_WIDTH + COL_GAP) * 3:
                self.rect.x += COL_WIDTH + COL_GAP

        def move_forward(self, offset):
            if self.rect.y < SCREEN_HEIGHT:
                self.rect.y += offset
                if self.rect.y >= SCREEN_HEIGHT - CAR_HEIGHT:
                    self.fade_out(5)  # 逐渐减小透明度

        def fade_out(self, fade_speed):
            self.alpha -= fade_speed
            if self.alpha < 0:
                self.alpha = 0


    # 初始化游戏
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    pygame.mixer.music.load('操作成功.wav')

    # 加载背景图像
    background_image_path = "road.png"  # 背景图像文件路径
    background_image = pygame.image.load(background_image_path)
    background_rect = background_image.get_rect()

    # 创建原始赛车对象
    original_car_image_path = "car1.png"  # 原始赛车的图像文件路径
    original_car = Car((SCREEN_WIDTH - CAR_WIDTH) // 2, SCREEN_HEIGHT // 2 - CAR_HEIGHT // 2, original_car_image_path)
    spawned_cars = []

    # 创建生成的赛车对象
    generated_car_image_path = "car2.png"  # 生成的赛车的图像文件路径

    # 加载顶部图片
    top_image_path = "data/images/9dbae68c4566.png"  # 顶部图片的文件路径
    top_image = pygame.image.load(top_image_path)
    # 设置目标尺寸
    target_width = 1000  # 目标宽度
    target_height = 500  # 目标高度

    # 调整图片尺寸
    top_image = pygame.transform.scale(top_image, (target_width, target_height))
    top_image_rect = top_image.get_rect()
    top_image_rect.topleft = (0, 0)  # 设置顶部图片的位置

    # 赛车出现的垂直位置列表
    spawn_y_positions = [50, 150, 250]

    # 当前垂直位置索引
    current_spawn_y_index = 0

    # 背景图像的初始位置
    background_y = 0

    # 生成赛车的垂直偏移量
    vertical_offset = 5

    # 生成赛车的水平偏移量
    left_spawn_offset = 0
    right_spawn_offset = 0

    # 生成赛车的垂直偏移量
    vertical_spawn_offset = 440

    # 游戏主循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    original_car.move_left()
                elif event.key == pygame.K_RIGHT:
                    original_car.move_right()
                elif event.key == pygame.K_UP:
                    original_car.move_forward(vertical_offset)
                elif event.key == pygame.K_0:
                    new_car = Car(original_car.rect.x - (COL_WIDTH + COL_GAP) + left_spawn_offset,
                                  spawn_y_positions[current_spawn_y_index] + vertical_spawn_offset,
                                  generated_car_image_path)
                    spawned_cars.append(new_car)
                    current_spawn_y_index = (current_spawn_y_index + 1) % len(spawn_y_positions)
                elif event.key == pygame.K_1:
                    new_car = Car(original_car.rect.x + (COL_WIDTH + COL_GAP) + right_spawn_offset,
                                  spawn_y_positions[current_spawn_y_index] + vertical_spawn_offset,
                                  generated_car_image_path)
                    spawned_cars.append(new_car)
                    current_spawn_y_index = (current_spawn_y_index + 1) % len(spawn_y_positions)
                elif event.key == pygame.K_SPACE:
                    recording = not recording  # 切换录音状态
                    record("record.wav", time=3)
                    result = model1.transcribe("record.wav")
                    sentence = result["text"]
                    os.remove("record.wav")
                    model = load_model('best_model.pth', 'cuda')
                    sentence = [sentence]
                    print(sentence)
                    predicted = predict(model, sentence, vision=vision)
                    act = 4  # 未知

                    if sentence[0].find("左") != -1:
                        act = 1;  # 左转
                    elif sentence[0].find("右") != -1:
                        act = 2;  # 右转
                    elif sentence[0].find("灯") != -1 or sentence[0].find("燈") != -1:
                        act = 3;  # 灯
                    if (predicted == 0):
                        act = 0  # 非法
                    if act==1:
                        original_car.move_left()
                        pygame.mixer.music.play(1, 0.0)
                    elif act==2:
                        original_car.move_right()
                        pygame.mixer.music.play(1, 0.0)
                    elif act==0:
                        pygame.mixer.music.load('危险操作.wav')
                        pygame.mixer.music.play(1, 0.0)
                        top_image_path = "runs/detect/exp/9dbae68c4566.png"  # 顶部图片的文件路径
                        top_image = pygame.image.load(top_image_path)
                        top_image = pygame.transform.scale(top_image, (target_width, target_height))
                    else:
                        pygame.mixer.music.load('未知操作.wav')
                        pygame.mixer.music.play(1, 0.0)





        # 更新背景图像的位置
        background_y += BACKGROUND_SPEED

        # 检查是否超出窗口高度，如果是则重置位置
        if background_y >= SCREEN_HEIGHT:
            background_y = 0

        screen.fill((0, 0, 0))  # 清空屏幕
        screen.blit(background_image, (0, background_y))  # 绘制背景图像
        screen.blit(background_image, (0, background_y - SCREEN_HEIGHT))  # 绘制重复的背景图像
        # 绘制顶部图片
        screen.blit(top_image, top_image_rect)
        # 绘制原始赛车
        screen.blit(original_car.image, original_car.rect)

        # 绘制生成的赛车并处理消失逻辑
        for spawned_car in spawned_cars:
            screen.blit(spawned_car.image, spawned_car.rect)
            spawned_car.move_forward(vertical_offset)  # 向下移动生成的赛车
            spawned_car.image.set_alpha(spawned_car.alpha)  # 设置赛车图像的透明度
            if spawned_car.rect.y >= SCREEN_HEIGHT:
                spawned_cars.remove(spawned_car)

        # 绘制录音状态提示文本
        font = pygame.font.Font(None, 36)
        text = font.render("Recording..." if recording else "Press SPACE to start recording", True, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH // 2, 50)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
