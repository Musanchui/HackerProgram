import pygame
import time

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
top_image = pygame.transform.scale(top_image, (1000,500))
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
                new_car = Car(original_car.rect.x - (COL_WIDTH + COL_GAP) + left_spawn_offset, spawn_y_positions[current_spawn_y_index] + vertical_spawn_offset, generated_car_image_path)
                spawned_cars.append(new_car)
                current_spawn_y_index = (current_spawn_y_index + 1) % len(spawn_y_positions)
            elif event.key == pygame.K_1:
                new_car = Car(original_car.rect.x + (COL_WIDTH + COL_GAP) + right_spawn_offset, spawn_y_positions[current_spawn_y_index] + vertical_spawn_offset, generated_car_image_path)
                spawned_cars.append(new_car)
                current_spawn_y_index = (current_spawn_y_index + 1) % len(spawn_y_positions)

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

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

