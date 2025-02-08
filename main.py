import pygame
import random
import numpy as np
from queue import PriorityQueue

# import copy
# from noise import snoise2

# 初始化Pygame
pygame.init()

# 设置游戏窗口
width, height = 1500, 900
block_size = 50  # 每个块的大小
map_width = width // block_size  # 地图宽度
map_height = height // block_size  # 地图高度
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("SparkleLoveCoins")
font = pygame.font.Font(None, 36)

# Constants
GROUND = 0
WALL = 1
BROKEN_WALL = 2


def load_image(name, scale=1):
    image = pygame.image.load(name)
    return pygame.transform.scale(
        image, (block_size * scale, block_size * scale)
    )  # 缩放图像至块大小


ground_image = load_image("resource/new/ground.png")
wall_image = load_image("resource/new/wall.jpg")
broken_wall1_image = load_image("resource/new/broken_wall1.png")
broken_wall2_image = load_image("resource/new/broken_wall2.png")
broken_wall3_image = load_image("resource/new/broken_wall3.png")
broken_wall4_image = load_image("resource/new/broken_wall4.png")
sparkle_image = load_image("resource/new/sparkle.png")
killer1_image = load_image("resource/new/killer1.png")
killer2_image = load_image("resource/new/killer2.png")
thief_image = load_image("resource/new/thief.png")
ghost_image = load_image("resource/new/ghost.png")
coin_image = load_image("resource/new/coin.png")
bomb_image = load_image("resource/new/bomb.png")
bomb1_image = load_image("resource/new/bomb1.png", scale=3)
bomb2_image = load_image("resource/new/bomb2.png", scale=3)
bomb3_image = load_image("resource/new/bomb3.png", scale=3)
bomb4_image = load_image("resource/new/bomb4.png", scale=3)
bomb5_image = load_image("resource/new/bomb5.png", scale=3)
invincible_image = load_image("resource/new/invincible.png")
# speed_up_image = load_image("resource/new/speed_up.png")
speed_down_image = load_image("resource/new/speed_down.png")
dizzy_image = load_image("resource/new/dizzy.png")


def fade(t):
    """用于平滑插值的fade函数
    在 0~1 之间接近线性
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    """线性插值函数"""
    return a + t * (b - a)


def grad(hash, x, y):
    """基于哈希值的梯度函数
    h = 0: u=x,v=y 返回 u+v
    h = 1: u=x,v=-y 返回 u-v
    h = 2: u=-x,v=y 返回 -u+v
    h = 3: u=-x,v=-y 返回 -u-v
    通过不同的条件组合，grad 生成的结果可以在大致的四个方向上变化：
    正 x / 负 x 和 正 y / 负 y 的组合
    """
    h = hash & 3
    u = x if h < 2 else y
    v = y if h == 1 or h == 2 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def perlin(x, y, perm):
    """生成Perlin噪声"""
    x0 = int(x) & 255
    y0 = int(y) & 255
    x -= int(x)
    y -= int(y)
    u = fade(x)
    v = fade(y)

    aa = perm[perm[x0] + y0]
    ab = perm[perm[x0] + y0 + 1]
    ba = perm[perm[x0 + 1] + y0]
    bb = perm[perm[x0 + 1] + y0 + 1]

    # grad(aa, x, y)可以看作以aa（左上角）为坐标原点，以x,y为坐标的点的梯度值
    # 原本的柏林噪声应当计算aa处的梯度向量与aa指向(x,y)的向量之间的点积
    # 此处grad函数对此做了简化，但是依然保留了柏林噪声的平滑特性
    # 由向量的运算法则，不难知道与ba处梯度值做点积的向量应该是x - 1, y
    # (x0 + x) - (x0 + 1) = x - 1
    return lerp(
        lerp(grad(aa, x, y), grad(ba, x - 1, y), u),
        lerp(grad(ab, x, y - 1), grad(bb, x - 1, y - 1), u),
        v,
    )


def generate_permutation():
    """生成一个随机的哈希表用于Perlin噪声"""
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    return np.stack([perm, perm]).flatten()


# 生成地图的函数
def generate_map():
    game_map = [[None for _ in range(map_width)] for _ in range(map_height)]
    scale = 0.55  # 控制噪声的细节级别
    threshold_wall = 0.55  # 决定墙体生成数量的阈值
    threshold_broken_wall = -0.4

    # 生成随机的哈希表用于Perlin噪声
    perm = generate_permutation()

    for y in range(map_height):
        for x in range(map_width):
            if (x, y) == (sparkle.x, sparkle.y):
                game_map[y][x] = GROUND
                grounds.add((x, y))
                continue
            if x == 0 or x == map_width - 1 or y == 0 or y == 1 or y == map_height - 1:
                game_map[y][x] = WALL
                walls.add((x, y))
                continue

            # 计算噪声值
            noise_value = perlin(x * scale, y * scale, perm)

            if noise_value > threshold_wall:
                game_map[y][x] = WALL
                walls.add((x, y))
            elif noise_value < threshold_broken_wall:
                game_map[y][x] = BROKEN_WALL
                broken_walls[(x, y)] = BrokenWall((x, y))
            else:
                game_map[y][x] = GROUND
                grounds.add((x, y))

    # 检查并修复孤立区域
    fix_isolated_grounds(game_map)

    return game_map


def fix_isolated_grounds(game_map):
    visited = set()

    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < map_width
                        and 0 <= ny < map_height
                        and game_map[ny][nx] == GROUND
                        and (nx, ny) not in visited
                    ):
                        stack.append((nx, ny))

    dfs(sparkle.x, sparkle.y)

    for y in range(map_height):
        for x in range(map_width):
            if game_map[y][x] == GROUND and (x, y) not in visited:
                game_map[y][x] = WALL
                grounds.discard((x, y))
                walls.add((x, y))


def draw_map(game_map):
    for y in range(1, map_height):
        for x in range(map_width):
            screen.blit(ground_image, (x * block_size, y * block_size))
            if game_map[y][x] == WALL:
                screen.blit(wall_image, (x * block_size, y * block_size))
            # elif game_map[y][x] == BROKEN_WALL:  已经在BrokenWall类内绘制
            #     screen.blit(broken_wall1_image, (x * block_size, y * block_size))


def astar(start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))  # (f_score, (x, y))
    came_from = {}

    g_score = {start: 0}
    f_score = {start: manhattan(start, goal)}  # 估计成本

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)  # 随机化方向顺序
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if game_map[neighbor[1]][neighbor[0]] == GROUND:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + manhattan(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set.queue]:
                        open_set.put((f_score[neighbor], neighbor))

    return []  # 返回空路径表示失败


def reconstruct_path(came_from, current):
    total_path = []
    while current in came_from:
        total_path.append(current)
        current = came_from[current]
    total_path.reverse()
    return total_path


class Sparkle:
    def __init__(self):
        self.x = random.randint(1, map_width - 2)
        self.y = random.randint(2, map_height - 2)
        self.direction = (0, 0)
        self.queued_direction = self.direction  # 初始化转弯请求为当前方向
        self.score = 0
        self.speed = 0.06
        self.normal_speed = 0.06
        self.down_speed = 0.03
        self.is_moving = False
        self.last_turn_time = 0  # 上次转弯请求时间
        self.turn_cooldown = (
            75 if self.speed == self.normal_speed else 180
        )  # 转弯请求的冷却时间
        self.score_deducted = False
        self.invincible_active = False
        self.invincible_timer = 0
        self.speed_down_active = False
        self.speed_down_timer = 0
        self.dizzy_active = False
        self.dizzy_timer = 0

    def move(self, game_map):
        if not self.is_moving:
            return
        global killer1, killer2, vertical_ghost, horizontal_ghost, thief, speed_down_items, dizzy_items
        
        new_x = self.x + self.direction[0] * self.speed
        new_y = self.y + self.direction[1] * self.speed

        # 当花火与两个方形区域相交时，其靠上或靠左的方形区域坐标
        grid_x = int(new_x)
        grid_y = int(new_y)

        # 判断是否拾取无敌道具
        for invincible in list(invincible_items):
            if abs(invincible.x - self.x) < 0.2 and abs(invincible.y - self.y) < 0.2:
                invincible_items.remove(invincible)
                invincible_positions.discard((invincible.x, invincible.y))
                occupied.discard((invincible.x, invincible.y))
                self.invincible_active = True
                self.invincible_timer = pygame.time.get_ticks()

        if self.invincible_active:
            current_time = pygame.time.get_ticks()
            if current_time - self.invincible_timer >= invincible_duration:
                self.invincible_active = False

        # 判断是否拾取减速道具
        for speed_down in list(speed_down_items):
            if abs(speed_down.x - self.x) < 0.2 and abs(speed_down.y - self.y) < 0.2:
                speed_down_items.remove(speed_down)
                speed_down_positions.discard((speed_down.x, speed_down.y))
                occupied.discard((speed_down.x, speed_down.y))
                if not self.invincible_active:
                    self.speed = self.down_speed
                    self.turn_cooldown = 180
                    self.speed_down_active = True
                    self.speed_down_timer = pygame.time.get_ticks()

        if self.speed_down_active:
            current_time = pygame.time.get_ticks()
            if current_time - self.speed_down_timer >= speed_down_duration:
                self.speed_down_active = False
                self.speed = self.normal_speed
                self.turn_cooldown = 75

        # 判断是否拾取晕眩道具
        for dizzy in list(dizzy_items):
            if abs(dizzy.x - self.x) < 0.2 and abs(dizzy.y - self.y) < 0.2:
                dizzy_items.remove(dizzy)
                dizzy_positions.discard((dizzy.x, dizzy.y))
                occupied.discard((dizzy.x, dizzy.y))
                if not self.invincible_active:
                    self.speed = 0
                    self.dizzy_active = True
                    self.dizzy_timer = pygame.time.get_ticks()

        if self.dizzy_active:
            current_time = pygame.time.get_ticks()
            if current_time - self.dizzy_timer >= dizzy_duration:
                self.dizzy_active = False
                self.speed = self.normal_speed

        # 判断是否撞击墙体
        if self.direction[0] == 1 or self.direction[0] == -1:
            if game_map[grid_y][int(new_x + 0.5)] == WALL:
                self.is_moving = False
                return True
            elif game_map[grid_y][int(new_x + 0.5)] == BROKEN_WALL:
                if self.invincible_active:
                    broken_walls[(int(new_x + 0.5), grid_y)].trigger_explosion()
                else:
                    self.is_moving = False
                    return True
        elif self.direction[1] == 1 or self.direction[1] == -1:
            if game_map[int(new_y + 0.5)][grid_x] == WALL:
                self.is_moving = False
                return True
            elif game_map[int(new_y + 0.5)][grid_x] == BROKEN_WALL:
                if self.invincible_active:
                    broken_walls[(grid_x, int(new_y + 0.5))].trigger_explosion()
                else:
                    self.is_moving = False
                    return True

        # 判断是否于killer1撞击
        if (
            killer1 is not None
            and abs(killer1.x - self.x) < 0.7
            and abs(killer1.y - self.y) < 0.7
        ):
            if self.invincible_active:
                self.score += 15
                killer1 = None
            else:
                self.is_moving = False
                return True

        # 判断是否与killer2撞击
        if (
            killer2 is not None
            and abs(killer2.x - self.x) < 0.7
            and abs(killer2.y - self.y) < 0.7
        ):
            if self.invincible_active:
                self.score += 15
                killer2 = None
            else:
                self.is_moving = False
                return True

        # 判断是否与vertical_ghost撞击
        if (
            vertical_ghost.visible
            and abs(vertical_ghost.x - self.x) < 0.7
            and abs(vertical_ghost.y - self.y) < 0.7
        ):
            if not self.invincible_active:
                self.is_moving = False
                return True

        # 判断是否与horizontal_ghost撞击
        if (
            horizontal_ghost.visible
            and abs(horizontal_ghost.x - self.x) < 0.7
            and abs(horizontal_ghost.y - self.y) < 0.7
        ):
            if not self.invincible_active:
                self.is_moving = False
                return True

        # 判断是否与thief撞击
        if thief is not None and grid_x == int(thief.x) and grid_y == int(thief.y):
            if self.invincible_active:
                self.score += 15
                thief = None
            elif self.score_deducted == False:
                self.score_deducted = True
                self.score -= 10
                self.score = max(0, self.score)
        else:
            self.score_deducted = False

        # 判断是否要转弯
        if self.queued_direction != self.direction:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_turn_time > self.turn_cooldown:
                self.last_turn_time = current_time

                # 计算中点
                mid_x = grid_x + 0.5
                mid_y = grid_y + 0.5

                if self.direction[0] == 1:
                    if new_x <= mid_x - 0.2:
                        new_x = grid_x
                        self.direction = self.queued_direction
                elif self.direction[0] == -1:
                    if new_x > mid_x + 0.2:
                        new_x = grid_x + 1
                        self.direction = self.queued_direction
                elif self.direction[1] == 1:
                    if new_y <= mid_y - 0.2:
                        new_y = grid_y
                        self.direction = self.queued_direction
                elif self.direction[1] == -1:
                    if new_y > mid_y + 0.2:
                        new_y = grid_y + 1
                        self.direction = self.queued_direction

        self.x, self.y = new_x, new_y

        # 判断是否吃到金币
        for coin in coins:
            if (
                abs(new_x - (coin.x, coin.y)[0]) < 0.2
                and abs(new_y - (coin.x, coin.y)[1]) < 0.2
            ):
                self.score += 10
                coins_positions.discard((coin.x, coin.y))
                coins.remove(coin)
                occupied.discard((coin.x, coin.y))
                break

        # 判断是否吃到炸弹
        for bomb in bombs:
            x0 = bomb.x
            y0 = bomb.y
            x1 = self.x
            y1 = self.y
            if euclidean(x0, y0, x1, y1) < 0.64:
                if len(bombs) == 2:
                    global bomb_timer
                    bomb_timer = 0
                # 判断能否杀死killer1
                if killer1 is not None and euclidean(x0, y0, killer1.x, killer1.y) < 9:
                    self.score += 15
                    killer1 = None
                # 判断能否杀死killer2
                if killer2 is not None and euclidean(x0, y0, killer2.x, killer2.y) < 9:
                    self.score += 15
                    killer2 = None
                # 判断是否能杀死thief
                if thief is not None and euclidean(x0, y0, thief.x, thief.y) < 9:
                    self.score += 15
                    thief = None
                # 清除范围内BROKEN_WALL speed_down 和 dizzy
                for j in range(-3, 4):
                    for i in range(-2, 3):
                        obj_x = int(bomb.x) + i
                        obj_y = int(bomb.y) + j
                        if (obj_x, obj_y) in broken_walls:
                            self.score += 2
                            broken_walls[(obj_x, obj_y)].trigger_explosion()
                        elif (obj_x, obj_y) in speed_down_positions:
                            speed_down_items = [
                                item
                                for item in speed_down_items
                                if (item.x, item.y) != (obj_x, obj_y)
                            ]
                            speed_down_positions.discard((obj_x, obj_y))
                            occupied.discard((obj_x, obj_y))
                        elif (obj_x, obj_y) in dizzy_positions:
                            dizzy_items = [
                                item
                                for item in dizzy_items
                                if (item.x, item.y) != (obj_x, obj_y)
                            ]
                            dizzy_positions.discard((obj_x, obj_y))
                            occupied.discard((obj_x, obj_y))

                explosions.append(Explosion((bomb.x - 1, bomb.y - 1)))
                bombs.remove(bomb)  # 移除被触碰的炸弹
                break  # 碰到炸弹只处理一个

        return False  # 没有碰撞

    def request_turn(self, new_direction):
        # 记录请求方向
        current_time = pygame.time.get_ticks()
        if current_time - self.last_turn_time > self.turn_cooldown:
            self.queued_direction = new_direction
            if not self.is_moving:
                self.is_moving = True
                self.direction = new_direction

    def draw(self):
        screen.blit(
            sparkle_image,
            (self.x * block_size, self.y * block_size),
        )


class Coin:
    def __init__(self, position: tuple[int, int]):
        self.x, self.y = position
        self.exist_time = 0

    def draw(self):
        screen.blit(coin_image, (self.x * block_size, self.y * block_size))

    def update(self, dt):
        self.exist_time += dt


class Bomb:
    def __init__(self, position):
        self.x, self.y = position
        self.speed = (0.015 * random.uniform(1, 2), 0.015 * random.uniform(1, 2))
        self.occupied_space = {
            (int(self.x), int(self.y)),
            (int(self.x), int(self.y) + 1),
            (int(self.x) + 1, int(self.y)),
            (int(self.x) + 1, int(self.y) + 1),
        }

    def move(self):
        new_x = self.x + self.speed[0]
        new_y = self.y + self.speed[1]

        if new_x < 1:
            new_x = 1
            self.speed = (-self.speed[0], self.speed[1])
        elif new_x + 1 > map_width - 1:
            new_x = map_width - 2
            self.speed = (-self.speed[0], self.speed[1])
        if new_y < 2:
            new_y = 2
            self.speed = (self.speed[0], -self.speed[1])
        elif new_y + 1 > map_height - 1:
            new_y = map_height - 2
            self.speed = (self.speed[0], -self.speed[1])

        # 炸弹左上角所处的格子坐标
        grid_x = int(new_x)
        grid_y = int(new_y)

        self.x, self.y = new_x, new_y
        self.occupied_space = {
            (grid_x, grid_y),
            (grid_x, grid_y + 1),
            (grid_x + 1, grid_y),
            (grid_x + 1, grid_y + 1),
        }

    def draw(self):
        screen.blit(bomb_image, (self.x * block_size, self.y * block_size))


class Invincible:
    def __init__(self, position):
        self.x, self.y = position

    def draw(self):
        screen.blit(invincible_image, (self.x * block_size, self.y * block_size))


class SpeedDown:
    def __init__(self, position):
        self.x, self.y = position

    def draw(self):
        screen.blit(speed_down_image, (self.x * block_size, self.y * block_size))


class Dizzy:
    def __init__(self, position):
        self.x, self.y = position

    def draw(self):
        screen.blit(dizzy_image, (self.x * block_size, self.y * block_size))


class Explosion:
    def __init__(self, position):
        self.x, self.y = position
        self.frames = [bomb1_image, bomb2_image, bomb3_image, bomb4_image, bomb5_image]
        self.current_frame = 0
        self.max_frames = len(self.frames)
        self.exist_time = 0
        self.lifetime = 500

    def update(self, dt):
        self.exist_time += dt
        if self.current_frame < self.max_frames - 1:
            self.current_frame = int(
                self.exist_time / (self.lifetime / self.max_frames)
            )

    def draw(self):
        screen.blit(
            self.frames[self.current_frame], (self.x * block_size, self.y * block_size)
        )

    def is_done(self):
        return self.exist_time >= self.lifetime


class BrokenWall:
    def __init__(self, position):
        self.x, self.y = position
        self.frames = [
            broken_wall1_image,
            broken_wall2_image,
            broken_wall3_image,
            broken_wall4_image,
        ]
        self.current_frame = 0
        self.max_frames = len(self.frames)
        self.exist_time = 0
        self.is_exploding = False
        self.explode_duration = 500

    def update(self, dt):
        if self.is_exploding:
            # print("exploding")
            self.exist_time += dt
            if self.exist_time < self.explode_duration:
                self.current_frame = int(
                    self.exist_time / (self.explode_duration / self.max_frames)
                )
            else:
                broken_walls.pop((self.x, self.y))
        else:
            self.current_frame = 0

    def trigger_explosion(self):
        self.is_exploding = True
        game_map[self.y][self.x] = GROUND
        occupied.discard((self.x, self.y))
        grounds.add((self.x, self.y))
        # self.exist_time = 0

    def draw(self):
        if self.current_frame < len(self.frames):
            screen.blit(
                self.frames[self.current_frame],
                (self.x * block_size, self.y * block_size),
            )


class Killer1:
    def __init__(self, position):
        self.x, self.y = position
        self.speed = 0.03
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.new_direction = self.direction
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
        }

    def is_in_sight(self, sparkle: Sparkle, game_map):
        # Bresenham
        start = (int(self.x), int(self.y))
        end = (int(sparkle.x), int(sparkle.y))

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True
        x_inc = dx / steps
        y_inc = dy / steps

        x, y = start
        for _ in range(steps):
            x += x_inc
            y += y_inc
            if game_map[int(round(y))][int(round(x))] in [
                WALL,
                BROKEN_WALL,
            ]:  # round会处理角色中心要加0.5的问题
                return False
        return True

    def move(self):
        grid_x, grid_y = (int(self.x), int(self.y))
        if (self.x - grid_x) < 0.05 and (
            self.y - grid_y
        ) < 0.05:  # 由于只有基本在格子中间时才会转弯，所以不用担心killer会偏移格子过多的问题但不宜让速度超过0.5，否则会导致无法进入if条件
            if self.is_in_sight(sparkle, game_map):
                target = (int(sparkle.x), int(sparkle.y))
                path = astar((grid_x, grid_y), target)

                if path:
                    next_pos = path[0]
                    self.direction = (next_pos[0] - grid_x, next_pos[1] - grid_y)

            next_grid_x, next_grid_y = (
                grid_x + self.direction[0],
                grid_y + self.direction[1],
            )
            if game_map[next_grid_y][next_grid_x] in [WALL, BROKEN_WALL]:
                valid_directions = [
                    (dx, dy)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    if game_map[grid_y + dy][grid_x + dx] == GROUND
                ]
                self.direction = random.choice(valid_directions)

        new_x = self.x + self.direction[0] * self.speed
        new_y = self.y + self.direction[1] * self.speed
        self.x, self.y = new_x, new_y
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
        }

    def draw(self):
        screen.blit(killer1_image, (self.x * block_size, self.y * block_size))


class Killer2:
    def __init__(self, position):
        self.x, self.y = position
        self.speed = 0.02
        self.direction = (0, 0)
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
        }

    def move(self):
        grid_x, grid_y = (int(self.x), int(self.y))
        target = (int(sparkle.x), int(sparkle.y))
        if self.x - grid_x < 0.05 and self.y - grid_y < 0.05:
            path = astar((grid_x, grid_y), target)

            if path:
                next_pos = path[0]
                self.direction = (next_pos[0] - grid_x, next_pos[1] - grid_y)

        new_x = self.x + self.direction[0] * self.speed
        new_y = self.y + self.direction[1] * self.speed
        self.x, self.y = new_x, new_y
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
        }

    def draw(self):
        screen.blit(killer2_image, (self.x * block_size, self.y * block_size))


class Ghost:
    def __init__(self, vertical=True):
        self.vertical = vertical
        self.x = self.y = 0
        self.speed = 0.15
        self.visible = False
        self.active = False
        self.appear_duration = 1500  # 显示警告幽灵的持续时间
        self.distance = (
            map_height - 4 if self.vertical else map_width - 3
        )  # map_height - 2 - 2 | map_width - 2 - 1
        self.moving_duration = self.distance / self.speed / 60 * 1000  # 除以帧数
        self.stop_duration = 1500  # 停留的时间
        self.appear_until = 0
        self.move_until = 0
        self.exists_until = 0
        self.direction = 0

    def reset(self, grid_x, grid_y):
        if self.vertical:
            self.x = grid_x
            self.y = random.choice([2, map_height - 2])
            self.direction = 1 if self.y == 2 else -1
        else:
            self.x = random.choice([1, map_width - 2])
            self.y = grid_y
            self.direction = 1 if self.x == 1 else -1
        self.appear_until = pygame.time.get_ticks() + self.appear_duration
        self.move_until = self.appear_until + self.moving_duration
        self.exists_until = (
            self.appear_until
            + max(map_height - 4, map_width - 3) / self.speed / 60 * 1000
            + self.stop_duration
        )
        self.visible = True
        self.active = True

    def update(self):
        if self.active:
            current_time = pygame.time.get_ticks()
            if current_time < self.appear_until:
                self.visible = True  # 在警告期间显示幽灵
            elif current_time < self.move_until:
                self.visible = True
                if self.vertical:
                    self.y += self.speed * self.direction
                else:
                    self.x += self.speed * self.direction
            elif current_time < self.exists_until:
                self.visible = True
            else:
                self.visible = False
                self.active = False

    def draw(self):
        if self.visible:
            screen.blit(ghost_image, (self.x * block_size, self.y * block_size))


class Thief:
    def __init__(self, position):
        self.x, self.y = position
        self.speed = 0.02
        self.direction = (0, 0)
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for j in range(-1, 2)
            for i in range(-1, 2)
        }

    def move(self):
        grid_x, grid_y = (int(self.x), int(self.y))
        closest_coin = None
        closest_distance = float("inf")
        for coin in coins:
            dist = euclidean(self.x, self.y, coin.x, coin.y)
            if dist < closest_distance:
                closest_distance = dist
                closest_coin = coin

        if closest_coin is not None:
            # 小偷吃到金币
            if (
                abs(self.x - closest_coin.x) < 0.2
                and abs(self.y - closest_coin.y) < 0.2
            ):
                coins_positions.discard((closest_coin.x, closest_coin.y))
                coins.remove(closest_coin)
                sparkle.score -= 5
                sparkle.score = max(0, sparkle.score)
            target = (closest_coin.x, closest_coin.y)
            if self.x - grid_x < 0.05 and self.y - grid_y < 0.05:
                path = astar((grid_x, grid_y), target)
                if path:
                    next_pos = path[0]
                    self.direction = (next_pos[0] - grid_x, next_pos[1] - grid_y)
        new_x = self.x + self.direction[0] * self.speed
        new_y = self.y + self.direction[1] * self.speed
        self.x, self.y = new_x, new_y
        self.occupied_space = {
            (int(self.x) + i, int(self.y) + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
        }

    def draw(self):
        screen.blit(thief_image, (self.x * block_size, self.y * block_size))


def manhattan(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def euclidean(x0, y0, x1, y1):
    return (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)


def generate_coin():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied and abs(sparkle.x - x) > 1 and abs(sparkle.y - y) > 1
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        coins_positions.add((x, y))
        coins.append(Coin((x, y)))
        occupied.add((x, y))  # 更新被占用的空间


def generate_broken_wall():
    invalid_positions = {
        (int(sparkle.x) + i, int(sparkle.y) + j)
        for i in range(-2, 3)
        for j in range(-2, 3)  # 建立一个 5x5 的范围
    }
    valid_ground_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in (invalid_positions | occupied)
    ]
    if valid_ground_positions:
        random.shuffle(valid_ground_positions)
        x, y = random.choice(valid_ground_positions)
        game_map[y][x] = BROKEN_WALL
        broken_walls[(x, y)] = BrokenWall((x, y))
        grounds.discard((x, y))  # 更新地面状态
        occupied.add((x, y))  # 更新被占用的空间


def generate_bomb():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied and abs(sparkle.x - x) > 3 and abs(sparkle.y - y) > 3
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        new_bomb = Bomb((x, y))
        bombs.append(new_bomb)
        occupied.union(new_bomb.occupied_space)  # 更新被占用的空间


def generate_invincible():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        invincible_items.append(Invincible((x, y)))
        invincible_positions.add((x, y))
        occupied.add((x, y))


def generate_killer1():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied and abs(sparkle.x - x) > 4 and abs(sparkle.y - y) > 4
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        global killer1
        killer1 = Killer1((x, y))
        occupied.union(killer1.occupied_space)


def generate_killer2():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied and abs(sparkle.x - x) > 4 and abs(sparkle.y - y) > 4
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        global killer2
        killer2 = Killer2((x, y))
        occupied.union(killer2.occupied_space)


def generate_thief():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied and abs(sparkle.x - x) > 4 and abs(sparkle.y - y) > 4
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        global thief
        thief = Thief((x, y))
        occupied.union(thief.occupied_space)


def generate_speed_down():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        speed_down_items.append(SpeedDown((x, y)))
        speed_down_positions.add((x, y))
        occupied.add((x, y))


def generate_dizzy():
    valid_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in occupied
    ]
    if valid_positions:
        random.shuffle(valid_positions)
        x, y = random.choice(valid_positions)
        dizzy_items.append(Dizzy((x, y)))
        dizzy_positions.add((x, y))
        occupied.add((x, y))


running = True
game_over = False
walls = set()
broken_walls: dict[tuple[int, int], BrokenWall] = {}
grounds = set()
sparkle = Sparkle()
game_map = generate_map()
occupied = set()
coins: list[Coin] = []
coins_positions = set()
coin_timer = 0
coin_interval = 5000
coin_lifetime = 30000
broken_wall_timer = 0
broken_wall_interval = 6000
bombs: list[Bomb] = []
bomb_timer = 0
bomb_cooling_period = 10000  # 冷却30s
max_bombs = 2
explosions: list[Explosion] = []
killer1: Killer1 = None
killer1_timer = 0
killer1_cooling_period = 30000
killer2: Killer2 = None
killer2_timer = 0
killer2_cooling_period = 20000
vertical_ghost: Ghost = Ghost(vertical=True)
horizontal_ghost: Ghost = Ghost(vertical=False)
vertical_flag = False
ghost_timer = 0
ghost_cooling_period = 10000
thief: Thief = None
thief_timer = 0
thief_cooling_period = 10000
invincible_items: list[Invincible] = []
invincible_positions = set()
invincible_duration = 8000
invincible_score_intervals = 100
last_invincible_score = 0
speed_down_items: list[SpeedDown] = []
speed_down_positions = set()
speed_down_duration = 2000  # 减速持续时间2秒
speed_down_timer = 0
speed_down_cooling_period = 15000  # 每10秒生成一个减速道具
dizzy_items: list[Dizzy] = []
dizzy_positions = set()
dizzy_duration = 2000  # 晕眩持续时间2秒
dizzy_timer = 0
dizzy_cooling_period = 15000  # 每15秒生成一个dizzy道具


# 初始化时钟和时间计数
clock = pygame.time.Clock()

while running:
    screen.fill((0, 0, 0))
    draw_map(game_map)

    broken_wall_instances = list(broken_walls.values())
    for broken_wall in broken_wall_instances:
        broken_wall.update(clock.get_time())
        broken_wall.draw()

    if not game_over:
        occupied: set = (
            walls
            | set(broken_walls.keys())
            | coins_positions
            | invincible_positions
            | speed_down_positions
            | dizzy_positions
        )
        for bomb in bombs:
            occupied.union(bomb.occupied_space)
        if killer1 is not None:
            occupied.union(killer1.occupied_space)
        if killer2 is not None:
            occupied.union(killer2.occupied_space)
        if thief is not None:
            occupied.union(thief.occupied_space)

        if sparkle.is_moving:
            grid_x, grid_y = int(sparkle.x), int(sparkle.y)
            for coin in list(
                coins
            ):  # 使用 list() 创建 coins 的副本，以便在迭代过程中修改 coins
                coin.update(clock.get_time())
                if coin.exist_time >= coin_lifetime:
                    coins_positions.discard((coin.x, coin.y))
                    coins.remove(coin)
                    occupied.discard((coin.x, coin.y))
                else:
                    coin.draw()

            broken_wall_timer += clock.get_time()
            if broken_wall_timer >= broken_wall_interval:
                generate_broken_wall()
                broken_wall_timer = 0

            coin_timer += clock.get_time()
            if coin_timer >= coin_interval or len(coins) == 0:
                generate_coin()
                coin_timer = 0

            if len(bombs) < max_bombs and sparkle.score >= 30:
                bomb_timer += clock.get_time()
                if bomb_timer >= bomb_cooling_period:
                    generate_bomb()
                    bomb_timer = 0

            for bomb in list(bombs):
                bomb.move()
                bomb.draw()

            for explosion in list(explosions):
                explosion.update(clock.get_time())
                explosion.draw()
                if explosion.is_done():
                    explosions.remove(explosion)

            if killer1 is None:
                killer1_timer += clock.get_time()
                if killer1_timer >= killer1_cooling_period:
                    generate_killer1()
                    killer1_timer = 0
            else:
                killer1.move()
                killer1.draw()

            if killer2 is None:
                killer2_timer += clock.get_time()
                if killer2_timer >= killer2_cooling_period:
                    generate_killer2()
                    killer2_timer = 0
            else:
                killer2.move()
                killer2.draw()

            ghost_timer += clock.get_time()
            if (
                ghost_timer >= ghost_cooling_period
                and 3 <= grid_x <= map_width - 4
                and 4 <= grid_y <= map_height - 4
            ):
                if vertical_flag:
                    vertical_ghost.reset(grid_x, grid_y)
                    vertical_flag = False
                else:
                    horizontal_ghost.reset(grid_x, grid_y)
                    vertical_flag = True
                ghost_timer = 0

            vertical_ghost.update()
            horizontal_ghost.update()
            vertical_ghost.draw()
            horizontal_ghost.draw()

            if thief is None:
                thief_timer += clock.get_time()
                if thief_timer >= thief_cooling_period:
                    generate_thief()
                    thief_timer = 0
            else:
                thief.move()
                thief.draw()

            if sparkle.score - last_invincible_score >= invincible_score_intervals:
                generate_invincible()
                last_invincible_score += invincible_score_intervals

            for invincible in invincible_items:
                invincible.draw()

            speed_down_timer += clock.get_time()
            if speed_down_timer >= speed_down_cooling_period:
                generate_speed_down()
                speed_down_timer = 0

            dizzy_timer += clock.get_time()
            if dizzy_timer >= dizzy_cooling_period:
                generate_dizzy()
                dizzy_timer = 0

            for speed_down in speed_down_items:
                speed_down.draw()

            for dizzy in dizzy_items:
                dizzy.draw()

        score_text = font.render(f"Score: {sparkle.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        # 绘制剩余无敌时间
        if sparkle.invincible_active:
            remaining_time = max(
                0,
                (
                    invincible_duration
                    - (pygame.time.get_ticks() - sparkle.invincible_timer)
                )
                / 1000,
            )
            invincible_time_text = font.render(
                f"Invincible: {remaining_time:.1f}s", True, (255, 255, 0)
            )
            text_rect = invincible_time_text.get_rect(center=(width // 4, 20))
            screen.blit(invincible_time_text, text_rect)

        # 绘制剩余减速时间
        if sparkle.speed_down_active:
            remaining_time = max(
                0,
                (
                    speed_down_duration
                    - (pygame.time.get_ticks() - sparkle.speed_down_timer)
                )
                / 1000,
            )
            speed_down_time_text = font.render(
                f"Speed Down: {remaining_time:.1f}s", True, (255, 255, 0)
            )
            text_rect = speed_down_time_text.get_rect(center=(width // 2, 20))
            screen.blit(speed_down_time_text, text_rect)

        # 绘制剩余眩晕时间
        if sparkle.dizzy_active:
            remaining_time = max(
                0,
                (dizzy_duration - (pygame.time.get_ticks() - sparkle.dizzy_timer))
                / 1000,
            )
            dizzy_time_text = font.render(
                f"Dizzy: {remaining_time:.1f}s", True, (255, 255, 0)
            )
            text_rect = dizzy_time_text.get_rect(center=(width // 4 * 3, 20))
            screen.blit(dizzy_time_text, text_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and sparkle.direction != (0, 1):
                    sparkle.request_turn((0, -1))
                elif event.key == pygame.K_DOWN and sparkle.direction != (0, -1):
                    sparkle.request_turn((0, 1))
                elif event.key == pygame.K_LEFT and sparkle.direction != (1, 0):
                    sparkle.request_turn((-1, 0))
                elif event.key == pygame.K_RIGHT and sparkle.direction != (-1, 0):
                    sparkle.request_turn((1, 0))

        if sparkle.move(game_map):  # 碰撞会返回True
            game_over = True

        sparkle.draw()

    else:
        game_over_text = font.render("Game Over!", True, (255, 0, 0))
        score_text = font.render(f"Final Score: {sparkle.score}", True, (255, 255, 255))
        restart_text = font.render("Press Space to Restart", True, (255, 255, 255))
        screen.blit(game_over_text, (width // 2 - 50, height // 2 - 30))
        screen.blit(score_text, (width // 2 - 60, height // 2))
        screen.blit(restart_text, (width // 2 - 90, height // 2 + 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # 按空格重启
                    occupied.clear()
                    grounds.clear()
                    walls.clear()
                    broken_walls.clear()
                    coins.clear()
                    coins_positions.clear()
                    bombs.clear()
                    coin_timer = 0
                    broken_wall_timer = 0
                    bomb_timer = 0
                    explosions.clear()
                    killer1 = None
                    killer1_timer = 0
                    killer2 = None
                    killer2_timer = 0
                    vertical_ghost: Ghost = Ghost(vertical=True)
                    horizontal_ghost: Ghost = Ghost(vertical=False)
                    ghost_timer = 0
                    thief = None
                    thief_timer = 0
                    invincible_items.clear()
                    invincible_positions.clear()
                    last_invincible_score = 0
                    speed_down_items.clear()
                    speed_down_positions.clear()
                    speed_down_timer = 0
                    dizzy_items.clear()
                    dizzy_positions.clear()
                    dizzy_timer = 0
                    game_over = False
                    sparkle = Sparkle()
                    game_map = generate_map()

    pygame.display.flip()
    clock.tick(60)  # 控制帧率为 60 FPS

pygame.quit()
