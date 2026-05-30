import math
import sys
import numpy as np
import pygame

# ═══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════
INFO_WIDTH   = 240       # ancho del panel lateral de información
FPS          = 60

# Colores (R, G, B)
COL_UNKNOWN  = (26,  26,  46)
COL_FREE     = (200, 230, 210)
COL_OCC      = (60,  60,  90)
COL_ROBOT    = (230, 57,  70)
COL_LIDAR    = (255, 200, 50)
COL_BG       = (15,  15,  30)
COL_PANEL    = (20,  20,  40)
COL_TEXT     = (200, 200, 210)
COL_OBST     = (100, 120, 160)


# ═══════════════════════════════════════════════════════════════════════════
#  BRESENHAM 
# ═══════════════════════════════════════════════════════════════════════════

def bresenham(x0: int, y0: int, x1: int, y1: int):
    """
    Genera todas las celdas (col, row) entre (x0,y0) y (x1,y1).
    La celda final NO se incluye (se trata aparte como ocupada).
    """
    cells = []
    dx =  abs(x1 - x0);  sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0);  sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if x0 == x1 and y0 == y1:
            break
        cells.append((x0, y0))
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy;  x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx;  y0 += sy

    return cells


# ═══════════════════════════════════════════════════════════════════════════
#  LiDAR VIRTUAL
# ═══════════════════════════════════════════════════════════════════════════

class VirtualLidar:

    def __init__(self, num_rays: int = 128, max_range: float = 3.0,
                 noise_std: float = 0.05):
        self.num_rays  = num_rays
        self.max_range = max_range
        self.noise_std = noise_std

    def _ray_vs_aabb(self, rx, ry, cos_a, sin_a,
                     ox, oy, ow, oh) -> float:

        ox1, ox2 = ox - ow / 2, ox + ow / 2
        oy1, oy2 = oy - oh / 2, oy + oh / 2
        best = math.inf
        if abs(cos_a) > 1e-9:
            for edge_x in (ox1, ox2):
                t = (edge_x - rx) / cos_a
                if t > 0:
                    hy = ry + t * sin_a
                    if oy1 - 0.01 <= hy <= oy2 + 0.01:
                        best = min(best, t)
        if abs(sin_a) > 1e-9:
            for edge_y in (oy1, oy2):
                t = (edge_y - ry) / sin_a
                if t > 0:
                    hx = rx + t * cos_a
                    if ox1 - 0.01 <= hx <= ox2 + 0.01:
                        best = min(best, t)
        return best

    def scan(self, robot_x: float, robot_y: float, robot_yaw: float,
             obstacles: list, world_half_w: float, world_half_h: float) -> list:

        hw = world_half_w - 0.05
        hh = world_half_h - 0.05
        results = []

        for i in range(self.num_rays):
            angle   = robot_yaw + (2 * math.pi * i / self.num_rays)
            cos_a   = math.cos(angle)
            sin_a   = math.sin(angle)
            min_dist = self.max_range

            # ── Intersección exacta con límites del mundo ───────────────

            world_min_x = 0.0
            world_max_x = 2 * hw

            world_min_y = 0.0
            world_max_y = 2 * hh

            t_values = []

            # Intersección con paredes verticales
            if abs(cos_a) > 1e-9:

                # pared izquierda
                t = (world_min_x - robot_x) / cos_a
                if t > 0:
                    y_hit = robot_y + t * sin_a
                    if world_min_y <= y_hit <= world_max_y:
                        t_values.append(t)

                # pared derecha
                t = (world_max_x - robot_x) / cos_a
                if t > 0:
                    y_hit = robot_y + t * sin_a
                    if world_min_y <= y_hit <= world_max_y:
                        t_values.append(t)

            # Intersección con paredes horizontales
            if abs(sin_a) > 1e-9:

                # pared inferior
                t = (world_min_y - robot_y) / sin_a
                if t > 0:
                    x_hit = robot_x + t * cos_a
                    if world_min_x <= x_hit <= world_max_x:
                        t_values.append(t)

                # pared superior
                t = (world_max_y - robot_y) / sin_a
                if t > 0:
                    x_hit = robot_x + t * cos_a
                    if world_min_x <= x_hit <= world_max_x:
                        t_values.append(t)

            if t_values:
                min_dist = min(min_dist, min(t_values))

            # ── Obstáculos ───────────────────────────────────────────────
            for obs in obstacles:
                d = self._ray_vs_aabb(robot_x, robot_y, cos_a, sin_a,
                                      obs['x'], obs['y'], obs['w'], obs['h'])
                min_dist = min(min_dist, d)

            # ── Ruido ────────────────────────────────────────────────────
            noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
            dist  = max(0.05, min_dist + noise)
            hit   = min_dist < self.max_range 

            results.append({
                'angle' : angle,
                'dist'  : dist,
                'hit'   : hit,
                'hit_x' : robot_x + dist * cos_a,
                'hit_y' : robot_y + dist * sin_a,
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID  (lógica idéntica al nodo ROS)
# ═══════════════════════════════════════════════════════════════════════════

class OccupancyGrid:

    def __init__(self, resolution: float = 0.05, world_h: float = 3.7, world_w: float = 4.8,
                 l_occ: float = 0.7,  l_free: float = -0.4,
                 l_min: float = -5.0, l_max: float =  5.0,
                 range_min: float = 0.05):

        self.resolution = resolution
        self.world_h    = world_h
        self.world_w    = world_w
        self.l_occ      = l_occ
        self.l_free     = l_free
        self.l_min      = l_min
        self.l_max      = l_max
        self.range_min  = range_min

        self.world_w = world_w
        self.world_h = world_h

        self.cols = int(world_w / resolution)
        self.rows = int(world_h / resolution)

        self.origin_x = 0.0
        self.origin_y = 0.0

        self.logodds = np.zeros((self.rows, self.cols), dtype=np.float32)

    def reset(self):
        self.logodds[:] = 0.0

    def world_to_cell(self, x: float, y: float):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def update(self, robot_x: float, robot_y: float, scan: list):

        rc, rr = self.world_to_cell(robot_x, robot_y)

        for ray in scan:
            dist = ray['dist']
            if dist < self.range_min:
                continue

            hit    = ray['hit']
            eff    = dist if hit else dist   # usa la distancia medida
            hx, hy = ray['hit_x'], ray['hit_y']

            hc, hr = self.world_to_cell(hx, hy)

            # Celdas libres a lo largo del rayo
            for col, row in bresenham(rc, rr, hc, hr):
                if self.in_bounds(col, row):
                    self.logodds[row, col] = np.clip(
                        self.logodds[row, col] + self.l_free,
                        self.l_min, self.l_max)

            # Celda de impacto → ocupada
            if hit and self.in_bounds(hc, hr):
                self.logodds[hr, hc] = np.clip(
                    self.logodds[hr, hc] + self.l_occ,
                    self.l_min, self.l_max)

    def to_rgb_array(self) -> np.ndarray:
        prob = 1.0 / (1.0 + np.exp(-self.logodds))   # sigmoid
        img  = np.full((self.rows, self.cols, 3),
                       COL_UNKNOWN, dtype=np.uint8)

        known = np.abs(self.logodds) > 0.01
        occ   = known & (prob > 0.6)
        free_ = known & (prob <= 0.6)

        img[free_] = COL_FREE
        img[occ]   = COL_OCC

        return img


# ═══════════════════════════════════════════════════════════════════════════
#  ROBOT
# ═══════════════════════════════════════════════════════════════════════════

class Robot:

    def __init__(self):

        # Pose REAL
        self.real_x = 4.45
        self.real_y = 0.4
        self.real_yaw = math.radians(180)

        # Pose ESTIMADA por odometría
        self.odom_x = 4.45
        self.odom_y = 0.4
        self.odom_yaw = math.radians(180)

        # Parámetros reales del robot
        self.radio = 0.0505
        self.length = 0.183

        # Velocidades ruedas
        self.wr = 0.0
        self.wl = 0.0

        # velocidades máximas
        self.max_wheel_speed = 8.0

    def update(self, keys, dt, hw, hh):

        # ----------------------------------------------------------
        # CONTROL TECLADO
        # ----------------------------------------------------------

        wr_cmd = 0.0
        wl_cmd = 0.0

        base = 1.5
        turn = 1

        if keys[pygame.K_w]:
            wr_cmd += base
            wl_cmd += base

        if keys[pygame.K_s]:
            wr_cmd -= base
            wl_cmd -= base

        if keys[pygame.K_a]:
            wr_cmd -= turn
            wl_cmd += turn

        if keys[pygame.K_d]:
            wr_cmd += turn
            wl_cmd -= turn

        self.wr = wr_cmd
        self.wl = wl_cmd

        # ----------------------------------------------------------
        # MOVIMIENTO REAL
        # ----------------------------------------------------------

        vr = self.radio * self.wr
        vl = self.radio * self.wl

        v = (vr + vl) / 2.0
        w = (vr - vl) / self.length

        self.real_x += v * math.cos(self.real_yaw) * dt
        self.real_y += v * math.sin(self.real_yaw) * dt
        self.real_yaw += w * dt

        self.real_yaw = (self.real_yaw + math.pi) % (2 * math.pi) - math.pi

        # ----------------------------------------------------------
        # ODOMETRÍA (CON ERROR)
        # ----------------------------------------------------------

        noisy_wr = self.wr + np.random.normal(0, 0.05)
        noisy_wl = self.wl + np.random.normal(0, 0.05)

        vr_o = self.radio * noisy_wr
        vl_o = self.radio * noisy_wl

        v_o = (vr_o + vl_o) / 2.0
        w_o = (vr_o - vl_o) / self.length

        self.odom_x += v_o * math.cos(self.odom_yaw) * dt
        self.odom_y += v_o * math.sin(self.odom_yaw) * dt
        self.odom_yaw += w_o * dt

        self.odom_yaw = (self.odom_yaw + math.pi) % (2 * math.pi) - math.pi

        # ----------------------------------------------------------
        # LIMITES
        # ----------------------------------------------------------
        lim_x = hw - 0.15
        lim_y = hh - 0.15

        self.real_x = max(0.0 + 0.15, min(2*hw - 0.15, self.real_x))
        self.real_y = max(0.0 + 0.15, min(2*hh - 0.15, self.real_y))


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class Sim:
    # ── Parámetros del mapa ────────────────────────────────────────────────
    RESOLUTION = 0.02     # m/celda
    WORLD_W = 5.458
    WORLD_H = 3.668

    # Log-odds (idénticos al nodo ROS)
    L_OCC  =  0.7
    L_FREE = -0.4
    L_MIN  = -5.0
    L_MAX  =  5.0

    # LiDAR
    NUM_RAYS   = 128
    MAX_RANGE  = 3.0      # metros
    NOISE_STD  = 0.01     # metros

    def __init__(self):
        pygame.init()
        self.map_px_x    = 546
        self.map_px_y    = 369
        self.screen = pygame.display.set_mode(
            (self.map_px_x * 2 + INFO_WIDTH, self.map_px_y))
        pygame.display.set_caption('PuzzleBot – Occupancy Grid Sim')
        self.clock     = pygame.time.Clock()
        self.font_sm   = pygame.font.SysFont('monospace', 13)
        self.font_md   = pygame.font.SysFont('monospace', 15, bold=True)

        self.grid  = OccupancyGrid(
            resolution=self.RESOLUTION, world_w=self.WORLD_W, world_h=self.WORLD_H,
            l_occ=self.L_OCC, l_free=self.L_FREE,
            l_min=self.L_MIN, l_max=self.L_MAX)

        self.lidar = VirtualLidar(
            num_rays=self.NUM_RAYS,
            max_range=self.MAX_RANGE,
            noise_std=self.NOISE_STD)

        self.robot = Robot()
        self.hw = self.WORLD_W / 2.0
        self.hh = self.WORLD_H / 2.0
        self.cell_px_x = self.map_px_x / self.grid.cols
        self.cell_px_y = self.map_px_y / self.grid.rows

        self.show_lidar = True
        self.last_scan  = []

        # Obstáculos iniciales (AABB: x, y = centro, w, h = tamaño)
        self.obstacles = [
            {'x': 1.32, 'y':  1.171, 'w': 1.050, 'h': 0.221},
            {'x': 1.32, 'y':  1.837, 'w': 1.050, 'h': 0.221},
            {'x': 1.32, 'y':  2.6075, 'w': 1.050, 'h': 0.221},
            {'x': 4.84, 'y':  3.368, 'w': 0.02, 'h': 0.600},
            {'x': 4.84, 'y':  1.9925, 'w': 0.02, 'h': 0.531},
            {'x': 5.15, 'y':  1.727, 'w': 0.608, 'h': 0.02},
            {'x': 5.15, 'y':  0.1725, 'w': 0.608, 'h': 0.345},
            {'x': 5.15, 'y':  0.780, 'w': 0.608, 'h': 0.150},
            {'x': 5.15, 'y':  1.290, 'w': 0.608, 'h': 0.150},
            {'x': -1.0, 'y':  2.0, 'w': 1.5, 'h': 0.3},
        ]

    def draw_ground_truth(self, surf):

        gt = pygame.Surface((self.map_px_x, self.map_px_y))
        gt.fill((30, 30, 40))

        # Obstáculos reales
        for obs in self.obstacles:

            ox, oy = self.world_to_px(
                obs['x'] - obs['w']/2,
                obs['y'] - obs['h']/2)

            ow = int(obs['w'] / self.RESOLUTION * self.cell_px_x)
            oh = int(obs['h'] / self.RESOLUTION * self.cell_px_y)

            pygame.draw.rect(gt, (180,180,220), (ox, oy, ow, oh))

        # Robot REAL
        rpx, rpy = self.world_to_px(
            self.robot.real_x,
            self.robot.real_y)

        pygame.draw.circle(gt, (255,70,70), (rpx,rpy), 8)

        # orientación
        fx = int(rpx + 18 * math.cos(self.robot.real_yaw))
        fy = int(rpy + 18 * math.sin(self.robot.real_yaw))

        pygame.draw.line(gt, (255,255,255), (rpx,rpy), (fx,fy), 2)

        # rayos lidar
        for ray in self.last_scan:

            hpx, hpy = self.world_to_px(ray['hit_x'], ray['hit_y'])

            pygame.draw.line(
                gt,
                (255,220,80),
                (rpx,rpy),
                (hpx,hpy),
                1)

        surf.blit(gt, (0,0))

    # ── Conversión coordenadas ─────────────────────────────────────────────

    def world_to_px(self, wx, wy):
        col = (wx - self.grid.origin_x) / self.RESOLUTION
        row = (wy - self.grid.origin_y) / self.RESOLUTION

        px = int(col * self.cell_px_x)
        py = int(row * self.cell_px_y)

        return px, py

    # ── Rendering ─────────────────────────────────────────────────────────

    def draw_map(self, surf: pygame.Surface):
        """Dibuja el occupancy grid como imagen escalada."""
        rgb = self.grid.to_rgb_array()
        # numpy array → pygame surface (transponer porque pygame es col-mayor)
        small = pygame.surfarray.make_surface(
            np.transpose(rgb, (1, 0, 2)))
        scaled = pygame.transform.scale(small, (self.map_px_x, self.map_px_y))
        surf.blit(scaled, (0, 0))

    def draw_obstacles_outline(self, surf: pygame.Surface):
        """Dibuja contorno semitransparente de obstáculos (referencia)."""
        for obs in self.obstacles:
            ox, oy = self.world_to_px(obs['x'] - obs['w'] / 2,
                                      obs['y'] - obs['h'] / 2)
            ow = int(obs['w'] / self.RESOLUTION * self.cell_px_x)
            oh = int(obs['h'] / self.RESOLUTION * self.cell_px_y)
            pygame.draw.rect(surf, COL_OBST, (ox, oy, ow, oh), 1)

    def draw_lidar(self, surf: pygame.Surface, x_offset=0):

        if not self.show_lidar or not self.last_scan:
            return

        rpx, rpy = self.world_to_px(
            self.robot.real_x,
            self.robot.real_y)

        rpx += x_offset

        for ray in self.last_scan:

            hpx, hpy = self.world_to_px(
                ray['hit_x'],
                ray['hit_y'])

            hpx += x_offset

            pygame.draw.line(
                surf,
                COL_LIDAR,
                (rpx, rpy),
                (hpx, hpy),
                1)

            if ray['hit']:
                pygame.draw.circle(
                    surf,
                    COL_LIDAR,
                    (hpx, hpy),
                    2)

    def draw_robot(self, surf: pygame.Surface):
        """Dibuja el robot como círculo con flecha de orientación."""
        rpx, rpy = self.world_to_px(self.robot.odom_x, self.robot.odom_y)
        r = 7
        pygame.draw.circle(surf, COL_ROBOT, (rpx, rpy), r)
        pygame.draw.circle(surf, (255, 255, 255), (rpx, rpy), r, 1)
        flen = 14
        fx = int(rpx + flen * math.cos(self.robot.odom_yaw))
        fy = int(rpy + flen * math.sin(self.robot.odom_yaw))
        pygame.draw.line(surf, (255, 255, 255), (rpx, rpy), (fx, fy), 2)

    def draw_info(self, surf: pygame.Surface):
        """Panel lateral con estadísticas y ayuda."""
        panel = pygame.Surface((INFO_WIDTH, self.map_px_y))
        panel.fill(COL_PANEL)

        def txt(text, y, color=COL_TEXT, font=None):
            f = font or self.font_sm
            s = f.render(text, True, color)
            panel.blit(s, (10, y))

        y = 10
        txt('PuzzleBot OGM Sim', y, (220, 200, 255), self.font_md); y += 28
        pygame.draw.line(panel, (60, 60, 90), (5, y), (INFO_WIDTH - 5, y)); y += 10

        txt('── Pose robot ──', y, (160, 160, 180)); y += 20
        txt(f'  x   = {self.robot.real_x:+.3f} m', y);      y += 18
        txt(f'  y   = {self.robot.real_y:+.3f} m', y);      y += 18
        txt(f'  yaw = {math.degrees(self.robot.real_yaw):+.1f}°', y); y += 24

        known  = np.abs(self.grid.logodds) > 0.01
        prob   = 1 / (1 + np.exp(-self.grid.logodds[known]))
        n_free = int(np.sum(prob <= 0.6))
        n_occ  = int(np.sum(prob > 0.6))
        total  = self.grid.rows * self.grid.cols

        txt('── Mapa ──', y, (160, 160, 180)); y += 20
        txt(f'  libres  = {n_free}', y);      y += 18
        txt(f'  ocupadas= {n_occ}', y);       y += 18
        txt(f'  desc.   = {total - known.sum()}', y); y += 18
        pct = 100 * known.sum() / total
        txt(f'  explor. = {pct:.1f}%', y);   y += 24

        txt('── LiDAR ──', y, (160, 160, 180)); y += 20
        txt(f'  rayos   = {self.lidar.num_rays}', y);  y += 18
        txt(f'  rango   = {self.lidar.max_range:.1f} m', y); y += 18
        txt(f'  ruido σ = {self.lidar.noise_std:.2f} m', y); y += 24

        pygame.draw.line(panel, (60, 60, 90), (5, y), (INFO_WIDTH - 5, y)); y += 10
        txt('── Controles ──', y, (160, 160, 180)); y += 20
        for line in [
            'W/S  avanzar/retroceder',
            'A/D  girar',
            'R    reiniciar mapa',
            'O    + obstáculo',
            'L    toggle LiDAR',
            'ESC  salir',
        ]:
            txt(f'  {line}', y); y += 17

        y += 10
        ld_col = (100, 220, 100) if self.show_lidar else (120, 80, 80)
        txt(f'  LiDAR: {"ON" if self.show_lidar else "OFF"}', y, ld_col)

        surf.blit(panel, (self.map_px_x  * 2, 0))

    # ── Loop principal ─────────────────────────────────────────────────────

    def add_random_obstacle(self):
        ang  = np.random.uniform(0, 2 * math.pi)
        dist = np.random.uniform(1.0, 3.0)
        self.obstacles.append({
            'x': float(np.clip(math.cos(ang) * dist, -4.0, 4.0)),
            'y': float(np.clip(math.sin(ang) * dist, -4.0, 4.0)),
            'w': float(np.random.uniform(0.2, 1.0)),
            'h': float(np.random.uniform(0.2, 1.0)),
        })

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            # ── Eventos ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        self.grid.reset()
                    if event.key == pygame.K_l:
                        self.show_lidar = not self.show_lidar
                    if event.key == pygame.K_o:
                        self.add_random_obstacle()

            # ── Movimiento del robot ────────────────────────────────────
            keys = pygame.key.get_pressed()
            self.robot.update(keys, dt, self.hw, self.hh)

            # ── LiDAR + actualización del mapa ─────────────────────────
            self.last_scan = self.lidar.scan(
                self.robot.real_x, self.robot.real_y, self.robot.real_yaw,
                self.obstacles, self.hw, self.hh)
            self.grid.update(self.robot.real_x, self.robot.real_y, self.last_scan)

            # ── Dibujo ─────────────────────────────────────────────────
            self.screen.fill(COL_BG)

            # izquierda = mundo real
            self.draw_ground_truth(self.screen)

            # derecha = occupancy grid
            map_surface = pygame.Surface((self.map_px_x, self.map_px_y))
            self.draw_map(map_surface)

            self.screen.blit(map_surface, (self.map_px_x, 0))

            self.draw_obstacles_outline(self.screen)
            self.draw_lidar(self.screen)
            self.draw_robot(self.screen)
            self.draw_info(self.screen)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


# ═══════════════════════════════════════════════════════════════════════════

def main():
    sim = Sim()
    sim.run()

if __name__ == '__main__':
    main()