#!/usr/bin/env python3
"""
semantic_planner.py  —  v2  (mapa dual)
═══════════════════════════════════════════════════════════════════════
Nodo 1 de 3 — Planificación semántica + rutas estructuradas

FILOSOFÍA DE PLANIFICACIÓN
───────────────────────────
Se usan dos mapas complementarios:

  semantic_map.png  — zonas navegables (verde/azul/rojo)
                      Se usa solo como "fallback" para tramos
                      fuera de las rutas estructuradas.

  route_map.png     — rutas predefinidas (morado 255,0,255 BGR)
                      Representan los pasillos ideales para el
                      montacargas (líneas rectas, sin curvas).

LÓGICA DE PLANIFICACIÓN (3 fases)
──────────────────────────────────
  FASE 0 — Incorporación a la ruta (si aplica)
    Si el robot NO está sobre una línea morada, se calcula un
    tramo corto con A* semántico desde la posición actual hasta
    el punto de ruta más cercano.

  FASE 1 — Navegación por rutas estructuradas
    Se recorre el grafo de rutas moradas desde el punto de entrada
    hasta el punto de ruta más cercano al destino final.
    El grafo se construye una sola vez al inicio desde los píxeles
    morados del route_map.

  FASE 2 — Llegada al destino (si aplica)
    Si el destino final NO está sobre una ruta morada, se calcula
    un tramo corto con A* semántico desde el punto de ruta más
    cercano hasta el destino.

SISTEMAS DE COORDENADAS
────────────────────────
  PIXEL  (px, py)  — origen esquina sup-izq, y crece hacia abajo
  WORLD  (wx, wy)  — origen esquina inf-izq, y crece hacia arriba
  ODOM   (ox, oy)  — origen en el CENTRO del mapa (frame del robot)

  odom → pixel:  odom_to_pixel()
  pixel → odom:  pixel_to_odom()

Tópicos suscritos:
  /odom        — nav_msgs/Odometry
  /goal_input  — std_msgs/String  ("carga" | "descarga" | "racks" | "x y")

Tópicos publicados:
  /plan        — nav_msgs/Path  (frame_id = "odom")

Servicios:
  /replan      — std_srvs/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy import qos
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

import cv2
import heapq
import numpy as np
import math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SEMANTIC_MAP_PATH = "/home/juanjo/semantic_map.png"
ROUTE_MAP_PATH    = "/home/juanjo/route_map.png"
MAP_RESOLUTION    = 0.05        # m/pixel

# Radio en píxeles para considerar que el robot "está sobre" una ruta
ROUTE_SNAP_RADIUS = 8           # ~0.4 m con res=0.05

# Submuestreo del path A* semántico (puntos de zona)
PATH_SUBSAMPLE    = 5

# Submuestreo del path de ruta (reduce waypoints intermedios en líneas rectas)
ROUTE_SUBSAMPLE   = 10

# Nombres de zona reconocidos
ZONE_NAMES = ("carga", "descarga", "racks")


# ═══════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════

class CoordConverter:
    def __init__(self, W: int, H: int, resolution: float):
        self.W         = W
        self.H         = H
        self.res       = resolution
        self.ox_offset = (W / 2) * resolution
        self.oy_offset = (H / 2) * resolution

    def pixel_to_world(self, px, py):
        return px * self.res, (self.H - 1 - py) * self.res

    def world_to_pixel(self, wx, wy):
        px = int(np.clip(int(wx / self.res),           0, self.W - 1))
        py = int(np.clip(int((self.H - 1) - wy / self.res), 0, self.H - 1))
        return px, py

    def odom_to_world(self, ox, oy):
        return ox + self.ox_offset, oy + self.oy_offset

    def world_to_odom(self, wx, wy):
        return wx - self.ox_offset, wy - self.oy_offset

    def odom_to_pixel(self, ox, oy):
        return self.world_to_pixel(*self.odom_to_world(ox, oy))

    def pixel_to_odom(self, px, py):
        return self.world_to_odom(*self.pixel_to_world(px, py))


# ═══════════════════════════════════════════════════════════════════════
#  CARGA DE MAPAS
# ═══════════════════════════════════════════════════════════════════════

def load_image(path: str, name: str = ""):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar {name or path}")
    return img


def build_semantic_masks(semantic: np.ndarray):
    """
    Devuelve:
      nav_mask  — uint8, 1 = navegable (verde + azul + rojo)
      zones     — dict con máscaras booleanas por zona
      dist_map  — distancia normalizada a obstáculos (para A*)
    """
    green = (semantic[:,:,1] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,2] < 80)
    blue  = (semantic[:,:,0] > 200) & (semantic[:,:,1] < 80) & (semantic[:,:,2] < 80)
    red   = (semantic[:,:,2] > 200) & (semantic[:,:,0] < 80) & (semantic[:,:,1] < 80)

    nav = np.zeros(semantic.shape[:2], dtype=np.uint8)
    nav[green] = 1
    nav[blue]  = 1
    nav[red]   = 1

    dist_map = cv2.distanceTransform(nav, cv2.DIST_L2, 5)
    max_d = dist_map.max()
    if max_d > 0:
        dist_map /= max_d

    return nav, {"carga": green, "racks": blue, "descarga": red}, dist_map


def build_route_mask(route_img: np.ndarray):
    """
    Extrae píxeles morados (255, 0, 255 en BGR).
    Devuelve máscara booleana H×W.
    """
    # Tolerancia ±30 por canal para cubrir compresión JPEG/PNG
    b, g, r = route_img[:,:,0], route_img[:,:,1], route_img[:,:,2]
    mask = (b > 180) & (g < 80) & (r > 180)
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  GRAFO DE RUTAS
# ═══════════════════════════════════════════════════════════════════════

class RouteGraph:
    """
    Construye un grafo ligero a partir de los píxeles de ruta.

    Estrategia:
      1. Submuestrea los píxeles de ruta (grid cada ROUTE_SUBSAMPLE px).
      2. Conecta nodos vecinos que tengan un píxel de ruta entre ellos
         (verificación por Bresenham simplificado).
      3. Búsqueda de camino mínimo con Dijkstra / A*.

    El resultado es una secuencia de píxeles que sigue la red de rutas.
    """

    def __init__(self, route_mask: np.ndarray, subsample: int = ROUTE_SUBSAMPLE):
        self.mask      = route_mask          # bool H×W
        self.H, self.W = route_mask.shape
        self.subsample = subsample

        # KD-tree ligero: lista de todos los píxeles de ruta
        ys, xs = np.where(route_mask)
        self.route_pixels = np.column_stack([xs, ys])  # (N,2) en orden (x,y)

        print(f"  RouteGraph: {len(self.route_pixels)} píxeles de ruta")

    def nearest_route_pixel(self, px: int, py: int):
        """Devuelve el píxel de ruta más cercano a (px, py)."""
        if len(self.route_pixels) == 0:
            return None
        pts = self.route_pixels.astype(np.float32)
        diffs = pts - np.array([px, py], dtype=np.float32)
        dists = np.hypot(diffs[:,0], diffs[:,1])
        idx   = np.argmin(dists)
        x, y  = self.route_pixels[idx]
        return (int(x), int(y)), float(dists[idx])

    def on_route(self, px: int, py: int, radius: int = ROUTE_SNAP_RADIUS) -> bool:
        """¿Está (px,py) sobre o cerca de una línea de ruta?"""
        _, dist = self.nearest_route_pixel(px, py)
        return dist <= radius

    def _line_on_route(self, p1, p2) -> bool:
        """
        Verifica que todos los puntos del segmento p1→p2 estén
        sobre la máscara de ruta (Bresenham).
        """
        x0, y0 = p1
        x1, y1 = p2
        dx, dy  = abs(x1 - x0), abs(y1 - y0)
        sx      = 1 if x0 < x1 else -1
        sy      = 1 if y0 < y1 else -1
        err     = dx - dy
        x, y    = x0, y0
        while True:
            if not (0 <= x < self.W and 0 <= y < self.H):
                return False
            if not self.mask[y, x]:
                return False
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy
        return True

    def find_path(self, start_px, goal_px):
        """
        A* sobre los píxeles de ruta.

        Para mantener el path sobre la red de rutas, los movimientos
        permitidos son a los 8-vecinos que también sean píxeles de ruta.

        Devuelve lista de (px, py) o None.
        """
        # Snap start y goal a la ruta
        start_snap, _ = self.nearest_route_pixel(*start_px)
        goal_snap,  _ = self.nearest_route_pixel(*goal_px)

        if start_snap is None or goal_snap is None:
            return None

        def h(a, b):
            return math.hypot(b[0]-a[0], b[1]-a[1])

        MOVES = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        open_set  = [(h(start_snap, goal_snap), start_snap)]
        came_from = {}
        g_score   = {start_snap: 0.0}
        closed    = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == goal_snap:
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start_snap)
                return path[::-1]

            if cur in closed:
                continue
            closed.add(cur)

            cx, cy = cur
            for dx, dy in MOVES:
                nx, ny = cx + dx, cy + dy
                nb = (nx, ny)
                if nb in closed:
                    continue
                if not (0 <= nx < self.W and 0 <= ny < self.H):
                    continue
                if not self.mask[ny, nx]:
                    continue
                # Diagonal: verificar esquinas
                if dx != 0 and dy != 0:
                    if not self.mask[cy, cx+dx] or not self.mask[cy+dy, cx]:
                        continue

                cost = math.hypot(dx, dy)
                tg   = g_score[cur] + cost
                if tg < g_score.get(nb, math.inf):
                    came_from[nb] = cur
                    g_score[nb]   = tg
                    heapq.heappush(open_set, (tg + h(nb, goal_snap), nb))

        return None   # sin camino


# ═══════════════════════════════════════════════════════════════════════
#  A* SEMÁNTICO (igual que v1, para tramos fuera de ruta)
# ═══════════════════════════════════════════════════════════════════════

def astar_semantic(nav, distance_map, start, goal):
    H, W    = nav.shape
    MOVES   = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    closed  = set()
    came_from = {}
    g     = {start: 0.0}
    f     = {start: math.hypot(goal[0]-start[0], goal[1]-start[1])}
    heap  = [(f[start], start)]

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur); cur = came_from[cur]
            path.append(start)
            return path[::-1]
        closed.add(cur)
        cx, cy = cur
        for dx, dy in MOVES:
            nx, ny = cx+dx, cy+dy
            nb = (nx, ny)
            if nb in closed: continue
            if not (0 <= nx < W and 0 <= ny < H): continue
            if nav[ny, nx] == 0: continue
            if dx and dy:
                if nav[cy, cx+dx] == 0 or nav[cy+dy, cx] == 0: continue
            penalty = 8.0 / (distance_map[ny, nx] + 1.0)
            tg = g[cur] + math.hypot(dx,dy) + penalty
            if tg < g.get(nb, math.inf):
                came_from[nb] = cur
                g[nb] = tg
                heapq.heappush(heap, (tg + math.hypot(goal[0]-nx, goal[1]-ny), nb))
    return None


# ═══════════════════════════════════════════════════════════════════════
#  UTILIDADES DE PATH
# ═══════════════════════════════════════════════════════════════════════

def subsample_path(path, step):
    if not path:
        return path
    sampled = path[::step]
    if sampled[-1] != path[-1]:
        sampled = sampled + [path[-1]]
    return sampled


def smooth_direction(path):
    """Elimina puntos intermedios redundantes (misma dirección)."""
    if len(path) < 3:
        return path
    result = [path[0]]
    for i in range(1, len(path)-1):
        pdx = path[i][0]   - path[i-1][0]
        pdy = path[i][1]   - path[i-1][1]
        ndx = path[i+1][0] - path[i][0]
        ndy = path[i+1][1] - path[i][1]
        if (pdx, pdy) != (ndx, ndy):
            result.append(path[i])
    result.append(path[-1])
    return result


def nearest_navigable(nav, conv, px, py, radius=20):
    best_dist, best = math.inf, None
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            nx, ny = px+dx, py+dy
            if 0 <= nx < conv.W and 0 <= ny < conv.H and nav[ny, nx]:
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist, best = d, (nx, ny)
    return best


def get_zone_center_px(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


# ═══════════════════════════════════════════════════════════════════════
#  NODO ROS
# ═══════════════════════════════════════════════════════════════════════

class SemanticPlannerNode(Node):

    def __init__(self):
        super().__init__('semantic_planner')

        # ── Cargar mapas ──────────────────────────────────────────────
        self.semantic_img = load_image(SEMANTIC_MAP_PATH, "semantic_map")
        self.route_img    = load_image(ROUTE_MAP_PATH,    "route_map")

        H, W, _ = self.semantic_img.shape
        self.conv = CoordConverter(W, H, MAP_RESOLUTION)

        self.nav, self.zones, self.dist_map = build_semantic_masks(self.semantic_img)
        route_mask = build_route_mask(self.route_img)
        self.route_graph = RouteGraph(route_mask)

        n_route = int(route_mask.sum())
        self.get_logger().info(
            f'Mapas cargados: {W}×{H} px  |  '
            f'Píxeles de ruta: {n_route}  |  '
            f'Offset odom: ({self.conv.ox_offset:.2f}, {self.conv.oy_offset:.2f}) m')

        # ── Estado ────────────────────────────────────────────────────
        self.robot_ox  = 0.0
        self.robot_oy  = 0.0
        self.odom_ok   = False
        self.path_msg  = None
        self.goal_desc = None

        # ── ROS ───────────────────────────────────────────────────────
        self.plan_pub = self.create_publisher(Path, '/plan', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb,
            qos.qos_profile_sensor_data)

        self.goal_sub = self.create_subscription(
            String, '/goal_input', self._goal_input_cb, 10)

        self.replan_srv = self.create_service(
            Trigger, '/replan', self._replan_cb)

        self.timer_pub = self.create_timer(1.0, self._publish_plan)

        self.get_logger().info(
            'semantic_planner v2 listo.\n'
            '  Envía el destino con:\n'
            '  ros2 topic pub --once /goal_input std_msgs/String "data: \'descarga\'"\n'
            '  ros2 topic pub --once /goal_input std_msgs/String "data: \'1.5 -0.8\'"')

    # ─────────────────────────────────────────────────────────────────
    #  ODOMETRÍA
    # ─────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.robot_ox = msg.pose.pose.position.x
        self.robot_oy = msg.pose.pose.position.y
        if not self.odom_ok:
            self.odom_ok = True
            self.get_logger().info(
                f'Primera odometría — '
                f'odom=({self.robot_ox:.2f}, {self.robot_oy:.2f}) m')

    # ─────────────────────────────────────────────────────────────────
    #  ENTRADA DE DESTINO
    # ─────────────────────────────────────────────────────────────────

    def _goal_input_cb(self, msg: String):
        raw = msg.data.strip().lower()
        self.get_logger().info(f'Destino recibido: "{raw}"')
        goal_px = self._parse_goal(raw)
        if goal_px is None:
            self.get_logger().warn(
                f'Destino no reconocido: "{raw}"  |  '
                f'Zonas: {", ".join(ZONE_NAMES)}  |  '
                'Coords: "x y" en metros')
            return
        self._plan(goal_px, description=raw)

    def _parse_goal(self, text: str):
        words = text.strip().split()
        for name in ZONE_NAMES:
            if name in words:
                center = get_zone_center_px(self.zones[name])
                if center is None:
                    self.get_logger().warn(f"Zona '{name}' vacía en el mapa")
                    return None
                return center
        if len(words) == 2:
            try:
                ox, oy = float(words[0]), float(words[1])
                return self.conv.odom_to_pixel(ox, oy)
            except ValueError:
                pass
        return None

    # ─────────────────────────────────────────────────────────────────
    #  PLANIFICACIÓN DUAL
    # ─────────────────────────────────────────────────────────────────

    def _plan(self, goal_px, description=""):
        """
        Planificación en tres fases:

          FASE 0 — Si el robot no está sobre la ruta:
                   A* semántico → punto de ruta más cercano al robot.

          FASE 1 — Navegación por el grafo de rutas:
                   punto de ruta inicio → punto de ruta más cercano al goal.

          FASE 2 — Si el goal no está sobre la ruta:
                   A* semántico → desde punto de ruta hasta goal.

        Las tres fases se concatenan en un único nav_msgs/Path.
        """
        if not self.odom_ok:
            self.get_logger().warn('Aún sin odometría — espera a /odom')
            return

        log = self.get_logger()
        start_px = self.conv.odom_to_pixel(self.robot_ox, self.robot_oy)
        gx, gy   = goal_px

        log.info(f'  Inicio pixel={start_px}  |  Goal pixel={goal_px}')

        # Asegurar navegabilidad del inicio
        sx, sy = start_px
        if self.nav[sy, sx] == 0:
            start_px = nearest_navigable(self.nav, self.conv, sx, sy)
            if start_px is None:
                log.error('No se encontró celda navegable cerca del inicio')
                return
            log.info(f'  Inicio ajustado a {start_px}')

        # Asegurar navegabilidad del goal en mapa semántico
        if self.nav[gy, gx] == 0:
            goal_px = nearest_navigable(self.nav, self.conv, gx, gy)
            if goal_px is None:
                log.error('No se encontró celda navegable cerca del goal')
                return
            log.info(f'  Goal ajustado a {goal_px}')
            gx, gy = goal_px

        full_path = []

        # ── FASE 0: incorporación a la ruta ──────────────────────────
        start_on_route = self.route_graph.on_route(*start_px)

        if start_on_route:
            # El robot ya está sobre la ruta; snap directo
            entry_snap, _ = self.route_graph.nearest_route_pixel(*start_px)
            log.info(f'  Fase 0: robot en ruta → snap a {entry_snap}')
            full_path.append(start_px)
        else:
            entry_snap, d_entry = self.route_graph.nearest_route_pixel(*start_px)
            log.info(f'  Fase 0: A* semántico inicio→ruta  '
                     f'(dist={d_entry:.1f} px, snap={entry_snap})')
            seg0 = astar_semantic(self.nav, self.dist_map, start_px, entry_snap)
            if seg0 is None:
                log.error('  Fase 0: A* semántico falló')
                return
            seg0_s = smooth_direction(subsample_path(seg0, PATH_SUBSAMPLE))
            log.info(f'    {len(seg0)} px crudos → {len(seg0_s)} waypoints')
            full_path.extend(seg0_s)

        # ── FASE 1: navegación por rutas ──────────────────────────────
        goal_on_route = self.route_graph.on_route(*goal_px)

        if goal_on_route:
            exit_snap, _ = self.route_graph.nearest_route_pixel(*goal_px)
        else:
            exit_snap, _ = self.route_graph.nearest_route_pixel(*goal_px)

        log.info(f'  Fase 1: ruta  {entry_snap} → {exit_snap}')

        if entry_snap != exit_snap:
            route_path = self.route_graph.find_path(entry_snap, exit_snap)
            if route_path is None:
                log.error('  Fase 1: no se encontró camino en el grafo de rutas')
                return
            route_s = subsample_path(route_path, ROUTE_SUBSAMPLE)
            log.info(f'    {len(route_path)} px de ruta → {len(route_s)} waypoints')
            # Evitar duplicar el punto de entrada que ya puede estar en full_path
            if full_path and route_s and route_s[0] == full_path[-1]:
                route_s = route_s[1:]
            full_path.extend(route_s)
        else:
            log.info('    Entrada y salida en el mismo nodo de ruta')

        # ── FASE 2: llegada al destino ────────────────────────────────
        if not goal_on_route:
            log.info(f'  Fase 2: A* semántico ruta→goal  '
                     f'(exit={exit_snap}, goal={goal_px})')
            seg2 = astar_semantic(self.nav, self.dist_map, exit_snap, goal_px)
            if seg2 is None:
                log.warn('  Fase 2: A* semántico falló — '
                         'el plan termina en el punto de ruta más cercano')
            else:
                seg2_s = smooth_direction(subsample_path(seg2, PATH_SUBSAMPLE))
                log.info(f'    {len(seg2)} px crudos → {len(seg2_s)} waypoints')
                if full_path and seg2_s and seg2_s[0] == full_path[-1]:
                    seg2_s = seg2_s[1:]
                full_path.extend(seg2_s)
        else:
            # Añadir goal exacto si no está ya
            if full_path[-1] != goal_px:
                full_path.append(goal_px)

        # ── Construir nav_msgs/Path ───────────────────────────────────
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = self.get_clock().now().to_msg()

        for px, py in full_path:
            ox, oy = self.conv.pixel_to_odom(px, py)
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp    = msg.header.stamp
            pose.pose.position.x = ox
            pose.pose.position.y = oy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_msg  = msg
        self.goal_desc = description

        self.plan_pub.publish(self.path_msg)
        log.info(f'  ✓ /plan publicado: {len(full_path)} waypoints  '
                 f'(fase0+fase1+fase2)  frame=odom')

    # ─────────────────────────────────────────────────────────────────
    #  PUBLICAR PLAN (latched 1 Hz)
    # ─────────────────────────────────────────────────────────────────

    def _publish_plan(self):
        if self.path_msg is None:
            return
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.plan_pub.publish(self.path_msg)

    # ─────────────────────────────────────────────────────────────────
    #  SERVICIO /replan
    # ─────────────────────────────────────────────────────────────────

    def _replan_cb(self, request, response):
        if self.goal_desc is None:
            response.success = False
            response.message = 'Sin destino previo'
            return response
        goal_px = self._parse_goal(self.goal_desc)
        if goal_px is None:
            response.success = False
            response.message = f'No se pudo resolver "{self.goal_desc}"'
            return response
        self._plan(goal_px, description=self.goal_desc)
        if self.path_msg:
            response.success = True
            response.message = f'{len(self.path_msg.poses)} waypoints'
        else:
            response.success = False
            response.message = 'Planificación falló'
        return response


# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SemanticPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
