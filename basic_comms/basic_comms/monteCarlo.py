import os, sys, time, threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import distance_transform_edt

# CoppeliaSim ZMQ remote API
ZMQAPI_PATH = os.path.expanduser(
    '~/Descargas/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04'
    '/programming/zmqRemoteApi/clients/python'
)
COPPELIA_AVAILABLE = False
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    COPPELIA_AVAILABLE = True
except ImportError:
    if os.path.isdir(ZMQAPI_PATH):
        sys.path.insert(0, ZMQAPI_PATH)
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            COPPELIA_AVAILABLE = True
        except ImportError:
            pass
if not COPPELIA_AVAILABLE:
    print("[WARN] coppeliasim_zmqremoteapi_client not found — running in SIMULATED mode.")

# ── Config ────────────────────────────────────────────────────────────────────
COPPELIA_HOST   = '10.22.153.25'
COPPELIA_PORT   = 23000

OBJ_ROBOT       = '/LineTracer'
OBJ_LEFT_JOINT  = '/DynamicLeftJoint'
OBJ_RIGHT_JOINT = '/DynamicRightJoint'
OBJ_LIDAR       = '/LaserScanner2D'
OBJ_LIDAR_JOINT = '/LaserScanner2D/joint'

WHEEL_RADIUS    = 0.027
WHEEL_DISTANCE  = 0.119

MAP_SIZE_CM     = 250
PIXELS_PER_CM   = 1
MAP_SIZE_PX     = MAP_SIZE_CM * PIXELS_PER_CM

N_PARTICLES     = 200
N_LIDAR_RAYS    = 16
N_LIDAR_REAL    = 32
LIDAR_MAX_CM    = 500

TRANS_NOISE     = 1.5
ROT_NOISE       = np.deg2rad(3)

SIGMA_INIT      = 25.0
SIGMA_MIN       = 1.0
SIGMA_DECAY     = 0.85

M_TO_CM         = 100
TOP_K_FRACTION  = 0.1
PIXEL_SCORE_RADIUS_CM = 5

OBSTACLES_CM = [
    (0,   225, 50,  25),
    (150, 150, 100, 100),
    (0,   100, 25,  25),
    (0,   0,   100, 100),
    (200, 0,   50,  50),
    (200, 100, 50,  50),
]

# ── Map ───────────────────────────────────────────────────────────────────────

def build_map(obstacles, size_px, px_per_cm):
    grid = np.zeros((size_px, size_px), dtype=np.uint8)
    grid[:, :2] = 1;  grid[:, -2:] = 1
    grid[:2, :] = 1;  grid[-2:, :] = 1
    for xc, yc, w, h in obstacles:
        x0, x1 = int(xc * px_per_cm), int((xc + w) * px_per_cm)
        y0, y1 = int(yc * px_per_cm), int((yc + h) * px_per_cm)
        grid[y0:y1, x0:x1] = 1
    return grid


def build_dist_transform(grid):
    free = (grid == 0).astype(np.float32)
    return distance_transform_edt(free).astype(np.float32)


# ── Ray casting (sphere tracing) ──────────────────────────────────────────────

def batch_lidar(grid, dt, xs, ys, thetas, n_rays, max_range, px_per_cm):
    size_px  = grid.shape[0]
    N        = len(xs)
    ray_angs = np.linspace(-np.deg2rad(45), np.deg2rad(45), n_rays, endpoint=True)
    ang      = thetas[:, None] + ray_angs[None, :]
    ddx, ddy = np.cos(ang), np.sin(ang)

    cx = (xs * px_per_cm).astype(np.float32)[:, None] * np.ones((1, n_rays), np.float32)
    cy = (ys * px_per_cm).astype(np.float32)[:, None] * np.ones((1, n_rays), np.float32)
    dx = (ddx * px_per_cm).astype(np.float32)
    dy = (ddy * px_per_cm).astype(np.float32)

    dists    = np.full((N, n_rays), max_range, dtype=np.float32)
    hit      = np.zeros((N, n_rays), dtype=bool)
    traveled = np.zeros((N, n_rays), dtype=np.float32)
    max_px   = int(max_range * px_per_cm)
    MIN_STEP = 0.5

    for _ in range(max_px):
        col = cx.astype(np.int32)
        row = cy.astype(np.int32)

        out     = (col < 0) | (col >= size_px) | (row < 0) | (row >= size_px)
        new_out = (~hit) & out
        dists[new_out] = traveled[new_out] / px_per_cm
        hit |= new_out
        if hit.all(): break

        col_c = np.clip(col, 0, size_px - 1)
        row_c = np.clip(row, 0, size_px - 1)

        obstacle = grid[row_c, col_c] > 0
        new_obs  = (~hit) & (~out) & obstacle
        dists[new_obs] = traveled[new_obs] / px_per_cm
        hit |= new_obs
        if hit.all(): break

        safe_px  = dt[row_c, col_c]
        step_px  = np.where(hit, 0.0,
                            np.maximum(MIN_STEP, np.minimum(safe_px, max_px - traveled)))
        traveled += step_px
        cx       += dx * (step_px / px_per_cm)
        cy       += dy * (step_px / px_per_cm)

        hit |= (~hit) & (traveled >= max_px)
        if hit.all(): break

    return dists


def single_lidar(grid, dt, x, y, theta, n_rays, max_range, px_per_cm):
    return batch_lidar(grid, dt,
                       np.array([x]), np.array([y]), np.array([theta]),
                       n_rays, max_range, px_per_cm)[0]


# ── Scoring ───────────────────────────────────────────────────────────────────

def pixel_score_particles(particles, obs_scan, grid, n_rays, px_per_cm, radius_cm=3):
    size_px  = grid.shape[0]
    N        = len(particles)
    ray_angs = np.linspace(-np.deg2rad(45), np.deg2rad(45), n_rays, endpoint=True)

    angles   = particles[:, 2:3] + ray_angs[None, :]
    impact_x = particles[:, 0:1] + obs_scan[None, :] * np.cos(angles)
    impact_y = particles[:, 1:2] + obs_scan[None, :] * np.sin(angles)

    col   = (impact_x * px_per_cm).astype(np.int32)
    row   = (impact_y * px_per_cm).astype(np.int32)
    valid = (col >= 0) & (col < size_px) & (row >= 0) & (row < size_px)
    col_c = np.clip(col, 0, size_px - 1)
    row_c = np.clip(row, 0, size_px - 1)

    r_px       = max(1, int(radius_cm * px_per_cm))
    pixel_hits = np.zeros((N, n_rays), dtype=np.float32)

    for dr in range(-r_px, r_px + 1):
        for dc in range(-r_px, r_px + 1):
            if dr * dr + dc * dc > r_px * r_px:
                continue
            r2 = np.clip(row_c + dr, 0, size_px - 1)
            c2 = np.clip(col_c + dc, 0, size_px - 1)
            pixel_hits += grid[r2, c2].astype(np.float32)

    pixel_hits[~valid] = 0.0
    area   = np.pi * r_px ** 2 + 1
    scores = pixel_hits.sum(axis=1) / (n_rays * area)
    return scores.astype(np.float64)


def score_particles_batch(particles, obs_scan, grid, dt,
                           n_rays, max_range, px_per_cm, sigma):
    xs, ys, thetas = particles[:, 0], particles[:, 1], particles[:, 2]
    sim_scans  = batch_lidar(grid, dt, xs, ys, thetas, n_rays, max_range, px_per_cm)
    diff       = np.abs(sim_scans - obs_scan[None, :])
    gauss      = np.mean(np.exp(-0.5 * (diff / sigma) ** 2), axis=1)
    px_score   = pixel_score_particles(
        particles, obs_scan, grid, n_rays, px_per_cm, PIXEL_SCORE_RADIUS_CM)
    return gauss * (1.0 + px_score)


# ── Particle filter ───────────────────────────────────────────────────────────

def sample_free(grid, map_size_cm, px_per_cm, n):
    size_px = grid.shape[0]
    pts     = []
    while len(pts) < n:
        bx   = np.random.uniform(5, map_size_cm - 5, n * 4)
        by   = np.random.uniform(5, map_size_cm - 5, n * 4)
        cols = (bx * px_per_cm).astype(np.int32)
        rows = (by * px_per_cm).astype(np.int32)
        valid = ((cols >= 0) & (cols < size_px) &
                 (rows >= 0) & (rows < size_px) &
                 (grid[rows, cols] == 0))
        for x, y in zip(bx[valid], by[valid]):
            pts.append([x, y, np.random.uniform(0, 2 * np.pi)])
            if len(pts) == n:
                break
    return np.array(pts[:n], dtype=np.float64)


def motion_update(particles, d_trans, d_rot):
    n   = len(particles)
    nt  = d_trans + np.random.normal(0, TRANS_NOISE, n)
    nr  = d_rot   + np.random.normal(0, ROT_NOISE,   n)
    p   = particles.copy()
    p[:, 0] += nt * np.cos(particles[:, 2] + nr / 2)
    p[:, 1] += nt * np.sin(particles[:, 2] + nr / 2)
    p[:, 2]  = (particles[:, 2] + nr) % (2 * np.pi)
    return p


def topk_filter_and_resample(particles, weights, grid, map_size_cm, px_per_cm,
                              top_k_frac=TOP_K_FRACTION, random_frac=0.05):
    N         = len(particles)
    k         = max(1, int(N * top_k_frac))
    n_random  = max(1, int(N * random_frac))
    order     = np.argsort(weights)[::-1]
    top_idx   = order[:k]
    rest_idx  = order[k:]

    new_p  = particles.copy()
    w_top  = weights[top_idx]
    w_top /= w_top.sum() + 1e-300
    chosen  = np.random.choice(k, size=len(rest_idx), replace=True, p=w_top)

    new_p[rest_idx]     = particles[top_idx[chosen]].copy()
    new_p[rest_idx, 0] += np.random.normal(0, 1.5,            len(rest_idx))
    new_p[rest_idx, 1] += np.random.normal(0, 1.5,            len(rest_idx))
    new_p[rest_idx, 2] += np.random.normal(0, np.deg2rad(3),  len(rest_idx))
    new_p[rest_idx, 2] %= (2 * np.pi)

    new_p[order[-n_random:]] = sample_free(grid, map_size_cm, px_per_cm, n_random)
    return new_p


def keep_in_bounds_vectorized(particles, grid, map_size_cm, px_per_cm):
    size_px = grid.shape[0]
    p       = particles.copy()
    p[:, 0] = np.clip(p[:, 0], 2, map_size_cm - 2)
    p[:, 1] = np.clip(p[:, 1], 2, map_size_cm - 2)

    cols    = np.clip((p[:, 0] * px_per_cm).astype(np.int32), 0, size_px - 1)
    rows    = np.clip((p[:, 1] * px_per_cm).astype(np.int32), 0, size_px - 1)
    in_wall = grid[rows, cols] > 0

    if in_wall.sum() > 0:
        p[in_wall] = sample_free(grid, map_size_cm, px_per_cm, in_wall.sum())
    return p


def weighted_mean_pose(particles, weights):
    w   = weights / (weights.sum() + 1e-300)
    ex  = np.sum(particles[:, 0] * w)
    ey  = np.sum(particles[:, 1] * w)
    eth = np.arctan2(
        np.sum(np.sin(particles[:, 2]) * w),
        np.sum(np.cos(particles[:, 2]) * w))
    return ex, ey, eth


def convergence_score(weights):
    w    = weights / (weights.sum() + 1e-300)
    neff = 1.0 / (np.sum(w ** 2) + 1e-300)
    return neff / len(weights)


# ── CoppeliaSim interface ─────────────────────────────────────────────────────

class CoppeliaInterface:
    _grid_ref = None
    _dt_ref   = None

    def __init__(self):
        self.connected  = False
        self.pose       = (MAP_SIZE_CM / 2, MAP_SIZE_CM / 2, 0.0)
        self.scan       = np.full(N_LIDAR_RAYS, LIDAR_MAX_CM)
        self._data_lock = threading.Lock()
        self._sim_lock  = threading.Lock()
        self._running   = True
        self.has_lidar  = False

        if not COPPELIA_AVAILABLE:
            return
        try:
            client   = RemoteAPIClient(COPPELIA_HOST, COPPELIA_PORT)
            self.sim = client.getObject('sim')
            print(f"[Coppelia] Connected  t={self.sim.getSimulationTime():.2f}s")

            self.robot_h     = self.sim.getObject(OBJ_ROBOT)
            self.lw          = self.sim.getObject(OBJ_LEFT_JOINT)
            self.rw          = self.sim.getObject(OBJ_RIGHT_JOINT)
            self.lidar_joint = self.sim.getObject(OBJ_LIDAR_JOINT)

            try:
                self.lidar_h      = self.sim.getObject(OBJ_LIDAR)
                self.lidar_sensor = self.sim.getObject('/LaserScanner2D/joint/sensor')
                self.has_lidar    = True
                print("[Coppelia] LIDAR found.")
            except Exception:
                print("[Coppelia] LIDAR not found — using simulated scan.")

            self.connected = True
        except Exception as e:
            print(f"[Coppelia] Connection failed: {e} — SIMULATED mode")

    def start(self):
        threading.Thread(target=self._pose_loop,  daemon=True).start()
        threading.Thread(target=self._lidar_loop, daemon=True).start()

    def _pose_loop(self):
        while self._running:
            if self.connected:
                try:
                    with self._sim_lock:
                        self._read_pose()
                except Exception as e:
                    print(f"[Pose] error: {e}")
            time.sleep(0.05)

    def _read_pose(self):
        pos  = self.sim.getObjectPosition(self.robot_h, -1)
        quat = self.sim.getObjectQuaternion(self.robot_h, -1)
        x, y, z, w = quat
        yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        x_map = pos[0] * M_TO_CM + MAP_SIZE_CM / 2 - 126.9
        y_map = pos[1] * M_TO_CM + MAP_SIZE_CM / 2 - 125
        with self._data_lock:
            self.pose = (x_map, y_map, yaw + np.pi)

    def _lidar_loop(self):
        while self._running:
            if self.connected:
                try:
                    with self._sim_lock:
                        self._read_lidar()
                except Exception as e:
                    print(f"[Lidar] error: {e}")
            time.sleep(0.05)

    def _read_lidar(self):
        if not self.has_lidar:
            with self._data_lock:
                x, y, th = self.pose
            sc = single_lidar(self._grid_ref, self._dt_ref,
                               x, y, th, N_LIDAR_RAYS, LIDAR_MAX_CM, PIXELS_PER_CM)
            with self._data_lock:
                self.scan = sc
            return

        try:
            h_angle   = np.deg2rad(90)
            start_ang = -h_angle / 2
            step      =  h_angle / (N_LIDAR_REAL - 1)
            raw_dist  = []
            p = start_ang
            for _ in range(N_LIDAR_REAL):
                self.sim.setJointPosition(self.lidar_joint, p)
                p  += step
                res = self.sim.handleProximitySensor(self.lidar_sensor)
                raw_dist.append(res[1] * M_TO_CM if (res[0] > 0 and res[1] > 0)
                                 else LIDAR_MAX_CM)

            raw    = np.array(raw_dist, dtype=np.float32)
            x_raw  = np.linspace(0, 1, N_LIDAR_REAL)
            x_full = np.linspace(0, 1, N_LIDAR_RAYS)
            scan   = np.clip(np.interp(x_full, x_raw, raw), 0.0, LIDAR_MAX_CM)
            with self._data_lock:
                self.scan = scan
        except Exception as e:
            print("LIDAR ERROR:", e)

    def get_pose(self):
        with self._data_lock:
            return self.pose

    def get_scan(self):
        with self._data_lock:
            return self.scan.copy()

    def stop(self):
        self._running = False


# ── Visualization ─────────────────────────────────────────────────────────────

def make_figure():
    fig    = plt.figure(figsize=(14, 7), facecolor='#0d1117')
    gs     = GridSpec(1, 2, figure=fig, wspace=0.04, width_ratios=[2, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_inf = fig.add_subplot(gs[1])
    for ax in (ax_map, ax_inf):
        ax.set_facecolor('#0d1117')
    return fig, ax_map, ax_inf


def draw_arrow(ax, x, y, theta, length=10, color='red', lw=2, zorder=6):
    ax.annotate('',
        xy=(x + length * np.cos(theta), y + length * np.sin(theta)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw),
        zorder=zorder)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)

    grid = build_map(OBSTACLES_CM, MAP_SIZE_PX, PIXELS_PER_CM)
    dt   = build_dist_transform(grid)

    CoppeliaInterface._grid_ref = grid
    CoppeliaInterface._dt_ref   = dt

    map_img = np.zeros((MAP_SIZE_PX, MAP_SIZE_PX, 3), dtype=np.uint8)
    map_img[grid > 0]  = [45,  105, 160]
    map_img[grid == 0] = [230, 230, 230]

    coppelia  = CoppeliaInterface()
    coppelia.start()

    particles = sample_free(grid, MAP_SIZE_CM, PIXELS_PER_CM, N_PARTICLES)

    state = {
        'it'       : 0,
        'prev_pose': coppelia.get_pose(),
        'trail'    : [],
        'sigma'    : SIGMA_INIT,
    }

    fig, ax_map, ax_inf = make_figure()

    def step(_frame):
        nonlocal particles

        ax_map.cla(); ax_inf.cla()
        ax_map.set_facecolor('#0d1117')
        ax_inf.set_facecolor('#0d1117')

        # Odometry delta
        x_r, y_r, th_r = coppelia.get_pose()
        px, py, pth    = state['prev_pose']
        d_trans = np.hypot(x_r - px, y_r - py)
        d_rot   = (th_r - pth + np.pi) % (2 * np.pi) - np.pi
        state['prev_pose'] = (x_r, y_r, th_r)
        state['trail'].append((x_r, y_r))

        # Motion update
        particles = motion_update(particles, d_trans, d_rot)
        particles = keep_in_bounds_vectorized(particles, grid, MAP_SIZE_CM, PIXELS_PER_CM)

        # Weight update
        obs_scan = coppelia.get_scan()
        sigma    = state['sigma']
        weights  = score_particles_batch(
            particles, obs_scan, grid, dt,
            N_LIDAR_RAYS, LIDAR_MAX_CM, PIXELS_PER_CM, sigma)

        # Resample
        neff      = convergence_score(weights)
        particles = topk_filter_and_resample(
            particles, weights, grid, MAP_SIZE_CM, PIXELS_PER_CM, TOP_K_FRACTION)
        particles = keep_in_bounds_vectorized(particles, grid, MAP_SIZE_CM, PIXELS_PER_CM)

        # Pose estimate
        est_x, est_y, est_th = weighted_mean_pose(particles, weights)
        err = np.hypot(x_r - est_x, y_r - est_y)

        # Adaptive sigma
        state['sigma'] = (max(SIGMA_MIN, sigma * SIGMA_DECAY)
                          if neff < 0.5 else
                          min(SIGMA_INIT, sigma * 1.5))

        # LIDAR rays overlay
        half     = np.deg2rad(45)
        ray_angs = np.linspace(-half, half, N_LIDAR_RAYS) + th_r
        for dist, rang in zip(obs_scan, ray_angs):
            ax_map.plot([x_r, x_r + dist * np.cos(rang)],
                        [y_r, y_r + dist * np.sin(rang)],
                        color='lime', alpha=0.25, linewidth=0.5, zorder=7)

        # Map render
        ax_map.imshow(map_img, extent=[0, MAP_SIZE_CM, 0, MAP_SIZE_CM],
                      origin='lower', interpolation='nearest')

        w_norm = weights / (weights.max() + 1e-9)
        sidx   = np.argsort(w_norm)
        ax_map.scatter(particles[sidx, 0], particles[sidx, 1],
                       c=plt.cm.plasma(w_norm[sidx]),
                       s=5, alpha=0.75, zorder=4, linewidths=0)

        ax_map.plot(est_x, est_y, '*', color='#00e5ff', markersize=13,
                    zorder=5, markeredgecolor='#0d1117', markeredgewidth=0.8)
        draw_arrow(ax_map, est_x, est_y, est_th, color='#00e5ff', lw=2, zorder=5)

        ax_map.plot(x_r, y_r, 'o', color='#ff4d4d', markersize=10,
                    zorder=6, markeredgecolor='white', markeredgewidth=1.5)
        draw_arrow(ax_map, x_r, y_r, th_r, color='#ff4d4d', lw=2, zorder=6)

        ax_map.set_xlim(0, MAP_SIZE_CM); ax_map.set_ylim(0, MAP_SIZE_CM)
        ax_map.set_xlabel('X (cm)', color='#aaa', fontsize=9)
        ax_map.set_ylabel('Y (cm)', color='#aaa', fontsize=9)
        ax_map.tick_params(colors='#666')
        ax_map.set_title(
            f'MCL — it {state["it"] + 1}  |  Error: {err:.1f} cm  |  σ={sigma:.1f} cm',
            color='white', fontsize=10, pad=8)
        for sp in ax_map.spines.values():
            sp.set_edgecolor('#333')

        # Info panel
        ax_inf.set_xlim(0, 1); ax_inf.set_ylim(0, 1); ax_inf.axis('off')

        def txt(x, y, s, **kw):
            kw.setdefault('color', 'white')
            kw.setdefault('fontfamily', 'monospace')
            ax_inf.text(x, y, s, transform=ax_inf.transAxes, **kw)

        txt(0.05, 0.97, '── MCL STATUS ──', fontsize=12, color='#00e5ff', fontweight='bold')
        txt(0.05, 0.91, f'Iteration   : {state["it"] + 1}', fontsize=10)
        txt(0.05, 0.86, f'Particles   : {N_PARTICLES}',     fontsize=10)
        txt(0.05, 0.81, f'LIDAR rays  : {N_LIDAR_RAYS}',    fontsize=10)
        cc = '#6bff9e' if coppelia.connected else '#ffd700'
        txt(0.05, 0.76,
            f'Coppelia    : {"YES ✓" if coppelia.connected else "NO (simulated)"}',
            fontsize=10, color=cc)
        txt(0.05, 0.71, f'σ current   : {sigma:.1f} cm', fontsize=10, color='#ffd700')
        txt(0.05, 0.61, f'Top-K frac  : {TOP_K_FRACTION}', fontsize=10, color='#ffd700')

        txt(0.05, 0.53, '── GROUND TRUTH ──', fontsize=11, color='#ff6b6b', fontweight='bold')
        txt(0.05, 0.47, f'x  = {x_r:7.1f} cm',             fontsize=10)
        txt(0.05, 0.42, f'y  = {y_r:7.1f} cm',             fontsize=10)
        txt(0.05, 0.37, f'θ  = {np.rad2deg(th_r):7.1f} °', fontsize=10)

        txt(0.05, 0.29, '── MCL ESTIMATE ──', fontsize=11, color='#00e5ff', fontweight='bold')
        txt(0.05, 0.23, f'x̂  = {est_x:7.1f} cm',             fontsize=10)
        txt(0.05, 0.18, f'ŷ  = {est_y:7.1f} cm',             fontsize=10)
        txt(0.05, 0.13, f'θ̂  = {np.rad2deg(est_th):7.1f} °', fontsize=10)

        ec = '#ff4d4d' if err > 20 else ('#ffd700' if err > 8 else '#6bff9e')
        txt(0.05, 0.05, f'Error : {err:.1f} cm', fontsize=13, color=ec, fontweight='bold')

        state['it'] += 1

    plt.suptitle('MCL — Differential Robot + LIDAR', color='#aaa', fontsize=9, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.998])

    FuncAnimation(fig, step, interval=30, cache_frame_data=False)
    plt.show()
    coppelia.stop()


if __name__ == '__main__':
    main()
