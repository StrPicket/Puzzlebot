"""
particle_filter.py  —  Monte Carlo Localization (MCL) para PuzzleBot
=====================================================================
Implementa un filtro de partículas estándar con:
  - Modelo de movimiento basado en odometría diferencial
  - Modelo de observación basado en ArUcos (distancia + bearing)
  - Re-muestreo sistemático con regularización (evita degeneración)

Uso:
    from particle_filter import ParticleFilter
    pf = ParticleFilter(n_particles=500, aruco_map=ARUCO_MAP)
    pf.predict(delta_rot1, delta_trans, delta_rot2)
    pf.update(observations)   # observations = [(id, dist, bearing), ...]
    x, y, theta = pf.estimate()
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

ArucoMap = Dict[int, Tuple[float, float, float]]   # {id: (x, y, yaw)}
Obs      = Tuple[int, float, float]                # (id, dist_m, bearing_rad)


# ═══════════════════════════════════════════════════════════════════════════
class ParticleFilter:
    """
    Filtro de partículas para localización 2-D.

    Estado por partícula: [x, y, theta]

    Modelo de movimiento: odometría diferencial estándar
      (Thrun, Burgard, Fox — Probabilistic Robotics, cap. 5)

    Modelo de observación: likelihood gaussiana sobre
      distancia y bearing a marcadores ArUco conocidos.
    """

    def __init__(
        self,
        n_particles: int = 500,
        aruco_map: ArucoMap = None,
        # Límites del mapa (m)
        x_min: float = 0.0, x_max: float = 4.8,
        y_min: float = 0.0, y_max: float = 3.7,
        # Ruido de movimiento (odometría diferencial)
        alpha1: float = 0.05,   # ruido rot  ← rot
        alpha2: float = 0.05,   # ruido rot  ← trans
        alpha3: float = 0.02,   # ruido trans ← trans
        alpha4: float = 0.02,   # ruido trans ← rot
        # Ruido del modelo de observación
        sigma_dist: float = 0.3,    # m
        sigma_bearing: float = 0.25, # rad
        # Dispersión inicial
        init_sigma_xy: float = 0.4,
        init_sigma_theta: float = math.pi / 4,
        # Fracción de partículas aleatorias inyectadas tras resample
        rand_frac: float = 0.05,
    ):
        self.N = n_particles
        self.aruco_map = aruco_map or {}
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.a1, self.a2 = alpha1, alpha2
        self.a3, self.a4 = alpha3, alpha4

        self.s_dist    = sigma_dist
        self.s_bearing = sigma_bearing

        self.init_sig_xy    = init_sigma_xy
        self.init_sig_theta = init_sigma_theta
        self.rand_frac      = rand_frac

        self.particles = None
        self.weights   = np.ones(n_particles) / n_particles
        self._estimate = np.zeros(3)

        # Odometría previa para calcular deltas
        self._prev_odom = None   # (x, y, theta)

        self.initialized = False

    # ── Inicialización ──────────────────────────────────────────────────────

    def init_at(self, x: float, y: float, theta: float):
        """Dispersa las partículas alrededor de una pose conocida."""
        self.particles = np.column_stack([
            np.random.normal(x,     self.init_sig_xy,    self.N),
            np.random.normal(y,     self.init_sig_xy,    self.N),
            np.random.normal(theta, self.init_sig_theta, self.N),
        ])
        self.particles[:, 2] = _wrap_arr(self.particles[:, 2])
        self._clip_particles()
        self.weights   = np.ones(self.N) / self.N
        self._estimate = np.array([x, y, theta])
        self.initialized = True

    def init_uniform(self):
        """Distribución uniforme en todo el mapa (localización global)."""
        self.particles = np.column_stack([
            np.random.uniform(self.x_min, self.x_max, self.N),
            np.random.uniform(self.y_min, self.y_max, self.N),
            np.random.uniform(-math.pi, math.pi, self.N),
        ])
        self.weights   = np.ones(self.N) / self.N
        self._estimate = np.array([
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            0.0,
        ])
        self.initialized = True

    # ── Modelo de movimiento ────────────────────────────────────────────────

    def predict_from_odom(self, x: float, y: float, theta: float):
        """
        Actualiza las partículas usando la pose odométrica absoluta.
        Calcula internamente (rot1, trans, rot2) respecto al paso anterior.
        """
        if not self.initialized:
            return
        if self._prev_odom is None:
            self._prev_odom = (x, y, theta)
            return

        px, py, pth = self._prev_odom
        self._prev_odom = (x, y, theta)

        dx    = x - px
        dy    = y - py
        dtheta = _wrap(theta - pth)
        delta_trans = math.sqrt(dx**2 + dy**2)

        if delta_trans < 1e-5:
            delta_rot1 = 0.0
        else:
            delta_rot1 = _wrap(math.atan2(dy, dx) - pth)

        delta_rot2 = _wrap(dtheta - delta_rot1)

        self._apply_motion(delta_rot1, delta_trans, delta_rot2)
        self._update_estimate() 

    def _apply_motion(self, rot1: float, trans: float, rot2: float):
        """Aplica ruido gaussiano al modelo de movimiento diferencial."""
        n = self.N
        eps = 1e-9

        std_r1 = math.sqrt(self.a1 * abs(rot1)  + self.a2 * abs(trans) + eps)
        std_tr = math.sqrt(self.a3 * abs(trans)  + self.a4 * (abs(rot1) + abs(rot2)) + eps)
        std_r2 = math.sqrt(self.a1 * abs(rot2)   + self.a2 * abs(trans) + eps)

        nr1 = np.random.normal(0, std_r1, n)
        ntr = np.random.normal(0, std_tr, n)
        nr2 = np.random.normal(0, std_r2, n)

        r1 = rot1  - nr1
        tr = trans - ntr
        r2 = rot2  - nr2

        th = self.particles[:, 2]
        self.particles[:, 0] += tr * np.cos(th + r1)
        self.particles[:, 1] += tr * np.sin(th + r1)
        self.particles[:, 2]  = _wrap_arr(th + r1 + r2)
        self._clip_particles()

    # ── Modelo de observación ───────────────────────────────────────────────

    def update(self, observations: List[Obs]):
        """
        Pondera partículas según las observaciones ArUco.

        observations: lista de (marker_id, dist_m, bearing_rad)
            dist_m   — distancia horizontal al marcador (m)
            bearing  — ángulo al marcador relativo al frente del robot (rad)
        """
        if not self.initialized or len(observations) == 0:
            return

        log_w = np.zeros(self.N)

        for mid, obs_dist, obs_bearing in observations:
            if mid not in self.aruco_map:
                continue

            mx, my, _ = self.aruco_map[mid]

            dx = mx - self.particles[:, 0]
            dy = my - self.particles[:, 1]

            exp_dist    = np.sqrt(dx**2 + dy**2)
            exp_bearing = _wrap_arr(np.arctan2(dy, dx) - self.particles[:, 2])

            err_d = obs_dist - exp_dist
            err_b = _wrap_arr(obs_bearing - exp_bearing)

            log_w += -0.5 * (err_d / self.s_dist)    ** 2
            log_w += -0.5 * (err_b / self.s_bearing)  ** 2

        # Normalizar (estabilidad numérica con log-sum-exp)
        log_w -= log_w.max()
        w = np.exp(log_w)
        total = w.sum()

        if total < 1e-30:
            # Colapso total — reinicializar
            self.init_uniform()
            return

        self.weights = w / total
        self._resample()
        self._update_estimate()

    # ── Resample ────────────────────────────────────────────────────────────

    def _resample(self):
        """Low-variance resampling + inyección de partículas aleatorias."""
        n_rand = max(1, int(self.N * self.rand_frac))
        n_keep = self.N - n_rand

        indices  = _systematic_resample(self.weights, n_keep)
        kept     = self.particles[indices].copy()

        rand_p = np.column_stack([
            np.random.uniform(self.x_min, self.x_max, n_rand),
            np.random.uniform(self.y_min, self.y_max, n_rand),
            np.random.uniform(-math.pi, math.pi, n_rand),
        ])

        self.particles = np.vstack([kept, rand_p])
        self.weights   = np.ones(self.N) / self.N

    # ── Estimación ──────────────────────────────────────────────────────────

    def _update_estimate(self):
        """Promedio ponderado; ángulo en círculo unitario."""
        wx  = np.sum(self.weights * self.particles[:, 0])
        wy  = np.sum(self.weights * self.particles[:, 1])
        s   = np.sum(self.weights * np.sin(self.particles[:, 2]))
        c   = np.sum(self.weights * np.cos(self.particles[:, 2]))
        wth = math.atan2(s, c)
        self._estimate = np.array([wx, wy, wth])

    def estimate(self) -> Tuple[float, float, float]:
        """Devuelve (x, y, theta) — estimación actual."""
        return float(self._estimate[0]), float(self._estimate[1]), float(self._estimate[2])

    @property
    def neff(self) -> float:
        """Número efectivo de partículas (1/N = degenerado, N = ideal)."""
        return float(1.0 / (np.sum(self.weights ** 2) + 1e-30))

    # ── Utilidades ──────────────────────────────────────────────────────────

    def _clip_particles(self):
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.x_min, self.x_max)
        self.particles[:, 1] = np.clip(self.particles[:, 1], self.y_min, self.y_max)

    def reset_odom(self, x: float, y: float, theta: float):
        """Reinicia la referencia de odometría."""
        self._prev_odom = (x, y, theta)


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    """Systematic resampling (low-variance). Selecciona n índices."""
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    step = 1.0 / n
    r    = np.random.uniform(0, step)
    pos  = r + step * np.arange(n)
    return np.clip(np.searchsorted(cumsum, pos), 0, len(weights) - 1)
