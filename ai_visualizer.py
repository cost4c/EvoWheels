"""
EvoWheels - Neural Inspector

Janela separada, somente leitura, que mostra o que o cerebro do carro
selecionado esta percebendo e decidindo agora.

Regras deste modulo:

    * nunca recalcula sensores
    * nunca chama Brain.pensar()
    * apenas le telemetria que o carro e o cerebro ja produziram

A simulacao tem prioridade. O inspector desenha cerca de 9 vezes por
segundo e mantem em cache tudo que nao muda entre as atualizacoes.
"""

import math
import os
import traceback

import numpy as np
import pygame

# Deixa a GPU suavizar o upscale quando a janela e maior que a resolucao
# interna de render. Precisa estar definido antes de criar o Renderer.
# Nao afeta a janela principal da simulacao.
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")

from pygame._sdl2 import Window, Renderer, Texture


# -------------------------------------------------------------------------
# TEXTOS DA INTERFACE
# -------------------------------------------------------------------------

# Tudo em um lugar so, para facilitar traducao.
L = {
    "title": "NEURAL INSPECTOR",
    "subtitle": "EvoWheels  -  leitura passiva da IA",

    "chip_track": "PISTA",
    "chip_car": "CARRO",
    "chip_mode": "MODO",
    "chip_fps": "FPS SIM",
    "chip_decisions": "DECISOES",

    "mode_best": "MELHOR VIVO",
    "mode_manual": "MANUAL",

    "perception": "PERCEPCAO",
    "perception_hint": "7 sensores frontais  -  alcance 100 px",
    "highest_risk": "MAIOR RISCO",

    "telemetry": "TELEMETRIA",
    "speed": "VELOCIDADE",

    "network": "REDE NEURAL",
    "decision": "DECISAO ATUAL",
    "neural_output": "SAIDA DA REDE",
    "applied": "COMANDO APLICADO",

    "meta": "REDE  /  EVOLUCAO",

    "footer": (
        "V  abrir e fechar inspector      "
        "TAB  proximo carro      "
        "B  seguir melhor vivo      "
        "F11  maximizar e restaurar"
    ),

    "legend_pos": "ciano = influencia positiva",
    "legend_neg": "laranja = influencia negativa",

    "waiting": "aguardando a primeira decisao do cerebro",
}

# Faixas de risco usadas no aviso de maior risco.
RISK_LEVELS = (
    (0.25, "LIVRE"),
    (0.50, "BAIXO"),
    (0.75, "ATENCAO"),
    (0.90, "PERIGO ALTO"),
    (2.00, "CRITICO"),
)

# Mesma ordem dos sensores do carro.
SENSOR_ANGLES = (-70, -45, -22, 0, 22, 45, 70)

SENSOR_SHORT = ("-70", "-45", "-22", "0", "+22", "+45", "+70")

SENSOR_NAMES = (
    "ESQUERDA  -70",
    "DIAGONAL ESQ.  -45",
    "FRENTE ESQ.  -22",
    "FRENTE  0",
    "FRENTE DIR.  +22",
    "DIAGONAL DIR.  +45",
    "DIREITA  +70",
)

# Direcao de cada sensor no radar, com o carro apontando para cima.
SENSOR_UNITS = tuple(
    (
        math.sin(math.radians(angle)),
        -math.cos(math.radians(angle)),
    )
    for angle in SENSOR_ANGLES
)

# Rotulos das 10 entradas, na ordem que o carro monta.
INPUT_LABELS = (
    "-70", "-45", "-22", "FRENTE", "+22", "+45", "+70",
    "VEL", "DIR", "TRAC",
)

OUTPUT_LABELS = ("ACELERAR", "FREAR", "VIRAR")

# Alcance real dos sensores do carro, usado apenas para desenhar.
SENSOR_RANGE = 100.0
SENSOR_START = 10.0


# -------------------------------------------------------------------------
# PALETA
# -------------------------------------------------------------------------

class Theme:

    BG_TOP = (9, 13, 20)
    BG_BOTTOM = (13, 19, 32)

    CARD = (21, 28, 39)
    CARD_SOFT = (25, 34, 49)

    BORDER = (41, 53, 72)
    BORDER_HI = (52, 68, 90)

    TEXT = (244, 247, 250)
    TEXT_DIM = (154, 168, 184)
    TEXT_FAINT = (104, 118, 136)

    CYAN = (38, 217, 240)
    BLUE = (52, 138, 247)
    GREEN = (61, 220, 151)
    YELLOW = (245, 196, 81)
    ORANGE = (255, 147, 77)
    RED = (255, 93, 103)

    # Neuronio em repouso
    IDLE = (44, 57, 76)

    TRACK_BG = (16, 22, 32)

    RISK_STOPS = (
        (0.00, GREEN),
        (0.50, YELLOW),
        (0.75, ORANGE),
        (1.00, RED),
    )

    @staticmethod
    def mix(color_a, color_b, amount):
        if amount <= 0.0:
            return color_a

        if amount >= 1.0:
            return color_b

        return (
            int(color_a[0] + (color_b[0] - color_a[0]) * amount),
            int(color_a[1] + (color_b[1] - color_a[1]) * amount),
            int(color_a[2] + (color_b[2] - color_a[2]) * amount),
        )

    @classmethod
    def over(cls, color, alpha, background=None):
        """Mistura manual, em vez de alpha de verdade.

        Linhas opacas ja misturadas com o fundo do card custam muito
        menos que criar uma superficie SRCALPHA inteira por frame.
        """
        if background is None:
            background = cls.CARD

        return cls.mix(background, color, alpha)

    @classmethod
    def risk(cls, value):
        value = 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))

        stops = cls.RISK_STOPS

        for index in range(len(stops) - 1):
            low, low_color = stops[index]
            high, high_color = stops[index + 1]

            if value <= high:
                span = high - low

                amount = 0.0 if span <= 0.0 else (value - low) / span

                return cls.mix(low_color, high_color, amount)

        return stops[-1][1]

    @classmethod
    def activation(cls, value):
        value = float(value)

        if value != value:
            return cls.IDLE

        value = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)

        intensity = abs(value) ** 0.72

        if value >= 0.0:
            return cls.mix(cls.IDLE, cls.CYAN, intensity)

        return cls.mix(cls.IDLE, cls.ORANGE, intensity)

    @staticmethod
    def risk_level(value):
        for limit, name in RISK_LEVELS:
            if value < limit:
                return name

        return RISK_LEVELS[-1][1]


# -------------------------------------------------------------------------
# FONTES E TEXTO EM CACHE
# -------------------------------------------------------------------------

class TextCache:
    """Cria as fontes uma vez e guarda os textos ja renderizados.

    Renderizar fonte e caro e os valores repetem muito ("0.00", "#37",
    "LIVRE"), entao o cache acerta quase sempre.
    """

    LIMIT = 1400

    def __init__(self, scale):
        self.fonts = {}
        self.cache = {}

        self.rebuild(scale)

    def rebuild(self, scale):
        def size(base):
            return max(9, int(round(base * scale)))

        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", size(23), bold=True),
            "card": pygame.font.SysFont("Segoe UI", size(13), bold=True),
            "body": pygame.font.SysFont("Segoe UI", size(13)),
            "body_bold": pygame.font.SysFont("Segoe UI", size(13), bold=True),
            "small": pygame.font.SysFont("Segoe UI", size(11)),
            "tiny": pygame.font.SysFont("Segoe UI", size(10), bold=True),
            "big": pygame.font.SysFont("Segoe UI", size(19), bold=True),
            "huge": pygame.font.SysFont("Segoe UI", size(34), bold=True),
            "mono": pygame.font.SysFont("Consolas", size(12)),
            "mono_bold": pygame.font.SysFont("Consolas", size(13), bold=True),
            "mono_big": pygame.font.SysFont("Consolas", size(16), bold=True),
        }

        self.cache = {}

    def render(self, font_key, text, color):
        key = (font_key, text, color)

        surface = self.cache.get(key)

        if surface is None:
            if len(self.cache) >= self.LIMIT:
                self.cache.clear()

            surface = self.fonts[font_key].render(text, True, color)

            self.cache[key] = surface

        return surface

    def blit(self, target, font_key, text, color, position, align="left"):
        surface = self.render(font_key, text, color)

        if align == "left":
            target.blit(surface, position)
            return surface

        rect = surface.get_rect()

        if align == "right":
            rect.topright = position
        elif align == "center":
            rect.midtop = position
        elif align == "midleft":
            rect.midleft = position
        elif align == "midright":
            rect.midright = position
        else:
            rect.center = position

        target.blit(surface, rect)

        return surface

    def height(self, font_key):
        return self.fonts[font_key].get_height()

    def width(self, font_key, text):
        return self.fonts[font_key].size(text)[0]


# -------------------------------------------------------------------------
# LAYOUT
# -------------------------------------------------------------------------

class Layout:
    """Todos os retangulos da interface, derivados do tamanho atual.

    Recalculado somente quando a janela muda de tamanho.
    """

    def __init__(self, width, height, text):
        self.width = width
        self.height = height

        margin = max(14, int(width * 0.012))
        gap = max(12, int(width * 0.010))

        header_height = max(68, int(height * 0.085))
        footer_height = max(22, text.height("small") + 8)

        self.header = pygame.Rect(
            margin, margin, width - margin * 2, header_height
        )

        body_top = self.header.bottom + gap
        body_height = height - body_top - footer_height - margin

        self.footer = pygame.Rect(
            margin,
            height - footer_height - int(margin * 0.4),
            width - margin * 2,
            footer_height,
        )

        left_width = int(min(380, max(268, width * 0.215)))
        right_width = int(min(360, max(252, width * 0.200)))

        center_width = width - margin * 2 - left_width - right_width - gap * 2

        # Em janelas estreitas o centro tem prioridade.
        if center_width < 420:
            missing = 420 - center_width

            shrink_left = max(0, min(missing // 2, left_width - 230))
            shrink_right = max(0, min(missing - shrink_left, right_width - 215))

            left_width -= shrink_left
            right_width -= shrink_right

            center_width = (
                width - margin * 2 - left_width - right_width - gap * 2
            )

        center_width = max(240, center_width)

        # ---- coluna esquerda
        perception_height = int(body_height * 0.60)

        self.perception = pygame.Rect(
            margin, body_top, left_width, perception_height
        )

        self.telemetry = pygame.Rect(
            margin,
            self.perception.bottom + gap,
            left_width,
            body_height - perception_height - gap,
        )

        # ---- centro
        self.network = pygame.Rect(
            self.perception.right + gap, body_top, center_width, body_height
        )

        # ---- coluna direita
        decision_height = int(body_height * 0.52)

        self.decision = pygame.Rect(
            self.network.right + gap, body_top, right_width, decision_height
        )

        self.meta = pygame.Rect(
            self.network.right + gap,
            self.decision.bottom + gap,
            right_width,
            body_height - decision_height - gap,
        )

        self.card_title_height = text.height("card") + 22
        self.pad = max(12, int(left_width * 0.052))

        self._build_perception(text)
        self._build_network()

    # ---------------------------------------------------------------------

    def inner(self, rect):
        """Area util de um card, abaixo do titulo."""

        return pygame.Rect(
            rect.x + self.pad,
            rect.y + self.card_title_height,
            rect.width - self.pad * 2,
            rect.height - self.card_title_height - self.pad,
        )

    def _build_perception(self, text):
        area = self.inner(self.perception)

        gap = 10

        callout_height = max(
            46, text.height("big") + text.height("tiny") + 18
        )

        row_height = max(16, min(27, int(area.height * 0.38 / 7)))

        meters_height = row_height * 7

        radar_height = area.height - callout_height - meters_height - gap * 2

        # Em janelas baixas as reguas cedem espaco antes do radar.
        if radar_height < 92:
            room = (row_height - 16) * 7

            give = min(92 - radar_height, room)

            row_height = max(16, row_height - int(math.ceil(give / 7.0)))

            meters_height = row_height * 7

            radar_height = (
                area.height - callout_height - meters_height - gap * 2
            )

        radar_height = max(60, radar_height)

        self.radar = pygame.Rect(area.x, area.y, area.width, radar_height)

        self.meters = pygame.Rect(
            area.x, self.radar.bottom + gap, area.width, meters_height
        )

        self.meter_row_height = row_height

        self.callout = pygame.Rect(
            area.x,
            self.meters.bottom + gap,
            area.width,
            max(40, area.bottom - self.meters.bottom - gap),
        )

        # Geometria do radar, fixa enquanto o tamanho nao mudar.
        origin_x = self.radar.centerx
        origin_y = self.radar.bottom - 16

        radius = min(
            self.radar.height - 26,
            int((self.radar.width * 0.5 - 10) / 0.94),
        )

        self.radar_origin = (origin_x, origin_y)
        self.radar_radius = max(34, radius)

    def _build_network(self):
        area = self.inner(self.network)

        label_left = 68
        label_right = 96

        self.network_area = area

        self.network_plot = pygame.Rect(
            area.x + label_left,
            area.y + 34,
            max(120, area.width - label_left - label_right),
            max(120, area.height - 42),
        )


# -------------------------------------------------------------------------
# GEOMETRIA DA REDE
# -------------------------------------------------------------------------

class NetworkGeometry:
    """Posicoes dos neuronios e pontas de cada conexao.

    As pontas ficam em duas listas planas, indexadas pelo mesmo indice
    global do vetor de pesos achatado. Assim o render nao faz nenhuma
    conta de indice por frame.
    """

    def __init__(self, architecture, plot_rect):
        self.architecture = tuple(architecture)
        self.rect = plot_rect

        self.positions = []
        self.radii = []
        self.columns = []

        self.point_a = []
        self.point_b = []

        self.offsets = [0]

        self._build_neurons()
        self._build_edges()

    # ---------------------------------------------------------------------

    @staticmethod
    def radius_for(count):
        if count <= 4:
            return 9
        if count <= 12:
            return 7
        if count <= 20:
            return 5
        if count <= 40:
            return 4

        return 3

    @staticmethod
    def max_spacing_for(count):
        if count <= 4:
            return 90
        if count <= 12:
            return 56

        return 46

    def _build_neurons(self):
        rect = self.rect

        layers = len(self.architecture)

        middle = rect.centery

        for index, count in enumerate(self.architecture):
            if layers == 1:
                x = rect.centerx
            else:
                x = rect.x + rect.width * index / (layers - 1)

            x = int(round(x))

            self.columns.append(x)
            self.radii.append(self.radius_for(count))

            if count <= 1:
                self.positions.append([(x, middle)])
                continue

            # Camadas grandes usam todo o espaco. Camadas pequenas
            # recebem um espacamento maximo para nao ficarem esticadas.
            spacing = min(
                rect.height / (count - 1), self.max_spacing_for(count)
            )

            top = middle - spacing * (count - 1) * 0.5

            self.positions.append(
                [
                    (x, int(round(top + spacing * neuron)))
                    for neuron in range(count)
                ]
            )

    def _build_edges(self):
        for index in range(len(self.architecture) - 1):
            source = self.positions[index]
            target = self.positions[index + 1]

            # Mesma ordem do achatamento C de uma matriz (saidas, entradas).
            for destination in target:
                for origin in source:
                    self.point_a.append(origin)
                    self.point_b.append(destination)

            self.offsets.append(len(self.point_a))

        self.total = len(self.point_a)


# -------------------------------------------------------------------------
# INFLUENCIA DAS CONEXOES
# -------------------------------------------------------------------------

class InfluenceProbe:
    """Acha as conexoes mais influentes da decisao atual.

    Influencia aproximada = ativacao da origem * peso.

    Tudo em NumPy, com buffers reaproveitados, para nao montar listas
    gigantes do Python a cada atualizacao.
    """

    def __init__(self, shapes):
        self.shapes = [tuple(shape) for shape in shapes]

        self.offsets = [0]

        for rows, columns in self.shapes:
            self.offsets.append(self.offsets[-1] + rows * columns)

        self.total = self.offsets[-1]

        self.values = np.empty(self.total, dtype=np.float32)
        self.magnitude = np.empty(self.total, dtype=np.float32)

        self.views = [
            self.values[
                self.offsets[index]:self.offsets[index + 1]
            ].reshape(shape)
            for index, shape in enumerate(self.shapes)
        ]

    def matches(self, shapes):
        return self.shapes == [tuple(shape) for shape in shapes]

    def top(self, weights, activations, count):
        if self.total <= 0 or len(activations) < len(weights):
            return None

        for index, matrix in enumerate(weights):
            source = activations[index]

            if source.shape[0] != matrix.shape[1]:
                return None

            # Broadcast por linha: contribuicao[o, i] = peso[o, i] * ativacao[i]
            np.multiply(matrix, source, out=self.views[index])

        np.abs(self.values, out=self.magnitude)

        count = max(1, min(int(count), self.total))

        if count >= self.total:
            order = np.argsort(self.magnitude)[::-1]
        else:
            cut = np.argpartition(self.magnitude, self.total - count)
            cut = cut[self.total - count:]

            order = cut[np.argsort(self.magnitude[cut])[::-1]]

        return (
            order.tolist(),
            self.values[order].tolist(),
            float(self.magnitude[order[0]]),
        )


# -------------------------------------------------------------------------
# JANELA
# -------------------------------------------------------------------------

class InspectorWindow:
    """Cuida da janela SDL2, do renderer e da textura reaproveitada."""

    # Resolucao interna maxima. Janelas maiores desenham um pouco
    # menores e a GPU amplia, para o upload da textura nao explodir em
    # monitores grandes.
    MAX_WIDTH = 2048
    MAX_HEIGHT = 1152

    MIN_WIDTH = 1040
    MIN_HEIGHT = 660

    def __init__(self, title, size):
        self.window = Window(title, size=size, resizable=True)

        self.renderer = Renderer(self.window)

        self.texture = None
        self.texture_size = None

        self.maximized = False

    # ---------------------------------------------------------------------

    def owns(self, event):
        target = getattr(event, "window", None)

        if target is None or self.window is None:
            return False

        if target is self.window:
            return True

        try:
            return int(getattr(target, "id", -1)) == int(self.window.id)
        except Exception:
            return False

    def size(self):
        if self.window is None:
            return None

        try:
            width, height = self.window.size
        except Exception:
            return None

        return (
            max(self.MIN_WIDTH, int(width)),
            max(self.MIN_HEIGHT, int(height)),
        )

    def render_size(self, width, height):
        factor = min(
            1.0,
            self.MAX_WIDTH / float(width),
            self.MAX_HEIGHT / float(height),
        )

        return (
            max(self.MIN_WIDTH, int(width * factor)),
            max(self.MIN_HEIGHT, int(height * factor)),
        )

    def maximize(self):
        try:
            self.window.maximize()
            self.maximized = True
        except Exception:
            pass

    def toggle_maximize(self):
        try:
            if self.maximized:
                self.window.restore()
                self.maximized = False
            else:
                self.window.maximize()
                self.maximized = True
        except Exception:
            pass

    def present(self, surface):
        size = surface.get_size()

        if self.texture is None or self.texture_size != size:
            self.texture = Texture(self.renderer, size)
            self.texture_size = size

        # Reaproveitar a textura evita alocar memoria de GPU a cada
        # atualizacao, que era o caminho antigo (Texture.from_surface).
        self.texture.update(surface)

        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    def destroy(self):
        self.texture = None
        self.renderer = None

        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass

        self.window = None


# -------------------------------------------------------------------------
# INSPECTOR
# -------------------------------------------------------------------------

class AIVisualizer:

    # Telemetria, nao jogo. Nove atualizacoes por segundo bastam.
    INSPECTOR_FPS = 9
    REFRESH_INTERVAL = 1.0 / INSPECTOR_FPS

    # Quantas conexoes acendem de verdade.
    ACTIVE_CONNECTIONS = 80

    # Conexoes de fundo desenhadas por quadro livre enquanto a malha
    # esta sendo reconstruida. Mantem o pico em torno de 3.5 ms.
    WIRE_CHUNK = 700

    # Quantos tons diferentes as conexoes de fundo podem ter.
    WIRE_BUCKETS = 5

    BEST_INTERVAL = 0.75

    FLASH_DECAY = 3.2

    DEFAULT_SIZE = (1620, 960)

    def __init__(self, evolution=None):
        self.evolution = evolution

        self.visible = False

        self.display = None

        self.width, self.height = self.DEFAULT_SIZE

        self.selected_index = 0
        self.follow_best = True

        self.best_timer = 0.0
        self.refresh_timer = 0.0
        self.poll_timer = 0.0

        # Desenho e upload da textura ficam em quadros separados.
        self.pending_present = False

        self.flash = 0.0
        self.last_counter = None

        self.text = TextCache(1.0)

        self.layout = None
        self.geometry = None
        self.probe = None

        self.chrome = None
        self.static = None
        self.frame = None
        self.radar_overlay = None

        # Guardamos a referencia forte do cerebro em cache. Comparar
        # somente id() seria furado, porque o Python reaproveita o id
        # de um objeto ja coletado.
        self.cached_brain = None
        self.cached_architecture = None

        self.wire_colors = None
        self.wire_cursor = 0
        self.wire_palette = []

        self.pending_size = None
        self.resize_timer = 0.0

        self.needs_chrome = True

        self.reported_error = False

    # =====================================================================
    # JANELA
    # =====================================================================

    def open(self):
        if self.visible:
            return

        try:
            self.display = InspectorWindow(
                "EvoWheels  -  Neural Inspector", self.DEFAULT_SIZE
            )
        except Exception:
            traceback.print_exc()
            self.display = None
            return

        self.visible = True

        self.refresh_timer = 0.0
        self.poll_timer = 0.0
        self.pending_present = False

        self.display.maximize()

        self.pending_size = None
        self.resize_timer = 0.0

        self.apply_size(force=True)

    def close(self):
        if self.display is not None:
            self.display.destroy()

        self.display = None
        self.visible = False

        self.pending_present = False

        # Libera as superficies grandes enquanto estiver fechado.
        self.chrome = None
        self.static = None
        self.frame = None
        self.radar_overlay = None

        self.cached_brain = None
        self.cached_architecture = None

        self.wire_colors = None
        self.wire_cursor = 0

        self.needs_chrome = True

    def toggle(self):
        if self.visible:
            self.close()
        else:
            self.open()

    # ---------------------------------------------------------------------

    def request_resize(self):
        if not self.visible or self.display is None:
            return

        size = self.display.size()

        if size is None:
            return

        if size == (self.width, self.height):
            self.pending_size = None
            return

        # Arrastar a borda gera muitos eventos seguidos. Esperamos o
        # tamanho parar de mudar antes de reconstruir os caches.
        self.pending_size = size
        self.resize_timer = 0.16

    def apply_size(self, force=False):
        if self.display is None:
            return

        size = self.pending_size or self.display.size() or self.DEFAULT_SIZE

        self.pending_size = None

        if not force and size == (self.width, self.height):
            return

        self.width, self.height = size

        render_width, render_height = self.display.render_size(
            self.width, self.height
        )

        self.frame = pygame.Surface((render_width, render_height))

        # A superficie trocou: nao faz sentido subir o que havia nela.
        self.pending_present = False

        scale = min(render_width / 1620.0, render_height / 960.0)
        scale = max(0.82, min(1.30, scale))

        self.text.rebuild(scale)

        self.layout = Layout(render_width, render_height, self.text)

        self.radar_overlay = pygame.Surface(
            self.layout.radar.size, pygame.SRCALPHA
        )

        self.geometry = None
        self.needs_chrome = True

    # =====================================================================
    # EVENTOS
    # =====================================================================

    def handle_event(self, event, cars):
        """Devolve True quando o evento foi consumido pelo inspector."""

        kind = event.type

        if kind == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                self.toggle()
                return True

            if not self.visible or self.display is None:
                return False

            if event.key == pygame.K_TAB:
                self.select_next(cars)
                return True

            if event.key == pygame.K_b:
                self.follow_best = not self.follow_best
                self.best_timer = 0.0
                return True

            if event.key == pygame.K_F11:
                # F11 vale para a janela que esta em foco.
                if not self.display_has_focus():
                    self.display.toggle_maximize()
                    self.request_resize()
                    return True

            return False

        if not self.visible or self.display is None:
            return False

        if kind == pygame.WINDOWCLOSE:
            # Fechar o inspector nao pode encerrar a simulacao.
            if self.display.owns(event):
                self.close()
                return True

            return False

        if kind in (
            pygame.WINDOWSIZECHANGED,
            pygame.WINDOWRESIZED,
            pygame.WINDOWMAXIMIZED,
            pygame.WINDOWRESTORED,
        ):
            if self.display.owns(event):
                if kind == pygame.WINDOWMAXIMIZED:
                    self.display.maximized = True
                elif kind == pygame.WINDOWRESTORED:
                    self.display.maximized = False

                self.request_resize()
                return True

        return False

    @staticmethod
    def display_has_focus():
        try:
            return bool(pygame.key.get_focused())
        except Exception:
            return False

    # =====================================================================
    # SELECAO DE CARRO
    # =====================================================================

    def select_next(self, cars):
        if not cars:
            return

        self.follow_best = False
        self.selected_index = (self.selected_index + 1) % len(cars)

    def selected_car(self, cars):
        if not cars:
            return None

        self.selected_index %= len(cars)

        return cars[self.selected_index]

    def update_best(self, cars, dt):
        if not self.follow_best or not cars:
            return

        self.best_timer -= dt

        if self.best_timer > 0.0:
            return

        # Nao trocar de carro toda hora, so a cada 0.75 s.
        self.best_timer = self.BEST_INTERVAL

        best_index = 0
        best_score = -1.0

        for index, car in enumerate(cars):
            score = car.get_evolution_score()

            if score > best_score:
                best_score = score
                best_index = index

        if best_index == self.selected_index:
            return

        current = cars[self.selected_index].get_evolution_score()

        # Margem pequena para nao pular entre carros empatados.
        if best_score > current + 1.0:
            self.selected_index = best_index

    # =====================================================================
    # CACHE
    # =====================================================================

    def ensure_cache(self, brain):
        """Reconstroi apenas o que ficou invalido.

        Chrome (fundo, cards, rotulos) depende do tamanho da janela.
        Conexoes de fundo dependem do cerebro selecionado.
        """

        architecture = tuple(brain.camadas)

        if (
            self.needs_chrome
            or self.chrome is None
            or self.cached_architecture != architecture
        ):
            self.build_chrome(architecture)

            # Forca o rebake das conexoes do cerebro atual.
            self.cached_brain = None

        if self.cached_brain is not brain:
            self.cached_brain = brain

            self.last_counter = None
            self.flash = 1.0

            self.build_wires(brain)

    # ---------------------------------------------------------------------

    def build_chrome(self, architecture):
        if self.layout is None or self.frame is None:
            return

        self.geometry = NetworkGeometry(architecture, self.layout.network_plot)

        shapes = [
            (architecture[index + 1], architecture[index])
            for index in range(len(architecture) - 1)
        ]

        if self.probe is None or not self.probe.matches(shapes):
            self.probe = InfluenceProbe(shapes)

        self.wire_palette = self.build_wire_palette()

        surface = pygame.Surface(self.frame.get_size())

        self.paint_background(surface)
        self.paint_header_frame(surface)
        self.paint_cards(surface)
        self.paint_radar_frame(surface)
        self.paint_network_frame(surface)
        self.paint_footer(surface)

        self.chrome = surface
        self.static = surface.copy()

        self.cached_architecture = architecture

        self.needs_chrome = False

        self.wire_colors = None
        self.wire_cursor = 0

    def build_wire_palette(self):
        palette = []

        for sign_index in range(2):
            base = Theme.BLUE if sign_index == 0 else Theme.RED

            for bucket in range(self.WIRE_BUCKETS):
                strength = (bucket + 1) / float(self.WIRE_BUCKETS)

                # Bem discreto: so sugere a malha, nao compete com as
                # conexoes acesas.
                palette.append(
                    Theme.over(base, 0.055 + 0.125 * strength, Theme.CARD)
                )

        return palette

    def build_wires(self, brain):
        """Pre-calcula a cor de cada conexao de fundo.

        As cores sao quantizadas em poucos tons, calculados de uma vez
        com NumPy, e so depois viram lista do Python.
        """

        if self.geometry is None or self.geometry.total <= 0:
            self.wire_colors = None
            self.wire_cursor = 0
            return

        flat = np.concatenate([matrix.ravel() for matrix in brain.pesos])

        if flat.shape[0] != self.geometry.total:
            self.wire_colors = None
            self.wire_cursor = 0
            return

        magnitude = np.abs(flat)

        peak = float(magnitude.max())

        normalized = (
            magnitude / peak if peak > 1e-9 else np.zeros_like(magnitude)
        )

        bucket = np.clip(
            (normalized * self.WIRE_BUCKETS).astype(np.int32),
            0,
            self.WIRE_BUCKETS - 1,
        )

        bucket += (flat < 0).astype(np.int32) * self.WIRE_BUCKETS

        palette = self.wire_palette

        self.wire_colors = [palette[index] for index in bucket.tolist()]

        # Volta ao chrome limpo: nenhuma conexao do cerebro antigo fica.
        self.static.blit(self.chrome, (0, 0))

        self.wire_cursor = 0

    def bake_wires(self):
        """Desenha as conexoes de fundo em pedacos.

        A malha pode ter milhares de linhas. Desenhar em pedacos evita
        um pico grande no mesmo frame e deixa a simulacao com prioridade.
        """

        if self.wire_colors is None:
            return

        total = len(self.wire_colors)

        if self.wire_cursor >= total:
            return

        end = min(total, self.wire_cursor + self.WIRE_CHUNK)

        surface = self.static
        line = pygame.draw.line

        points_a = self.geometry.point_a
        points_b = self.geometry.point_b
        colors = self.wire_colors

        for index in range(self.wire_cursor, end):
            line(surface, colors[index], points_a[index], points_b[index], 1)

        self.wire_cursor = end

    # =====================================================================
    # PINTURA ESTATICA
    # =====================================================================

    def paint_background(self, surface):
        strip = pygame.Surface((1, 2))

        strip.set_at((0, 0), Theme.BG_TOP)
        strip.set_at((0, 1), Theme.BG_BOTTOM)

        surface.blit(
            pygame.transform.smoothscale(strip, surface.get_size()), (0, 0)
        )

    def card(self, surface, rect, title, hint=None):
        pygame.draw.rect(surface, Theme.CARD, rect, border_radius=12)
        pygame.draw.rect(surface, Theme.BORDER, rect, 1, border_radius=12)

        self.text.blit(
            surface,
            "card",
            title,
            Theme.TEXT,
            (rect.x + self.layout.pad, rect.y + 13),
        )

        if hint:
            self.text.blit(
                surface,
                "small",
                hint,
                Theme.TEXT_FAINT,
                (rect.right - self.layout.pad, rect.y + 15),
                align="right",
            )

        baseline = rect.y + self.layout.card_title_height - 9

        pygame.draw.line(
            surface,
            Theme.BORDER,
            (rect.x + self.layout.pad, baseline),
            (rect.right - self.layout.pad, baseline),
            1,
        )

    def paint_header_frame(self, surface):
        rect = self.layout.header

        self.text.blit(
            surface, "title", L["title"], Theme.TEXT, (rect.x + 2, rect.y + 2)
        )

        self.text.blit(
            surface,
            "small",
            L["subtitle"],
            Theme.TEXT_FAINT,
            (rect.x + 3, rect.y + 6 + self.text.height("title")),
        )

        pygame.draw.line(
            surface,
            Theme.BORDER,
            (rect.x, rect.bottom - 1),
            (rect.right, rect.bottom - 1),
            1,
        )

    def paint_cards(self, surface):
        hint = (
            L["perception_hint"] if self.layout.perception.width > 330 else None
        )

        self.card(surface, self.layout.perception, L["perception"], hint)
        self.card(surface, self.layout.telemetry, L["telemetry"])
        self.card(surface, self.layout.decision, L["decision"])
        self.card(surface, self.layout.meta, L["meta"])

        rect = self.layout.network

        pygame.draw.rect(surface, Theme.CARD, rect, border_radius=12)
        pygame.draw.rect(surface, Theme.BORDER, rect, 1, border_radius=12)

    def paint_footer(self, surface):
        rect = self.layout.footer

        self.text.blit(
            surface, "small", L["footer"], Theme.TEXT_FAINT, (rect.x + 2, rect.y)
        )

    # ---------------------------------------------------------------------

    def paint_radar_frame(self, surface):
        """Arcos de alcance, eixos e o carro. Nada disso muda."""

        origin_x, origin_y = self.layout.radar_origin
        radius = self.layout.radar_radius

        for step in (0.33, 0.66, 1.0):
            size = int(radius * step)

            box = pygame.Rect(
                origin_x - size, origin_y - size, size * 2, size * 2
            )

            color = Theme.over(
                Theme.BORDER_HI, 0.80 if step >= 1.0 else 0.45, Theme.CARD
            )

            pygame.draw.arc(
                surface, color, box, math.radians(20), math.radians(160), 1
            )

        faint = Theme.over(Theme.BORDER_HI, 0.55, Theme.CARD)

        for unit_x, unit_y in SENSOR_UNITS:
            pygame.draw.line(
                surface,
                faint,
                (origin_x, origin_y),
                (
                    int(origin_x + unit_x * radius),
                    int(origin_y + unit_y * radius),
                ),
                1,
            )

        # Carro visto de cima, na escala real do alcance do sensor.
        scale = radius / SENSOR_RANGE

        length = max(7, int(28 * scale))
        width = max(4, int(13 * scale))

        body = pygame.Rect(0, 0, width, length)
        body.center = (origin_x, origin_y)

        pygame.draw.rect(surface, Theme.TEXT_DIM, body, border_radius=3)

        nose = max(3, int(5 * scale))

        pygame.draw.polygon(
            surface,
            Theme.CYAN,
            (
                (origin_x, body.top - nose),
                (origin_x - max(3, width // 2), body.top + 1),
                (origin_x + max(3, width // 2), body.top + 1),
            ),
        )

        self.text.blit(
            surface,
            "tiny",
            "100 px",
            Theme.TEXT_FAINT,
            (origin_x, origin_y - radius - self.text.height("tiny") - 2),
            align="center",
        )

    def paint_network_frame(self, surface):
        """Titulo, nomes das camadas e rotulos de entrada e saida."""

        area = self.layout.network_area
        geometry = self.geometry

        architecture = geometry.architecture

        biases = sum(architecture[1:])

        rect = self.layout.network

        self.text.blit(
            surface, "card", L["network"], Theme.TEXT, (area.x, rect.y + 13)
        )

        self.text.blit(
            surface,
            "mono",
            "  ".join(str(count) for count in architecture),
            Theme.CYAN,
            (area.right, rect.y + 14),
            align="right",
        )

        # Numeros derivados das matrizes reais do cerebro.
        self.text.blit(
            surface,
            "small",
            (
                f"{geometry.total} conexoes   -   {biases} biases   -   "
                f"top {min(self.ACTIVE_CONNECTIONS, geometry.total)} acesas"
            ),
            Theme.TEXT_FAINT,
            (area.right, rect.y + 15 + self.text.height("card")),
            align="right",
        )

        baseline = rect.y + self.layout.card_title_height - 9

        pygame.draw.line(
            surface,
            Theme.BORDER,
            (area.x, baseline),
            (area.x + 170, baseline),
            1,
        )

        # ---- nome de cada camada
        names = ["INPUT"]

        for index in range(len(architecture) - 2):
            names.append(f"H{index + 1}")

        names.append("OUTPUT")

        label_y = geometry.rect.y - self.text.height("tiny") - 11

        for index, column in enumerate(geometry.columns):
            name = names[index] if index < len(names) else f"L{index}"

            self.text.blit(
                surface, "tiny", name, Theme.TEXT_DIM, (column, label_y),
                align="center",
            )

            self.text.blit(
                surface,
                "small",
                str(architecture[index]),
                Theme.TEXT_FAINT,
                (column, label_y + self.text.height("tiny") - 1),
                align="center",
            )

        # ---- rotulos das entradas
        for index, position in enumerate(geometry.positions[0]):
            if index >= len(INPUT_LABELS):
                break

            self.text.blit(
                surface,
                "small",
                INPUT_LABELS[index],
                Theme.TEXT_DIM if index == 3 else Theme.TEXT_FAINT,
                (position[0] - geometry.radii[0] - 9, position[1]),
                align="midright",
            )

        # ---- rotulos das saidas
        for index, position in enumerate(geometry.positions[-1]):
            if index >= len(OUTPUT_LABELS):
                break

            self.text.blit(
                surface,
                "tiny",
                OUTPUT_LABELS[index],
                Theme.TEXT_DIM,
                (position[0] + geometry.radii[-1] + 10, position[1] - 8),
                align="midleft",
            )

    # =====================================================================
    # PINTURA DINAMICA
    # =====================================================================

    def chip(self, label, value, value_color, right, top, width, height):
        rect = pygame.Rect(right - width, top, width, height)

        pygame.draw.rect(self.frame, Theme.CARD_SOFT, rect, border_radius=8)
        pygame.draw.rect(self.frame, Theme.BORDER, rect, 1, border_radius=8)

        self.text.blit(
            self.frame, "tiny", label, Theme.TEXT_FAINT, (rect.x + 10, rect.y + 6)
        )

        self.text.blit(
            self.frame,
            "body_bold",
            value,
            value_color,
            (rect.x + 10, rect.y + 5 + self.text.height("tiny")),
        )

    def draw_header(self, car, track, fps):
        rect = self.layout.header

        height = min(
            rect.height - 6,
            self.text.height("tiny") + self.text.height("body_bold") + 13,
        )

        top = rect.bottom - height - 6

        right = rect.right

        decisions = getattr(car.brain, "contador_decisoes", 0)

        chips = (
            (L["chip_decisions"], str(decisions), Theme.TEXT, 104),
            (
                L["chip_fps"],
                f"{fps:.0f}",
                Theme.GREEN if fps >= 50.0 else Theme.YELLOW,
                84,
            ),
            (
                L["chip_mode"],
                L["mode_best"] if self.follow_best else L["mode_manual"],
                Theme.CYAN if self.follow_best else Theme.TEXT_DIM,
                126,
            ),
            (L["chip_car"], f"#{car.identifier}", Theme.TEXT, 78),
            (L["chip_track"], str(track.name), Theme.TEXT, 142),
        )

        for label, value, color, width in chips:
            if right - width < rect.x + 330:
                break

            self.chip(label, value, color, right, top, width, height)

            right -= width + 8

    # ---------------------------------------------------------------------

    def draw_perception(self, car):
        risks = getattr(car, "last_sensor_risks", None) or ()

        # Somente leitura do que o carro ja calculou.
        values = [
            float(risks[index]) if index < len(risks) else 0.0
            for index in range(len(SENSOR_ANGLES))
        ]

        self.draw_radar(values)
        self.draw_meters(values)
        self.draw_callout(values)

    def draw_radar(self, values):
        origin_x, origin_y = self.layout.radar_origin
        radius = self.layout.radar_radius

        hits = []

        for index, risk in enumerate(values):
            # Mesma conta que o carro usou para gerar o risco, invertida:
            # distancia livre = START + (1 - risco) * (RANGE - START)
            free = SENSOR_START + (1.0 - risk) * (SENSOR_RANGE - SENSOR_START)

            length = radius * (free / SENSOR_RANGE)

            unit_x, unit_y = SENSOR_UNITS[index]

            hits.append(
                (
                    int(origin_x + unit_x * length),
                    int(origin_y + unit_y * length),
                    risk,
                )
            )

        # Area livre percebida, translucida. A superficie e pequena e
        # reaproveitada, entao o custo e irrelevante.
        radar = self.layout.radar

        overlay = self.radar_overlay

        overlay.fill((0, 0, 0, 0))

        polygon = [(origin_x - radar.x, origin_y - radar.y)]

        for x, y, _ in hits:
            polygon.append((x - radar.x, y - radar.y))

        pygame.draw.polygon(overlay, (38, 217, 240, 26), polygon)

        self.frame.blit(overlay, radar.topleft)

        # Raio de cada sensor e a parede que ele encontrou.
        for x, y, risk in hits:
            color = Theme.risk(risk)

            pygame.draw.line(
                self.frame, color, (origin_x, origin_y), (x, y), 2
            )

            if risk > 0.02:
                pygame.draw.circle(self.frame, color, (x, y), 3)

    def draw_meters(self, values):
        rect = self.layout.meters
        row = self.layout.meter_row_height

        label_width = max(34, self.text.width("small", "FRENTE") + 6)
        value_width = max(34, self.text.width("mono", "0.00") + 8)

        bar_x = rect.x + 14 + label_width
        bar_width = max(24, rect.right - bar_x - value_width - 6)

        bar_height = max(6, min(10, row - 9))

        for index, risk in enumerate(values):
            middle = rect.y + row * index + row // 2

            color = Theme.risk(risk)

            pygame.draw.circle(self.frame, color, (rect.x + 5, middle), 3)

            is_front = index == 3

            self.text.blit(
                self.frame,
                "small",
                "FRENTE" if is_front else SENSOR_SHORT[index],
                Theme.TEXT if is_front else Theme.TEXT_DIM,
                (rect.x + 14, middle),
                align="midleft",
            )

            bar = pygame.Rect(
                bar_x, middle - bar_height // 2, bar_width, bar_height
            )

            pygame.draw.rect(self.frame, Theme.TRACK_BG, bar, border_radius=4)

            # Barra cheia = risco alto. Vazia = caminho livre.
            filled = int(bar_width * min(1.0, max(0.0, risk)))

            if filled > 1:
                pygame.draw.rect(
                    self.frame,
                    color,
                    (bar.x, bar.y, filled, bar_height),
                    border_radius=4,
                )

            self.text.blit(
                self.frame,
                "mono",
                f"{risk:.2f}",
                color,
                (rect.right, middle),
                align="midright",
            )

    def draw_callout(self, values):
        rect = self.layout.callout

        worst = 0
        worst_value = -1.0

        for index, risk in enumerate(values):
            if risk > worst_value:
                worst_value = risk
                worst = index

        color = Theme.risk(worst_value)

        pygame.draw.rect(self.frame, Theme.CARD_SOFT, rect, border_radius=9)
        pygame.draw.rect(self.frame, Theme.BORDER, rect, 1, border_radius=9)

        pygame.draw.rect(
            self.frame,
            color,
            (rect.x, rect.y + 7, 3, rect.height - 14),
            border_radius=2,
        )

        self.text.blit(
            self.frame,
            "tiny",
            L["highest_risk"],
            Theme.TEXT_FAINT,
            (rect.x + 13, rect.y + 7),
        )

        self.text.blit(
            self.frame,
            "small",
            SENSOR_NAMES[worst],
            Theme.TEXT,
            (rect.x + 13, rect.y + 7 + self.text.height("tiny")),
        )

        self.text.blit(
            self.frame,
            "big",
            f"{worst_value:.2f}",
            color,
            (rect.right - 13, rect.y + 6),
            align="right",
        )

        self.text.blit(
            self.frame,
            "tiny",
            Theme.risk_level(worst_value),
            color,
            (rect.right - 13, rect.bottom - self.text.height("tiny") - 8),
            align="right",
        )

    # ---------------------------------------------------------------------

    def draw_telemetry(self, car, track):
        rect = self.layout.inner(self.layout.telemetry)

        max_speed = float(getattr(car, "MAX_SPEED", 175.0)) or 1.0

        speed = float(car.speed)

        ratio = max(0.0, min(1.0, speed / max_speed))

        self.text.blit(
            self.frame, "tiny", L["speed"], Theme.TEXT_FAINT, (rect.x, rect.y)
        )

        value_top = rect.y + self.text.height("tiny") + 2

        big = self.text.blit(
            self.frame, "huge", f"{speed:.0f}", Theme.TEXT, (rect.x, value_top)
        )

        # Nao existe velocidade obrigatoria: mostramos so o teto.
        self.text.blit(
            self.frame,
            "body",
            f"/ {max_speed:.0f}",
            Theme.TEXT_FAINT,
            (
                rect.x + big.get_width() + 8,
                value_top + big.get_height() - self.text.height("body") - 5,
            ),
        )

        self.text.blit(
            self.frame,
            "mono_big",
            f"{ratio * 100.0:.0f}%",
            Theme.CYAN,
            (rect.right, value_top + 4),
            align="right",
        )

        bar_y = value_top + big.get_height() + 6

        pygame.draw.rect(
            self.frame,
            Theme.TRACK_BG,
            (rect.x, bar_y, rect.width, 6),
            border_radius=3,
        )

        filled = int(rect.width * ratio)

        if filled > 1:
            pygame.draw.rect(
                self.frame,
                Theme.mix(Theme.BLUE, Theme.CYAN, ratio),
                (rect.x, bar_y, filled, 6),
                border_radius=3,
            )

        checkpoints = len(getattr(track, "checkpoints", ()))

        rows = (
            ("Carro", f"#{car.identifier}", Theme.TEXT, "body"),
            ("Pista", str(track.name), Theme.TEXT, "body"),
            ("Score", f"{car.get_evolution_score():.1f}", Theme.CYAN, "body"),
            (
                "Bonus velocidade",
                f"{float(getattr(car, 'speed_fitness', 0.0)):.1f}",
                Theme.GREEN,
                "body",
            ),
            ("Melhor score", f"{car.best_fitness:.1f}", Theme.TEXT, "body"),
            (
                "Checkpoint",
                f"{car.checkpoint_atual}/{checkpoints}",
                Theme.TEXT,
                "body",
            ),
            (
                "Voltas",
                str(car.voltas),
                Theme.GREEN if car.voltas else Theme.TEXT,
                "body",
            ),
            ("Mortes", str(car.deaths), Theme.TEXT, "body"),
            ("Tempo vivo", f"{car.time_alive:.1f}s", Theme.TEXT, "body"),
            ("Direcao", f"{car.steering:+.2f}", Theme.TEXT, "mono_bold"),
            ("Tracao", f"{car.last_traction:+.2f}", Theme.TEXT, "mono_bold"),
        )

        top = bar_y + 16

        line = max(15, min(24, int((rect.bottom - top) / len(rows))))

        for index, (label, value, color, font_key) in enumerate(rows):
            y = top + line * index

            if y + line > rect.bottom + 4:
                break

            self.text.blit(
                self.frame, "small", label, Theme.TEXT_DIM, (rect.x, y)
            )

            self.text.blit(
                self.frame,
                font_key,
                value,
                color,
                (rect.right, y - 1),
                align="right",
            )

    # ---------------------------------------------------------------------

    def bar_row(self, x, y, width, label, value, signed, color):
        """Uma linha com rotulo, valor e barra. Devolve o y final."""

        self.text.blit(self.frame, "small", label, Theme.TEXT_DIM, (x, y))

        self.text.blit(
            self.frame,
            "mono_bold",
            f"{value:+.2f}" if signed else f"{value:.2f}",
            color,
            (x + width, y - 1),
            align="right",
        )

        bar_y = y + self.text.height("small") + 4

        pygame.draw.rect(
            self.frame, Theme.TRACK_BG, (x, bar_y, width, 8), border_radius=4
        )

        if signed:
            center = x + width // 2

            pygame.draw.line(
                self.frame,
                Theme.BORDER_HI,
                (center, bar_y - 2),
                (center, bar_y + 10),
                1,
            )

            amount = int(min(1.0, abs(value)) * (width / 2 - 1))

            if amount > 1:
                left = center if value >= 0.0 else center - amount

                pygame.draw.rect(
                    self.frame, color, (left, bar_y, amount, 8), border_radius=4
                )
        else:
            amount = int(max(0.0, min(1.0, value)) * width)

            if amount > 1:
                pygame.draw.rect(
                    self.frame, color, (x, bar_y, amount, 8), border_radius=4
                )

        return bar_y + 8

    def draw_decision(self, car, brain):
        rect = self.layout.inner(self.layout.decision)

        decision = getattr(brain, "ultima_decisao", None) or {}

        raw = getattr(brain, "ultima_saida_bruta", None)

        gap = max(7, int(rect.height * 0.026))

        # Altura de uma linha completa, usada para nao passar do card
        # em janelas baixas.
        row = self.text.height("small") + 12 + gap

        y = rect.y

        # ---- o que a rede produziu
        self.text.blit(
            self.frame, "tiny", L["neural_output"], Theme.TEXT_FAINT, (rect.x, y)
        )

        y += self.text.height("tiny") + 6

        y = gap + self.bar_row(
            rect.x, y, rect.width, "ACELERAR",
            float(decision.get("acelerar", 0.0)), False, Theme.GREEN,
        )

        if y + row > rect.bottom:
            return

        y = gap + self.bar_row(
            rect.x, y, rect.width, "FREAR",
            float(decision.get("frear", 0.0)), False, Theme.RED,
        )

        if y + row > rect.bottom:
            return

        y = gap + self.bar_row(
            rect.x, y, rect.width, "VIRAR",
            float(decision.get("virar", 0.0)), True, Theme.CYAN,
        )

        if y + row > rect.bottom:
            return

        if raw is not None and len(raw) >= 3:
            self.text.blit(
                self.frame,
                "mono",
                (
                    f"tanh  {float(raw[0]):+.2f}  "
                    f"{float(raw[1]):+.2f}  {float(raw[2]):+.2f}"
                ),
                Theme.TEXT_FAINT,
                (rect.x, y),
            )

            y += self.text.height("mono") + gap

        pygame.draw.line(
            self.frame, Theme.BORDER, (rect.x, y), (rect.right, y), 1
        )

        y += gap + 2

        # ---- o que o carro realmente aplicou, depois de ganho,
        #      suavizacao e tracao
        self.text.blit(
            self.frame, "tiny", L["applied"], Theme.TEXT_FAINT, (rect.x, y)
        )

        y += self.text.height("tiny") + 6

        y = gap + self.bar_row(
            rect.x, y, rect.width, "ACELERADOR",
            float(car.accelerator), False, Theme.GREEN,
        )

        if y + row > rect.bottom:
            return

        y = gap + self.bar_row(
            rect.x, y, rect.width, "FREIO", float(car.brake), False, Theme.RED
        )

        if y + row > rect.bottom:
            return

        y = gap + self.bar_row(
            rect.x, y, rect.width, "DIRECAO", float(car.steering), True,
            Theme.CYAN,
        )

        if y + row <= rect.bottom:
            self.bar_row(
                rect.x, y, rect.width, "TRACAO",
                float(car.last_traction), True, Theme.BLUE,
            )

    # ---------------------------------------------------------------------

    def draw_meta(self, brain, track):
        rect = self.layout.inner(self.layout.meta)

        architecture = self.geometry.architecture

        rows = [
            (
                "Arquitetura",
                "-".join(str(count) for count in architecture),
                Theme.TEXT,
            ),
            ("Neuronios", str(sum(architecture)), Theme.TEXT),
            ("Camadas ocultas", str(max(0, len(architecture) - 2)), Theme.TEXT),
            ("Conexoes", str(self.geometry.total), Theme.TEXT),
            ("Biases", str(sum(architecture[1:])), Theme.TEXT),
            ("Decisoes", str(getattr(brain, "contador_decisoes", 0)), Theme.CYAN),
        ]

        evolution = self.evolution

        if evolution is not None:
            try:
                elites = evolution.quantidade_elites()
                limit = getattr(evolution, "elite_size", elites)

                stale = evolution.avaliacoes_sem_recorde

                rows.extend(
                    (
                        (None, None, None),
                        (
                            "Recorde global",
                            f"{evolution.melhor_fitness():.1f}",
                            Theme.GREEN,
                        ),
                        (
                            "Recorde da pista",
                            f"{evolution.melhor_fitness_pista(track.name):.1f}",
                            Theme.TEXT,
                        ),
                        ("Elites", f"{elites}/{limit}", Theme.TEXT),
                        (
                            "Persistencia",
                            (
                                "CARREGADA"
                                if getattr(
                                    evolution,
                                    "carregado_do_disco",
                                    False
                                )
                                else "ATIVA"
                            ),
                            Theme.GREEN,
                        ),
                        (
                            "Avaliacoes",
                            str(evolution.total_avaliacoes),
                            Theme.TEXT,
                        ),
                        (
                            "Sem recorde",
                            str(stale),
                            Theme.YELLOW if stale > 600 else Theme.TEXT,
                        ),
                    )
                )
            except Exception:
                pass

        legend_height = self.text.height("small") * 2 + 10

        available = rect.height - legend_height

        line = max(14, min(23, int(available / max(1, len(rows)))))

        y = rect.y

        for label, value, color in rows:
            if label is None:
                pygame.draw.line(
                    self.frame,
                    Theme.BORDER,
                    (rect.x, y + line // 2),
                    (rect.right, y + line // 2),
                    1,
                )

                y += line
                continue

            if y + line > rect.y + available + 4:
                break

            self.text.blit(
                self.frame, "small", label, Theme.TEXT_DIM, (rect.x, y)
            )

            self.text.blit(
                self.frame, "body", value, color, (rect.right, y - 1),
                align="right",
            )

            y += line

        legend_y = rect.bottom - legend_height + 4

        self.text.blit(
            self.frame, "small", L["legend_pos"], Theme.CYAN, (rect.x, legend_y)
        )

        self.text.blit(
            self.frame,
            "small",
            L["legend_neg"],
            Theme.ORANGE,
            (rect.x, legend_y + self.text.height("small") + 2),
        )

    # ---------------------------------------------------------------------

    def draw_active_connections(self, brain):
        """Acende apenas as conexoes mais influentes da decisao atual.

        Influencia = ativacao da origem * peso. E uma aproximacao de
        contribuicao, nao uma explicacao causal.
        """

        activations = getattr(brain, "ultimas_ativacoes", None)

        if not activations:
            return False

        result = self.probe.top(
            brain.pesos, activations, self.ACTIVE_CONNECTIONS
        )

        if result is None:
            return False

        indexes, values, strongest = result

        if strongest <= 1e-6:
            return True

        pulse = 0.34 + 0.66 * self.flash

        inverse = 1.0 / strongest

        points_a = self.geometry.point_a
        points_b = self.geometry.point_b

        line = pygame.draw.line
        surface = self.frame

        over = Theme.over

        for position, index in enumerate(indexes):
            contribution = values[position]

            ratio = abs(contribution) * inverse

            base = Theme.CYAN if contribution >= 0.0 else Theme.ORANGE

            intensity = ratio * pulse

            start = points_a[index]
            end = points_b[index]

            # Halo largo e nucleo fino, ambos ja misturados com o fundo.
            line(surface, over(base, 0.10 + 0.26 * intensity), start, end, 3)

            line(
                surface,
                over(base, 0.34 + 0.66 * intensity),
                start,
                end,
                2 if ratio > 0.55 else 1,
            )

        return True

    def draw_neurons(self, brain):
        activations = getattr(brain, "ultimas_ativacoes", None) or ()

        geometry = self.geometry

        surface = self.frame

        circle = pygame.draw.circle

        for layer, positions in enumerate(geometry.positions):
            radius = geometry.radii[layer]

            values = (
                activations[layer].tolist() if layer < len(activations) else None
            )

            for index, position in enumerate(positions):
                value = (
                    values[index]
                    if values is not None and index < len(values)
                    else 0.0
                )

                color = Theme.activation(value)

                circle(surface, Theme.CARD, position, radius + 2)
                circle(surface, color, position, radius)

                # Anel fino somente nos neuronios bem ativados.
                if abs(value) >= 0.70:
                    circle(
                        surface,
                        Theme.mix(Theme.TEXT, color, 0.35),
                        position,
                        radius + 2,
                        1,
                    )

    def draw_output_values(self, brain):
        activations = getattr(brain, "ultimas_ativacoes", None) or ()

        if not activations:
            return

        values = activations[-1]

        geometry = self.geometry

        offset = geometry.radii[-1] + 10

        for index, position in enumerate(geometry.positions[-1]):
            if index >= len(values):
                break

            value = float(values[index])

            self.text.blit(
                self.frame,
                "mono_bold",
                f"{value:+.2f}",
                Theme.activation(value) if abs(value) > 0.08 else Theme.TEXT_DIM,
                (position[0] + offset, position[1] + 8),
                align="midleft",
            )

    def draw_network(self, brain):
        active = self.draw_active_connections(brain)

        self.draw_neurons(brain)
        self.draw_output_values(brain)

        if not active:
            self.text.blit(
                self.frame,
                "small",
                L["waiting"],
                Theme.TEXT_FAINT,
                self.layout.network_plot.center,
                align="midcenter",
            )

    # =====================================================================
    # RENDER
    # =====================================================================

    def compose(self, car, track, fps):
        """Monta o quadro na superficie interna. Nao sobe para a GPU.

        Desenhar custa cerca de 5 ms e subir a textura cerca de 8 ms.
        Juntos passariam do orcamento de um frame de 60 FPS, por isso
        as duas etapas ficam em quadros diferentes.
        """

        brain = car.brain

        self.ensure_cache(brain)

        if self.frame is None or self.static is None or self.geometry is None:
            return False

        # Base pronta: copia simples, sem alpha.
        self.frame.blit(self.static, (0, 0))

        self.draw_header(car, track, fps)
        self.draw_perception(car)
        self.draw_telemetry(car, track)
        self.draw_network(brain)
        self.draw_decision(car, brain)
        self.draw_meta(brain, track)

        return True

    # =====================================================================
    # UPDATE
    # =====================================================================

    def update(self, cars, track, dt, fps=0.0):
        if not self.visible or self.display is None:
            return

        if self.pending_size is not None:
            self.resize_timer -= dt

            if self.resize_timer <= 0.0:
                self.apply_size()
        else:
            # Rede de seguranca caso algum evento de resize nao chegue.
            self.poll_timer -= dt

            if self.poll_timer <= 0.0:
                self.poll_timer = 0.5
                self.request_resize()

        if not cars:
            return

        self.update_best(cars, dt)

        car = self.selected_car(cars)

        if car is None:
            return

        # O brilho decai no ritmo da simulacao, mesmo com o inspector
        # desenhando menos vezes.
        counter = getattr(car.brain, "contador_decisoes", 0)

        if counter != self.last_counter:
            self.last_counter = counter
            self.flash = 1.0
        else:
            self.flash = max(0.0, self.flash - dt * self.FLASH_DECAY)

        self.refresh_timer -= dt

        try:
            # Quadro seguinte ao desenho: so sobe a textura e mostra.
            if self.pending_present:
                self.pending_present = False

                self.display.present(self.frame)
                return

            if self.refresh_timer > 0.0:
                # Quadros livres entre as atualizacoes adiantam a malha
                # de conexoes de fundo, que e a parte mais pesada.
                self.bake_wires()
                return

            # Sem acumular atraso: se a simulacao travou, apenas pula.
            self.refresh_timer = self.REFRESH_INTERVAL

            self.pending_present = self.compose(car, track, fps)
        except Exception:
            # O inspector nunca pode derrubar a simulacao.
            if not self.reported_error:
                self.reported_error = True
                traceback.print_exc()

            self.close()
