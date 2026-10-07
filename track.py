# Importar as bibliotecas
import math
import pygame


# Classe da pista
class Track:
    def __init__(self, image_path, screen_width, screen_height):
        self.name = image_path.stem
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Carregar a imagem preservando a transparencia do PNG
        self.original_surface = pygame.image.load(str(image_path)).convert_alpha()
        self.surface = None
        self.mask = None
        self.rect = None
        self.spawn_point = (0, 0)
        self.spawn_angle = 0.0
        self.load_and_fit()

    def load_and_fit(self):
        # Ajustar a pista ao tamanho da janela sem deformar
        width, height = self.original_surface.get_size()
        scale = min(
            (self.screen_width * 0.85) / width,
            (self.screen_height * 0.90) / height,
        )
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

        self.surface = pygame.transform.smoothscale(self.original_surface, new_size)
        self.rect = self.surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2)
        )

        # Na mascara, 1 significa pista visivel e 0 significa fora da pista
        self.mask = pygame.mask.from_surface(self.surface, 180)
        self.spawn_point, self.spawn_angle = self.find_spawn_point()

    def is_point_on_track(self, point):
        # Converter coordenadas da janela em coordenadas da imagem
        x = int(point[0] - self.rect.left)
        y = int(point[1] - self.rect.top)
        width, height = self.mask.get_size()

        if not (0 <= x < width and 0 <= y < height):
            return False
        return bool(self.mask.get_at((x, y)))

    def is_car_on_track(self, car_mask, car_rect):
        # Exige que TODOS os pixels do carrinho estejam dentro da pista
        offset = (car_rect.left - self.rect.left, car_rect.top - self.rect.top)
        pixels_inside = self.mask.overlap_area(car_mask, offset)
        return pixels_inside == car_mask.count()

    def _safe_local_spawn(self, x, y, horizontal):
        # Evitar largadas encostadas na borda (carro: 14 x 8 pixels)
        width, height = self.mask.get_size()
        if horizontal:
            points = [(0, 0), (-14, -7), (-14, 7), (14, -7), (14, 7),
                      (30, 0), (-30, 0)]
        else:
            points = [(0, 0), (-7, -14), (7, -14), (-7, 14), (7, 14),
                      (0, 30), (0, -30)]

        for dx, dy in points:
            px, py = x + dx, y + dy
            if not (0 <= px < width and 0 <= py < height):
                return False
            if not self.mask.get_at((px, py)):
                return False
        return True

    def find_spawn_point(self):
        # Buscar um trecho longo e seguro nas direcoes horizontal e vertical
        width, height = self.mask.get_size()
        best = None

        for horizontal in (True, False):
            line_count = height if horizontal else width
            line_length = width if horizontal else height

            for line in range(0, line_count, 4):
                start = None

                for pos in range(line_length + 1):
                    on_track = False
                    if pos < line_length:
                        x, y = (pos, line) if horizontal else (line, pos)
                        on_track = bool(self.mask.get_at((x, y)))

                    if on_track and start is None:
                        start = pos
                    elif not on_track and start is not None:
                        length = pos - start
                        mid = (start + pos - 1) // 2
                        x, y = (mid, line) if horizontal else (line, mid)

                        if (best is None or length > best[0]) and self._safe_local_spawn(x, y, horizontal):
                            angle = 0.0 if horizontal else math.pi / 2
                            best = (length, x, y, angle)
                        start = None

        if best is None:
            raise ValueError(f"Nao foi possivel encontrar uma largada segura em {self.name}.")

        _, x, y, angle = best
        return (self.rect.left + x, self.rect.top + y), angle
