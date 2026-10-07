# Importar as bibliotecas
import math
import pygame

import numpy as np
from skimage.morphology import skeletonize


# Classe da pista
class Track:

    # Quantidade de checkpoints criados automaticamente
    CHECKPOINT_COUNT = 40

    # Distancia que o carro precisa chegar do checkpoint
    CHECKPOINT_RADIUS = 18

    def __init__(self, image_path, screen_width, screen_height):
        self.name = image_path.stem

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Carregar imagem da pista com transparencia
        self.original_surface = pygame.image.load(
            str(image_path)
        ).convert_alpha()

        self.surface = None
        self.mask = None
        self.rect = None

        # Ponto de largada
        self.spawn_point = (0, 0)
        self.spawn_angle = 0.0

        # Linha central
        self.centerline = None
        self.centerline_points = []

        # Checkpoints
        self.checkpoints = []

        # Imagens usadas somente para debug
        self.centerline_debug = None
        self.checkpoints_debug = None

        # Preparar pista
        self.load_and_fit()

    # ---------------------------------------------------------
    # CARREGAR E PREPARAR A PISTA
    # ---------------------------------------------------------

    def load_and_fit(self):

        # Tamanho original da imagem
        width, height = self.original_surface.get_size()

        # Ajustar a pista na tela sem deformar
        scale = min(
            (self.screen_width * 0.85) / width,
            (self.screen_height * 0.90) / height,
        )

        new_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )

        # Redimensionar pista
        self.surface = pygame.transform.smoothscale(
            self.original_surface,
            new_size
        )

        # Centralizar pista
        self.rect = self.surface.get_rect(
            center=(
                self.screen_width // 2,
                self.screen_height // 2
            )
        )

        # Criar mascara da pista
        # 1 = pista
        # 0 = fora da pista
        self.mask = pygame.mask.from_surface(
            self.surface,
            180
        )

        # Encontrar local da largada
        self.spawn_point, self.spawn_angle = (
            self.find_spawn_point()
        )

        # Criar linha central
        self.generate_centerline()

        # Criar checkpoints
        self.generate_checkpoints()

        # Criar imagem de debug dos checkpoints
        self.create_checkpoints_debug()

    # ---------------------------------------------------------
    # LINHA CENTRAL
    # ---------------------------------------------------------

    def generate_centerline(self):

        # Pegar transparencia da imagem
        alpha = pygame.surfarray.array_alpha(
            self.surface
        )

        # Pixels visiveis sao considerados pista
        track_area = alpha > 180

        # Pygame trabalha X,Y
        # Numpy trabalha Y,X
        track_area = track_area.T

        # Criar esqueleto da pista
        skeleton = skeletonize(
            track_area
        )

        # Remover pequenas pontas indesejadas
        skeleton = self.remove_small_branches(
            skeleton
        )

        self.centerline = skeleton

        # Ordenar os pontos da linha central
        self.centerline_points = (
            self.order_centerline()
        )

        # Criar imagem de debug
        self.centerline_debug = pygame.Surface(
            self.surface.get_size(),
            pygame.SRCALPHA
        )

        ys, xs = np.where(
            self.centerline
        )

        # Desenhar uma unica vez
        for x, y in zip(xs, ys):

            self.centerline_debug.set_at(
                (x, y),
                pygame.Color("red")
            )

    # ---------------------------------------------------------
    # REMOVER PEQUENAS RAMIFICACOES
    # ---------------------------------------------------------

    def remove_small_branches(
        self,
        skeleton,
        passes=8
    ):

        result = skeleton.copy()

        height, width = result.shape

        # Fazer poucas passagens para eliminar
        # pequenas pontas criadas pelo skeletonize
        for _ in range(passes):

            remove = []

            ys, xs = np.where(result)

            for y, x in zip(ys, xs):

                x1 = max(0, x - 1)
                x2 = min(width, x + 2)

                y1 = max(0, y - 1)
                y2 = min(height, y + 2)

                area = result[
                    y1:y2,
                    x1:x2
                ]

                # Quantidade de vizinhos
                neighbors = (
                    np.count_nonzero(area) - 1
                )

                # Ponta da linha
                if neighbors <= 1:
                    remove.append(
                        (x, y)
                    )

            # Nao existe mais nada para remover
            if not remove:
                break

            for x, y in remove:
                result[y, x] = False

        return result

    # ---------------------------------------------------------
    # PEGAR VIZINHOS DE UM PONTO DA LINHA
    # ---------------------------------------------------------

    def get_centerline_neighbors(
        self,
        point,
        points
    ):

        x, y = point

        neighbors = []

        # Verificar os 8 pixels ao redor
        for dy in (-1, 0, 1):

            for dx in (-1, 0, 1):

                if dx == 0 and dy == 0:
                    continue

                candidate = (
                    x + dx,
                    y + dy
                )

                if candidate in points:
                    neighbors.append(
                        candidate
                    )

        return neighbors

    # ---------------------------------------------------------
    # PEGAR MAIOR PARTE CONECTADA DA LINHA
    # ---------------------------------------------------------

    def largest_centerline_component(
        self,
        points
    ):

        remaining = set(points)

        largest = set()

        while remaining:

            start = remaining.pop()

            component = {
                start
            }

            stack = [
                start
            ]

            while stack:

                current = stack.pop()

                for neighbor in (
                    self.get_centerline_neighbors(
                        current,
                        remaining
                    )
                ):

                    if neighbor in remaining:

                        remaining.remove(
                            neighbor
                        )

                        component.add(
                            neighbor
                        )

                        stack.append(
                            neighbor
                        )

            if len(component) > len(largest):
                largest = component

        return largest

    # ---------------------------------------------------------
    # ORDENAR A LINHA CENTRAL
    # ---------------------------------------------------------

    def order_centerline(self):

        ys, xs = np.where(
            self.centerline
        )

        all_points = {
            (int(x), int(y))
            for x, y in zip(xs, ys)
        }

        if not all_points:
            raise ValueError(
                f"Nao foi possivel criar a centerline de {self.name}."
            )

        # Usar somente o maior caminho conectado
        points = self.largest_centerline_component(
            all_points
        )

        if not points:
            raise ValueError(
                f"Centerline vazia em {self.name}."
            )

        # Converter largada para coordenadas locais
        spawn_local = (
            self.spawn_point[0] - self.rect.left,
            self.spawn_point[1] - self.rect.top,
        )

        # Encontrar ponto da centerline mais perto da largada
        start = min(
            points,
            key=lambda point:
                (
                    point[0] - spawn_local[0]
                ) ** 2
                +
                (
                    point[1] - spawn_local[1]
                ) ** 2
        )

        ordered = [
            start
        ]

        visited = {
            start
        }

        previous = None
        current = start

        # Direcao inicial da pista
        spawn_direction = (
            math.cos(self.spawn_angle),
            math.sin(self.spawn_angle)
        )

        # Caminhar pela linha central
        for _ in range(
            len(points) + 100
        ):

            neighbors = (
                self.get_centerline_neighbors(
                    current,
                    points
                )
            )

            # Vizinho ainda nao visitado
            candidates = [
                point
                for point in neighbors
                if point not in visited
            ]

            if not candidates:
                break

            # Primeiro passo segue a direcao da largada
            if previous is None:

                def first_score(point):

                    dx = (
                        point[0] - current[0]
                    )

                    dy = (
                        point[1] - current[1]
                    )

                    return (
                        dx * spawn_direction[0]
                        +
                        dy * spawn_direction[1]
                    )

                next_point = max(
                    candidates,
                    key=first_score
                )

            else:

                # Depois tentar continuar na direcao
                # mais suave possivel
                old_dx = (
                    current[0] - previous[0]
                )

                old_dy = (
                    current[1] - previous[1]
                )

                old_length = math.hypot(
                    old_dx,
                    old_dy
                )

                if old_length == 0:
                    old_length = 1

                old_dx /= old_length
                old_dy /= old_length

                def direction_score(point):

                    new_dx = (
                        point[0] - current[0]
                    )

                    new_dy = (
                        point[1] - current[1]
                    )

                    new_length = math.hypot(
                        new_dx,
                        new_dy
                    )

                    if new_length == 0:
                        return -999

                    new_dx /= new_length
                    new_dy /= new_length

                    # Quanto maior, menor a mudanca
                    # brusca de direcao
                    return (
                        old_dx * new_dx
                        +
                        old_dy * new_dy
                    )

                next_point = max(
                    candidates,
                    key=direction_score
                )

            previous = current
            current = next_point

            ordered.append(
                current
            )

            visited.add(
                current
            )

        if len(ordered) < 100:

            raise ValueError(
                "A linha central encontrada em "
                f"{self.name} ficou muito pequena."
            )

        return ordered

    # ---------------------------------------------------------
    # CRIAR CHECKPOINTS
    # ---------------------------------------------------------

    def generate_checkpoints(self):

        self.checkpoints = []

        total_points = len(
            self.centerline_points
        )

        if total_points == 0:
            return

        # Dividir a linha em partes iguais
        step = (
            total_points
            / self.CHECKPOINT_COUNT
        )

        # Comecar um pouco depois da largada
        # O ultimo checkpoint volta para perto da largada
        for number in range(
            1,
            self.CHECKPOINT_COUNT + 1
        ):

            index = round(
                number * step
            ) % total_points

            local_x, local_y = (
                self.centerline_points[
                    index
                ]
            )

            # Converter para coordenadas da tela
            world_x = (
                self.rect.left
                + local_x
            )

            world_y = (
                self.rect.top
                + local_y
            )

            self.checkpoints.append(
                (
                    float(world_x),
                    float(world_y)
                )
            )

    # ---------------------------------------------------------
    # VERIFICAR CHECKPOINT
    # ---------------------------------------------------------

    def checkpoint_reached(
        self,
        position,
        checkpoint_index
    ):

        if not self.checkpoints:
            return False

        checkpoint_index %= len(
            self.checkpoints
        )

        checkpoint_x, checkpoint_y = (
            self.checkpoints[
                checkpoint_index
            ]
        )

        dx = (
            position[0]
            - checkpoint_x
        )

        dy = (
            position[1]
            - checkpoint_y
        )

        distance_squared = (
            dx * dx
            +
            dy * dy
        )

        return (
            distance_squared
            <=
            self.CHECKPOINT_RADIUS
            ** 2
        )

    # ---------------------------------------------------------
    # DEBUG DOS CHECKPOINTS
    # ---------------------------------------------------------

    def create_checkpoints_debug(self):

        self.checkpoints_debug = pygame.Surface(
            (
                self.screen_width,
                self.screen_height
            ),
            pygame.SRCALPHA
        )

        for index, point in enumerate(
            self.checkpoints
        ):

            x = round(point[0])
            y = round(point[1])

            # Primeiro checkpoint amarelo
            if index == 0:
                color = pygame.Color(
                    "yellow"
                )

            else:
                color = pygame.Color(
                    "cyan"
                )

            pygame.draw.circle(
                self.checkpoints_debug,
                color,
                (x, y),
                5
            )

    # Mostrar linha central
    def draw_centerline_debug(
        self,
        screen
    ):

        screen.blit(
            self.centerline_debug,
            self.rect
        )

    # Mostrar checkpoints
    def draw_checkpoints_debug(
        self,
        screen
    ):

        screen.blit(
            self.checkpoints_debug,
            (0, 0)
        )

    # ---------------------------------------------------------
    # SABER SE UM PONTO ESTA NA PISTA
    # ---------------------------------------------------------

    def is_point_on_track(
        self,
        point
    ):

        x = int(
            point[0]
            - self.rect.left
        )

        y = int(
            point[1]
            - self.rect.top
        )

        width, height = (
            self.mask.get_size()
        )

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            return False

        return bool(
            self.mask.get_at(
                (x, y)
            )
        )

    # ---------------------------------------------------------
    # VERIFICAR SE O CARRO ESTA COMPLETAMENTE NA PISTA
    # ---------------------------------------------------------

    def is_car_on_track(
        self,
        car_mask,
        car_rect
    ):

        offset = (
            car_rect.left
            - self.rect.left,

            car_rect.top
            - self.rect.top
        )

        pixels_inside = (
            self.mask.overlap_area(
                car_mask,
                offset
            )
        )

        return (
            pixels_inside
            ==
            car_mask.count()
        )

    # ---------------------------------------------------------
    # VERIFICAR SE A LARGADA E SEGURA
    # ---------------------------------------------------------

    def _safe_local_spawn(
        self,
        x,
        y,
        horizontal
    ):

        width, height = (
            self.mask.get_size()
        )

        if horizontal:

            points = [
                (0, 0),
                (-14, -7),
                (-14, 7),
                (14, -7),
                (14, 7),
                (30, 0),
                (-30, 0),
            ]

        else:

            points = [
                (0, 0),
                (-7, -14),
                (7, -14),
                (-7, 14),
                (7, 14),
                (0, 30),
                (0, -30),
            ]

        for dx, dy in points:

            px = x + dx
            py = y + dy

            if not (
                0 <= px < width
                and
                0 <= py < height
            ):
                return False

            if not self.mask.get_at(
                (px, py)
            ):
                return False

        return True

    # ---------------------------------------------------------
    # ENCONTRAR LARGADA
    # ---------------------------------------------------------

    def find_spawn_point(self):

        width, height = (
            self.mask.get_size()
        )

        best = None

        # Procurar trechos horizontais e verticais
        for horizontal in (
            True,
            False
        ):

            if horizontal:
                line_count = height
                line_length = width

            else:
                line_count = width
                line_length = height

            # Andar de 4 em 4 para reduzir processamento
            for line in range(
                0,
                line_count,
                4
            ):

                start = None

                for pos in range(
                    line_length + 1
                ):

                    on_track = False

                    if pos < line_length:

                        if horizontal:

                            x = pos
                            y = line

                        else:

                            x = line
                            y = pos

                        on_track = bool(
                            self.mask.get_at(
                                (x, y)
                            )
                        )

                    # Entrou na pista
                    if (
                        on_track
                        and
                        start is None
                    ):

                        start = pos

                    # Saiu da pista
                    elif (
                        not on_track
                        and
                        start is not None
                    ):

                        length = (
                            pos - start
                        )

                        mid = (
                            start
                            + pos
                            - 1
                        ) // 2

                        if horizontal:

                            x = mid
                            y = line

                        else:

                            x = line
                            y = mid

                        better = (
                            best is None
                            or
                            length > best[0]
                        )

                        if (
                            better
                            and
                            self._safe_local_spawn(
                                x,
                                y,
                                horizontal
                            )
                        ):

                            if horizontal:
                                angle = 0.0

                            else:
                                angle = (
                                    math.pi / 2
                                )

                            best = (
                                length,
                                x,
                                y,
                                angle
                            )

                        start = None

        if best is None:

            raise ValueError(
                "Nao foi possivel encontrar "
                f"uma largada segura em {self.name}."
            )

        _, x, y, angle = best

        return (
            (
                self.rect.left + x,
                self.rect.top + y
            ),
            angle
        )