import math
from pathlib import Path

import numpy as np
import pygame

from skimage.measure import label
from skimage.morphology import skeletonize


class Track:

    ALPHA_THRESHOLD = 180

    CHECKPOINT_COUNT = 40
    CHECKPOINT_RADIUS = 18

    SCREEN_WIDTH_PERCENT = 0.85
    SCREEN_HEIGHT_PERCENT = 0.90

    # ---------------------------------------------------------
    # INICIALIZAR
    # ---------------------------------------------------------

    def __init__(
        self,
        image_path,
        screen_width,
        screen_height
    ):

        self.path = Path(
            image_path
        )

        self.name = (
            self.path.stem
        )

        self.screen_width = (
            screen_width
        )

        self.screen_height = (
            screen_height
        )

        # -----------------------------------------------------
        # CARREGAR IMAGEM
        # -----------------------------------------------------

        source = pygame.image.load(
            str(
                self.path
            )
        ).convert_alpha()

        # Remover bordas transparentes desnecessarias
        bounds = source.get_bounding_rect(
            min_alpha=1
        )

        if (
            bounds.width > 0
            and bounds.height > 0
        ):

            source = source.subsurface(
                bounds
            ).copy()

        source_width = (
            source.get_width()
        )

        source_height = (
            source.get_height()
        )

        max_width = int(
            screen_width
            * self.SCREEN_WIDTH_PERCENT
        )

        max_height = int(
            screen_height
            * self.SCREEN_HEIGHT_PERCENT
        )

        scale = min(
            max_width
            / source_width,

            max_height
            / source_height
        )

        new_width = max(
            1,
            round(
                source_width
                * scale
            )
        )

        new_height = max(
            1,
            round(
                source_height
                * scale
            )
        )

        if (
            new_width != source_width
            or new_height != source_height
        ):

            source = pygame.transform.smoothscale(
                source,
                (
                    new_width,
                    new_height
                )
            )

        self.surface = (
            source
        )

        self.rect = (
            self.surface.get_rect(
                center=(
                    screen_width // 2,
                    screen_height // 2
                )
            )
        )

        self.width = (
            self.surface.get_width()
        )

        self.height = (
            self.surface.get_height()
        )

        # -----------------------------------------------------
        # MAPA DA PISTA
        # -----------------------------------------------------

        alpha = pygame.surfarray.array_alpha(
            self.surface
        )

        # pygame entrega [x, y].
        # Internamente usamos [y, x].
        driveable = (
            alpha.T
            >= self.ALPHA_THRESHOLD
        )

        self.driveable_map = (
            np.ascontiguousarray(
                driveable,
                dtype=np.uint8
            )
        )

        # Bytes simples para as consultas feitas
        # milhares de vezes por segundo.
        self._driveable_bytes = (
            self.driveable_map.tobytes()
        )

        # Distancia ate a borda em cada pixel. Preparada UMA vez por pista.
        # Durante a corrida, evita testar sete pontos do carro para cada
        # posicao simulada nas previsoes de curva.
        from scipy.ndimage import distance_transform_edt
        borda = np.pad(self.driveable_map, 1, mode="constant")
        distancia = distance_transform_edt(borda)[1:-1, 1:-1]
        self._safety_distance_bytes = np.minimum(
            distancia, 255
        ).astype(np.uint8).tobytes()


        # Mantido por compatibilidade,
        # mas a simulacao principal nao depende dele.
        self.mask = pygame.mask.from_surface(
            self.surface,
            self.ALPHA_THRESHOLD
        )

        # -----------------------------------------------------
        # SPAWN
        # -----------------------------------------------------

        rough_spawn, rough_angle = (
            self.find_spawn(
                self.driveable_map
            )
        )

        # -----------------------------------------------------
        # CENTRO DA PISTA
        # -----------------------------------------------------

        skeleton = (
            self.create_skeleton(
                self.driveable_map
            )
        )

        ordered_line = (
            self.order_centerline(
                skeleton,
                rough_spawn,
                rough_angle
            )
        )

        # Se a linha foi encontrada,
        # usar o proprio centro dela como largada.
        if len(
            ordered_line
        ) >= 2:

            spawn_local = (
                ordered_line[0]
            )

            direction_index = min(
                12,
                len(
                    ordered_line
                ) - 1
            )

            direction_point = (
                ordered_line[
                    direction_index
                ]
            )

            dx = (
                direction_point[0]
                - spawn_local[0]
            )

            dy = (
                direction_point[1]
                - spawn_local[1]
            )

            if (
                dx != 0
                or dy != 0
            ):

                spawn_angle = (
                    math.atan2(
                        dy,
                        dx
                    )
                )

            else:

                spawn_angle = (
                    rough_angle
                )

        else:

            spawn_local = (
                rough_spawn
            )

            spawn_angle = (
                rough_angle
            )

        self.spawn_point = (
            self.rect.x
            + spawn_local[0],

            self.rect.y
            + spawn_local[1]
        )

        self.spawn_angle = (
            spawn_angle
        )

        # Linha em coordenadas da tela
        self.centerline = [
            (
                self.rect.x + x,
                self.rect.y + y
            )
            for x, y in ordered_line
        ]

        # -----------------------------------------------------
        # CHECKPOINTS
        # -----------------------------------------------------

        self.checkpoints = (
            self.create_checkpoints(
                ordered_line
            )
        )

        # Portais completos de checkpoint, cruzando toda a largura da pista.
        # Um raio pequeno no centro faz os carros perderem checkpoints
        # quando passam pela faixa externa das curvas.
        self.checkpoint_gates = self.create_checkpoint_gates(ordered_line)

        self.checkpoint_radius_squared = (
            self.CHECKPOINT_RADIUS
            * self.CHECKPOINT_RADIUS
        )

        # Debug somente se for solicitado
        self._debug_surface = None

    # ---------------------------------------------------------
    # CONSULTA RAPIDA
    # ---------------------------------------------------------

    def is_point_on_track_xy(
        self,
        x,
        y
    ):

        local_x = (
            int(x)
            - self.rect.x
        )

        local_y = (
            int(y)
            - self.rect.y
        )

        if (
            local_x < 0
            or local_y < 0
            or local_x >= self.width
            or local_y >= self.height
        ):

            return False

        index = (
            local_y
            * self.width
            + local_x
        )

        return (
            self._driveable_bytes[
                index
            ]
            != 0
        )

    def is_point_on_track(
        self,
        point
    ):

        return self.is_point_on_track_xy(
            point[0],
            point[1]
        )

    # ---------------------------------------------------------
    # COMPATIBILIDADE COM MASK
    # ---------------------------------------------------------

    def is_car_on_track(
        self,
        car_mask,
        car_rect
    ):

        offset = (
            car_rect.x
            - self.rect.x,

            car_rect.y
            - self.rect.y
        )

        overlap = (
            self.mask.overlap_area(
                car_mask,
                offset
            )
        )

        return (
            overlap
            == car_mask.count()
        )

    # ---------------------------------------------------------
    # MAIOR TRECHO HORIZONTAL
    # ---------------------------------------------------------

    @staticmethod
    def widest_horizontal_run(
        driveable
    ):

        height, width = (
            driveable.shape
        )

        current = np.zeros(
            height,
            dtype=np.int32
        )

        best = np.zeros(
            height,
            dtype=np.int32
        )

        best_end = np.zeros(
            height,
            dtype=np.int32
        )

        for x in range(
            width
        ):

            active = (
                driveable[
                    :,
                    x
                ]
                != 0
            )

            current = np.where(
                active,
                current + 1,
                0
            )

            improved = (
                current > best
            )

            best[
                improved
            ] = current[
                improved
            ]

            best_end[
                improved
            ] = x

        row = int(
            np.argmax(
                best
            )
        )

        length = int(
            best[
                row
            ]
        )

        end = int(
            best_end[
                row
            ]
        )

        start = (
            end
            - length
            + 1
        )

        center_x = (
            start + end
        ) // 2

        return (
            length,
            center_x,
            row
        )

    # ---------------------------------------------------------
    # MAIOR TRECHO VERTICAL
    # ---------------------------------------------------------

    @staticmethod
    def widest_vertical_run(
        driveable
    ):

        height, width = (
            driveable.shape
        )

        current = np.zeros(
            width,
            dtype=np.int32
        )

        best = np.zeros(
            width,
            dtype=np.int32
        )

        best_end = np.zeros(
            width,
            dtype=np.int32
        )

        for y in range(
            height
        ):

            active = (
                driveable[
                    y,
                    :
                ]
                != 0
            )

            current = np.where(
                active,
                current + 1,
                0
            )

            improved = (
                current > best
            )

            best[
                improved
            ] = current[
                improved
            ]

            best_end[
                improved
            ] = y

        column = int(
            np.argmax(
                best
            )
        )

        length = int(
            best[
                column
            ]
        )

        end = int(
            best_end[
                column
            ]
        )

        start = (
            end
            - length
            + 1
        )

        center_y = (
            start + end
        ) // 2

        return (
            length,
            column,
            center_y
        )

    # ---------------------------------------------------------
    # LARGADA
    # ---------------------------------------------------------

    def find_spawn(
        self,
        driveable
    ):

        (
            horizontal_length,
            horizontal_x,
            horizontal_y
        ) = self.widest_horizontal_run(
            driveable
        )

        (
            vertical_length,
            vertical_x,
            vertical_y
        ) = self.widest_vertical_run(
            driveable
        )

        if (
            horizontal_length
            >= vertical_length
        ):

            return (
                (
                    horizontal_x,
                    horizontal_y
                ),
                0.0
            )

        return (
            (
                vertical_x,
                vertical_y
            ),
            math.pi / 2.0
        )

    # ---------------------------------------------------------
    # MAIOR COMPONENTE
    # ---------------------------------------------------------

    @staticmethod
    def largest_component(
        skeleton
    ):

        labels = label(
            skeleton,
            connectivity=2
        )

        if labels.max() <= 0:

            return skeleton

        counts = np.bincount(
            labels.ravel()
        )

        # Fundo nao conta
        counts[0] = 0

        largest = int(
            np.argmax(
                counts
            )
        )

        return (
            labels == largest
        )

    # ---------------------------------------------------------
    # REMOVER PONTAS CURTAS
    # ---------------------------------------------------------

    @staticmethod
    def prune_endpoints(
        skeleton,
        passes=8
    ):

        current = (
            skeleton.copy()
        )

        for _ in range(
            passes
        ):

            padded = np.pad(
                current.astype(
                    np.uint8
                ),
                1
            )

            neighbors = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]

                + padded[1:-1, :-2]
                + padded[1:-1, 2:]

                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            )

            endpoints = (
                current
                & (
                    neighbors <= 1
                )
            )

            if not np.any(
                endpoints
            ):
                break

            current[
                endpoints
            ] = False

        return current

    # ---------------------------------------------------------
    # SKELETON
    # ---------------------------------------------------------

    def create_skeleton(
        self,
        driveable
    ):

        skeleton = skeletonize(
            driveable.astype(
                bool
            )
        )

        skeleton = (
            self.largest_component(
                skeleton
            )
        )

        skeleton = (
            self.prune_endpoints(
                skeleton,
                passes=8
            )
        )

        skeleton = (
            self.largest_component(
                skeleton
            )
        )

        return skeleton

    # ---------------------------------------------------------
    # VIZINHOS
    # ---------------------------------------------------------

    @staticmethod
    def neighbors_of(
        point,
        pixels
    ):

        y, x = point

        result = []

        for dy in (
            -1,
            0,
            1
        ):

            for dx in (
                -1,
                0,
                1
            ):

                if (
                    dx == 0
                    and dy == 0
                ):
                    continue

                candidate = (
                    y + dy,
                    x + dx
                )

                if candidate in pixels:
                    result.append(
                        candidate
                    )

        return result

    # ---------------------------------------------------------
    # TRAÇAR UMA DIRECAO
    # ---------------------------------------------------------

    def trace_path(
        self,
        start,
        first,
        pixels
    ):

        path = [
            start,
            first
        ]

        visited = {
            start,
            first
        }

        previous = (
            start
        )

        current = (
            first
        )

        while len(
            visited
        ) < len(
            pixels
        ):

            candidates = [
                point
                for point in self.neighbors_of(
                    current,
                    pixels
                )
                if point not in visited
            ]

            if not candidates:
                break

            current_direction_x = (
                current[1]
                - previous[1]
            )

            current_direction_y = (
                current[0]
                - previous[0]
            )

            best_candidate = None
            best_score = -999.0

            for candidate in candidates:

                dx = (
                    candidate[1]
                    - current[1]
                )

                dy = (
                    candidate[0]
                    - current[0]
                )

                length_a = math.sqrt(
                    current_direction_x
                    * current_direction_x
                    + current_direction_y
                    * current_direction_y
                )

                length_b = math.sqrt(
                    dx * dx
                    + dy * dy
                )

                if (
                    length_a <= 0.0
                    or length_b <= 0.0
                ):

                    score = -1.0

                else:

                    score = (
                        current_direction_x * dx
                        + current_direction_y * dy
                    ) / (
                        length_a
                        * length_b
                    )

                if score > best_score:

                    best_score = score
                    best_candidate = (
                        candidate
                    )

            if best_candidate is None:
                break

            previous = current
            current = best_candidate

            visited.add(
                current
            )

            path.append(
                current
            )

        return path

    # ---------------------------------------------------------
    # ORDENAR LINHA CENTRAL
    # ---------------------------------------------------------

    def order_centerline(
        self,
        skeleton,
        spawn,
        spawn_angle
    ):

        coordinates = np.argwhere(
            skeleton
        )

        if len(
            coordinates
        ) == 0:

            return [
                spawn
            ]

        spawn_x = (
            spawn[0]
        )

        spawn_y = (
            spawn[1]
        )

        distances = (
            (
                coordinates[:, 1]
                - spawn_x
            ) ** 2

            + (
                coordinates[:, 0]
                - spawn_y
            ) ** 2
        )

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        start_y = int(
            coordinates[
                nearest_index,
                0
            ]
        )

        start_x = int(
            coordinates[
                nearest_index,
                1
            ]
        )

        start = (
            start_y,
            start_x
        )

        pixels = set(
            map(
                tuple,
                coordinates.tolist()
            )
        )

        first_options = (
            self.neighbors_of(
                start,
                pixels
            )
        )

        if not first_options:

            return [
                (
                    start_x,
                    start_y
                )
            ]

        desired_x = math.cos(
            spawn_angle
        )

        desired_y = math.sin(
            spawn_angle
        )

        paths = []

        for first in first_options:

            path = self.trace_path(
                start,
                first,
                pixels
            )

            dx = (
                first[1]
                - start[1]
            )

            dy = (
                first[0]
                - start[0]
            )

            length = math.sqrt(
                dx * dx
                + dy * dy
            )

            if length > 0:

                alignment = (
                    desired_x * dx
                    + desired_y * dy
                ) / length

            else:

                alignment = -1.0

            paths.append(
                (
                    path,
                    alignment
                )
            )

        # Preferir caminhos longos.
        longest = max(
            len(
                item[0]
            )
            for item in paths
        )

        minimum_good_length = (
            longest * 0.95
        )

        good_paths = [
            item
            for item in paths
            if len(
                item[0]
            )
            >= minimum_good_length
        ]

        # Entre caminhos praticamente iguais,
        # escolher a direcao alinhada com a largada.
        best_path, _ = max(
            good_paths,
            key=lambda item: item[1]
        )

        # Converter y,x para x,y
        return [
            (
                point[1],
                point[0]
            )
            for point in best_path
        ]

    # ---------------------------------------------------------
    # CHECKPOINTS
    # ---------------------------------------------------------

    def create_checkpoints(
        self,
        ordered_line
    ):

        if not ordered_line:

            return []

        amount = len(
            ordered_line
        )

        checkpoints = []

        # O primeiro checkpoint fica a frente
        # da largada e o ultimo volta para ela.
        for index in range(
            1,
            self.CHECKPOINT_COUNT + 1
        ):

            line_index = int(
                (
                    index
                    * amount
                )
                / self.CHECKPOINT_COUNT
            ) % amount

            x, y = (
                ordered_line[
                    line_index
                ]
            )

            checkpoints.append(
                (
                    self.rect.x + x,
                    self.rect.y + y
                )
            )

        return checkpoints

    # ---------------------------------------------------------
    # PORTAIS DE CHECKPOINT
    # ---------------------------------------------------------

    def create_checkpoint_gates(self, ordered_line):
        if not ordered_line:
            return []
        gates = []
        total = len(ordered_line)
        for k in range(1, self.CHECKPOINT_COUNT + 1):
            idx = int(k * total / self.CHECKPOINT_COUNT) % total
            px, py = ordered_line[idx]
            before = ordered_line[(idx - 14) % total]
            after = ordered_line[(idx + 14) % total]
            dx = after[0] - before[0]
            dy = after[1] - before[1]
            length = math.hypot(dx, dy)
            if length < 1.0:
                dx, dy = 1.0, 0.0
            else:
                dx, dy = dx / length, dy / length
            nx, ny = -dy, dx
            x, y = self.rect.x + px, self.rect.y + py
            sides = []
            # Mede ate a parede em ambos os lados do checkpoint.
            for sign in (-1, 1):
                width = 0.0
                for step in range(2, 125, 2):
                    if not self.is_point_on_track_xy(
                        x + nx * step * sign,
                        y + ny * step * sign
                    ):
                        width = float(step)
                        break
                else:
                    width = 124.0
                sides.append(max(10.0, width + 4.0))
            gates.append((x, y, dx, dy, nx, ny, sides[0], sides[1]))
        return gates

    def checkpoint_crossed(self, previous, current, checkpoint_index):
        gates = getattr(self, 'checkpoint_gates', ())
        if not gates:
            return False
        x, y, dx, dy, nx, ny, negative_width, positive_width = (
            gates[checkpoint_index % len(gates)]
        )
        ax, ay = previous
        bx, by = current
        before = (ax - x) * dx + (ay - y) * dy
        after = (bx - x) * dx + (by - y) * dy
        if not (before <= 0.0 <= after and after > before):
            return False
        # Interpolacao na reta do portal para medir a largura correta.
        portion = -before / (after - before)
        crossing_x = ax + (bx - ax) * portion
        crossing_y = ay + (by - ay) * portion
        side_distance = (crossing_x - x) * nx + (crossing_y - y) * ny
        return -negative_width <= side_distance <= positive_width

    # ---------------------------------------------------------
    # CHECAR CHECKPOINT
    # ---------------------------------------------------------

    def checkpoint_reached(
        self,
        position,
        checkpoint_index
    ):

        if not self.checkpoints:
            return False

        checkpoint = (
            self.checkpoints[
                checkpoint_index
                % len(
                    self.checkpoints
                )
            ]
        )

        dx = (
            position[0]
            - checkpoint[0]
        )

        dy = (
            position[1]
            - checkpoint[1]
        )

        # Sem sqrt
        return (
            dx * dx
            + dy * dy
            <= self.checkpoint_radius_squared
        )

    # ---------------------------------------------------------
    # DEBUG OPCIONAL
    # ---------------------------------------------------------

    def create_debug_surface(
        self
    ):

        if (
            self._debug_surface
            is not None
        ):

            return (
                self._debug_surface
            )

        surface = pygame.Surface(
            (
                self.screen_width,
                self.screen_height
            ),
            pygame.SRCALPHA
        )

        # Linha central
        if len(
            self.centerline
        ) >= 2:

            pygame.draw.lines(
                surface,
                (
                    0,
                    220,
                    255,
                    140
                ),
                False,
                self.centerline,
                1
            )

        # Checkpoints
        for index, checkpoint in enumerate(
            self.checkpoints
        ):

            pygame.draw.circle(
                surface,
                (
                    255,
                    200,
                    40,
                    170
                ),
                (
                    int(
                        checkpoint[0]
                    ),
                    int(
                        checkpoint[1]
                    )
                ),
                5,
                1
            )

        self._debug_surface = (
            surface
        )

        return surface

    def draw_debug(
        self,
        screen
    ):

        screen.blit(
            self.create_debug_surface(),
            (
                0,
                0
            )
        )