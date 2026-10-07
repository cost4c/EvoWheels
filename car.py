import math
from pathlib import Path

import pygame

from brain import Brain


class Car:

    # ---------------------------------------------------------
    # SENSORES SOMENTE NA FRENTE
    # ---------------------------------------------------------

    SENSOR_ANGLES = (
        -70,
        -45,
        -22,
        0,
        22,
        45,
        70,
    )

    SENSOR_ANGLES_RAD = tuple(
        math.radians(angle)
        for angle in SENSOR_ANGLES
    )

    SENSOR_DIRECTIONS = tuple(
        (
            math.cos(angle),
            math.sin(angle),
        )
        for angle in SENSOR_ANGLES_RAD
    )

    # Alcance pedido
    SENSOR_RANGE = 100

    # Comecar um pouco fora do centro
    SENSOR_START = 10

    # Testar de 5 em 5 pixels
    SENSOR_STEP = 5

    SENSOR_DISTANCES = tuple(
        range(
            SENSOR_START,
            SENSOR_RANGE + 1,
            SENSOR_STEP
        )
    )

    # ---------------------------------------------------------
    # VELOCIDADE
    # ---------------------------------------------------------

    # Nao existe velocidade minima fixa.
    MAX_SPEED = 175.0

    # Velocidade de largada, nao e um limite minimo
    START_SPEED = 90.0

    # Motor forte e freio rapido para atacar reta e curva
    ACCELERATION = 340.0
    BRAKING = 390.0

    FRICTION = 1.5

    TURN_SPEED = 2.75

    # ---------------------------------------------------------
    # SISTEMA ANTI ARRASTO
    # ---------------------------------------------------------

    # Abaixo disso o motor ajuda o carro
    # a recuperar movimento.
    ANTI_CRAWL_SPEED = 22.0

    # Forca extra progressiva.
    ANTI_CRAWL_ACCELERATION = 130.0

    # Se insistir em se arrastar, a tentativa termina
    CRAWL_SPEED = 10.0

    CRAWL_TIME_LIMIT = 0.90

    # Se praticamente parar
    STOPPED_SPEED = 2.5

    STOPPED_TIME_LIMIT = 0.35

    # ---------------------------------------------------------
    # IA
    # ---------------------------------------------------------

    # Dez decisoes por segundo
    THINK_INTERVAL = 0.10

    COMMAND_GAIN = 1.5

    # Instinto de corrida. Nao fixa velocidade.
    # Mantem tracao forte enquanto os sensores mostram pista livre.
    CLEAR_THROTTLE = 0.78
    CAUTION_THROTTLE = 0.52
    APPROACH_THROTTLE = 0.28

    CLEAR_RISK = 0.35
    CAUTION_RISK = 0.55
    DANGER_RISK = 0.72

    # Em pista livre, reduz esterco aleatorio para evitar circulos.
    # Quando aparece parede, devolve autoridade ao cerebro.
    CLEAR_STEERING_AUTHORITY = 0.18

    # ---------------------------------------------------------
    # ANTI LOOP
    # ---------------------------------------------------------

    NO_PROGRESS_LIMIT = 1.8

    MIN_PROGRESS_DELTA = 0.25

    # ---------------------------------------------------------
    # FITNESS
    # ---------------------------------------------------------

    CHECKPOINT_REWARD = 100
    LAP_REWARD = 1000

    # Bonus por avancar rapido sem premiar velocidade inutil
    SPEED_PROGRESS_REWARD = 35.0
    CHECKPOINT_SPEED_REWARD = 15.0

    # ---------------------------------------------------------
    # TAMANHO DO CARRO
    # ---------------------------------------------------------

    CAR_LENGTH = 28

    HITBOX_HALF_LENGTH = 11.5
    HITBOX_HALF_WIDTH = 5.0

    # ---------------------------------------------------------
    # CACHE DA IMAGEM
    # ---------------------------------------------------------

    base_image = None

    rotation_cache = {}

    cache_ready = False

    # ---------------------------------------------------------
    # INICIALIZAR
    # ---------------------------------------------------------

    def __init__(
        self,
        track,
        identifier,
        evolution
    ):

        self.identifier = (
            identifier
        )

        self.evolution = (
            evolution
        )

        # Usa o aprendizado salvo quando ele ja existe
        if (
            self.evolution is not None
            and hasattr(
                self.evolution,
                "criar_cerebro_inicial"
            )
        ):

            self.brain = (
                self.evolution.criar_cerebro_inicial(
                    getattr(
                        track,
                        "name",
                        None
                    )
                )
            )

        else:

            self.brain = Brain()

        # Fitness
        self.fitness = 0.0
        self.best_fitness = 0.0

        # Corrida
        self.checkpoint_atual = 0
        self.voltas = 0
        self.deaths = 0

        # Progresso
        self.distancia_inicio_segmento = 1.0
        self.menor_distancia_segmento = 1.0

        # Velocidade so vale quando existe progresso real
        self.speed_fitness = 0.0

        self.tempo_sem_progresso = 0.0
        self.melhor_score_tentativa = 0.0

        # Anti arrasto
        self.tempo_arrastando = 0.0
        self.tempo_parado = 0.0

        # 7 sensores
        self.last_sensor_risks = [
            0.0
        ] * len(
            self.SENSOR_ANGLES
        )

        # -----------------------------------------------------
        # IMAGEM
        # -----------------------------------------------------

        if Car.base_image is None:

            Car.base_image = (
                self.load_car_sprite()
            )

        self.original_image = (
            Car.base_image
        )

        if not Car.cache_ready:

            self.create_rotation_cache()

        self.image = (
            self.original_image
        )

        self.sprite_angle = None

        self.reset(
            track
        )

    # ---------------------------------------------------------
    # CARREGAR IMAGEM
    # ---------------------------------------------------------

    def load_car_sprite(
        self
    ):

        car_path = (
            Path(__file__).resolve().parent
            / "assets"
            / "car"
            / "car1.png"
        )

        if not car_path.exists():

            raise FileNotFoundError(
                f"Imagem do carro nao encontrada: "
                f"{car_path}"
            )

        image = pygame.image.load(
            str(car_path)
        ).convert_alpha()

        bounds = (
            image.get_bounding_rect(
                min_alpha=80
            )
        )

        if (
            bounds.width > 0
            and bounds.height > 0
        ):

            image = (
                image.subsurface(
                    bounds
                ).copy()
            )

        # Frente para direita
        image = pygame.transform.rotate(
            image,
            -90
        )

        width, height = (
            image.get_size()
        )

        scale = (
            self.CAR_LENGTH
            / width
        )

        new_size = (
            self.CAR_LENGTH,

            max(
                1,
                round(
                    height
                    * scale
                )
            )
        )

        return pygame.transform.smoothscale(
            image,
            new_size
        )

    # ---------------------------------------------------------
    # CACHE DE ROTACOES
    # ---------------------------------------------------------

    def create_rotation_cache(
        self
    ):

        for angle in range(
            360
        ):

            Car.rotation_cache[
                angle
            ] = pygame.transform.rotate(
                self.original_image,
                -angle
            )

        Car.cache_ready = True

    # ---------------------------------------------------------
    # ATUALIZAR IMAGEM
    # ---------------------------------------------------------

    def update_image(
        self
    ):

        angle_degrees = (
            round(
                math.degrees(
                    self.angle
                )
            )
            % 360
        )

        if (
            self.sprite_angle
            == angle_degrees
        ):
            return

        self.image = (
            Car.rotation_cache[
                angle_degrees
            ]
        )

        self.sprite_angle = (
            angle_degrees
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(
        self,
        track
    ):

        score_anterior = (
            self.get_evolution_score()
        )

        self.best_fitness = max(
            self.best_fitness,
            score_anterior
        )

        self.fitness = 0.0
        self.speed_fitness = 0.0

        self.x = float(
            track.spawn_point[0]
        )

        self.y = float(
            track.spawn_point[1]
        )

        self.angle = float(
            track.spawn_angle
        )

        # Isso e apenas a velocidade de largada.
        # Nao e velocidade obrigatoria.
        self.speed = (
            self.START_SPEED
        )

        self.steering = 0.0

        self.accelerator = 0.0

        self.brake = 0.0

        self.last_traction = 0.0

        # Distribuir pensamento entre frames
        self.think_timer = (
            self.identifier % 10
        ) * (
            self.THINK_INTERVAL / 10.0
        )

        self.time_alive = 0.0

        self.distance_traveled = 0.0

        self.checkpoint_atual = 0

        self.voltas = 0

        self.tempo_sem_progresso = 0.0

        self.melhor_score_tentativa = 0.0

        self.tempo_arrastando = 0.0

        self.tempo_parado = 0.0

        self.last_sensor_risks = [
            0.0
        ] * len(
            self.SENSOR_ANGLES
        )

        self.reset_segment_progress(
            track
        )

        self.sprite_angle = None

        self.update_image()

    # ---------------------------------------------------------
    # SENSORES
    # ---------------------------------------------------------

    def get_sensors(
        self,
        track
    ):

        readings = []

        risks = []

        # Calcular apenas uma vez
        cos_car = math.cos(
            self.angle
        )

        sin_car = math.sin(
            self.angle
        )

        for (
            relative_cos,
            relative_sin
        ) in self.SENSOR_DIRECTIONS:

            direction_x = (
                cos_car
                * relative_cos

                - sin_car
                * relative_sin
            )

            direction_y = (
                sin_car
                * relative_cos

                + cos_car
                * relative_sin
            )

            # 0 = seguro
            risco = 0.0

            for distance in (
                self.SENSOR_DISTANCES
            ):

                x = (
                    self.x
                    + direction_x
                    * distance
                )

                y = (
                    self.y
                    + direction_y
                    * distance
                )

                if not track.is_point_on_track_xy(
                    x,
                    y
                ):

                    distancia_util = (
                        distance
                        - self.SENSOR_START
                    )

                    alcance_util = (
                        self.SENSOR_RANGE
                        - self.SENSOR_START
                    )

                    livre = (
                        distancia_util
                        / alcance_util
                    )

                    # Quanto mais perto,
                    # maior o risco
                    risco = (
                        1.0 - livre
                    )

                    risco = max(
                        0.0,
                        min(
                            1.0,
                            risco
                        )
                    )

                    break

            readings.append(
                risco
            )

            risks.append(
                risco
            )

        self.last_sensor_risks = (
            risks
        )

        # -----------------------------------------------------
        # ENTRADA 8
        # VELOCIDADE
        # -----------------------------------------------------

        readings.append(
            self.speed
            / self.MAX_SPEED
        )

        # -----------------------------------------------------
        # ENTRADA 9
        # DIRECAO ANTERIOR
        # -----------------------------------------------------

        readings.append(
            self.steering
        )

        # -----------------------------------------------------
        # ENTRADA 10
        # TRACAO ANTERIOR
        # -----------------------------------------------------

        readings.append(
            self.last_traction
        )

        return readings

    # ---------------------------------------------------------
    # DISTANCIA PARA CHECKPOINT
    # ---------------------------------------------------------

    def distance_to_checkpoint(
        self,
        track
    ):

        if not track.checkpoints:

            return 0.0

        checkpoint = (
            track.checkpoints[
                self.checkpoint_atual
            ]
        )

        dx = (
            self.x
            - checkpoint[0]
        )

        dy = (
            self.y
            - checkpoint[1]
        )

        return math.sqrt(
            dx * dx
            + dy * dy
        )

    # ---------------------------------------------------------
    # INICIAR PROGRESSO DO SEGMENTO
    # ---------------------------------------------------------

    def reset_segment_progress(
        self,
        track
    ):

        if not track.checkpoints:

            self.distancia_inicio_segmento = 1.0

            self.menor_distancia_segmento = 1.0

            return

        distancia = (
            self.distance_to_checkpoint(
                track
            )
        )

        self.distancia_inicio_segmento = max(
            1.0,
            distancia
        )

        self.menor_distancia_segmento = (
            self.distancia_inicio_segmento
        )

    # ---------------------------------------------------------
    # ATUALIZAR PROGRESSO
    # ---------------------------------------------------------

    def update_segment_progress(
        self,
        track
    ):

        if not track.checkpoints:
            return

        distancia = (
            self.distance_to_checkpoint(
                track
            )
        )

        if (
            distancia
            < self.menor_distancia_segmento
        ):

            inicio = max(
                1.0,
                self.distancia_inicio_segmento
            )

            progresso_anterior = (
                inicio
                - self.menor_distancia_segmento
            ) / inicio

            self.menor_distancia_segmento = (
                distancia
            )

            progresso_atual = (
                inicio
                - self.menor_distancia_segmento
            ) / inicio

            progresso_anterior = max(
                0.0,
                min(0.999, progresso_anterior)
            )

            progresso_atual = max(
                0.0,
                min(0.999, progresso_atual)
            )

            delta_progresso = max(
                0.0,
                progresso_atual
                - progresso_anterior
            )

            # Premiar velocidade apenas quando esta indo para frente na pista
            if delta_progresso > 0.0:

                velocidade = max(
                    0.0,
                    min(
                        1.0,
                        self.speed / self.MAX_SPEED
                    )
                )

                self.speed_fitness += (
                    delta_progresso
                    * self.SPEED_PROGRESS_REWARD
                    * (velocidade ** 1.5)
                )

        score = (
            self.get_evolution_score()
        )

        if (
            score
            > self.best_fitness
        ):

            self.best_fitness = (
                score
            )

    # ---------------------------------------------------------
    # SCORE
    # ---------------------------------------------------------

    def get_evolution_score(
        self
    ):

        inicio = max(
            1.0,
            self.distancia_inicio_segmento
        )

        progresso = (
            inicio
            - self.menor_distancia_segmento
        ) / inicio

        progresso = max(
            0.0,
            min(
                0.999,
                progresso
            )
        )

        parcial = (
            progresso
            * self.CHECKPOINT_REWARD
        )

        return (
            float(
                self.fitness
            )
            + self.speed_fitness
            + parcial
        )

    # ---------------------------------------------------------
    # CHECKPOINT
    # ---------------------------------------------------------

    def update_checkpoint(
        self,
        track
    ):

        if not track.checkpoints:
            return

        if not track.checkpoint_reached(
            (
                self.x,
                self.y
            ),
            self.checkpoint_atual
        ):

            return

        self.fitness += (
            self.CHECKPOINT_REWARD
        )

        # Cruzar o checkpoint rapido vale mais do que apenas correr sem rumo
        velocidade = max(
            0.0,
            min(
                1.0,
                self.speed / self.MAX_SPEED
            )
        )

        self.speed_fitness += (
            self.CHECKPOINT_SPEED_REWARD
            * (velocidade ** 1.5)
        )

        self.checkpoint_atual += 1

        if (
            self.checkpoint_atual
            >= len(
                track.checkpoints
            )
        ):

            self.voltas += 1

            self.fitness += (
                self.LAP_REWARD
            )

            self.checkpoint_atual = 0

        self.best_fitness = max(
            self.best_fitness,
            self.fitness
        )

        self.reset_segment_progress(
            track
        )

    # ---------------------------------------------------------
    # COLISAO
    # ---------------------------------------------------------

    def is_on_track(
        self,
        track
    ):

        cos_a = math.cos(
            self.angle
        )

        sin_a = math.sin(
            self.angle
        )

        side_x = (
            -sin_a
        )

        side_y = (
            cos_a
        )

        front_x = (
            self.x
            + cos_a
            * self.HITBOX_HALF_LENGTH
        )

        front_y = (
            self.y
            + sin_a
            * self.HITBOX_HALF_LENGTH
        )

        rear_x = (
            self.x
            - cos_a
            * self.HITBOX_HALF_LENGTH
        )

        rear_y = (
            self.y
            - sin_a
            * self.HITBOX_HALF_LENGTH
        )

        width = (
            self.HITBOX_HALF_WIDTH
        )

        points = (
            # Centro
            (
                self.x,
                self.y
            ),

            # Frente
            (
                front_x,
                front_y
            ),

            # Traseira
            (
                rear_x,
                rear_y
            ),

            # Frente esquerda
            (
                front_x
                + side_x * width,

                front_y
                + side_y * width
            ),

            # Frente direita
            (
                front_x
                - side_x * width,

                front_y
                - side_y * width
            ),

            # Traseira esquerda
            (
                rear_x
                + side_x * width,

                rear_y
                + side_y * width
            ),

            # Traseira direita
            (
                rear_x
                - side_x * width,

                rear_y
                - side_y * width
            ),
        )

        for x, y in points:

            if not track.is_point_on_track_xy(
                x,
                y
            ):

                return False

        return True

    # ---------------------------------------------------------
    # MORTE
    # ---------------------------------------------------------

    def die(
        self,
        track
    ):

        score = (
            self.get_evolution_score()
        )

        self.best_fitness = max(
            self.best_fitness,
            score
        )

        self.deaths += 1

        # Receber novo cerebro com conhecimento
        # da elite da populacao
        self.brain = (
            self.evolution.criar_descendente(
                self.brain,
                score,
                getattr(
                    track,
                    "name",
                    None
                )
            )
        )

        self.reset(
            track
        )

    # ---------------------------------------------------------
    # DECISAO DA IA
    # ---------------------------------------------------------

    def think(
        self,
        track
    ):

        entradas = (
            self.get_sensors(
                track
            )
        )

        decisao = (
            self.brain.pensar(
                entradas
            )
        )

        acelerar = max(
            0.0,
            min(
                1.0,
                decisao[
                    "acelerar"
                ]
                * self.COMMAND_GAIN
            )
        )

        frear = max(
            0.0,
            min(
                1.0,
                decisao[
                    "frear"
                ]
                * self.COMMAND_GAIN
            )
        )

        # Aceleracao base de corrida.
        # O cerebro continua livre para frear quando o risco aumenta.
        riscos = self.last_sensor_risks

        risco_frente = max(
            riscos[2],
            riscos[3],
            riscos[4],
        )

        if risco_frente < self.CLEAR_RISK:
            tracao_base = self.CLEAR_THROTTLE
        elif risco_frente < self.CAUTION_RISK:
            tracao_base = self.CAUTION_THROTTLE
        elif risco_frente < self.DANGER_RISK:
            tracao_base = self.APPROACH_THROTTLE
        else:
            tracao_base = 0.0

        tracao_neural = (
            acelerar
            - frear
        )

        # Em pista livre, todos correm.
        # Perto da parede, a rede neural assume o controle total.
        if risco_frente < self.DANGER_RISK:
            tracao_desejada = max(
                tracao_neural,
                tracao_base
            )
        else:
            tracao_desejada = tracao_neural

        tracao = (
            self.last_traction
            * 0.22

            + tracao_desejada
            * 0.78
        )

        if abs(
            tracao
        ) < 0.02:

            tracao = 0.0

        self.last_traction = (
            tracao
        )

        if tracao >= 0.0:

            self.accelerator = (
                tracao
            )

            self.brake = 0.0

        else:

            self.accelerator = 0.0

            self.brake = (
                -tracao
            )

        # -----------------------------------------------------
        # DIRECAO
        # -----------------------------------------------------

        direcao_desejada = max(
            -1.0,
            min(
                1.0,
                decisao[
                    "virar"
                ]
                * self.COMMAND_GAIN
            )
        )

        # Sem parede por perto, o carro tende a atacar em linha reta.
        # O sensor nao escolhe o lado da curva; apenas libera mais esterco.
        risco_lateral = max(
            riscos[0],
            riscos[1],
            riscos[5],
            riscos[6],
        )

        risco_direcao = max(
            risco_frente,
            risco_lateral
        )

        autoridade = (
            self.CLEAR_STEERING_AUTHORITY
            + (1.0 - self.CLEAR_STEERING_AUTHORITY)
            * min(1.0, risco_direcao / self.DANGER_RISK)
        )

        direcao_desejada *= autoridade

        if abs(
            direcao_desejada
        ) < 0.025:

            direcao_desejada = 0.0

        self.steering = (
            self.steering
            * 0.18

            + direcao_desejada
            * 0.82
        )

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    def update(
        self,
        dt,
        track
    ):

        dt = max(
            0.0,
            min(
                dt,
                0.05
            )
        )

        if dt <= 0.0:
            return

        self.time_alive += dt

        # -----------------------------------------------------
        # CEREBRO
        # -----------------------------------------------------

        self.think_timer -= dt

        if (
            self.think_timer
            <= 0.0
        ):

            self.think(
                track
            )

            self.think_timer += (
                self.THINK_INTERVAL
            )

            if (
                self.think_timer
                <= 0.0
            ):

                self.think_timer = (
                    self.THINK_INTERVAL
                )

        # -----------------------------------------------------
        # FISICA NORMAL
        # -----------------------------------------------------

        acceleration = (
            self.accelerator
            * self.ACCELERATION

            - self.brake
            * self.BRAKING

            - self.FRICTION
        )

        # -----------------------------------------------------
        # ANTI ARRASTO
        # -----------------------------------------------------

        # Isso NAO fixa a velocidade.
        #
        # Apenas da uma ajuda progressiva
        # caso o carro esteja ficando lento demais.
        if (
            self.speed
            < self.ANTI_CRAWL_SPEED
        ):

            falta_velocidade = (
                self.ANTI_CRAWL_SPEED
                - self.speed
            )

            proporcao = (
                falta_velocidade
                / self.ANTI_CRAWL_SPEED
            )

            ajuda = (
                proporcao
                * self.ANTI_CRAWL_ACCELERATION
            )

            acceleration += (
                ajuda
            )

        # -----------------------------------------------------
        # APLICAR VELOCIDADE
        # -----------------------------------------------------

        self.speed += (
            acceleration
            * dt
        )

        # Nunca pode andar para tras
        if self.speed < 0.0:

            self.speed = 0.0

        if (
            self.speed
            > self.MAX_SPEED
        ):

            self.speed = (
                self.MAX_SPEED
            )

        # -----------------------------------------------------
        # DETECTAR ARRASTO
        # -----------------------------------------------------

        if (
            self.speed
            < self.CRAWL_SPEED
        ):

            self.tempo_arrastando += (
                dt
            )

        else:

            self.tempo_arrastando = 0.0

        # Insistiu em andar muito devagar
        if (
            self.tempo_arrastando
            >= self.CRAWL_TIME_LIMIT
        ):

            self.die(
                track
            )

            return

        # -----------------------------------------------------
        # DETECTAR PARADO
        # -----------------------------------------------------

        if (
            self.speed
            < self.STOPPED_SPEED
        ):

            self.tempo_parado += (
                dt
            )

        else:

            self.tempo_parado = 0.0

        if (
            self.tempo_parado
            >= self.STOPPED_TIME_LIMIT
        ):

            self.die(
                track
            )

            return

        # -----------------------------------------------------
        # DIRECAO
        # -----------------------------------------------------

        speed_ratio = (
            self.speed
            / self.MAX_SPEED
        )

        steering_control = (
            1.0
            - 0.48
            * speed_ratio
        )

        steering_control = max(
            0.52,
            steering_control
        )

        self.angle += (
            self.steering
            * self.TURN_SPEED
            * steering_control
            * dt
        )

        self.update_image()

        # -----------------------------------------------------
        # MOVIMENTO
        # -----------------------------------------------------

        distance = (
            self.speed
            * dt
        )

        steps = max(
            1,
            math.ceil(
                distance / 4.0
            )
        )

        step_distance = (
            distance / steps
        )

        cos_a = math.cos(
            self.angle
        )

        sin_a = math.sin(
            self.angle
        )

        dx = (
            cos_a
            * step_distance
        )

        dy = (
            sin_a
            * step_distance
        )

        for _ in range(
            steps
        ):

            self.x += dx
            self.y += dy

            if not self.is_on_track(
                track
            ):

                self.die(
                    track
                )

                return

        self.distance_traveled += (
            distance
        )

        # -----------------------------------------------------
        # PROGRESSO
        # -----------------------------------------------------

        self.update_segment_progress(
            track
        )

        self.update_checkpoint(
            track
        )

        score_atual = (
            self.get_evolution_score()
        )

        # -----------------------------------------------------
        # ANTI LOOP
        # -----------------------------------------------------

        if (
            score_atual
            >= self.melhor_score_tentativa
            + self.MIN_PROGRESS_DELTA
        ):

            self.melhor_score_tentativa = (
                score_atual
            )

            self.tempo_sem_progresso = 0.0

        else:

            self.tempo_sem_progresso += (
                dt
            )

        if (
            self.tempo_sem_progresso
            >= self.NO_PROGRESS_LIMIT
        ):

            self.die(
                track
            )

            return

    # ---------------------------------------------------------
    # DESENHAR
    # ---------------------------------------------------------

    def draw(
        self,
        screen
    ):

        rect = (
            self.image.get_rect(
                center=(
                    round(
                        self.x
                    ),
                    round(
                        self.y
                    )
                )
            )
        )

        screen.blit(
            self.image,
            rect
        )