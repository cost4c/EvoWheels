import math
from pathlib import Path

import pygame

from brain import Brain
from curve_safety import CurveSafety


class Car:
    # O turbo prepara o sprite somente quando vai desenhar.
    RENDER_SPRITES = True


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
    SENSOR_RANGE = 180

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

    MAX_SPEED = 175.0

    # Velocidade de largada, nao e um limite minimo
    START_SPEED = 55.0

    # Motor forte e freio rapido para atacar reta e curva
    ACCELERATION = 230.0
    BRAKING = 270.0

    FRICTION = 3.0

    TURN_SPEED = 2.75

    # ---------------------------------------------------------
    # LIMITE FISICO DE MOVIMENTO
    # ---------------------------------------------------------

    # Regra fisica: o carro nunca pode parar.
    # A IA continua escolhendo a velocidade acima deste piso.
    MIN_ROLLING_SPEED = 25.0

    # Ritmo adaptativo. Nao existe velocidade-alvo fixa.
    # O sistema calcula uma velocidade segura a partir do espaco real
    # visto pelos sensores e so ajuda quando o cerebro esta muito abaixo
    # do ritmo que aquele trecho permite.
    PACE_SAFETY_BUFFER = 20.0
    PACE_BRAKE_MARGIN = 0.36
    PACE_RESPONSE_TIME = 0.60
    PACE_MAX_ASSIST = 0.90
    PACE_ASSIST_START_DISTANCE = 55.0
    PACE_FULL_ASSIST_DISTANCE = 130.0

    # A evolucao ganha pontos extras quando o proprio cerebro consegue
    # manter o ritmo sem depender da ajuda adaptativa.
    AUTONOMY_PROGRESS_REWARD = 20.0
    CHECKPOINT_AUTONOMY_REWARD = 12.0

    # ---------------------------------------------------------
    # IA
    # ---------------------------------------------------------

    # Dez decisoes por segundo
    THINK_INTERVAL = 0.10

    COMMAND_GAIN = 1.5

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

    # Depois que aprende o caminho, passar pelos mesmos pontos em menos
    # tempo vale mais. Isso faz a evolucao buscar voltas mais rapidas.
    CHECKPOINT_TIME_REWARD = 24.0
    CHECKPOINT_TIME_BONUS_MAX = 45.0
    LAP_TIME_REWARD = 1500.0
    LAP_TIME_BONUS_MAX = 500.0

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
        self.autonomy_fitness = 0.0

        # Ritmo adaptativo. A ajuda mantem a corrida viva agora, mas a
        # evolucao recebe vantagem quando aprende a nao precisar dela.
        self.dynamic_safe_speed = self.START_SPEED
        self.safety_active = False
        self.safety_interventions = 0
        self.safety_assist_ema = 0.0
        self.last_pace_assist = 0.0
        self.last_pace_assist_ratio = 0.0
        self.pace_assist_ema = 0.0
        self.pace_assist_sum = 0.0
        self.pace_assist_samples = 0
        self.last_neural_traction = 0.0

        # Metricas usadas para premiar eficiencia e guardar memoria.
        self.checkpoints_total = 0
        self.tempo_segmento = 0.0
        self.tempo_volta = 0.0
        self.melhor_tempo_volta = None

        self.tempo_sem_progresso = 0.0
        self.melhor_score_tentativa = 0.0

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
        self.autonomy_fitness = 0.0

        self.dynamic_safe_speed = self.START_SPEED
        self.safety_active = False
        self.safety_interventions = 0
        self.safety_assist_ema = 0.0
        self.last_pace_assist = 0.0
        self.last_pace_assist_ratio = 0.0
        self.pace_assist_ema = 0.0
        self.pace_assist_sum = 0.0
        self.pace_assist_samples = 0
        self.last_neural_traction = 0.0

        self.checkpoints_total = 0
        self.tempo_segmento = 0.0
        self.tempo_volta = 0.0
        self.melhor_tempo_volta = None

        self.x = float(
            track.spawn_point[0]
        )

        self.y = float(
            track.spawn_point[1]
        )

        self.previous_position = (self.x, self.y)

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

        # Uma nova vida nunca herda previsões feitas na vida anterior.
        self._curve_safety_cache = None

        self.distance_traveled = 0.0

        self.checkpoint_atual = 0

        self.voltas = 0

        self.tempo_sem_progresso = 0.0

        self.melhor_score_tentativa = 0.0

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
                    * (1.0 - 0.65 * self.safety_assist_ema)
                )

                # Se dois carros avancam parecido, o que aprendeu a manter
                # velocidade sozinho recebe vantagem evolutiva. A ajuda de
                # ritmo nao vira um atalho permanente para a rede.
                autonomia = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - self.pace_assist_ema
                    )
                )

                self.autonomy_fitness += (
                    delta_progresso
                    * self.AUTONOMY_PROGRESS_REWARD
                    * autonomia
                    * velocidade
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
            + self.autonomy_fitness
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

        reached = track.checkpoint_reached(
            (self.x, self.y), self.checkpoint_atual
        )
        if not reached and hasattr(track, "checkpoint_crossed"):
            reached = track.checkpoint_crossed(
                self.previous_position,
                (self.x, self.y),
                self.checkpoint_atual
            )
        if not reached:
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
            * (1.0 - 0.65 * self.safety_assist_ema)
        )

        autonomia = max(
            0.0,
            min(
                1.0,
                1.0 - self.pace_assist_ema
            )
        )

        self.autonomy_fitness += (
            self.CHECKPOINT_AUTONOMY_REWARD
            * autonomia
        )

        # O mesmo checkpoint vale mais quando e alcancado em menos tempo.
        # O bonus e limitado para nunca superar a importancia do progresso.
        tempo_segmento = max(
            0.25,
            self.tempo_segmento
        )

        self.speed_fitness += min(
            self.CHECKPOINT_TIME_BONUS_MAX,
            self.CHECKPOINT_TIME_REWARD / tempo_segmento
            * (1.0 - 0.65 * self.safety_assist_ema)
        )

        self.tempo_segmento = 0.0
        self.checkpoints_total += 1
        self.checkpoint_atual += 1

        if (
            self.checkpoint_atual
            >= len(
                track.checkpoints
            )
        ):

            self.voltas += 1

            tempo_volta = max(
                1.0,
                self.tempo_volta
            )

            if (
                self.melhor_tempo_volta is None
                or tempo_volta < self.melhor_tempo_volta
            ):
                self.melhor_tempo_volta = tempo_volta

            self.fitness += (
                self.LAP_REWARD
            )

            # Uma volta completa mais rapida recebe mais fitness.
            self.speed_fitness += min(
                self.LAP_TIME_BONUS_MAX,
                self.LAP_TIME_REWARD / tempo_volta
            )

            self.tempo_volta = 0.0
            self.checkpoint_atual = 0

        self.best_fitness = max(
            self.best_fitness,
            self.fitness
        )

        self.reset_segment_progress(
            track
        )

        # Salva marcos importantes no momento em que acontecem.
        # Assim um carro nao precisa morrer para a evolucao lembrar dele.
        if (
            self.evolution is not None
            and hasattr(
                self.evolution,
                "registrar_memoria"
            )
        ):
            self.evolution.registrar_memoria(
                self.brain,
                self.get_evolution_score(),
                getattr(
                    track,
                    "name",
                    None
                ),
                self.get_learning_metrics()
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
    # METRICAS DE APRENDIZADO
    # ---------------------------------------------------------

    def get_learning_metrics(
        self
    ):
        if self.pace_assist_samples > 0:
            assistencia_media = (
                self.pace_assist_sum
                / self.pace_assist_samples
            )
        else:
            assistencia_media = 0.0

        return {
            "checkpoints": self.checkpoints_total,
            "voltas": self.voltas,
            "melhor_tempo_volta": self.melhor_tempo_volta,
            "assistencia_media": assistencia_media,
            "correcoes_curva": self.safety_interventions,
            "ajuda_curva": self.safety_assist_ema,
            "autonomia_ritmo": max(
                0.0,
                min(1.0, 1.0 - assistencia_media)
            ),
        }

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
                ),
                metricas=self.get_learning_metrics()
            )
        )

        self.reset(
            track
        )

    # ---------------------------------------------------------
    # RITMO ADAPTATIVO
    # ---------------------------------------------------------

    def sensor_risk_to_distance(
        self,
        risk
    ):
        risk = max(
            0.0,
            min(1.0, float(risk))
        )

        return (
            self.SENSOR_START
            + (1.0 - risk)
            * (
                self.SENSOR_RANGE
                - self.SENSOR_START
            )
        )

    def calculate_dynamic_pace(
        self
    ):
        # Em vez de uma regra binaria de "livre/perigo", usamos a
        # distancia real percebida. O sensor central pesa mais e o lado
        # mais aberto ajuda o carro a entender que existe continuacao
        # em uma curva.
        distances = [
            self.sensor_risk_to_distance(risk)
            for risk in self.last_sensor_risks
        ]

        front = distances[3]
        near_open = max(
            distances[2],
            distances[4]
        )
        wide_open = max(
            distances[1],
            distances[5]
        )

        corridor = (
            front * 0.68
            + near_open * 0.24
            + wide_open * 0.08
        )

        usable = max(
            0.0,
            corridor - self.PACE_SAFETY_BUFFER
        )

        # Velocidade que ainda deixa espaco fisico para reduzir usando
        # a capacidade real de frenagem do carro.
        safe_speed = math.sqrt(
            2.0
            * self.BRAKING
            * usable
            * self.PACE_BRAKE_MARGIN
        )

        safe_speed = max(
            self.MIN_ROLLING_SPEED,
            min(self.MAX_SPEED, safe_speed)
        )

        span = max(
            1.0,
            self.PACE_FULL_ASSIST_DISTANCE
            - self.PACE_ASSIST_START_DISTANCE
        )

        confidence = (
            corridor
            - self.PACE_ASSIST_START_DISTANCE
        ) / span

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return safe_speed, confidence

    def calculate_pace_assist(
        self,
        neural_traction
    ):
        safe_speed, confidence = (
            self.calculate_dynamic_pace()
        )

        self.dynamic_safe_speed = safe_speed
        self.last_neural_traction = neural_traction

        speed_gap = max(
            0.0,
            safe_speed - self.speed
        )

        if speed_gap <= 0.0 or confidence <= 0.0:
            self.last_pace_assist = 0.0
            self.last_pace_assist_ratio = 0.0
            return 0.0

        desired_acceleration = (
            speed_gap
            / self.PACE_RESPONSE_TIME
        )

        assist_floor = (
            desired_acceleration
            + self.FRICTION
        ) / self.ACCELERATION

        assist_floor = max(
            0.0,
            min(
                self.PACE_MAX_ASSIST,
                assist_floor
            )
        )

        # A confianca cresce de forma continua conforme aumenta o espaco
        # disponivel. Nao existe mais o antigo corte seco de risco 0.30.
        assist_floor *= confidence

        assist_used = max(
            0.0,
            assist_floor - neural_traction
        )

        if assist_floor > 0.02:
            assist_ratio = max(
                0.0,
                min(
                    1.0,
                    assist_used / assist_floor
                )
            )

            self.pace_assist_ema = (
                self.pace_assist_ema * 0.92
                + assist_ratio * 0.08
            )

            self.pace_assist_sum += assist_ratio
            self.pace_assist_samples += 1
        else:
            assist_ratio = 0.0

        self.last_pace_assist = assist_used
        self.last_pace_assist_ratio = assist_ratio

        return assist_floor

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

        # A rede continua decidindo aceleracao e freio.
        tracao_neural = (
            acelerar
            - frear
        )

        # O ritmo adaptativo nao manda fazer uma velocidade fixa. Ele usa
        # os 180 px de visao e a capacidade de frenagem para calcular
        # quanto aquele trecho realmente permite. Se o cerebro ja acelera
        # o suficiente, nenhuma ajuda e aplicada.
        pace_floor = self.calculate_pace_assist(
            tracao_neural
        )

        tracao_desejada = max(
            tracao_neural,
            pace_floor
        )

        tracao = (
            self.last_traction
            * 0.30

            + tracao_desejada
            * 0.70
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

        if abs(
            direcao_desejada
        ) < 0.02:

            direcao_desejada = 0.0

        # Olha a curvatura dos proximos trechos: frear antes e melhor
        # do que perceber a parede quando ja nao e possivel virar.
        limite_curva = CurveSafety.curve_speed_limit(self, track)
        tracao_limitada = CurveSafety.cap_traction(self, tracao, limite_curva)
        reduziu_na_curva = tracao_limitada < tracao - 0.01
        tracao = tracao_limitada

        # A direcao permanece neural se a trajetoria prevista for livre.
        # So procura alternativas se a colisao estiver prevista.
        direcao_desejada, tracao_segura, precisa_ajuda, _ = CurveSafety.choose(
            self, track, direcao_desejada, tracao
        )

        self.safety_active = precisa_ajuda or reduziu_na_curva
        self.safety_assist_ema = (
            self.safety_assist_ema * 0.96
            + (1.0 if self.safety_active else 0.0) * 0.04
        )
        if self.safety_active:
            self.safety_interventions += 1
            # Desliga a assistencia de ritmo na curva critica.
            # Sem filtro de tracao: o freio precisa agir imediatamente.
            self.last_traction = tracao_segura
            self.last_pace_assist = 0.0
            self.last_pace_assist_ratio = 0.0
            self.accelerator = max(0.0, tracao_segura)
            self.brake = max(0.0, -tracao_segura)
            self.dynamic_safe_speed = min(
                self.dynamic_safe_speed, self.speed
            )

        elif reduziu_na_curva:
            self.last_traction = tracao
            self.accelerator = max(0.0, tracao)
            self.brake = max(0.0, -tracao)
            self.last_pace_assist = 0.0
            self.last_pace_assist_ratio = 0.0

        self.steering = (
            self.steering * 0.25
            + direcao_desejada * 0.75
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
        self.tempo_segmento += dt
        self.tempo_volta += dt

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
        # APLICAR VELOCIDADE
        # -----------------------------------------------------

        self.speed += (
            acceleration
            * dt
        )

        # Regra absoluta: NAO PODE PARAR.
        # Abaixo de 25 px/s, a velocidade fica em 25 px/s.
        if (
            self.speed
            < self.MIN_ROLLING_SPEED
        ):

            self.speed = (
                self.MIN_ROLLING_SPEED
            )

        if (
            self.speed
            > self.MAX_SPEED
        ):

            self.speed = (
                self.MAX_SPEED
            )

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

        if self.RENDER_SPRITES:
            self.update_image()

        # -----------------------------------------------------
        # MOVIMENTO
        # -----------------------------------------------------

        self.previous_position = (self.x, self.y)

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

        if not self.RENDER_SPRITES:
            self.update_image()

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