# Importar as bibliotecas
import math
from pathlib import Path

import pygame

# Importar o cerebro do carrinho
from brain import Brain


# Classe do carrinho
class Car:

    # Angulos dos sensores em relacao a frente do carro
    SENSOR_ANGLES = (
        -160, -125, -95, -70, -45, -22,
        0,
        22, 45, 70, 95, 125, 160
    )

    # Converter os angulos para radianos apenas uma vez
    SENSOR_ANGLES_RAD = tuple(
        math.radians(angle)
        for angle in SENSOR_ANGLES
    )

    # Sensores
    SENSOR_RANGE = 170
    SENSOR_STEP = 4

    # Movimento
    MAX_SPEED = 175.0
    START_SPEED = 85.0

    ACCELERATION = 240.0
    BRAKING = 310.0
    FRICTION = 8.0
    TURN_SPEED = 2.7

    # O cerebro pensa 10 vezes por segundo
    THINK_INTERVAL = 0.10

    # Aumentar a resposta inicial da rede
    COMMAND_GAIN = 8.0

    # Tempo maximo parado
    STOPPED_LIMIT = 2.0

    # Pontuacao
    CHECKPOINT_REWARD = 100
    LAP_REWARD = 1000

    # Tamanho do carro dentro do jogo
    CAR_LENGTH = 28

    # Imagem principal compartilhada por todos os carros
    base_image = None

    # Cache das 360 rotacoes
    rotation_cache = {}
    mask_cache = {}
    cache_ready = False

    def __init__(self, track, identifier):

        self.identifier = identifier

        # Cada carro possui seu proprio cerebro
        # O cerebro nao e apagado quando o carro bate
        self.brain = Brain()

        # Pontuacao
        self.fitness = 0
        self.best_fitness = 0

        # Progresso na pista
        self.checkpoint_atual = 0
        self.voltas = 0

        # Quantidade de mortes
        self.deaths = 0

        # Carregar a imagem do carro apenas uma vez
        if Car.base_image is None:
            Car.base_image = self.load_car_sprite()

        self.original_image = Car.base_image

        # Tamanho real depois do redimensionamento
        self.width, self.height = (
            self.original_image.get_size()
        )

        # Criar as 360 rotacoes apenas uma vez
        if not Car.cache_ready:
            self.create_rotation_cache()

        self.image = self.original_image

        self.car_mask = pygame.mask.from_surface(
            self.original_image,
            180
        )

        self.sprite_angle = None

        # Colocar o carro na pista
        self.reset(track)

    # ---------------------------------------------------------
    # CARREGAR IMAGEM DO CARRO
    # ---------------------------------------------------------

    def load_car_sprite(self):

        # Caminho do arquivo
        car_path = (
            Path(__file__).resolve().parent
            / "assets"
            / "car"
            / "car1.png"
        )

        # Verificar se a imagem existe
        if not car_path.exists():
            raise FileNotFoundError(
                f"Imagem do carro nao encontrada: {car_path}"
            )

        # Carregar mantendo transparencia
        image = pygame.image.load(
            str(car_path)
        ).convert_alpha()

        # Remover espaco transparente desnecessario
        bounds = image.get_bounding_rect(
            min_alpha=80
        )

        if bounds.width > 0 and bounds.height > 0:
            image = image.subsurface(
                bounds
            ).copy()

        # A imagem esta com a frente apontando para cima.
        # No jogo, angulo 0 aponta para a direita.
        image = pygame.transform.rotate(
            image,
            -90
        )

        # Manter proporcao ao diminuir o carro
        width, height = image.get_size()

        scale = (
            self.CAR_LENGTH
            / width
        )

        new_size = (
            self.CAR_LENGTH,
            max(
                1,
                round(height * scale)
            )
        )

        # Redimensionar com boa qualidade
        image = pygame.transform.smoothscale(
            image,
            new_size
        )

        return image

    # ---------------------------------------------------------
    # CRIAR CACHE DAS ROTACOES
    # ---------------------------------------------------------

    def create_rotation_cache(self):

        # Criar uma versao do carro para cada grau
        for angle in range(360):

            image = pygame.transform.rotate(
                self.original_image,
                -angle
            )

            # Criar mascara ignorando partes transparentes
            mask = pygame.mask.from_surface(
                image,
                180
            )

            Car.rotation_cache[angle] = image
            Car.mask_cache[angle] = mask

        Car.cache_ready = True

    # ---------------------------------------------------------
    # REINICIAR O CARRO
    # ---------------------------------------------------------

    def reset(self, track):

        # Guardar melhor pontuacao
        self.best_fitness = max(
            self.best_fitness,
            self.fitness
        )

        # Nova tentativa começa com zero
        self.fitness = 0

        # Voltar para a largada
        self.x, self.y = map(
            float,
            track.spawn_point
        )

        self.angle = float(
            track.spawn_angle
        )

        # Velocidade inicial
        self.speed = self.START_SPEED

        # Comandos
        self.steering = 0.0
        self.accelerator = 0.0
        self.brake = 0.0

        self.last_traction = 0.0

        # Espalhar o processamento dos cerebros
        # para os carros nao pensarem todos juntos
        self.think_timer = (
            self.identifier % 10
        ) * (
            self.THINK_INTERVAL / 10
        )

        # Tempos
        self.time_alive = 0.0
        self.time_stopped = 0.0

        # Distancia da tentativa atual
        self.distance_traveled = 0.0

        # Progresso
        self.checkpoint_atual = 0
        self.voltas = 0

        # Atualizar imagem inicial
        self.sprite_angle = None
        self.update_image()

    # ---------------------------------------------------------
    # ATUALIZAR IMAGEM
    # ---------------------------------------------------------

    def update_image(self):

        # Converter o angulo para graus de 0 a 359
        angle_degrees = (
            round(math.degrees(self.angle))
            % 360
        )

        # Nao atualizar se o angulo nao mudou
        if self.sprite_angle == angle_degrees:
            return

        # Usar imagem pronta do cache
        self.image = Car.rotation_cache[
            angle_degrees
        ]

        # Usar mascara pronta do cache
        self.car_mask = Car.mask_cache[
            angle_degrees
        ]

        self.sprite_angle = angle_degrees

    # ---------------------------------------------------------
    # SENSORES
    # ---------------------------------------------------------

    def get_sensors(self, track):

        readings = []

        # Criar os 13 sensores
        for relative_angle in self.SENSOR_ANGLES_RAD:

            ray_angle = (
                self.angle
                + relative_angle
            )

            direction_x = math.cos(
                ray_angle
            )

            direction_y = math.sin(
                ray_angle
            )

            # 1 significa que nao encontrou borda
            reading = 1.0

            for distance in range(
                0,
                self.SENSOR_RANGE + 1,
                self.SENSOR_STEP
            ):

                point = (
                    self.x
                    + direction_x * distance,

                    self.y
                    + direction_y * distance
                )

                # Encontrou a borda da pista
                if not track.is_point_on_track(
                    point
                ):

                    reading = (
                        distance
                        / self.SENSOR_RANGE
                    )

                    break

            readings.append(
                reading
            )

        # Entrada 14
        # Velocidade atual
        readings.append(
            self.speed / self.MAX_SPEED
        )

        # Entrada 15
        # Direcao usada anteriormente
        readings.append(
            self.steering
        )

        # Entrada 16
        # Aceleracao menos freio
        readings.append(
            self.last_traction
        )

        return readings

    # ---------------------------------------------------------
    # CHECKPOINTS
    # ---------------------------------------------------------

    def update_checkpoint(self, track):

        # Se nao existem checkpoints
        if not track.checkpoints:
            return

        # Verificar somente o checkpoint esperado
        reached = track.checkpoint_reached(
            (self.x, self.y),
            self.checkpoint_atual
        )

        if not reached:
            return

        # Ganhar pontos
        self.fitness += (
            self.CHECKPOINT_REWARD
        )

        # Atualizar melhor pontuacao
        self.best_fitness = max(
            self.best_fitness,
            self.fitness
        )

        # Ir para o proximo checkpoint
        self.checkpoint_atual += 1

        # Verificar se completou todos
        if self.checkpoint_atual >= len(
            track.checkpoints
        ):

            # Completou uma volta
            self.voltas += 1

            # Bonus da volta
            self.fitness += (
                self.LAP_REWARD
            )

            self.best_fitness = max(
                self.best_fitness,
                self.fitness
            )

            # Nova volta
            self.checkpoint_atual = 0

    # ---------------------------------------------------------
    # MORTE
    # ---------------------------------------------------------

    def die(self, track):

        # Contar morte
        self.deaths += 1

        # Somente este carro reinicia
        # O cerebro continua igual
        self.reset(track)

    # ---------------------------------------------------------
    # ATUALIZAR O CARRO
    # ---------------------------------------------------------

    def update(self, dt, track):

        # Evitar saltos grandes se houver travamento
        dt = max(
            0.0,
            min(dt, 0.05)
        )

        if dt == 0:
            return

        # Tempo vivo
        self.time_alive += dt

        # Tempo ate a proxima decisao
        self.think_timer -= dt

        # -----------------------------------------------------
        # CEREBRO
        # -----------------------------------------------------

        if self.think_timer <= 0.0:

            # Ler os sensores
            entradas = self.get_sensors(
                track
            )

            # Tomar uma decisao
            decisao = self.brain.pensar(
                entradas
            )

            # Acelerar
            self.accelerator = max(
                0.0,
                min(
                    1.0,
                    decisao["acelerar"]
                    * self.COMMAND_GAIN
                )
            )

            # Frear
            self.brake = max(
                0.0,
                min(
                    1.0,
                    decisao["frear"]
                    * self.COMMAND_GAIN
                )
            )

            # Virar
            self.steering = max(
                -1.0,
                min(
                    1.0,
                    decisao["virar"]
                    * self.COMMAND_GAIN
                )
            )

            # Guardar tracao
            self.last_traction = (
                self.accelerator
                - self.brake
            )

            # Proxima decisao
            self.think_timer += (
                self.THINK_INTERVAL
            )

        # -----------------------------------------------------
        # FISICA
        # -----------------------------------------------------

        acceleration = (
            self.accelerator
            * self.ACCELERATION

            - self.brake
            * self.BRAKING

            - self.FRICTION
        )

        # Atualizar velocidade
        self.speed = max(
            0.0,
            min(
                self.MAX_SPEED,
                self.speed
                + acceleration * dt
            )
        )

        # Virar
        if self.speed > 0.0:

            speed_factor = (
                0.35
                + 0.65
                * (
                    self.speed
                    / self.MAX_SPEED
                )
            )

            self.angle += (
                self.steering
                * self.TURN_SPEED
                * speed_factor
                * dt
            )

        # Atualizar imagem
        self.update_image()

        # -----------------------------------------------------
        # MOVIMENTO
        # -----------------------------------------------------

        distance = (
            self.speed
            * dt
        )

        # Dividir em pequenos passos
        # para nao atravessar a borda
        steps = max(
            1,
            math.ceil(
                distance / 2.0
            )
        )

        dx = (
            math.cos(self.angle)
            * distance
            / steps
        )

        dy = (
            math.sin(self.angle)
            * distance
            / steps
        )

        # Mover passo a passo
        for _ in range(steps):

            self.x += dx
            self.y += dy

            rect = self.image.get_rect(
                center=(
                    round(self.x),
                    round(self.y)
                )
            )

            # Saiu da pista
            if not track.is_car_on_track(
                self.car_mask,
                rect
            ):

                self.die(track)
                return

        # Somar distancia percorrida
        self.distance_traveled += (
            distance
        )

        # -----------------------------------------------------
        # CHECKPOINT
        # -----------------------------------------------------

        self.update_checkpoint(
            track
        )

        # -----------------------------------------------------
        # CARRO PARADO
        # -----------------------------------------------------

        if self.speed < 3.0:

            self.time_stopped += dt

            # Se ficar parado por muito tempo
            if (
                self.time_stopped
                >= self.STOPPED_LIMIT
            ):

                self.die(track)
                return

        else:

            self.time_stopped = 0.0

    # ---------------------------------------------------------
    # DESENHAR
    # ---------------------------------------------------------

    def draw(self, screen):

        rect = self.image.get_rect(
            center=(
                round(self.x),
                round(self.y)
            )
        )

        screen.blit(
            self.image,
            rect
        )