# Importar as bibliotecas
import math
import pygame

# Importar o cerebro dos carrinhos
from brain import Brain


# Classe do carrinho
class Car:
    # Sensores de distancia, em graus, em relacao a frente do carrinho
    SENSOR_ANGLES = (-160, -125, -95, -70, -45, -22, 0, 22, 45, 70, 95, 125, 160)
    SENSOR_RANGE = 170
    SENSOR_STEP = 4

    # Configuracoes da movimentacao
    MAX_SPEED = 175.0
    START_SPEED = 85.0
    ACCELERATION = 240.0
    BRAKING = 310.0
    FRICTION = 8.0
    TURN_SPEED = 2.7

    # A rede decide a cada 0.1 segundo; a fisica continua em todos os frames
    THINK_INTERVAL = 0.10
    COMMAND_GAIN = 8.0
    STOPPED_LIMIT = 2.0

    def __init__(self, track, identifier):
        self.identifier = identifier
        self.brain = Brain()
        self.width = 14
        self.height = 8

        self.fitness = 0
        self.best_fitness = 0

        # Cada carrinho recebe uma cor fixa, sem depender de sorteio
        self.color = pygame.Color("#00FF00")

        # Criar a imagem do carrinho uma unica vez
        self.original_image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(
            self.original_image,
            self.color,
            (0, 0, self.width, self.height),
            border_radius=2,
        )
        pygame.draw.rect(
            self.original_image,
            "white",
            (self.width - 4, 1, 3, self.height - 2),
            border_radius=1,
        )

        self.deaths = 0
        self.image = self.original_image
        self.car_mask = pygame.mask.from_surface(self.original_image)
        self.sprite_angle = None
        self.reset(track)

    # Reiniciar somente este carrinho, mantendo o mesmo cerebro
    def reset(self, track):
        self.x, self.y = map(float, track.spawn_point)
        self.angle = float(track.spawn_angle)
        self.speed = self.START_SPEED
        self.steering = 0.0
        self.accelerator = 0.0
        self.brake = 0.0
        self.last_traction = 0.0
        self.think_timer = 0.0
        self.time_alive = 0.0
        self.time_stopped = 0.0
        self.distance_traveled = 0.0
        self.sprite_angle = None
        self.update_image()

        # Atualizar a melhor pontuacao, se necessario, e zerar a pontuacao atual a cada batida
        self.best_fitness = max(self.best_fitness, self.fitness)
        self.fitness = 0

    # Ajustar a imagem e a mascara quando o angulo mudar
    def update_image(self):
        angle_degrees = round(math.degrees(self.angle)) % 360
        if self.sprite_angle == angle_degrees:
            return

        self.image = pygame.transform.rotate(self.original_image, -angle_degrees)
        self.car_mask = pygame.mask.from_surface(self.image)
        self.sprite_angle = angle_degrees

    # Verificar a distancia das bordas em 13 direcoes
    def get_sensors(self, track):
        readings = []

        for relative_degrees in self.SENSOR_ANGLES:
            ray_angle = self.angle + math.radians(relative_degrees)
            direction_x = math.cos(ray_angle)
            direction_y = math.sin(ray_angle)
            reading = 1.0

            for distance in range(0, self.SENSOR_RANGE + 1, self.SENSOR_STEP):
                point = (
                    self.x + direction_x * distance,
                    self.y + direction_y * distance,
                )

                if not track.is_point_on_track(point):
                    reading = distance / self.SENSOR_RANGE
                    break

            readings.append(reading)

        # 13 sensores + velocidade + direcao anterior + tracao anterior = 16 entradas
        readings.append(self.speed / self.MAX_SPEED)
        readings.append(self.steering)
        readings.append(self.last_traction)
        return readings

    # Atualizar a posicao usando apenas as decisoes da rede neural
    def update(self, dt, track):
        dt = max(0.0, min(dt, 0.05))
        if dt == 0:
            return

        self.time_alive += dt
        self.think_timer -= dt

        # Consultar o cerebro periodicamente, sem sorteios de movimento
        if self.think_timer <= 0.0:
            entradas = self.get_sensors(track)
            decisao = self.brain.pensar(entradas)

            # Aumentar a sensibilidade das saidas iniciais da rede
            self.accelerator = max(0.0, min(1.0, decisao["acelerar"] * self.COMMAND_GAIN))
            self.brake = max(0.0, min(1.0, decisao["frear"] * self.COMMAND_GAIN))
            self.steering = max(-1.0, min(1.0, decisao["virar"] * self.COMMAND_GAIN))
            self.last_traction = self.accelerator - self.brake
            self.think_timer = self.THINK_INTERVAL

        # Aplicar aceleracao, freio e resistencia ao movimento
        acceleration = (
            self.accelerator * self.ACCELERATION
            - self.brake * self.BRAKING
            - self.FRICTION
        )
        self.speed = max(0.0, min(self.MAX_SPEED, self.speed + acceleration * dt))

        # Girar conforme o comando neural
        if self.speed > 0.0:
            speed_factor = 0.35 + 0.65 * (self.speed / self.MAX_SPEED)
            self.angle += self.steering * self.TURN_SPEED * speed_factor * dt

        self.update_image()

        # Verificar a colisao em pequenos passos, sem atravessar as bordas
        distance = self.speed * dt
        steps = max(1, math.ceil(distance / 2.0))
        dx = math.cos(self.angle) * distance / steps
        dy = math.sin(self.angle) * distance / steps

        for _ in range(steps):
            self.x += dx
            self.y += dy

            rect = self.image.get_rect(center=(round(self.x), round(self.y)))
            if not track.is_car_on_track(self.car_mask, rect):
                self.deaths += 1
                self.reset(track)
                return

        self.distance_traveled += distance

        # Evitar que um carrinho fique parado para sempre
        if self.speed < 3.0:
            self.time_stopped += dt
            if self.time_stopped >= self.STOPPED_LIMIT:
                self.deaths += 1
                self.reset(track)
        else:
            self.time_stopped = 0.0

    # Desenhar o carrinho
    def draw(self, screen):
        rect = self.image.get_rect(center=(round(self.x), round(self.y)))
        screen.blit(self.image, rect)
