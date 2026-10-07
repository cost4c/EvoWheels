# Importar as bibliotecas e classes
from pathlib import Path
import random
import pygame

from track import Track
from car import Car


# Inicializar o pygame e configuracoes de tela, titulo, fundo e FPS
pygame.init()
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
NUM_CARS = 100

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("EvoWheels")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)


# Carregar as pistas transparentes
tracks_path = Path(__file__).resolve().parent / "assets" / "tracks"
track_files = sorted(tracks_path.glob("pista_*.png"))

if not track_files:
    pygame.quit()
    raise FileNotFoundError("Coloque as imagens pista_01.png ... em assets/tracks/")

tracks = [Track(path, SCREEN_WIDTH, SCREEN_HEIGHT) for path in track_files]
current_index = random.randrange(len(tracks))
current_track = tracks[current_index]

# Criar 250 carrinhos independentes
cars = [Car(current_track, i + 1) for i in range(NUM_CARS)]


# Trocar de pista pela tecla R
def change_track():
    global current_index, current_track

    available = [i for i in range(len(tracks)) if i != current_index]
    if available:
        current_index = random.choice(available)

    current_track = tracks[current_index]

    # Ao mudar a pista, todos precisam ir para a nova largada
    for car in cars:
        car.reset(current_track)


# Game Loop - Manter a janela em execucao
running = True

while running:
    dt = clock.tick(FPS) / 1000.0

    # Verificar os eventos da janela
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                change_track()

    # Pintar o fundo e desenhar a pista
    screen.fill(pygame.Color("black"))
    screen.blit(current_track.surface, current_track.rect)

    # Atualizar cada carrinho individualmente
    for car in cars:
        car.update(dt, current_track)
        car.draw(screen)

    # Mostrar informacoes da simulacao
    total_deaths = sum(car.deaths for car in cars)
    info = (
        f"Pista: {current_track.name}   |   "
        f"Carrinhos: {NUM_CARS}   |   "
        f"Mortes: {total_deaths}   |   R: trocar pista"
    )
    text = font.render(info, True, "white")
    screen.blit(text, (15, 15))

    # Atualizar a tela
    pygame.display.flip()

# Encerrar o pygame
pygame.quit()
