from pathlib import Path
import random

import pygame

from track import Track
from car import Car
from evolution import Evolution
from ai_visualizer import AIVisualizer


# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------

pygame.init()

display_info = pygame.display.Info()

SCREEN_WIDTH = max(1280, display_info.current_w - 120)
SCREEN_HEIGHT = max(720, display_info.current_h - 140)

FPS = 60
NUM_CARS = 250


# ---------------------------------------------------------
# JANELA PRINCIPAL
# ---------------------------------------------------------

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.RESIZABLE
)

pygame.display.set_caption("EvoWheels")

fullscreen = False


def toggle_fullscreen():
    """F11 na janela da corrida.

    As pistas nao sao reconstruidas nem reposicionadas, porque o
    retangulo da pista define as coordenadas do mundo.
    """

    global screen, fullscreen

    fullscreen = not fullscreen

    try:
        if fullscreen:
            screen = pygame.display.set_mode(
                (0, 0),
                pygame.FULLSCREEN
            )
        else:
            screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                pygame.RESIZABLE
            )
    except Exception:
        fullscreen = not fullscreen

clock = pygame.time.Clock()

font = pygame.font.SysFont("Segoe UI", 18)
font_small = pygame.font.SysFont("Segoe UI", 15)


# ---------------------------------------------------------
# PISTAS
# ---------------------------------------------------------

tracks_path = Path(__file__).resolve().parent / "assets" / "tracks"
track_files = sorted(tracks_path.glob("pista_*.png"))

if not track_files:
    pygame.quit()
    raise FileNotFoundError(
        "Coloque as imagens pista_01.png, pista_02.png... em assets/tracks/"
    )

tracks = [
    Track(path, SCREEN_WIDTH, SCREEN_HEIGHT)
    for path in track_files
]

current_index = random.randrange(len(tracks))
current_track = tracks[current_index]


# ---------------------------------------------------------
# EVOLUÇÃO
# ---------------------------------------------------------

evolution = Evolution(elite_size=10)


# ---------------------------------------------------------
# CARROS
# ---------------------------------------------------------

cars = [
    Car(current_track, i + 1, evolution)
    for i in range(NUM_CARS)
]


# ---------------------------------------------------------
# VISUALIZADOR
# ---------------------------------------------------------

ai_visualizer = AIVisualizer(evolution=evolution)


# ---------------------------------------------------------
# TROCAR PISTA
# ---------------------------------------------------------

def change_track():
    global current_index, current_track

    available = [
        i for i in range(len(tracks))
        if i != current_index
    ]

    if available:
        current_index = random.choice(available)

    current_track = tracks[current_index]

    # Mantém o aprendizado e só reposiciona
    for car in cars:
        car.reset(current_track)


# ---------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------

running = True

while running:
    dt = clock.tick(FPS) / 1000.0

    # -----------------------------------------------------
    # EVENTOS
    # -----------------------------------------------------

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            continue

        # O inspector ve o evento primeiro e diz se usou.
        if ai_visualizer.handle_event(event, cars):
            continue

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                change_track()

            elif event.key == pygame.K_F11:
                toggle_fullscreen()

    # -----------------------------------------------------
    # DESENHAR FUNDO
    # -----------------------------------------------------

    screen.fill((0, 0, 0))
    screen.blit(current_track.surface, current_track.rect)

    # -----------------------------------------------------
    # ATUALIZAR CARROS
    # -----------------------------------------------------

    for car in cars:
        car.update(dt, current_track)
        car.draw(screen)

    # -----------------------------------------------------
    # HUD
    # -----------------------------------------------------

    total_deaths = sum(car.deaths for car in cars)

    best_alive = max(
        car.get_evolution_score()
        for car in cars
    )

    best_track_score = evolution.melhor_fitness_pista(current_track.name)

    info_1 = (
        f"Pista: {current_track.name}   |   "
        f"Carros: {NUM_CARS}   |   "
        f"Mortes: {total_deaths}   |   "
        f"FPS: {clock.get_fps():.0f}"
    )

    info_2 = (
        f"Avaliações: {evolution.total_avaliacoes}   |   "
        f"Elites: {evolution.quantidade_elites()}/10   |   "
        f"Recorde global: {evolution.melhor_fitness():.1f}"
    )

    info_3 = (
        f"Recorde da pista: {best_track_score:.1f}   |   "
        f"Melhor vivo: {best_alive:.1f}   |   "
        f"R: trocar pista   |   V: inspector   |   F11: tela cheia"
    )

    hud_rect = pygame.Rect(10, 10, 860, 78)

    pygame.draw.rect(screen, (10, 14, 18), hud_rect, border_radius=10)
    pygame.draw.rect(screen, (48, 58, 70), hud_rect, 1, border_radius=10)

    text_1 = font.render(info_1, True, (245, 245, 245))
    text_2 = font_small.render(info_2, True, (195, 205, 215))
    text_3 = font_small.render(info_3, True, (195, 205, 215))

    screen.blit(text_1, (22, 18))
    screen.blit(text_2, (22, 43))
    screen.blit(text_3, (22, 62))

    pygame.display.flip()

    # -----------------------------------------------------
    # ATUALIZAR INSPECTOR
    # -----------------------------------------------------

    ai_visualizer.update(cars, current_track, dt, clock.get_fps())


# ---------------------------------------------------------
# FINALIZAR
# ---------------------------------------------------------

ai_visualizer.close()
pygame.quit()