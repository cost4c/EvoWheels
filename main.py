"""EvoWheels: simulação fluida e treino rápido com limite de processamento."""
from pathlib import Path
import random
import time

import pygame

from track import Track
from car import Car
from evolution import Evolution
from ai_visualizer import AIVisualizer


pygame.init()
info = pygame.display.Info()
SCREEN_WIDTH = max(1280, info.current_w - 120)
SCREEN_HEIGHT = max(720, info.current_h - 140)
FPS = 60

# Quantidade que você está usando atualmente.
NUM_CARS = 100

FAST_STEPS_OPTIONS = (2, 4, 8, 12)
FAST_STEPS_INDEX = 1
FIXED_DT = 1.0 / FPS

# O treino usa no máximo 12 ms por rodada, mesmo no nível x12.
# Se o computador não acompanhar, prefere manter a janela respondendo.
FAST_WORK_BUDGET = 0.012
FAST_RENDER_INTERVAL = 1.0 / 22.0
HUD_REFRESH_INTERVAL = 0.35

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE
)
pygame.display.set_caption("EvoWheels")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Segoe UI", 14)
font_small = pygame.font.SysFont("Segoe UI", 12)

fullscreen = False
fast_training = False
show_hud_details = False


def toggle_fullscreen():
    global screen, fullscreen
    fullscreen = not fullscreen
    try:
        if fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE
            )
    except Exception:
        fullscreen = not fullscreen


tracks_path = Path(__file__).resolve().parent / "assets" / "tracks"
track_files = sorted(tracks_path.glob("pista_*.png"))
if not track_files:
    pygame.quit()
    raise FileNotFoundError(
        "Coloque as imagens pista_01.png, pista_02.png... em assets/tracks/"
    )

tracks = [Track(path, SCREEN_WIDTH, SCREEN_HEIGHT) for path in track_files]
current_index = random.randrange(len(tracks))
current_track = tracks[current_index]
evolution = Evolution(elite_size=10)
cars = [Car(current_track, i + 1, evolution) for i in range(NUM_CARS)]
ai_visualizer = AIVisualizer(evolution=evolution)


def change_track():
    global current_index, current_track
    indices = [i for i in range(len(tracks)) if i != current_index]
    if indices:
        current_index = random.choice(indices)
    current_track = tracks[current_index]
    for car in cars:
        car.reset(current_track)


def curto(numero):
    # Números grandes ficam legíveis sem ocupar a pista toda.
    if numero >= 1_000_000:
        return f"{numero / 1_000_000:.2f} mi"
    if numero >= 10_000:
        return f"{numero / 1000:.0f} mil"
    return str(numero)


def ajustar_texto(fonte, mensagem, largura):
    if fonte.size(mensagem)[0] <= largura:
        return mensagem
    while mensagem and fonte.size(mensagem + "...")[0] > largura:
        mensagem = mensagem[:-1]
    return mensagem + "..."


def criar_hud(fps_visual, ritmo_real):
    # HUD compacto é o padrão; H mostra estatísticas adicionais.
    width = max(280, min(628, screen.get_width() - 20))
    height = 94 if show_hud_details else 52
    hud = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(
        hud, (9, 13, 19, 230), hud.get_rect(), border_radius=9
    )
    pygame.draw.rect(
        hud, (47, 60, 75, 220), hud.get_rect(), 1, border_radius=9
    )

    if fast_training:
        modo = f"TURBO x{FAST_STEPS_OPTIONS[FAST_STEPS_INDEX]}"
        modo += f" (real x{ritmo_real:.1f})"
    else:
        modo = "NORMAL"

    pista = current_track.name.replace("pista_", "")
    linha1 = (
        f"Pista {pista}   |   {len(cars)} carros   |   "
        f"{modo}   |   {fps_visual:.0f} FPS"
    )
    tempo = evolution.melhor_tempo_volta(current_track.name)
    volta = "--" if tempo is None else f"{tempo:.2f}s"
    linha2 = (
        f"{curto(evolution.total_avaliacoes)} avaliações   |   "
        f"Recorde {evolution.melhor_fitness_pista(current_track.name):.0f}   |   "
        f"Volta {volta}   |   Memórias {evolution.quantidade_memorias()}"
    )

    hud.blit(
        font.render(
            ajustar_texto(font, linha1, width - 20), True, (240, 245, 250)
        ), (10, 7)
    )
    hud.blit(
        font_small.render(
            ajustar_texto(font_small, linha2, width - 20),
            True, (173, 191, 207)
        ), (10, 29)
    )

    if show_hud_details:
        medias = (
            sum(c.speed for c in cars) / len(cars),
            100 * sum(c.pace_assist_ema for c in cars) / len(cars),
            100 * sum(c.safety_assist_ema for c in cars) / len(cars),
        )
        linha3 = (
            f"Mortes {sum(c.deaths for c in cars)}   |   "
            f"Vel. {medias[0]:.0f} px/s   |   "
            f"Ritmo {medias[1]:.0f}%   |   Curvas {medias[2]:.0f}%"
        )
        linha4 = "T Turbo  |  [ ] Nivel  |  S Salvar  |  R Pista  |  V Inspector  |  F11 Tela"
        hud.blit(font_small.render(
            ajustar_texto(font_small, linha3, width - 20),
            True, (185, 204, 219)
        ), (10, 49))
        hud.blit(font_small.render(
            ajustar_texto(font_small, linha4, width - 20),
            True, (141, 163, 185)
        ), (10, 70))

    return hud


# Contadores de velocidade real, não estimativa pelo multiplicador do turbo.
last_meter = time.perf_counter()
window_sim_time = 0.0
window_frames = 0
fps_visual = 0.0
ritmo_real = 1.0
next_render = 0.0
last_hud_refresh = 0.0
hud_surface = None
inspector_dt = 0.0
running = True

try:
    while running:
        if fast_training:
            clock.tick(0)
            dt = FIXED_DT
        else:
            # Picos externos não devem avançar demais a física de uma vez.
            dt = min(0.05, clock.tick(FPS) / 1000.0)

        # Processar teclado e fechar janela a cada rodada, inclusive no turbo.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if ai_visualizer.handle_event(event, cars):
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_r:
                change_track()
            elif event.key == pygame.K_F11:
                toggle_fullscreen()
                hud_surface = None
            elif event.key == pygame.K_t:
                fast_training = not fast_training
                next_render = 0.0
                hud_surface = None
            elif event.key == pygame.K_LEFTBRACKET:
                FAST_STEPS_INDEX = max(0, FAST_STEPS_INDEX - 1)
                hud_surface = None
            elif event.key == pygame.K_RIGHTBRACKET:
                FAST_STEPS_INDEX = min(
                    len(FAST_STEPS_OPTIONS) - 1, FAST_STEPS_INDEX + 1
                )
                hud_surface = None
            elif event.key == pygame.K_s:
                evolution.salvar(force=True)
            elif event.key == pygame.K_h:
                show_hud_details = not show_hud_details
                hud_surface = None

        if not running:
            break

        Car.RENDER_SPRITES = not fast_training
        started = time.perf_counter()
        simulation_steps = 0
        max_steps = FAST_STEPS_OPTIONS[FAST_STEPS_INDEX] if fast_training else 1
        while simulation_steps < max_steps:
            for car in cars:
                car.update(dt, current_track)
            simulation_steps += 1
            # Não deixa x12 transformar um único quadro em uma pausa longa.
            if fast_training and time.perf_counter() - started >= FAST_WORK_BUDGET:
                break

        advanced = dt * simulation_steps
        inspector_dt += advanced
        window_sim_time += advanced
        now = time.perf_counter()

        if now - last_meter >= 1.0:
            elapsed = now - last_meter
            ritmo_real = window_sim_time / elapsed
            fps_visual = window_frames / elapsed
            window_sim_time = 0.0
            window_frames = 0
            last_meter = now
            hud_surface = None

        # Em turbo desenha em função do tempo real, não do número de passos.
        if fast_training and now < next_render:
            time.sleep(0)  # Deixa a janela e o disco receberem CPU.
            continue
        next_render = now + FAST_RENDER_INTERVAL

        screen.fill((0, 0, 0))
        screen.blit(current_track.surface, current_track.rect)
        for car in cars:
            car.draw(screen)

        if hud_surface is None or now - last_hud_refresh >= HUD_REFRESH_INTERVAL:
            hud_surface = criar_hud(fps_visual, ritmo_real)
            last_hud_refresh = now
        screen.blit(hud_surface, (10, 10))
        pygame.display.flip()
        window_frames += 1

        ai_visualizer.update(
            cars, current_track, inspector_dt, fps_visual
        )
        inspector_dt = 0.0
        if fast_training:
            time.sleep(0)
finally:
    # Espera o autosave em andamento e grava os últimos avanços.
    evolution.salvar(force=True)
    ai_visualizer.close()
    pygame.quit()
