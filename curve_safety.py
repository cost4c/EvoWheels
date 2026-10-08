"""Protecao de curvas baseada em trajetorias fisicamente possiveis.

O cerebro dirige normalmente. Este modulo so intervem quando a trajetoria
prevista sairia da pista; avalia alternativas sem regras fixas de esquerda
ou direita. Nao altera os pesos da rede nem o metodo de evolucao.
"""

import math


class CurveSafety:
    # A previsao e curta para nao assumir o papel do piloto neural.
    HORIZON = 1.05
    STEP_TIME = 0.080
    MOVE_STEP = 6.0
    CANDIDATE_STEERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    BRAKING_OPTIONS = (-0.30, -0.65, -1.0)

    @staticmethod
    def _radius(p0, p1, p2):
        ax, ay = p0
        bx, by = p1
        cx, cy = p2
        ab = math.hypot(bx - ax, by - ay)
        bc = math.hypot(cx - bx, cy - by)
        ac = math.hypot(cx - ax, cy - ay)
        twice_area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if twice_area < 1e-5:
            return 100000.0
        return ab * bc * ac / (2.0 * twice_area)

    @classmethod
    def _curve_samples(cls, track):
        """Prepara a curvatura da linha central uma so vez por pista."""
        if hasattr(track, '_curve_safety_samples'):
            return track._curve_safety_samples
        line = getattr(track, 'centerline', None) or ()
        if len(line) < 80:
            track._curve_safety_samples = []
            return track._curve_safety_samples
        # A linha central tem pontos aproximadamente a cada pixel.
        stride = 22
        total = len(line)
        samples = []
        for i in range(0, total, stride):
            point = line[i]
            radius = cls._radius(
                line[(i - stride) % total],
                point,
                line[(i + stride) % total]
            )
            samples.append((point[0], point[1], radius))
        track._curve_safety_samples = samples
        return samples

    @classmethod
    def curve_speed_limit(cls, car, track):
        """Limite fisico previsto nas proximas curvas, nunca velocidade-alvo.

        Usa a curvatura da pista somente para impedir uma entrada
        fisicamente impossivel. Nao escolhe o volante para o cerebro.
        """
        samples = cls._curve_samples(track)
        if not samples:
            return car.MAX_SPEED
        count = len(samples)
        points = getattr(track, 'checkpoints', None) or ()
        if points:
            expected = int(car.checkpoint_atual * count / len(points)) % count
            positions = ((expected + d) % count for d in range(-8, 9))
        else:
            positions = range(count)
        nearest = min(
            positions,
            key=lambda i: (samples[i][0]-car.x)**2 + (samples[i][1]-car.y)**2
        )
        limit = car.MAX_SPEED
        previous = (car.x, car.y)
        distance_ahead = 0.0
        for step in range(10):
            x, y, radius = samples[(nearest + step) % count]
            if step == 0:
                # A distorção lateral nao pode parecer espaco de frenagem.
                ahead = 0.0
            else:
                x0, y0, _ = samples[(nearest + step - 1) % count]
                distance_ahead += math.hypot(x - x0, y - y0)
                ahead = distance_ahead
            if ahead > car.SENSOR_RANGE:
                break
            # Reserva folga para carroceria e para o atraso entre decisoes.
            usable = max(0.0, ahead - car.HITBOX_HALF_LENGTH
                         - car.speed * car.THINK_INTERVAL)
            effective_radius = max(8.0, radius - car.HITBOX_HALF_WIDTH - 3.0)
            max_yaw = effective_radius * car.TURN_SPEED * 0.95
            corner_limit = max_yaw / (1.0 + 0.48 * max_yaw / car.MAX_SPEED)
            allowed = math.sqrt(corner_limit * corner_limit
                                + 2.0 * car.BRAKING * usable)
            limit = min(limit, allowed)
        return max(car.MIN_ROLLING_SPEED, min(car.MAX_SPEED, limit))

    @staticmethod
    def cap_traction(car, traction, speed_limit):
        """Impede acelerar para uma curva que ainda nao pode ser feita.

        Permite frear mais do que o minimo calculado; nunca acelera
        sozinho nem introduz velocidade minima alem do piso existente.
        """
        desired_acceleration = (speed_limit - car.speed) / 0.25
        cap = max(-1.0, min(1.0,
                  (desired_acceleration + car.FRICTION) /
                  (car.ACCELERATION if desired_acceleration >= 0 else car.BRAKING)))
        return min(traction, cap)

    @staticmethod
    def _on_track(track, x, y, angle, half_length, half_width):
        ca = math.cos(angle)
        sa = math.sin(angle)
        fx = ca * half_length
        fy = sa * half_length
        sx = -sa * half_width
        sy = ca * half_width
        inside = track.is_point_on_track_xy
        # Primeiro testa as extremidades que normalmente atingem a parede.
        return (
            inside(x + fx + sx, y + fy + sy)
            and inside(x + fx - sx, y + fy - sy)
            and inside(x - fx + sx, y - fy + sy)
            and inside(x - fx - sx, y - fy - sy)
            and inside(x + fx, y + fy)
            and inside(x - fx, y - fy)
            and inside(x, y)
        )

    @classmethod
    def predict(cls, car, track, steer, traction):
        """Prevê até 1,05 s sem mover o carro verdadeiro.

        A maior parte das posições usa uma consulta em um mapa de folga
        pré-calculado. Perto de paredes, mantém o teste exato da carroceria.
        """
        x, y = car.x, car.y
        angle, speed = car.angle, car.speed
        used_steer = car.steering * 0.25 + steer * 0.75
        elapsed = 0.0
        traveled = 0.0
        distance_map = getattr(track, '_safety_distance_bytes', None)
        if distance_map is not None:
            left = track.rect.x
            top = track.rect.y
            width = track.width
            height = track.height
            # Esse raio inclui toda a carroceria e margem de segurança.
            clearance = math.ceil(math.hypot(
                car.HITBOX_HALF_LENGTH + 2.5,
                car.HITBOX_HALF_WIDTH + 2.5,
            )) + 1

        traction_acceleration = (
            max(0.0, traction) * car.ACCELERATION
            - max(0.0, -traction) * car.BRAKING
            - car.FRICTION
        )
        half_length = car.HITBOX_HALF_LENGTH + 2.5
        half_width = car.HITBOX_HALF_WIDTH + 2.5
        on_track = cls._on_track
        # Loop pequeno e com passo maior, mas testa a borda exata quando
        # a folga do centro nao garante a passagem da carroceria.
        for _ in range(math.ceil(cls.HORIZON / cls.STEP_TIME)):
            dt = min(cls.STEP_TIME, cls.HORIZON - elapsed)
            if dt <= 0:
                break
            speed = max(car.MIN_ROLLING_SPEED,
                        min(car.MAX_SPEED, speed + traction_acceleration * dt))
            steering_control = max(0.52, 1.0 - 0.48 * (speed / car.MAX_SPEED))
            angle += used_steer * car.TURN_SPEED * steering_control * dt
            distance = speed * dt
            subdivisions = max(1, math.ceil(distance / cls.MOVE_STEP))
            dx = math.cos(angle) * distance / subdivisions
            dy = math.sin(angle) * distance / subdivisions
            for i in range(subdivisions):
                x += dx
                y += dy
                traveled += distance / subdivisions
                if distance_map is not None:
                    px = int(x) - left
                    py = int(y) - top
                    if (0 <= px < width and 0 <= py < height
                            and distance_map[py * width + px] >= clearance):
                        continue
                if not on_track(track, x, y, angle, half_length, half_width):
                    return elapsed + dt * (i + 1) / subdivisions, x, y, speed, traveled
            elapsed += dt
        return cls.HORIZON, x, y, speed, traveled

    @classmethod
    def choose(cls, car, track, neural_steer, neural_traction):
        """Reaproveita uma previsão recente quando o controle quase não muda.

        A previsão cobre 1,05 s. Reusar somente a decisão seguinte (0,1 s)
        evita refazer o mesmo cálculo, sem deixar uma curva desprotegida.
        Uma mudança brusca de comando ou de dinâmica sempre recalcula.
        """
        cache = getattr(car, '_curve_safety_cache', None)
        if cache is not None and cache[0] == id(track) and cache[1] > 0:
            (_, _, last_time, last_x, last_y, last_angle,
             last_speed, last_steer, last_traction, decision) = cache
            if (0 <= car.time_alive - last_time <= 0.15
                    and (car.x - last_x)**2 + (car.y - last_y)**2 <= 24**2
                    and abs(car.angle - last_angle) < 0.30
                    and abs(car.speed - last_speed) <= 25.0
                    and abs(neural_steer - last_steer) <= 0.08
                    and abs(neural_traction - last_traction) <= 0.16):
                car._curve_safety_cache = (
                    cache[0], 0, *cache[2:]
                )
                if decision[2]:
                    # A proteção anterior segue válida; nunca ignora
                    # uma frenagem mais forte exigida neste instante.
                    traction = min(decision[1], neural_traction)
                    return decision[0], traction, True, traction < neural_traction
                return neural_steer, neural_traction, False, False

        decision = cls._choose_uncached(car, track, neural_steer, neural_traction)
        car._curve_safety_cache = (
            id(track), 1, car.time_alive, car.x, car.y, car.angle,
            car.speed, neural_steer, neural_traction, decision
        )
        return decision

    @classmethod
    def _choose_uncached(cls, car, track, neural_steer, neural_traction):
        """Retorna (direcao, tracao, intervenção, frenagem)."""
        neural_steer = max(-1.0, min(1.0, neural_steer))
        neural_traction = max(-1.0, min(1.0, neural_traction))
        # Em reta muito aberta, a proxima parede esta fora dos 180 px.
        # Evita previsoes caras em carros que nao correm risco imediato.
        risks = getattr(car, 'last_sensor_risks', ())
        if (len(risks) == 7 and max(risks[2:5]) < 0.005
                and abs(neural_steer) < 0.20):
            return neural_steer, neural_traction, False, False

        baseline = cls.predict(car, track, neural_steer, neural_traction)

        # Enquanto a rede escolhe uma rota segura, ela tem controle total.
        if baseline[0] >= cls.HORIZON - 1e-6:
            return neural_steer, neural_traction, False, False

        # A trajetória neural bate. Começa com os dois esterços capazes
        # de fechar uma curva e só testa esterços intermediários se preciso.
        # Assim, não simula dezenas de caminhos equivalentes por decisão.
        start_distance = car.distance_to_checkpoint(track)
        best_score = -float('inf')
        best_steer = neural_steer
        best_traction = -1.0

        if neural_traction > 0.0:
            traction_levels = (
                neural_traction,
                min(neural_traction, 0.36),
                min(neural_traction, 0.07),
                0.0,
                -0.5,
                -1.0,
            )
        else:
            traction_levels = (neural_traction, -0.5, -1.0)

        if track.checkpoints:
            px, py = track.checkpoints[car.checkpoint_atual]
        else:
            px = py = None

        for traction in dict.fromkeys(traction_levels):
            level_best = None
            level_score = -float('inf')
            found_safe = False
            # Primeiro testa a intenção da rede e esterços completos.
            core = tuple(dict.fromkeys((neural_steer, -1.0, 1.0)))
            extra = (-0.5, 0.5)
            for group in (core, extra):
                for steer in group:
                    if steer in core and group is extra:
                        continue
                    free_time, x, y, final_speed, _ = cls.predict(
                        car, track, steer, traction
                    )
                    if px is not None:
                        progress = start_distance - math.hypot(x - px, y - py)
                    else:
                        progress = 0.0
                    safe = free_time >= cls.HORIZON - 1e-6
                    score = (
                        (2000.0 if safe else free_time * 1000.0)
                        + max(-50.0, min(50.0, progress)) * 1.5
                        + final_speed * 0.10
                        - abs(steer - neural_steer) * 5.0
                    )
                    if score > level_score:
                        level_score = score
                        level_best = (steer, traction, safe)
                    if score > best_score:
                        best_score = score
                        best_steer, best_traction = steer, traction
                    found_safe = found_safe or safe
                if found_safe:
                    break
            if level_best[2]:
                return (level_best[0], level_best[1], True,
                        traction < neural_traction)

        return best_steer, best_traction, True, True
