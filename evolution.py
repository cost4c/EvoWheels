import random

from brain import Brain


class Evolution:

    def __init__(
        self,
        elite_size=10
    ):

        self.elite_size = (
            elite_size
        )

        # Melhores gerais
        self.elites = []

        # Melhores de cada pista
        self.elites_por_pista = {}

        # Estatisticas
        self.total_avaliacoes = 0

        self.melhor_score = 0.0

        self.avaliacoes_sem_recorde = 0

    # ---------------------------------------------------------
    # VERIFICAR SE MERECE ENTRAR
    # ---------------------------------------------------------

    def qualifies(
        self,
        elite_list,
        score
    ):

        if score <= 0.0:
            return False

        if (
            len(
                elite_list
            )
            < self.elite_size
        ):
            return True

        return (
            score
            > elite_list[-1]["score"]
        )

    # ---------------------------------------------------------
    # INSERIR SEM ORDENAR A LISTA INTEIRA
    # ---------------------------------------------------------

    def insert_record(
        self,
        elite_list,
        record
    ):

        score = (
            record["score"]
        )

        position = len(
            elite_list
        )

        for index, existing in enumerate(
            elite_list
        ):

            if (
                score
                > existing["score"]
            ):

                position = index
                break

        elite_list.insert(
            position,
            record
        )

        if (
            len(
                elite_list
            )
            > self.elite_size
        ):

            elite_list.pop()

    # ---------------------------------------------------------
    # REGISTRAR
    # ---------------------------------------------------------

    def registrar(
        self,
        brain,
        score,
        pista=None
    ):

        score = float(
            score
        )

        self.total_avaliacoes += 1

        # Recorde global
        if (
            score
            > self.melhor_score
        ):

            self.melhor_score = (
                score
            )

            self.avaliacoes_sem_recorde = 0

        else:

            self.avaliacoes_sem_recorde += 1

        if score <= 0.0:
            return

        # Elite da pista
        track_elite = None

        if pista is not None:

            track_elite = (
                self.elites_por_pista.setdefault(
                    pista,
                    []
                )
            )

        global_qualifies = (
            self.qualifies(
                self.elites,
                score
            )
        )

        track_qualifies = (
            track_elite is not None
            and self.qualifies(
                track_elite,
                score
            )
        )

        # -----------------------------------------------------
        # OTIMIZACAO IMPORTANTE
        # -----------------------------------------------------

        # Se nao vai entrar em nenhuma elite,
        # nem copiar o cerebro.
        if (
            not global_qualifies
            and not track_qualifies
        ):

            return

        # Apenas UMA copia.
        snapshot = (
            brain.copiar()
        )

        if global_qualifies:

            global_record = {
                "score": score,
                "brain": snapshot,
                "pista": pista,
            }

            self.insert_record(
                self.elites,
                global_record
            )

        if track_qualifies:

            track_record = {
                "score": score,
                "brain": snapshot,
                "pista": pista,
            }

            self.insert_record(
                track_elite,
                track_record
            )

    # ---------------------------------------------------------
    # SELECAO PONDERADA
    # ---------------------------------------------------------

    @staticmethod
    def weighted_choice(
        elite_list
    ):

        count = len(
            elite_list
        )

        if count == 0:
            return None

        # Melhor = peso maior.
        #
        # Exemplo 4:
        # 16, 9, 4, 1
        weights = [
            (
                count - index
            ) ** 2
            for index in range(
                count
            )
        ]

        total = sum(
            weights
        )

        target = (
            random.random()
            * total
        )

        accumulated = 0.0

        for record, weight in zip(
            elite_list,
            weights
        ):

            accumulated += weight

            if (
                target
                <= accumulated
            ):

                return (
                    record["brain"]
                )

        return (
            elite_list[-1][
                "brain"
            ]
        )

    # ---------------------------------------------------------
    # ESCOLHER CONHECIMENTO
    # ---------------------------------------------------------

    def escolher_elite(
        self,
        pista=None
    ):

        if not self.elites:
            return None

        # 80% das vezes aprende com alguem
        # que foi bom nesta mesma pista.
        if (
            pista is not None
            and pista
            in self.elites_por_pista
            and self.elites_por_pista[
                pista
            ]
            and random.random()
            < 0.80
        ):

            chosen = (
                self.weighted_choice(
                    self.elites_por_pista[
                        pista
                    ]
                )
            )

            if chosen is not None:
                return chosen

        # O restante usa conhecimento global
        return self.weighted_choice(
            self.elites
        )

    # ---------------------------------------------------------
    # EXPLORACAO ADAPTATIVA
    # ---------------------------------------------------------

    def fator_exploracao(
        self
    ):

        if (
            self.avaliacoes_sem_recorde
            >= 1200
        ):

            return 1.80

        if (
            self.avaliacoes_sem_recorde
            >= 600
        ):

            return 1.50

        if (
            self.avaliacoes_sem_recorde
            >= 300
        ):

            return 1.25

        return 1.0

    # ---------------------------------------------------------
    # CRIAR NOVO CEREBRO
    # ---------------------------------------------------------

    def criar_descendente(
        self,
        brain_atual,
        score_atual,
        pista=None
    ):

        # O score ja inclui progresso, checkpoints e velocidade util.
        # Aqui a selecao espalha os melhores comportamentos.
        self.registrar(
            brain_atual,
            score_atual,
            pista
        )

        # -----------------------------------------------------
        # AINDA NAO EXISTE ELITE
        # -----------------------------------------------------

        if not self.elites:

            filho = (
                brain_atual.copiar()
            )

            filho.mutar(
                taxa=0.14,
                intensidade=0.22
            )

            return filho

        chance = (
            random.random()
        )

        fator = (
            self.fator_exploracao()
        )

        # -----------------------------------------------------
        # 18%
        # COPIA DIRETA DE UM BOM CEREBRO
        # -----------------------------------------------------

        if chance < 0.18:

            pai = (
                self.escolher_elite(
                    pista
                )
            )

            return (
                pai.copiar()
            )

        # -----------------------------------------------------
        # 70%
        # PEQUENA EVOLUCAO DE UM BOM CEREBRO
        # -----------------------------------------------------

        if chance < 0.88:

            pai = (
                self.escolher_elite(
                    pista
                )
            )

            filho = (
                pai.copiar()
            )

            filho.mutar(
                taxa=min(
                    0.10,
                    0.035
                    * fator
                ),
                intensidade=min(
                    0.22,
                    0.075
                    * fator
                )
            )

            return filho

        # -----------------------------------------------------
        # 9%
        # EXPLORACAO MAIS FORTE
        # -----------------------------------------------------

        if chance < 0.97:

            pai = (
                self.escolher_elite(
                    pista
                )
            )

            filho = (
                pai.copiar()
            )

            filho.mutar(
                taxa=min(
                    0.28,
                    0.11
                    * fator
                ),
                intensidade=min(
                    0.45,
                    0.20
                    * fator
                )
            )

            return filho

        # -----------------------------------------------------
        # 3%
        # GENETICA TOTALMENTE NOVA
        # -----------------------------------------------------

        return Brain()

    # ---------------------------------------------------------
    # ESTATISTICAS
    # ---------------------------------------------------------

    def quantidade_elites(
        self
    ):

        return len(
            self.elites
        )

    def melhor_fitness(
        self
    ):

        if not self.elites:
            return 0.0

        return float(
            self.elites[0][
                "score"
            ]
        )

    def melhor_fitness_pista(
        self,
        pista
    ):

        elite = (
            self.elites_por_pista.get(
                pista,
                []
            )
        )

        if not elite:
            return 0.0

        return float(
            elite[0][
                "score"
            ]
        )