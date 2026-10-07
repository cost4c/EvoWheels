import atexit
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from brain import Brain


class Evolution:

    STATE_VERSION = 1
    SAVE_INTERVAL = 3.0

    def __init__(
        self,
        elite_size=10,
        storage_path=None,
        auto_load=True
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

        # Persistencia
        if storage_path is None:

            storage_path = (
                Path(__file__).resolve().parent
                / "models"
                / "evolution_state.npz"
            )

        self.storage_path = Path(
            storage_path
        )

        self.carregado_do_disco = False
        self._dirty = False
        self._last_save = 0.0

        if auto_load:
            self.carregar()

        # Salva tambem quando o programa fecha normalmente
        atexit.register(
            self.salvar
        )

    # ---------------------------------------------------------
    # PERSISTENCIA
    # ---------------------------------------------------------

    @staticmethod
    def _brain_to_arrays(
        arrays,
        prefix,
        brain
    ):

        for indice, peso in enumerate(
            brain.pesos
        ):

            arrays[
                f"{prefix}_w_{indice}"
            ] = np.asarray(
                peso,
                dtype=np.float32
            )

        for indice, bias in enumerate(
            brain.biases
        ):

            arrays[
                f"{prefix}_b_{indice}"
            ] = np.asarray(
                bias,
                dtype=np.float32
            )

    @staticmethod
    def _brain_from_arrays(
        data,
        prefix
    ):

        pesos = []
        biases = []

        for indice in range(
            len(Brain.ARQUITETURA) - 1
        ):

            pesos.append(
                data[
                    f"{prefix}_w_{indice}"
                ]
            )

            biases.append(
                data[
                    f"{prefix}_b_{indice}"
                ]
            )

        return Brain.criar_com_parametros(
            pesos,
            biases
        )

    def salvar(
        self,
        force=True
    ):

        if not force and not self._dirty:
            return False

        if not self._dirty and self.storage_path.exists():
            return False

        try:
            self.storage_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            metadata = {
                "version": self.STATE_VERSION,
                "architecture": list(
                    Brain.ARQUITETURA
                ),
                "elite_size": self.elite_size,
                "total_avaliacoes": self.total_avaliacoes,
                "melhor_score": self.melhor_score,
                "avaliacoes_sem_recorde": self.avaliacoes_sem_recorde,
                "global": [],
                "tracks": [],
            }

            arrays = {}

            for indice, record in enumerate(
                self.elites
            ):

                metadata[
                    "global"
                ].append(
                    {
                        "score": float(
                            record["score"]
                        ),
                        "pista": record.get(
                            "pista"
                        ),
                    }
                )

                self._brain_to_arrays(
                    arrays,
                    f"g_{indice}",
                    record["brain"]
                )

            for pista_indice, (pista, records) in enumerate(
                self.elites_por_pista.items()
            ):

                track_meta = {
                    "pista": pista,
                    "records": [],
                }

                for record_indice, record in enumerate(
                    records
                ):

                    track_meta[
                        "records"
                    ].append(
                        {
                            "score": float(
                                record["score"]
                            )
                        }
                    )

                    self._brain_to_arrays(
                        arrays,
                        f"t_{pista_indice}_{record_indice}",
                        record["brain"]
                    )

                metadata[
                    "tracks"
                ].append(
                    track_meta
                )

            arrays[
                "metadata"
            ] = np.array(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
            )

            temp_path = self.storage_path.with_suffix(
                self.storage_path.suffix + ".tmp"
            )

            with open(
                temp_path,
                "wb"
            ) as file:

                np.savez_compressed(
                    file,
                    **arrays
                )

            os.replace(
                temp_path,
                self.storage_path
            )

            self._dirty = False
            self._last_save = time.monotonic()

            return True

        except Exception as error:

            print(
                "[EvoWheels] Nao foi possivel salvar o aprendizado: "
                f"{error}"
            )

            return False

    def _salvar_se_preciso(self):

        if not self._dirty:
            return

        agora = time.monotonic()

        if (
            agora - self._last_save
            >= self.SAVE_INTERVAL
        ):

            self.salvar(
                force=False
            )

    def carregar(self):

        if not self.storage_path.exists():
            return False

        try:
            with np.load(
                self.storage_path,
                allow_pickle=False
            ) as data:

                metadata = json.loads(
                    str(
                        data["metadata"].item()
                    )
                )

                if metadata.get(
                    "version"
                ) != self.STATE_VERSION:

                    raise ValueError(
                        "versao do arquivo de treino incompativel"
                    )

                arquitetura = tuple(
                    metadata.get(
                        "architecture",
                        ()
                    )
                )

                if arquitetura != tuple(
                    Brain.ARQUITETURA
                ):

                    raise ValueError(
                        "arquitetura salva diferente da arquitetura atual"
                    )

                elites = []

                for indice, record_meta in enumerate(
                    metadata.get(
                        "global",
                        []
                    )[:self.elite_size]
                ):

                    brain = self._brain_from_arrays(
                        data,
                        f"g_{indice}"
                    )

                    elites.append(
                        {
                            "score": float(
                                record_meta["score"]
                            ),
                            "brain": brain,
                            "pista": record_meta.get(
                                "pista"
                            ),
                        }
                    )

                elites_por_pista = {}

                for pista_indice, track_meta in enumerate(
                    metadata.get(
                        "tracks",
                        []
                    )
                ):

                    pista = track_meta.get(
                        "pista"
                    )

                    if pista is None:
                        continue

                    records = []

                    for record_indice, record_meta in enumerate(
                        track_meta.get(
                            "records",
                            []
                        )[:self.elite_size]
                    ):

                        brain = self._brain_from_arrays(
                            data,
                            f"t_{pista_indice}_{record_indice}"
                        )

                        records.append(
                            {
                                "score": float(
                                    record_meta["score"]
                                ),
                                "brain": brain,
                                "pista": pista,
                            }
                        )

                    if records:
                        elites_por_pista[
                            pista
                        ] = records

                self.elites = elites
                self.elites_por_pista = elites_por_pista

                self.total_avaliacoes = int(
                    metadata.get(
                        "total_avaliacoes",
                        0
                    )
                )

                self.melhor_score = float(
                    metadata.get(
                        "melhor_score",
                        0.0
                    )
                )

                self.avaliacoes_sem_recorde = int(
                    metadata.get(
                        "avaliacoes_sem_recorde",
                        0
                    )
                )

                if self.elites:
                    self.melhor_score = max(
                        self.melhor_score,
                        float(
                            self.elites[0][
                                "score"
                            ]
                        )
                    )

            self.carregado_do_disco = True
            self._dirty = False
            self._last_save = time.monotonic()

            print(
                "[EvoWheels] Aprendizado carregado: "
                f"{len(self.elites)} elites globais, "
                f"recorde {self.melhor_score:.1f}."
            )

            return True

        except Exception as error:

            print(
                "[EvoWheels] O aprendizado salvo foi ignorado: "
                f"{error}"
            )

            return False

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
        self._dirty = True

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
            self._salvar_se_preciso()
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

        # Se nao vai entrar em nenhuma elite,
        # nem copiar o cerebro
        if (
            not global_qualifies
            and not track_qualifies
        ):

            self._salvar_se_preciso()
            return

        # Apenas uma copia
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

        self._salvar_se_preciso()

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

        # Melhor = peso maior
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

        # Prioriza quem foi bom na mesma pista
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

        return self.weighted_choice(
            self.elites
        )

    # ---------------------------------------------------------
    # CEREBRO INICIAL
    # ---------------------------------------------------------

    def criar_cerebro_inicial(
        self,
        pista=None
    ):

        # Primeira execucao: ainda nao existe experiencia salva
        if not self.elites:
            return Brain()

        pai = self.escolher_elite(
            pista
        )

        if pai is None:
            return Brain()

        chance = random.random()

        # Parte da populacao preserva exatamente o que ja funcionou
        if chance < 0.20:
            return pai.copiar()

        # A maioria nasce perto dos melhores, mas com diversidade
        if chance < 0.92:

            filho = pai.copiar()

            filho.mutar(
                taxa=0.03,
                intensidade=0.06
            )

            return filho

        # Mantem algumas geneticas novas na populacao
        return Brain()

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

        # O score ja inclui progresso, checkpoints e velocidade util
        self.registrar(
            brain_atual,
            score_atual,
            pista
        )

        # Ainda nao existe elite
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

        # 18%: copia direta de um bom cerebro
        if chance < 0.18:

            pai = (
                self.escolher_elite(
                    pista
                )
            )

            return (
                pai.copiar()
            )

        # 70%: pequena evolucao de um bom cerebro
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

        # 9%: exploracao mais forte
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

        # 3%: genetica totalmente nova
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
