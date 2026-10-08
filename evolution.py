import atexit
import json
import os
import random
import time
import threading
from pathlib import Path

import numpy as np

from brain import Brain


class Evolution:
    """Neuroevolucao com elite e memoria protegida de marcos importantes."""

    STATE_VERSION = 3
    COMPATIBLE_STATE_VERSIONS = (2, 3)
    SAVE_INTERVAL = 15.0

    def __init__(
        self,
        elite_size=10,
        storage_path=None,
        auto_load=True
    ):
        self.elite_size = elite_size

        # Melhores avaliados normalmente.
        self.elites = []
        self.elites_por_pista = {}

        # Memorias protegidas. Elas nao somem quando a elite muda.
        self.memorias_importantes = []

        self.total_avaliacoes = 0
        self.melhor_score = 0.0
        self.avaliacoes_sem_recorde = 0

        if storage_path is None:
            storage_path = (
                Path(__file__).resolve().parent
                / "models"
                / "evolution_state.npz"
            )

        self.storage_path = Path(storage_path)
        self.carregado_do_disco = False
        self.migrado_de_v2 = False
        self._dirty = False
        self._last_save = time.monotonic()
        self._save_thread = None

        if auto_load:
            self.carregar()

        atexit.register(self.salvar)

    # ---------------------------------------------------------
    # SERIALIZACAO DO CEREBRO
    # ---------------------------------------------------------

    @staticmethod
    def _brain_to_arrays(arrays, prefix, brain):
        for indice, peso in enumerate(brain.pesos):
            arrays[f"{prefix}_w_{indice}"] = np.asarray(
                peso,
                dtype=np.float32
            )

        for indice, bias in enumerate(brain.biases):
            arrays[f"{prefix}_b_{indice}"] = np.asarray(
                bias,
                dtype=np.float32
            )

    @staticmethod
    def _brain_from_arrays(data, prefix):
        pesos = []
        biases = []

        for indice in range(len(Brain.ARQUITETURA) - 1):
            pesos.append(data[f"{prefix}_w_{indice}"])
            biases.append(data[f"{prefix}_b_{indice}"])

        return Brain.criar_com_parametros(pesos, biases)

    # ---------------------------------------------------------
    # PERSISTENCIA
    # ---------------------------------------------------------

    def _montar_snapshot(self):
        # Prepara uma fotografia independente para gravar sem parar a corrida.
        metadata = {
            "version": self.STATE_VERSION,
            "architecture": list(Brain.ARQUITETURA),
            "elite_size": self.elite_size,
            "total_avaliacoes": self.total_avaliacoes,
            "melhor_score": self.melhor_score,
            "avaliacoes_sem_recorde": self.avaliacoes_sem_recorde,
            "global": [], "tracks": [], "memories": [],
        }
        arrays = {}

        for indice, record in enumerate(self.elites):
            metadata["global"].append({
                "score": float(record["score"]),
                "pista": record.get("pista"),
            })
            self._brain_to_arrays(arrays, f"g_{indice}", record["brain"])

        for pista_indice, (pista, records) in enumerate(
            self.elites_por_pista.items()
        ):
            track_meta = {"pista": pista, "records": []}
            for record_indice, record in enumerate(records):
                track_meta["records"].append({
                    "score": float(record["score"])
                })
                self._brain_to_arrays(
                    arrays, f"t_{pista_indice}_{record_indice}",
                    record["brain"]
                )
            metadata["tracks"].append(track_meta)

        for indice, record in enumerate(self.memorias_importantes):
            metadata["memories"].append({
                "tipo": record["tipo"],
                "pista": record.get("pista"),
                "score": float(record["score"]),
                "checkpoints": int(record.get("checkpoints", 0)),
                "voltas": int(record.get("voltas", 0)),
                "melhor_tempo_volta": record.get("melhor_tempo_volta"),
            })
            self._brain_to_arrays(arrays, f"m_{indice}", record["brain"])

        arrays["metadata"] = np.array(
            json.dumps(metadata, ensure_ascii=False)
        )
        # As elites sao fotografias imutaveis, mas o disco recebe suas
        # proprias matrizes para que a thread nao dependa da simulacao.
        return {chave: valor.copy() for chave, valor in arrays.items()}

    def _gravar_snapshot(self, arrays):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.storage_path.with_suffix(
            self.storage_path.suffix + ".tmp"
        )
        try:
            with open(temp_path, "wb") as arquivo:
                # Continua NPZ, mas sem compactacao cara a cada autosave.
                # np.load() le normalmente os formatos comprimido e normal.
                np.savez(arquivo, **arrays)
            os.replace(temp_path, self.storage_path)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _gravar_em_segundo_plano(self, arrays):
        try:
            self._gravar_snapshot(arrays)
        except Exception as erro:
            # Se falhar, novas avaliacoes tentam salvar novamente.
            self._dirty = True
            print(f"[EvoWheels] Falha no autosave: {erro}")

    def salvar(self, force=True):
        # Salvamento manual e encerramento sempre aguardam o autosave.
        thread = self._save_thread
        if thread is not None:
            thread.join()
            self._save_thread = None

        if not self._dirty and self.storage_path.exists():
            return False
        if not force and not self._dirty:
            return False

        try:
            snapshot = self._montar_snapshot()
            self._gravar_snapshot(snapshot)
            self._dirty = False
            self._last_save = time.monotonic()
            return True
        except Exception as erro:
            print(f"[EvoWheels] Nao foi possivel salvar o aprendizado: {erro}")
            return False

    def _salvar_se_preciso(self):
        if not self._dirty:
            return
        agora = time.monotonic()
        if agora - self._last_save < self.SAVE_INTERVAL:
            return

        # Nunca cria fila de gravacoes. So uma tarefa por vez.
        thread = self._save_thread
        if thread is not None:
            if thread.is_alive():
                return
            thread.join()
            self._save_thread = None

        try:
            snapshot = self._montar_snapshot()
            self._dirty = False
            self._last_save = agora
            self._save_thread = threading.Thread(
                target=self._gravar_em_segundo_plano,
                args=(snapshot,),
                name="EvoWheels-AutoSave",
                daemon=True,
            )
            self._save_thread.start()
        except Exception as erro:
            self._dirty = True
            print(f"[EvoWheels] Nao foi possivel iniciar autosave: {erro}")

    def carregar(self):
        if not self.storage_path.exists():
            return False

        try:
            with np.load(
                self.storage_path,
                allow_pickle=False
            ) as data:
                metadata = json.loads(
                    str(data["metadata"].item())
                )

                version = int(metadata.get("version", -1))
                if version not in self.COMPATIBLE_STATE_VERSIONS:
                    raise ValueError(
                        "versao do arquivo de treino incompativel"
                    )

                arquitetura = tuple(
                    metadata.get("architecture", ())
                )
                if arquitetura != tuple(Brain.ARQUITETURA):
                    raise ValueError(
                        "arquitetura salva diferente da arquitetura atual"
                    )

                elites = []
                for indice, record_meta in enumerate(
                    metadata.get("global", [])[:self.elite_size]
                ):
                    elites.append(
                        {
                            "score": float(record_meta["score"]),
                            "brain": self._brain_from_arrays(
                                data,
                                f"g_{indice}"
                            ),
                            "pista": record_meta.get("pista"),
                        }
                    )

                elites_por_pista = {}
                for pista_indice, track_meta in enumerate(
                    metadata.get("tracks", [])
                ):
                    pista = track_meta.get("pista")
                    if pista is None:
                        continue

                    records = []
                    for record_indice, record_meta in enumerate(
                        track_meta.get("records", [])[:self.elite_size]
                    ):
                        records.append(
                            {
                                "score": float(record_meta["score"]),
                                "brain": self._brain_from_arrays(
                                    data,
                                    f"t_{pista_indice}_{record_indice}"
                                ),
                                "pista": pista,
                            }
                        )

                    if records:
                        elites_por_pista[pista] = records

                memorias = []
                if version >= 3:
                    for indice, memory_meta in enumerate(
                        metadata.get("memories", [])
                    ):
                        memorias.append(
                            {
                                "tipo": memory_meta["tipo"],
                                "pista": memory_meta.get("pista"),
                                "score": float(memory_meta["score"]),
                                "checkpoints": int(
                                    memory_meta.get("checkpoints", 0)
                                ),
                                "voltas": int(
                                    memory_meta.get("voltas", 0)
                                ),
                                "melhor_tempo_volta": memory_meta.get(
                                    "melhor_tempo_volta"
                                ),
                                "brain": self._brain_from_arrays(
                                    data,
                                    f"m_{indice}"
                                ),
                            }
                        )

                self.elites = elites
                self.elites_por_pista = elites_por_pista
                self.memorias_importantes = memorias
                self.total_avaliacoes = int(
                    metadata.get("total_avaliacoes", 0)
                )
                self.melhor_score = float(
                    metadata.get("melhor_score", 0.0)
                )
                self.avaliacoes_sem_recorde = int(
                    metadata.get("avaliacoes_sem_recorde", 0)
                )

                if self.elites:
                    self.melhor_score = max(
                        self.melhor_score,
                        float(self.elites[0]["score"])
                    )

                # A versao 2 ja tinha elites validas. Em vez de apagar,
                # transforma os melhores em memorias protegidas.
                if version == 2:
                    self._migrar_elites_para_memoria()
                    self.migrado_de_v2 = True

            self.carregado_do_disco = True
            self._last_save = time.monotonic()

            if self.migrado_de_v2:
                self._dirty = True
                self.salvar(force=False)
                print(
                    "[EvoWheels] Aprendizado V2 preservado e migrado "
                    "para a memoria automatica V3."
                )
            else:
                self._dirty = False

            print(
                "[EvoWheels] Aprendizado carregado: "
                f"{len(self.elites)} elites, "
                f"{len(self.memorias_importantes)} memorias, "
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
    # ELITE NORMAL
    # ---------------------------------------------------------

    def qualifies(self, elite_list, score):
        if score <= 0.0:
            return False

        if len(elite_list) < self.elite_size:
            return True

        return score > elite_list[-1]["score"]

    @staticmethod
    def insert_record(elite_list, record):
        score = record["score"]
        position = len(elite_list)

        for index, existing in enumerate(elite_list):
            if score > existing["score"]:
                position = index
                break

        elite_list.insert(position, record)

    def _trim_elite(self, elite_list):
        if len(elite_list) > self.elite_size:
            del elite_list[self.elite_size:]

    # ---------------------------------------------------------
    # MEMORIA IMPORTANTE
    # ---------------------------------------------------------

    @staticmethod
    def _normalizar_metricas(metricas):
        metricas = metricas or {}

        tempo = metricas.get("melhor_tempo_volta")
        if tempo is not None:
            tempo = float(tempo)
            if tempo <= 0.0:
                tempo = None

        return {
            "checkpoints": max(
                0,
                int(metricas.get("checkpoints", 0))
            ),
            "voltas": max(
                0,
                int(metricas.get("voltas", 0))
            ),
            "melhor_tempo_volta": tempo,
        }

    def _buscar_memoria(self, tipo, pista):
        for record in self.memorias_importantes:
            if (
                record["tipo"] == tipo
                and record.get("pista") == pista
            ):
                return record
        return None

    def _substituir_memoria(self, tipo, pista, record):
        for indice, existing in enumerate(self.memorias_importantes):
            if (
                existing["tipo"] == tipo
                and existing.get("pista") == pista
            ):
                self.memorias_importantes[indice] = record
                return

        self.memorias_importantes.append(record)

    def _tipos_memoria_para_atualizar(
        self,
        score,
        pista,
        metricas
    ):
        tipos = []

        global_best = self._buscar_memoria(
            "recorde_global",
            None
        )
        if global_best is None or score > global_best["score"]:
            tipos.append(("recorde_global", None))

        if pista is None:
            return tipos

        track_best = self._buscar_memoria(
            "melhor_score",
            pista
        )
        if track_best is None or score > track_best["score"]:
            tipos.append(("melhor_score", pista))

        progress_best = self._buscar_memoria(
            "maior_progresso",
            pista
        )
        current_progress = (
            metricas["voltas"],
            metricas["checkpoints"],
            score,
        )

        if progress_best is None:
            if metricas["checkpoints"] > 0 or metricas["voltas"] > 0:
                tipos.append(("maior_progresso", pista))
        else:
            previous_progress = (
                int(progress_best.get("voltas", 0)),
                int(progress_best.get("checkpoints", 0)),
                float(progress_best.get("score", 0.0)),
            )
            if current_progress > previous_progress:
                tipos.append(("maior_progresso", pista))

        tempo = metricas["melhor_tempo_volta"]
        if tempo is not None:
            fastest = self._buscar_memoria(
                "volta_mais_rapida",
                pista
            )
            if (
                fastest is None
                or fastest.get("melhor_tempo_volta") is None
                or tempo < float(fastest["melhor_tempo_volta"])
            ):
                tipos.append(("volta_mais_rapida", pista))

        return tipos

    def registrar_memoria(
        self,
        brain,
        score,
        pista=None,
        metricas=None
    ):
        score = float(score)
        if score <= 0.0:
            return False

        metricas = self._normalizar_metricas(metricas)
        tipos = self._tipos_memoria_para_atualizar(
            score,
            pista,
            metricas
        )

        if not tipos:
            return False

        # Uma unica copia pode ocupar mais de um marco porque ela nunca
        # e mutada dentro da memoria. Filhos sempre recebem outra copia.
        snapshot = brain.copiar()

        for tipo, memoria_pista in tipos:
            self._substituir_memoria(
                tipo,
                memoria_pista,
                {
                    "tipo": tipo,
                    "pista": memoria_pista,
                    "score": score,
                    "checkpoints": metricas["checkpoints"],
                    "voltas": metricas["voltas"],
                    "melhor_tempo_volta": metricas[
                        "melhor_tempo_volta"
                    ],
                    "brain": snapshot,
                }
            )

        # A gravação ocorre depois de registrar também a elite.
        # Assim a memória e a elite saem na mesma fotografia.
        self._dirty = True
        return True

    def _migrar_elites_para_memoria(self):
        if self.memorias_importantes:
            return

        if self.elites:
            best = self.elites[0]
            self.memorias_importantes.append(
                {
                    "tipo": "recorde_global",
                    "pista": None,
                    "score": float(best["score"]),
                    "checkpoints": 0,
                    "voltas": 0,
                    "melhor_tempo_volta": None,
                    "brain": best["brain"].copiar(),
                }
            )

        for pista, records in self.elites_por_pista.items():
            if not records:
                continue
            best = records[0]
            self.memorias_importantes.append(
                {
                    "tipo": "melhor_score",
                    "pista": pista,
                    "score": float(best["score"]),
                    "checkpoints": 0,
                    "voltas": 0,
                    "melhor_tempo_volta": None,
                    "brain": best["brain"].copiar(),
                }
            )

    # ---------------------------------------------------------
    # REGISTRAR AVALIACAO
    # ---------------------------------------------------------

    def registrar(
        self,
        brain,
        score,
        pista=None,
        metricas=None
    ):
        score = float(score)
        self.total_avaliacoes += 1
        self._dirty = True

        if score > self.melhor_score:
            self.melhor_score = score
            self.avaliacoes_sem_recorde = 0
        else:
            self.avaliacoes_sem_recorde += 1

        self.registrar_memoria(
            brain,
            score,
            pista,
            metricas
        )

        if score <= 0.0:
            self._salvar_se_preciso()
            return

        track_elite = None
        if pista is not None:
            track_elite = self.elites_por_pista.setdefault(
                pista,
                []
            )

        global_qualifies = self.qualifies(
            self.elites,
            score
        )
        track_qualifies = (
            track_elite is not None
            and self.qualifies(track_elite, score)
        )

        if not global_qualifies and not track_qualifies:
            self._salvar_se_preciso()
            return

        snapshot = brain.copiar()

        if global_qualifies:
            self.insert_record(
                self.elites,
                {
                    "score": score,
                    "brain": snapshot,
                    "pista": pista,
                }
            )
            self._trim_elite(self.elites)

        if track_qualifies:
            self.insert_record(
                track_elite,
                {
                    "score": score,
                    "brain": snapshot,
                    "pista": pista,
                }
            )
            self._trim_elite(track_elite)

        self._salvar_se_preciso()

    # ---------------------------------------------------------
    # ESCOLHER CONHECIMENTO
    # ---------------------------------------------------------

    @staticmethod
    def weighted_choice(elite_list):
        count = len(elite_list)
        if count == 0:
            return None

        weights = [
            count - index
            for index in range(count)
        ]
        total = sum(weights)
        target = random.random() * total
        accumulated = 0.0

        for record, weight in zip(elite_list, weights):
            accumulated += weight
            if target <= accumulated:
                return record["brain"]

        return elite_list[-1]["brain"]

    def escolher_elite(self, pista=None):
        if not self.elites:
            return None

        if (
            pista is not None
            and pista in self.elites_por_pista
            and self.elites_por_pista[pista]
            and random.random() < 0.80
        ):
            chosen = self.weighted_choice(
                self.elites_por_pista[pista]
            )
            if chosen is not None:
                return chosen

        return self.weighted_choice(self.elites)

    def escolher_memoria(self, pista=None):
        if not self.memorias_importantes:
            return None

        locais = [
            record
            for record in self.memorias_importantes
            if pista is not None and record.get("pista") == pista
        ]

        candidatos = locais if locais else self.memorias_importantes

        # A volta mais rapida e o maior progresso aparecem um pouco mais
        # para que velocidade e conhecimento de percurso sejam lembrados.
        ponderados = []
        for record in candidatos:
            peso = 1
            if record["tipo"] in (
                "maior_progresso",
                "volta_mais_rapida",
            ):
                peso = 2
            ponderados.extend([record] * peso)

        return random.choice(ponderados)["brain"]

    def escolher_base(self, pista=None):
        memoria = self.escolher_memoria(pista)
        elite = self.escolher_elite(pista)

        if memoria is not None and (
            elite is None or random.random() < 0.40
        ):
            return memoria

        if elite is not None:
            return elite

        return memoria

    # ---------------------------------------------------------
    # NOVOS CEREBROS
    # ---------------------------------------------------------

    def criar_cerebro_inicial(self, pista=None):
        pai = self.escolher_base(pista)
        if pai is None:
            return Brain()

        chance = random.random()

        if chance < 0.15:
            return pai.copiar()

        if chance < 0.85:
            filho = pai.copiar()
            filho.mutar(
                taxa=0.04,
                intensidade=0.08
            )
            return filho

        return Brain()

    def fator_exploracao(self):
        if self.avaliacoes_sem_recorde >= 1200:
            return 1.80
        if self.avaliacoes_sem_recorde >= 600:
            return 1.50
        if self.avaliacoes_sem_recorde >= 300:
            return 1.25
        return 1.0

    def criar_descendente(
        self,
        brain_atual,
        score_atual,
        pista=None,
        metricas=None
    ):
        self.registrar(
            brain_atual,
            score_atual,
            pista,
            metricas
        )

        pai = self.escolher_base(pista)
        if pai is None:
            filho = brain_atual.copiar()
            filho.mutar(
                taxa=0.14,
                intensidade=0.22
            )
            return filho

        chance = random.random()
        fator = self.fator_exploracao()

        # Preserva uma pequena parcela de conhecimento exatamente.
        if chance < 0.10:
            return pai.copiar()

        # Maior parte explora perto de elite ou memoria importante.
        if chance < 0.65:
            filho = pai.copiar()
            filho.mutar(
                taxa=min(0.12, 0.05 * fator),
                intensidade=min(0.26, 0.10 * fator)
            )
            return filho

        # Mantem a propria linhagem para nao virar uma unica familia.
        if chance < 0.85:
            filho = brain_atual.copiar()
            filho.mutar(
                taxa=min(0.30, 0.12 * fator),
                intensidade=min(0.48, 0.22 * fator)
            )
            return filho

        # Sempre existe exploracao totalmente nova.
        return Brain()

    # ---------------------------------------------------------
    # ESTATISTICAS
    # ---------------------------------------------------------

    def quantidade_elites(self):
        return len(self.elites)

    def quantidade_memorias(self):
        return len(self.memorias_importantes)

    def melhor_fitness(self):
        if not self.elites:
            return 0.0
        return float(self.elites[0]["score"])

    def melhor_fitness_pista(self, pista):
        elite = self.elites_por_pista.get(pista, [])
        if not elite:
            return 0.0
        return float(elite[0]["score"])

    def melhor_tempo_volta(self, pista):
        record = self._buscar_memoria(
            "volta_mais_rapida",
            pista
        )
        if record is None:
            return None
        return record.get("melhor_tempo_volta")
