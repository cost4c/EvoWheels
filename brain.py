import numpy as np


class Brain:

    ARQUITETURA = (
        10,
        24,
        48,
        24,
        12,
        3,
    )

    def __init__(self):

        # 7 sensores
        # velocidade
        # esterco
        # tracao
        #
        # Total: 10 entradas
        self.camadas = list(
            self.ARQUITETURA
        )

        self.pesos = []
        self.biases = []

        # -----------------------------------------------------
        # CRIAR REDE
        # -----------------------------------------------------

        for entradas, saidas in zip(
            self.camadas[:-1],
            self.camadas[1:]
        ):

            escala = (
                1.0
                / np.sqrt(entradas)
            )

            pesos = np.random.uniform(
                -1.0,
                1.0,
                (
                    saidas,
                    entradas
                )
            ).astype(
                np.float32
            )

            pesos *= escala

            biases = np.zeros(
                saidas,
                dtype=np.float32
            )

            self.pesos.append(
                pesos
            )

            self.biases.append(
                biases
            )

        self.limpar_telemetria()

    # ---------------------------------------------------------
    # TELEMETRIA
    # ---------------------------------------------------------

    def limpar_telemetria(self):

        self.ultima_entrada = np.zeros(
            self.camadas[0],
            dtype=np.float32
        )

        self.ultimas_ativacoes = []

        self.ultima_saida_bruta = np.zeros(
            self.camadas[-1],
            dtype=np.float32
        )

        self.ultima_decisao = {
            "acelerar": 0.0,
            "frear": 0.0,
            "virar": 0.0,
        }

        self.contador_decisoes = 0

    # ---------------------------------------------------------
    # PENSAR
    # ---------------------------------------------------------

    def pensar(
        self,
        entradas
    ):

        if len(entradas) != self.camadas[0]:

            raise ValueError(
                f"O cerebro precisa receber "
                f"{self.camadas[0]} entradas, "
                f"mas recebeu {len(entradas)}."
            )

        resultado = np.asarray(
            entradas,
            dtype=np.float32
        )

        # Uma copia so para desacoplar da lista que chegou
        resultado = (
            resultado.copy()
        )

        self.ultima_entrada = (
            resultado
        )

        self.ultimas_ativacoes = [
            resultado
        ]

        # Toda a rede e calculada pelo NumPy
        for pesos, biases in zip(
            self.pesos,
            self.biases
        ):

            resultado = np.tanh(
                pesos @ resultado
                + biases
            )

            self.ultimas_ativacoes.append(
                resultado
            )

        self.ultima_saida_bruta = (
            resultado
        )

        acelerar = max(
            0.0,
            float(
                resultado[0]
            )
        )

        frear = max(
            0.0,
            float(
                resultado[1]
            )
        )

        virar = float(
            resultado[2]
        )

        self.ultima_decisao = {
            "acelerar": acelerar,
            "frear": frear,
            "virar": virar,
        }

        self.contador_decisoes += 1

        return self.ultima_decisao

    # ---------------------------------------------------------
    # COPIAR
    # ---------------------------------------------------------

    def copiar(self):

        novo = Brain.__new__(
            Brain
        )

        novo.camadas = (
            self.camadas.copy()
        )

        novo.pesos = [
            pesos.copy()
            for pesos in self.pesos
        ]

        novo.biases = [
            biases.copy()
            for biases in self.biases
        ]

        novo.limpar_telemetria()

        return novo

    # ---------------------------------------------------------
    # CARREGAR PARAMETROS
    # ---------------------------------------------------------

    @classmethod
    def criar_com_parametros(
        cls,
        pesos,
        biases
    ):

        arquitetura = list(
            cls.ARQUITETURA
        )

        if (
            len(pesos)
            != len(arquitetura) - 1
            or len(biases)
            != len(arquitetura) - 1
        ):

            raise ValueError(
                "Quantidade de camadas incompativel com o cerebro atual."
            )

        novos_pesos = []
        novos_biases = []

        for indice, (entradas, saidas) in enumerate(
            zip(
                arquitetura[:-1],
                arquitetura[1:]
            )
        ):

            peso = np.asarray(
                pesos[indice],
                dtype=np.float32
            )

            bias = np.asarray(
                biases[indice],
                dtype=np.float32
            )

            if peso.shape != (
                saidas,
                entradas
            ):

                raise ValueError(
                    f"Peso da camada {indice} tem formato invalido: "
                    f"{peso.shape}."
                )

            if bias.shape != (
                saidas,
            ):

                raise ValueError(
                    f"Bias da camada {indice} tem formato invalido: "
                    f"{bias.shape}."
                )

            novos_pesos.append(
                peso.copy()
            )

            novos_biases.append(
                bias.copy()
            )

        novo = cls.__new__(
            cls
        )

        novo.camadas = arquitetura
        novo.pesos = novos_pesos
        novo.biases = novos_biases

        novo.limpar_telemetria()

        return novo

    # ---------------------------------------------------------
    # MUTAR
    # ---------------------------------------------------------

    def mutar(
        self,
        taxa=0.05,
        intensidade=0.15
    ):

        for indice in range(
            len(self.pesos)
        ):

            # -------------------------------------------------
            # PESOS
            # -------------------------------------------------

            pesos = self.pesos[
                indice
            ]

            mascara = (
                np.random.random(
                    pesos.shape
                )
                < taxa
            )

            quantidade = int(
                mascara.sum()
            )

            if quantidade > 0:

                pesos[
                    mascara
                ] += np.random.normal(
                    0.0,
                    intensidade,
                    quantidade
                ).astype(
                    np.float32
                )

            # -------------------------------------------------
            # BIASES
            # -------------------------------------------------

            biases = self.biases[
                indice
            ]

            mascara_bias = (
                np.random.random(
                    biases.shape
                )
                < taxa
            )

            quantidade_bias = int(
                mascara_bias.sum()
            )

            if quantidade_bias > 0:

                biases[
                    mascara_bias
                ] += np.random.normal(
                    0.0,
                    intensidade,
                    quantidade_bias
                ).astype(
                    np.float32
                )
