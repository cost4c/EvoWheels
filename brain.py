import numpy as np


class Brain:

    def __init__(self):

        # 7 sensores
        # velocidade
        # esterco
        # tracao
        #
        # Total: 10 entradas
        self.camadas = [
            10,
            24,
            48,
            24,
            12,
            3,
        ]

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

        # Instinto de corrida: nasce acelerando e com a direcao centrada
        self.pesos[-1][0] *= 0.15
        self.pesos[-1][1] *= 0.15
        self.pesos[-1][2] *= 0.08

        self.biases[-1][0] = 1.15
        self.biases[-1][1] = -1.00
        self.biases[-1][2] = 0.00

        # -----------------------------------------------------
        # TELEMETRIA
        # -----------------------------------------------------

        self.ultima_entrada = np.zeros(
            self.camadas[0],
            dtype=np.float32
        )

        self.ultimas_ativacoes = []

        self.ultima_saida_bruta = np.zeros(
            3,
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

        # Uma copia só para desacoplar da lista que chegou.
        resultado = (
            resultado.copy()
        )

        self.ultima_entrada = (
            resultado
        )

        self.ultimas_ativacoes = [
            resultado
        ]

        # Toda a rede e calculada pelo NumPy.
        #
        # Cada np.tanh ja devolve um array novo e nada o altera
        # depois, entao guardar a telemetria nao precisa copiar
        # de novo.
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

        novo.ultima_entrada = np.zeros(
            novo.camadas[0],
            dtype=np.float32
        )

        novo.ultimas_ativacoes = []

        novo.ultima_saida_bruta = np.zeros(
            3,
            dtype=np.float32
        )

        novo.ultima_decisao = {
            "acelerar": 0.0,
            "frear": 0.0,
            "virar": 0.0,
        }

        novo.contador_decisoes = 0

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