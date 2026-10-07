# Importar as bibliotecas necessárias
import math
import random
import copy


# Criar um neurônio artificial
def neuronio(entradas, pesos, bias):
    resultado = sum(
        entrada * peso
        for entrada, peso in zip(entradas, pesos)
    )

    resultado += bias

    return math.tanh(resultado)


# Criar uma camada de neurônios
def camada(entradas, pesos, biases):
    resultados = []

    for peso, bias in zip(pesos, biases):
        resultado = neuronio(entradas, peso, bias)
        resultados.append(resultado)

    return resultados


# Criar o cérebro dos carrinhos
class Brain:
    def __init__(self):

        # Quantidade de neurônios em cada camada
        self.camadas = [16, 32, 64, 32, 16, 16, 3]

        self.pesos = []
        self.biases = []

        # Criar os pesos e biases iniciais
        for i in range(len(self.camadas) - 1):

            quantidade_entradas = self.camadas[i]
            quantidade_neuronios = self.camadas[i + 1]

            pesos_camada = []
            biases_camada = []

            for _ in range(quantidade_neuronios):

                pesos_neuronio = []

                for _ in range(quantidade_entradas):
                    peso = random.uniform(-1, 1)
                    peso /= math.sqrt(quantidade_entradas)

                    pesos_neuronio.append(peso)

                pesos_camada.append(pesos_neuronio)
                biases_camada.append(0.0)

            self.pesos.append(pesos_camada)
            self.biases.append(biases_camada)


    # Processar as informações recebidas pelos sensores
    def pensar(self, entradas):

        if len(entradas) != self.camadas[0]:
            raise ValueError("Quantidade de entradas incorreta")

        resultado = entradas

        for pesos, biases in zip(self.pesos, self.biases):
            resultado = camada(resultado, pesos, biases)

        # Resultado das decisões do carrinho
        return {
            "acelerar": max(0.0, resultado[0]),
            "frear": max(0.0, resultado[1]),
            "virar": resultado[2]
        }


    # Criar uma cópia do cérebro
    def copiar(self):
        return copy.deepcopy(self)


    # Aplicar mutações nos neurônios
    def mutar(self, taxa=0.05, intensidade=0.15):

        for i in range(len(self.pesos)):

            for j in range(len(self.pesos[i])):

                for k in range(len(self.pesos[i][j])):

                    if random.random() < taxa:
                        self.pesos[i][j][k] += random.gauss(
                            0, intensidade
                        )

                # Mutação dos biases
                if random.random() < taxa:
                    self.biases[i][j] += random.gauss(
                        0, intensidade
                    )


# Testar o cérebro sem precisar iniciar o jogo
if __name__ == "__main__":

    cerebro = Brain()

    # Simular informações dos 16 sensores
    sensores = [0.5] * 16

    # Processar informações
    decisao = cerebro.pensar(sensores)

    print("Decisão do carrinho:")
    print(decisao)

    # Criar um descendente com mutações
    filho = cerebro.copiar()
    filho.mutar()

    print("Decisão do descendente:")
    print(filho.pensar(sensores))