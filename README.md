
# 🏎️ EvoWheels

Simulador de carros autônomos com inteligência artificial e aprendizado por neuroevolução.

Os carros utilizam sensores para identificar obstáculos, tomar decisões e melhorar seu desempenho ao longo das tentativas.

## 🚀 Instalação

**Requisito:** Python 3.12.10

Crie o ambiente virtual:

```powershell
py -3.12 -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

## ▶️ Executar

Com o ambiente virtual ativado:

```powershell
python main.py
```

## 🎮 Controles

### 🏁 Simulação principal

| Tecla | Função |
|---|---|
| `T` | Ativar/desativar treinamento rápido |
| `[` | Diminuir intensidade do treinamento |
| `]` | Aumentar intensidade do treinamento |
| `H` | Abrir/fechar detalhes do painel de estatísticas |
| `S` | Salvar o aprendizado manualmente |
| `R` | Trocar de pista |
| `V` | Abrir/fechar o Neural Inspector |
| `F11` | Alternar tela cheia |

### 🧠 Neural Inspector

Pressione **V** para abrir o painel neural em uma janela separada.

| Tecla | Função |
|---|---|
| `V` | Abrir/fechar o Neural Inspector |
| `TAB` | Selecionar o próximo carro |
| `B` | Ativar/desativar acompanhamento do melhor carro |
| `F11` | Maximizar/restaurar a janela do Inspector |

O Neural Inspector permite visualizar:

- Os 7 sensores frontais do carro.
- A distância e o risco de colisão.
- A velocidade atual.
- A arquitetura da rede neural.
- Os neurônios e suas ativações.
- As conexões neurais em funcionamento.
- As decisões de acelerar, frear e virar.
- As informações da evolução.

**Observação:** o Inspector é apenas visual. Ele não interfere diretamente nas decisões da IA.

## ⚡ Treinamento rápido

Pressione `T` para ativar ou desativar.

Intensidades disponíveis:

- `x2`
- `x4`
- `x8`
- `x12`

Use `[` e `]` para ajustar.

No modo rápido, o simulador prioriza o processamento dos carros e atualiza a tela com menor frequência.

O multiplicador representa o limite de passos por rodada. A aceleração real depende do desempenho do computador.

## 📊 Painel de estatísticas

Pressione **H** para mostrar ou ocultar as informações detalhadas.

O painel apresenta dados como:

| Indicador | Significado |
|---|---|
| FPS | Quadros por segundo |
| Carros | Quantidade de carros simulados |
| Mortes | Tentativas encerradas |
| Avaliações | Total de tentativas avaliadas |
| Elites | Melhores cérebros selecionados |
| Memórias | Aprendizados importantes preservados |
| Recorde global | Maior pontuação alcançada |
| Recorde da pista | Melhor resultado na pista atual |
| Melhor volta | Menor tempo de volta registrado |
| Velocidade média | Velocidade média dos carros |
| Dependência do ritmo | Uso da assistência de aceleração |
| Ajuda nas curvas | Uso da proteção de trajetória |

## 🧠 Como funciona o aprendizado

1. Cada carro possui uma rede neural própria.
2. Os sensores identificam as condições da pista.
3. A IA decide acelerar, frear e virar.
4. O sistema avalia progresso, checkpoints e desempenho.
5. Os melhores cérebros são selecionados.
6. Novos carros herdam e modificam os cérebros selecionados.
7. A evolução continua buscando melhores resultados.

O aprendizado ocorre pela seleção e mutação entre tentativas.

## 💾 Memória automática

O EvoWheels salva automaticamente os conhecimentos importantes, incluindo os melhores cérebros e recordes.

O arquivo de aprendizado fica em:

```text
models/evolution_state.npz
```

Ao reiniciar o programa, o aprendizado salvo é carregado automaticamente.

**Importante:** não apague a pasta `models` se quiser preservar o treinamento.

Você também pode pressionar `S` para salvar manualmente.

## 🚗 Velocidade e sensores

| Configuração | Valor |
|---|---|
| Quantidade de carros | 100 |
| Sensores frontais | 7 |
| Alcance dos sensores | 180 px |
| Velocidade mínima | 25 px/s |
| Velocidade máxima | 175 px/s |
| Direção | Controlada pela IA |
| Proteção de curvas | Ativa |

## 🏁 Como acompanhar a evolução

Observe principalmente:

- Aumento do número de checkpoints alcançados.
- Melhora dos recordes de cada pista.
- Redução dos tempos de volta.
- Redução da dependência das assistências.
- Maior consistência nas curvas.

**Atenção:** muitas avaliações não significam necessariamente que os carros estão aprendendo. O progresso deve ser medido pelos resultados.

## ⏹️ Encerrar

Feche a janela normalmente para encerrar o simulador e salvar os resultados.

Para salvar sem encerrar, pressione `S`.

---

**EvoWheels — Inteligência artificial que aprende a correr.**
