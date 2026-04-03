# 🍎 Fruit Classification: Fresh vs Rotten
### Deep Learning | Transfer Learning | Data Augmentation

Este projeto foi desenvolvido como parte de uma avaliação técnica da **NVIDIA Deep Learning Institute (DLI)**. O objetivo é classificar imagens de frutas entre frescas e estragadas, utilizando uma rede neural profunda com foco em alta acurácia e generalização.

---

## 📋 Visão Geral
O classificador foi treinado para identificar 6 categorias distintas de frutas:
* **Maçãs**: Frescas e Estragadas
* **Bananas**: Frescas e Estragadas
* **Laranjas**: Frescas e Estragadas

A meta estabelecida para aprovação era uma acurácia de validação de pelo menos **92%**.

---

## 🏗️ Arquitetura do Modelo
Para este desafio, utilizei a técnica de **Transfer Learning** (Aprendizado por Transferência).

* **Base Model**: VGG16 pré-treinada no dataset ImageNet.
* **Feature Extraction**: As camadas convolucionais da VGG16 foram mantidas para extrair características complexas das imagens.
* **Custom Head**: 
    * `GlobalAveragePooling2D` para redução de dimensionalidade.
    * Camada Densa com 256 neurônios e ativação `ReLU`.
    * `Dropout (0.5)` para prevenir overfitting.
    * Camada de saída com 6 neurônios e ativação `Softmax`.

---

## 🛠️ Técnicas Aplicadas

### 1. Data Augmentation
Devido ao tamanho do dataset, apliquei aumento de dados em tempo real para tornar o modelo mais robusto contra variações de ângulo, zoom e posição:
* Rotação, Zoom, Deslocamento e Horizontal Flip.

### 2. Fine-Tuning e Otimização
* **Optimizer**: `RMSprop` com uma taxa de aprendizado reduzida ($10^{-5}$) para garantir estabilidade durante o ajuste fino dos pesos.
* **Loss Function**: `Categorical Crossentropy` (apropriada para múltiplas classes).

---

## 📈 Resultados
O modelo alcançou uma performance sólida, superando os requisitos da avaliação:
* **Acurácia de Validação**: ~93.75%
* **Loss Final**: ~0.17

![Gráfico de Perda](loss_graph.png)
> *Convergência estável observada entre o erro de treinamento e validação ao longo de 16 épocas.*

---

## 💻 Como Executar
1. Clone o repositório.
2. Certifique-se de ter as bibliotecas instaladas:
   ```bash
   pip install tensorflow matplotlib