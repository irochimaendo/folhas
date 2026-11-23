# Classificador de doenças em folhas (Naive Bayes)

Site local em Flask que recebe a foto de uma folha e indica a classe mais provável
(planta e doença, ou folha saudável), com a probabilidade estimada.

**Como funciona:** a foto passa por um extrator de características (ConvNeXt-Tiny pré-treinada,
histograma de cor HSV e textura LBP, 810 números no total) e esse vetor é classificado por um
modelo **Naive Bayes** (`PowerTransformer` + `PCA` + `GaussianNB`, com probabilidades calibradas).
O dataset de treino foi o PlantVillage.

## Estrutura esperada da pasta

```
folhas/
├── app.py
├── extrator_personalizado.py
├── templates/
│   └── index.html
└── saida_treinamento/
    └── modelo_folhas.joblib
```

Os nomes precisam ser exatamente estes. O `index.html` tem que ficar dentro de `templates`, e o
arquivo do extrator não pode ter sufixos como `(1)` ou `_1`.

## Requisitos

- Python 3.10 ou mais novo
- Cerca de 2 GB livres para as bibliotecas (o PyTorch é a maior)
- Internet na primeira execução, para baixar os pesos da ConvNeXt (uns 100 MB)

## Instalação (uma vez só)

Abra um terminal **dentro da pasta `folhas`**. No Windows, dá para digitar `cmd` na barra de
endereço do Explorer e apertar Enter.

```
py -m venv venv
venv\Scripts\activate
pip install flask joblib numpy opencv-python scikit-image scikit-learn torch torchvision
```

No Linux ou macOS, troque as duas primeiras linhas por:

```
python3 -m venv venv
source venv/bin/activate
```

O terminal mostra `(venv)` no início da linha quando o ambiente está ativo. Nas próximas vezes, só
é preciso ativá-lo de novo (`venv\Scripts\activate`), sem reinstalar nada.

## Como rodar

```
python app.py
```

Quando aparecer `Running on http://127.0.0.1:5000`, abra esse endereço no navegador.

Para encerrar, volte ao terminal e aperte `Ctrl+C`.

## Como usar o site

Há quatro formas de enviar a imagem:

1. **Escolher arquivo:** clique na área tracejada e selecione uma imagem.
2. **Ctrl+V:** cole uma imagem copiada de outro site, do chat ou de um print.
3. **Botão "Colar imagem":** lê a imagem direto da área de transferência. O navegador pode pedir
   permissão na primeira vez.
4. **Arrastar e soltar:** arraste um arquivo para a área tracejada.

Depois, clique em **Prever**. A página mostra a classe mais provável, a probabilidade estimada e as
três classes mais prováveis.

Se a maior probabilidade ficar abaixo de 70%, o site avisa que não tem certeza. Nesse caso, tente
outra foto.

## Dicas para um bom resultado

- Fotografe **uma única folha**, de perto, com boa luz e a folha inteira no quadro.
- Fundo liso ajuda. O modelo foi treinado com fotos de laboratório (fundo uniforme), então fotos
  de campo, com solo, mãos e sombras, tendem a ter resultado pior.
- O modelo só conhece as classes do dataset de treino. Uma planta ou doença fora dessa lista
  receberá a classe parecida mais próxima, geralmente com baixa confiança.
- Para conferir se está funcionando, envie fotos do próprio dataset e veja se a classe prevista
  bate com a pasta de onde a foto veio.

## Limitações

- Esta é uma ferramenta de apoio e **não substitui a avaliação de um agrônomo**.
- A primeira previsão demora mais, porque o extrator carrega (e baixa, na primeira vez) a rede
  ConvNeXt. As seguintes são rápidas.
- O site roda em modo de desenvolvimento do Flask, só para uso local.

## Problemas comuns

| Mensagem ou sintoma | O que fazer |
|---|---|
| `ModuleNotFoundError: extrator_personalizado` | O arquivo precisa se chamar exatamente `extrator_personalizado.py` e estar na mesma pasta do `app.py`. |
| `ModuleNotFoundError` de outra biblioteca | O ambiente virtual não está ativo (`venv\Scripts\activate`) ou falta instalar o pacote. |
| `TemplateNotFound: index.html` | Crie a pasta `templates` e coloque o `index.html` dentro dela. |
| `FileNotFoundError: ... modelo_folhas.joblib` | Confira se o arquivo está em `saida_treinamento/` e se o `app.py` carrega `saida_treinamento/modelo_folhas.joblib`. |
| `InconsistentVersionWarning` no terminal | É só um aviso: o modelo foi treinado com o scikit-learn 1.6.1 e você tem outra versão. Para eliminá-lo, tente `pip install scikit-learn==1.6.1`. Se der erro de instalação (comum em Python muito novo, como o 3.14), valide o site com fotos do dataset e ignore o aviso, ou use um ambiente com Python 3.12 ou 3.13. |
| Falha ao instalar o `torch` | Confira a versão do Python (`python --version`). Versões muito novas podem não ter pacote pronto; use o Python 3.12 ou 3.13. |
| A página não atualiza depois de editar o `index.html` | Recarregue com `Ctrl+F5`. |
| O botão "Colar imagem" não funciona | Use `Ctrl+V` com a página em foco, ou arraste o arquivo. |
| "Não foi possível ler a imagem" | Envie um arquivo de imagem comum (JPG, PNG, WEBP). |
