"""Extrator de características para classificação de doenças foliares.

O vetor de saída mantém 810 dimensões:
  768 ConvNeXt-Tiny (ImageNet) + 32 histograma HSV + 10 histograma LBP.

IMPORTANTE: use este mesmo arquivo no treinamento e na inferência do site.
As características são extraídas da imagem original; o treinamento não faz
uma segunda passagem desnecessária com imagens espelhadas.
"""

from threading import Lock

import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern
from torchvision import models

TAM = 224
DIM_CNN = 768
DIM_HSV = 32
DIM_LBP = 10
TOTAL_FEATURES = DIM_CNN + DIM_HSV + DIM_LBP
VERSAO_EXTRATOR = "2.0.0"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_MODEL_LOCK = Lock()

# Em GPU, AMP reduz memória e costuma acelerar a inferência. O resultado final
# é convertido para float32. Como treino e site importam este mesmo módulo,
# ambos usam exatamente a mesma configuração.
USAR_AMP = _DEVICE.type == "cuda"


def dispositivo():
    """Retorna o dispositivo usado pelo extrator (cuda ou cpu)."""
    return str(_DEVICE)


def _obter_modelo():
    """Carrega a ConvNeXt somente na primeira chamada."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                pesos = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                modelo = models.convnext_tiny(weights=pesos)
                modelo.classifier[2] = torch.nn.Identity()
                modelo.eval().to(_DEVICE)
                _MODEL = modelo
    return _MODEL


def _redimensionar_preservando(img):
    """Centraliza a imagem em um quadrado com padding neutro e redimensiona."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Imagem vazia ou inválida.")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Formato de imagem não suportado. Esperado BGR, BGRA ou escala de cinza.")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    if h == TAM and w == TAM:
        return np.ascontiguousarray(img)

    lado = max(h, w)
    top = (lado - h) // 2
    bottom = lado - h - top
    left = (lado - w) // 2
    right = lado - w - left
    quadrada = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return cv2.resize(quadrada, (TAM, TAM), interpolation=cv2.INTER_AREA)


def ler_imagem(origem):
    """Lê uma imagem de caminho ou bytes e devolve imagem BGR 224x224."""
    if isinstance(origem, (bytes, bytearray, memoryview)):
        img = cv2.imdecode(np.frombuffer(origem, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(origem), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {origem}")
    return _redimensionar_preservando(img)


def segmentar(img):
    """Cria máscara aproximada da folha usando saturação, Otsu e morfologia."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturacao = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, mascara = cv2.threshold(
        saturacao, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    if mascara.mean() / 255.0 < 0.10:
        mascara = np.full_like(mascara, 255)
    return mascara


def _normalizar_histograma(vetor):
    return vetor / (float(vetor.sum()) + 1e-9)


def _caracteristicas_cor(img, mascara):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mascara, [16], [0, 180]).ravel()
    hist_s = cv2.calcHist([hsv], [1], mascara, [8], [0, 256]).ravel()
    hist_v = cv2.calcHist([hsv], [2], mascara, [8], [0, 256]).ravel()
    return np.concatenate(
        [
            _normalizar_histograma(hist_h),
            _normalizar_histograma(hist_s),
            _normalizar_histograma(hist_v),
        ]
    ).astype(np.float32)


def _caracteristicas_textura(img, mascara):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(cinza, P=8, R=1, method="uniform")
    valores = lbp[mascara > 0].astype(np.int32)
    hist = np.bincount(valores, minlength=10)[:10].astype(np.float32)
    return _normalizar_histograma(hist).astype(np.float32)


def _tensor_imagens(imagens):
    """Converte imagens BGR uint8 em tensor RGB normalizado para ImageNet."""
    rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imagens], axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
    tensor = tensor.to(device=_DEVICE, dtype=torch.float32).div_(255.0)
    media = torch.tensor([0.485, 0.456, 0.406], device=_DEVICE).view(1, 3, 1, 1)
    desvio = torch.tensor([0.229, 0.224, 0.225], device=_DEVICE).view(1, 3, 1, 1)
    return (tensor - media) / desvio


def _forward_cnn(imagens):
    modelo = _obter_modelo()
    tensor = _tensor_imagens(imagens)
    with torch.inference_mode():
        if USAR_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                saida = modelo(tensor)
        else:
            saida = modelo(tensor)
    saida = saida.float().cpu().numpy()
    if saida.ndim != 2 or saida.shape[1] != DIM_CNN:
        raise RuntimeError(
            f"A ConvNeXt retornou formato inesperado {saida.shape}; esperado (N, {DIM_CNN})."
        )
    norma = np.linalg.norm(saida, axis=1, keepdims=True) + 1e-9
    return (saida / norma).astype(np.float32)


def _embedding_lote(imagens, batch_size=None):
    if not imagens:
        return np.empty((0, DIM_CNN), dtype=np.float32)

    if batch_size is None:
        batch_size = 32 if _DEVICE.type == "cuda" else 16
    tamanho = max(1, int(batch_size))
    partes = []
    inicio = 0

    while inicio < len(imagens):
        bloco = imagens[inicio : inicio + tamanho]
        try:
            partes.append(_forward_cnn(bloco))
            inicio += len(bloco)
        except RuntimeError as erro:
            texto = str(erro).lower()
            falta_memoria = "out of memory" in texto or "cuda error: out of memory" in texto
            if not falta_memoria or tamanho <= 1:
                raise
            if _DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            tamanho = max(1, tamanho // 2)
            print(f"Memória insuficiente na CNN; reduzindo lote para {tamanho}.")

    return np.vstack(partes).astype(np.float32)


def extrair_lote(imgs_bgr, espelhar=False, batch_size=None):
    """Extrai matriz (N, 810) de imagens BGR.

    Args:
        imgs_bgr: lista/tupla de arrays BGR; podem ter qualquer tamanho.
        espelhar: mantido por compatibilidade; use False no treinamento normal.
        batch_size: lote máximo enviado à CNN por vez.
    """
    imagens = [_redimensionar_preservando(im) for im in imgs_bgr]
    if espelhar:
        imagens = [cv2.flip(im, 1) for im in imagens]
    if not imagens:
        return np.empty((0, TOTAL_FEATURES), dtype=np.float32)

    embeddings = _embedding_lote(imagens, batch_size=batch_size)
    extras = []
    for img in imagens:
        mascara = segmentar(img)
        cor = _caracteristicas_cor(img, mascara)
        textura = _caracteristicas_textura(img, mascara)
        extras.append(np.concatenate([cor, textura]))

    matriz = np.hstack([embeddings, np.asarray(extras, dtype=np.float32)])
    if matriz.shape[1] != TOTAL_FEATURES:
        raise RuntimeError(
            f"Extrator produziu {matriz.shape[1]} características; esperado {TOTAL_FEATURES}."
        )
    return matriz.astype(np.float32, copy=False)


def extrair(img_bgr, espelhar=False):
    """Extrai um vetor de 810 características de uma única imagem BGR."""
    return extrair_lote([img_bgr], espelhar=espelhar)[0]


if __name__ == "__main__":
    print(f"Extrator de folhas v{VERSAO_EXTRATOR}")
    print(f"Dispositivo: {dispositivo()} | AMP: {USAR_AMP}")
    print(f"Dimensão do vetor: {TOTAL_FEATURES}")
