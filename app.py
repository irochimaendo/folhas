"""Site: envia a foto, o Flask extrai as características e o Naive Bayes classifica.
Usa o extrator_personalizado.py e o modelo_folhas.joblib gerado por treinar_personalizado.py."""
import base64

import joblib
import numpy as np
from flask import Flask, render_template, request

from extrator_personalizado import extrair, ler_imagem

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

modelo = joblib.load("saida_treinamento/modelo_folhas.joblib")
LIMIAR = 0.70   # ajuste usando a tabela de limiares que o treino imprime


def nome_legivel(classe):
    return classe.replace("___", " — ").replace("_", " ")


def prever(dados):
    img = ler_imagem(dados)
    X = np.stack([extrair(img), extrair(img, espelhar=True)])   # foto + versão espelhada
    proba = modelo.predict_proba(X).mean(axis=0)
    ordem = np.argsort(proba)[::-1][:3]
    top = [(nome_legivel(modelo.classes_[i]), round(float(proba[i]) * 100, 1)) for i in ordem]
    return top, float(proba[ordem[0]]) >= LIMIAR


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {}
    if request.method == "POST":
        arq = request.files.get("foto")
        if not arq or arq.filename == "":
            ctx["erro"] = "Escolha uma foto antes de prever."
        else:
            dados = arq.read()
            try:
                top, confiavel = prever(dados)
                ctx.update(top=top, confiavel=confiavel,
                           imagem=base64.b64encode(dados).decode(),
                           mime=arq.mimetype or "image/jpeg")
            except ValueError as e:
                ctx["erro"] = str(e)
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
