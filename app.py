import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")

from sistema_forcas import SistemaDeForcas

MAX_FORCAS = 10

EXEMPLOS = {
    "-- selecione um exemplo --": None,

    "n = 1 (caso trivial)": dict(
        forcas=[[10, 0, 0]], pontos=[[1, 2, 0]],
        Q=[0, 0, 0], A=[1, 0, 0], u=[0, 1, 0],
    ),
    "a) Sistema nulo": dict(
        forcas=[[5, 0, 0], [-5, 0, 0]], pontos=[[0, 0, 0], [3, 0, 0]],
        Q=[1, 1, 1], A=[2, -1, 0], u=[0, 0, 1],
    ),
    "b) Binário (conjugado)": dict(
        forcas=[[0, 0, 10], [0, 0, -10]], pontos=[[0, 0, 0], [2, 0, 0]],
        Q=[0, 0, 0], A=[5, 5, 5], u=[1, 1, 0],
    ),
    "c1) Força única - concorrentes": dict(
        forcas=[[4, 0, 0], [0, 5, 0], [0, 0, 6]],
        pontos=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        Q=[0, 0, 0], A=[2, 2, 2], u=[1, 0, 0],
    ),
    "c2) Força única - coplanares": dict(
        forcas=[[3, 2, 0], [-1, 4, 0], [2, -5, 0]],
        pontos=[[0, 0, 0], [2, 0, 0], [1, 3, 0]],
        Q=[0, 0, 0], A=[4, 1, 0], u=[0, 1, 0],
    ),
    "c3) Força única - paralelas": dict(
        forcas=[[0, 0, 4], [0, 0, -1], [0, 0, 3]],
        pontos=[[0, 0, 0], [2, 0, 0], [1, 2, 0]],
        Q=[0, 0, 0], A=[1, 1, 0], u=[1, 0, 0],
    ),
    "c4) Força única - caso geral (I=0)": dict(
        forcas=[[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        pontos=[[0, 1, 0], [1, 0, 0], [1, 1, 0]],
        Q=[0, 0, 0], A=[3, 3, 3], u=[0, 0, 1],
    ),
    "d) Força + binário (caso geral)": dict(
        forcas=[[2, -1, 3], [0, 4, -2], [-1, 1, 5]],
        pontos=[[1, 0, 0], [0, 2, 1], [2, 2, -1]],
        Q=[0, 0, 0], A=[1, 1, 1], u=[1, 1, 1],
    ),
}


def _parse_vetor(texto, nome):
    partes = [p.strip() for p in texto.replace(";", ",").split(",") if p.strip() != ""]
    if len(partes) != 3:
        raise ValueError(f"O campo '{nome}' deve conter exatamente 3 números "
                          f"separados por vírgula (ex.: 1, 2, 3). Recebido: '{texto}'")
    try:
        return [float(x) for x in partes]
    except ValueError:
        raise ValueError(f"O campo '{nome}' contém valor(es) não numérico(s): '{texto}'")


def atualizar_visibilidade(n):
    n = int(n)
    return [gr.update(visible=(i < n)) for i in range(MAX_FORCAS)]


def carregar_exemplo(nome_exemplo):
    ex = EXEMPLOS.get(nome_exemplo)
    saida_campos = [gr.update() for _ in range(MAX_FORCAS * 6)]
    if ex is None:
        return (gr.update(), *saida_campos, gr.update(), gr.update(), gr.update(), "")

    n = len(ex["forcas"])
    for i in range(n):
        Fx, Fy, Fz = ex["forcas"][i]
        x, y, z = ex["pontos"][i]
        base = i * 6
        saida_campos[base + 0] = gr.update(value=Fx)
        saida_campos[base + 1] = gr.update(value=Fy)
        saida_campos[base + 2] = gr.update(value=Fz)
        saida_campos[base + 3] = gr.update(value=x)
        saida_campos[base + 4] = gr.update(value=y)
        saida_campos[base + 5] = gr.update(value=z)
    for i in range(n, MAX_FORCAS):
        base = i * 6
        for k in range(6):
            saida_campos[base + k] = gr.update(value=0.0)

    Qx, Qy, Qz = ex["Q"]
    Ax, Ay, Az = ex["A"]
    ux, uy, uz = ex["u"]
    return (
        n,
        *saida_campos,
        f"{Qx}, {Qy}, {Qz}",
        f"{Ax}, {Ay}, {Az}",
        f"{ux}, {uy}, {uz}",
        f"Exemplo carregado: {nome_exemplo}",
    )

def calcular(n, *args):
    """
    args = [Fx1,Fy1,Fz1,x1,y1,z1,  Fx2,Fy2,Fz2,x2,y2,z2, ... até MAX_FORCAS]
            + (texto_Q, texto_A, texto_u)
    """
    try:
        campos = args[: MAX_FORCAS * 6]
        texto_Q, texto_A, texto_u = args[MAX_FORCAS * 6: MAX_FORCAS * 6 + 3]

        n = int(n)
        if n < 1:
            raise ValueError("n deve ser >= 1.")
        if n > MAX_FORCAS:
            raise ValueError(f"n máximo suportado pela interface é {MAX_FORCAS}.")

        forcas, pontos = [], []
        for i in range(n):
            base = i * 6
            valores = campos[base: base + 6]
            valores = [0.0 if v is None else float(v) for v in valores]
            forcas.append(valores[0:3])
            pontos.append(valores[3:6])

        Q = _parse_vetor(texto_Q, "Q")
        A = _parse_vetor(texto_A, "A")
        u = _parse_vetor(texto_u, "u")

        sist = SistemaDeForcas(
            forcas=forcas, pontos=pontos, Q=Q, A=A, u=u,
            nome="Sistema de Forças",
        )

        def v(x):
            return [round(float(x[0]), 4), round(float(x[1]), 4), round(float(x[2]), 4)]

        linhas_resultado = [
            ["Resultante R", *v(sist.R), round(sist.normR, 4)],
            ["Momento M_Q", *v(sist.M_Q), round(sist.normMQ, 4)],
            ["Momento M_A", *v(sist.M_A), round(float(np.linalg.norm(sist.M_A)), 4)],
            ["Torque eixo Au", round(sist.torque_Au, 4), "", "", "", ""],
            ["Invariante I = R.M_Q", round(sist.I, 4), "", "", "", ""],
        ]
        if sist.tem_eixo_central:
            linhas_resultado.append(["Ponto P0 do eixo central", *v(sist.P0), ""])
            linhas_resultado.append(["Direção do eixo central", *v(sist.dir_eixo), ""])
            linhas_resultado.append(["Momento mínimo M_E", *v(sist.M_min),
                                      round(float(np.linalg.norm(sist.M_min)), 4)])

        df_resultado = [linha + [""] * (6 - len(linha)) for linha in linhas_resultado]

        texto_classificacao = f"### Classificação: {sist.classe}"
        if sist.classe == "Redutível a um binário (conjugado)":
            texto_classificacao += "\n\nO momento é o mesmo em qualquer polo. Não existe eixo central."
        elif sist.classe == "Sistema nulo (equilibrado)":
            texto_classificacao += "\n\nCorpo em equilíbrio estático quanto a este sistema."
        elif sist.tem_eixo_central:
            texto_classificacao += (
                f"\n\n**Eixo central** passa por P0 = {tuple(round(float(x),4) for x in sist.P0)}, "
                f"direção {tuple(round(float(x),4) for x in sist.dir_eixo)}."
            )

        fig1 = sist.desenhar_principal(fechar=False)
        if sist.tem_eixo_central:
            fig2 = sist.desenhar_eixo_central(fechar=False)
        else:
            fig2 = None

        return (
            texto_classificacao,
            df_resultado,
            fig1,
            fig2,
            "", 
        )

    except Exception as e:
        return (
            "### Erro",
            [],
            None,
            None,
            f"**Erro:** {e}",
        )


with gr.Blocks(title="Sistemas de Forças em Corpo Rígido") as demo:
    gr.Markdown(
        "# Análise de Sistemas de Forças em Corpo Rígido\n"
        "Preencha os dados abaixo e clique em **Calcular**."
    )

    with gr.Row():
        exemplo_dd = gr.Dropdown(
            choices=list(EXEMPLOS.keys()), value="-- selecione um exemplo --",
            label="Exemplos prontos (opcional)"
        )
    with gr.Row():
        n_num = gr.Number(value=3, precision=0, minimum=1, maximum=MAX_FORCAS,
                           label=f"n (número de forças, 1 a {MAX_FORCAS})")

    gr.Markdown("### Forças F_i e pontos de aplicação P_i")

    linhas_forca = []
    campos_forca = []
    for i in range(MAX_FORCAS):
        with gr.Row(visible=(i < 3)) as linha:
            gr.Markdown(f"**F{i+1}**", scale=1)
            fx = gr.Number(value=0.0, label="Fx", scale=2)
            fy = gr.Number(value=0.0, label="Fy", scale=2)
            fz = gr.Number(value=0.0, label="Fz", scale=2)
            x = gr.Number(value=0.0, label="x", scale=2)
            y = gr.Number(value=0.0, label="y", scale=2)
            z = gr.Number(value=0.0, label="z", scale=2)
        linhas_forca.append(linha)
        campos_forca += [fx, fy, fz, x, y, z]

    with gr.Row():
        Q_txt = gr.Textbox(label="Polo Q (x, y, z)", value="0, 0, 0")
        A_txt = gr.Textbox(label="Polo A (x, y, z)", value="1, 1, 1")
        u_txt = gr.Textbox(label="Versor u (x, y, z) -- não precisa estar normalizado", value="1, 0, 0")

    calcular_btn = gr.Button("Calcular", variant="primary")
    erro_txt = gr.Markdown()

    classificacao_md = gr.Markdown()
    resultado_df = gr.Dataframe(
        headers=["Grandeza", "X / valor", "Y", "Z", "Módulo", ""],
        label="Resultados",
    )

    with gr.Row():
        fig1_plot = gr.Plot(label="Figura 1 - Sistema completo (h)")
        fig2_plot = gr.Plot(label="Figura 2 - Eixo central (i)")

    n_num.change(atualizar_visibilidade, inputs=n_num, outputs=linhas_forca)

    exemplo_dd.change(
        carregar_exemplo, inputs=exemplo_dd,
        outputs=[n_num, *campos_forca, Q_txt, A_txt, u_txt, erro_txt],
    ).then(atualizar_visibilidade, inputs=n_num, outputs=linhas_forca)

    calcular_btn.click(
        calcular,
        inputs=[n_num, *campos_forca, Q_txt, A_txt, u_txt],
        outputs=[classificacao_md, resultado_df, fig1_plot, fig2_plot, erro_txt],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
