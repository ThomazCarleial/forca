import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
EPS_ABS = 1e-9

def _tol(*vecs, rel=1e-9):
    """Tolerância relativa baseada na escala das grandezas envolvidas,
    de modo a evitar problemas numéricos (inclusive no caso trivial n=1)."""
    escala = max([np.linalg.norm(v) for v in vecs] + [1.0])
    return max(EPS_ABS, rel * escala)

def unit(v):
    n = np.linalg.norm(v)
    if n < EPS_ABS:
        raise ValueError("Não é possível normalizar um vetor nulo.")
    return v / n

def fmt_vec(v, casas=4):
    return "({: .{c}f}, {: .{c}f}, {: .{c}f})".format(v[0], v[1], v[2], c=casas)

class SistemaDeForcas:
    """
    Representa um sistema de n forças aplicadas a um corpo rígido e realiza
    toda a análise estática solicitada.
    """
    def __init__(self, forcas, pontos, Q, A, u, nome="Sistema de Forças"):
        """
        forcas : lista/array (n,3) -> componentes (Fx,Fy,Fz) de cada força
        pontos : lista/array (n,3) -> coordenadas (x,y,z) do ponto de
                 aplicação de cada força
        Q      : (3,) coordenadas do polo Q
        A      : (3,) coordenadas do polo A
        u      : (3,) componentes do versor u (não precisa estar
                 normalizado -- o programa normaliza)
        """
        self.nome = nome
        self.F = np.atleast_2d(np.array(forcas, dtype=float))
        self.P = np.atleast_2d(np.array(pontos, dtype=float))
        if self.F.shape != self.P.shape:
            raise ValueError("forcas e pontos devem ter o mesmo formato (n,3).")
        self.n = self.F.shape[0]
        self.Q = np.array(Q, dtype=float)
        self.A = np.array(A, dtype=float)
        u = np.array(u, dtype=float)
        if np.linalg.norm(u) < EPS_ABS:
            raise ValueError("O versor u não pode ser nulo.")
        self.u = unit(u)

        self._calcular()
    def _calcular(self):
        self.R = self.F.sum(axis=0)
        self.M_Q = np.zeros(3)
        for i in range(self.n):
            self.M_Q += np.cross(self.P[i] - self.Q, self.F[i])

        self.M_A = self.M_Q + np.cross(self.Q - self.A, self.R)

        self.torque_Au = float(np.dot(self.M_A, self.u))

        self.I = float(np.dot(self.R, self.M_Q))

        self.tolR = _tol(self.R, *self.F)
        self.tolM = _tol(self.M_Q, self.M_A, *[np.cross(p, f) for p, f in zip(self.P, self.F)])
        self.normR = np.linalg.norm(self.R)
        self.normMQ = np.linalg.norm(self.M_Q)

        self.tolI = max(EPS_ABS, 1e-9 * max(self.normR * self.normMQ, 1.0))

        self.classe = self._classificar()

        self.tem_eixo_central = self.classe in (
            "Redutível a uma força única (resultante)",
            "Redutível a uma força + um binário (caso geral)",
        )
        if self.tem_eixo_central:
            self.P0, self.dir_eixo, self.M_min = self._eixo_central()
        else:
            self.P0 = self.dir_eixo = self.M_min = None

    def _classificar(self):
        R_nulo = self.normR < self.tolR
        M_nulo = self.normMQ < self.tolM
        if R_nulo and M_nulo:
            return "Sistema nulo (equilibrado)"
        if R_nulo and not M_nulo:
            return "Redutível a um binário (conjugado)"
        if (not R_nulo) and abs(self.I) < self.tolI:
            return "Redutível a uma força única (resultante)"
        return "Redutível a uma força + um binário (caso geral)"

    def _eixo_central(self):
        """Retorna (ponto sobre o eixo mais próximo de Q, versor do eixo,
        vetor momento mínimo M_E)."""
        R = self.R
        normR2 = np.dot(R, R)
        d = np.cross(R, self.M_Q) / normR2          # vetor de Q até o eixo
        P0 = self.Q + d
        dir_eixo = unit(R)
        M_min = (self.I / normR2) * R
        return P0, dir_eixo, M_min

    def relatorio(self):
        linhas = []
        linhas.append("=" * 78)
        linhas.append(f"SISTEMA: {self.nome}   (n = {self.n} força{'s' if self.n != 1 else ''})")
        linhas.append("=" * 78)

        cab = f"{'i':>2} | {'Fx':>10} {'Fy':>10} {'Fz':>10} | {'x':>10} {'y':>10} {'z':>10}"
        linhas.append(cab)
        linhas.append("-" * len(cab))
        for i in range(self.n):
            Fx, Fy, Fz = self.F[i]
            x, y, z = self.P[i]
            linhas.append(
                f"{i+1:>2} | {Fx:10.4f} {Fy:10.4f} {Fz:10.4f} | "
                f"{x:10.4f} {y:10.4f} {z:10.4f}"
            )
        linhas.append("-" * len(cab))
        linhas.append(f"Polo Q = {fmt_vec(self.Q)}    Polo A = {fmt_vec(self.A)}    "
                       f"versor u = {fmt_vec(self.u)}")
        linhas.append("")

        res = [
            ("a) Resultante R", fmt_vec(self.R), f"|R| = {self.normR:.4f}"),
            ("b) Momento M_Q", fmt_vec(self.M_Q), f"|M_Q| = {self.normMQ:.4f}"),
            ("c) Momento M_A", fmt_vec(self.M_A), f"|M_A| = {np.linalg.norm(self.M_A):.4f}"),
            ("d) Torque eixo Au", f"{self.torque_Au:.4f}", ""),
            ("e) Invariante I = R.M_Q", f"{self.I:.4f}", ""),
        ]
        w1 = max(len(r[0]) for r in res)
        w2 = max(len(r[1]) for r in res)
        for label, val, extra in res:
            linhas.append(f"{label:<{w1}} : {val:<{w2}}   {extra}")

        linhas.append("")
        linhas.append(f"f) Classificação: {self.classe}")

        if self.classe == "Sistema nulo (equilibrado)":
            linhas.append("   -> Corpo em equilíbrio estático quanto a este sistema de forças.")
        elif self.classe == "Redutível a um binário (conjugado)":
            linhas.append("   -> O momento do sistema é o mesmo em qualquer polo "
                           f"(conferir M_A = {fmt_vec(self.M_A)}).")
            linhas.append("   -> Não existe eixo central (nenhuma força única equivalente).")
        else:
            linhas.append(f"g) Eixo central: passa por P0 = {fmt_vec(self.P0)}")
            linhas.append(f"                 com direção (versor) = {fmt_vec(self.dir_eixo)}")
            linhas.append(f"   Momento mínimo M_E = {fmt_vec(self.M_min)}   "
                           f"(|M_E| = {np.linalg.norm(self.M_min):.4f})")
            if self.classe == "Redutível a uma força única (resultante)":
                linhas.append("   -> M_E = 0 : o sistema equivale a uma força única "
                               "R aplicada sobre o eixo central.")
        linhas.append("=" * 78)
        texto = "\n".join(linhas)
        print(texto)
        return texto

    def _bbox(self, extras=None):
        pts = list(self.P) + [self.Q, self.A]
        if extras:
            pts += list(extras)
        pts = np.array(pts)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = mx - mn
        span = np.where(span < 1e-6, 1.0, span)
        centro = (mx + mn) / 2
        mn = centro - span / 2
        mx = centro + span / 2
        return mn, mx

    @staticmethod
    def _desenhar_caixa(ax, mn, mx, cor="0.6"):
        """Desenha um paralelepípedo (wireframe) delimitando a região de
        trabalho, para dar noção de profundidade/escala (item 3)."""
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        arestas = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in arestas:
            ax.plot(*zip(vertices[i], vertices[j]), color=cor, lw=0.8, ls="--")

    def _escala_setas(self, mn, mx):
        diag = np.linalg.norm(mx - mn)
        return max(diag, 1.0)

    def _eixos_Oxyz(self, ax, mn, mx):
        L = 1.15 * self._escala_setas(mn, mx) / 2.6
        origem = np.zeros(3)
        for vetor, nome, cor in zip(np.eye(3), "xyz", ["red", "green", "blue"]):
            ax.quiver(*origem, *(vetor * L), color=cor, arrow_length_ratio=0.12, lw=1.5)
            ax.text(*(vetor * L * 1.1), nome, color=cor, fontsize=11, fontweight="bold")
        ax.scatter(*origem, color="black", s=25)
        ax.text(0, 0, -0.06 * L, "O", fontsize=10)

    def desenhar_principal(self, elev=22, azim=-55, salvar=None, mostrar=False, fechar=True):
        extras = [self.A + self.u, self.A - self.u]
        if self.normR > self.tolR:
            extras.append(self.Q + unit(self.R))
        mn, mx = self._bbox(extras)
        escala = self._escala_setas(mn, mx)
        fF = 0.30 * escala
        fM = 0.30 * escala
        fEixo = 0.65 * escala

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        self._desenhar_caixa(ax, mn, mx)
        self._eixos_Oxyz(ax, mn, mx)

        for i in range(self.n):
            P = self.P[i]
            F = self.F[i]
            ax.scatter(*P, color="black", s=25)
            ax.text(*(P + 0.02 * escala), f"P{i+1}", fontsize=9)
            nF = np.linalg.norm(F)
            if nF > EPS_ABS:
                Fd = F / nF * fF
                ax.quiver(*P, *Fd, color="orange", lw=2, arrow_length_ratio=0.2)
                ax.text(*(P + Fd * 1.08), f"F{i+1}", color="darkorange", fontsize=9)

        ax.scatter(*self.Q, color="purple", s=45, marker="s")
        ax.text(*(self.Q + 0.02 * escala), "Q", color="purple", fontsize=11, fontweight="bold")

        ax.scatter(*self.A, color="teal", s=45, marker="^")
        ax.text(*(self.A + 0.02 * escala), "A", color="teal", fontsize=11, fontweight="bold")
        p1 = self.A - self.u * fEixo
        p2 = self.A + self.u * fEixo
        ax.plot(*zip(p1, p2), color="teal", lw=1.5, ls="-.")
        ax.text(*(p2 + 0.02 * escala), "eixo Au", color="teal", fontsize=9)

        if self.normR > self.tolR:
            Rd = unit(self.R) * fM * 1.3
            ax.quiver(*self.Q, *Rd, color="crimson", lw=3, arrow_length_ratio=0.18)
            ax.text(*(self.Q + Rd * 1.1), "R", color="crimson", fontsize=12, fontweight="bold")

        if self.normMQ > self.tolM:
            Md = unit(self.M_Q) * fM
            ax.quiver(*self.Q, *Md, color="blue", lw=2.4, arrow_length_ratio=0.22, linestyle="dashed")
            ax.quiver(*self.Q, *(Md * 0.75), color="blue", lw=2.4, arrow_length_ratio=0.30)
            ax.text(*(self.Q + Md * 1.15), "M_Q", color="blue", fontsize=11, fontweight="bold")

        normMA = np.linalg.norm(self.M_A)
        if normMA > self.tolM:
            Md = unit(self.M_A) * fM
            ax.quiver(*self.A, *Md, color="green", lw=2.4, arrow_length_ratio=0.22, linestyle="dashed")
            ax.quiver(*self.A, *(Md * 0.75), color="green", lw=2.4, arrow_length_ratio=0.30)
            ax.text(*(self.A + Md * 1.15), "M_A", color="green", fontsize=11, fontweight="bold")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{self.nome}\nSistema de forças, polos Q e A, eixo Au")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        fig.tight_layout()
        if salvar:
            fig.savefig(salvar, dpi=140)
            print(f"[figura salva em: {salvar}]")
        if mostrar:
            plt.show()
        if fechar:
            plt.close(fig)
        return fig

    def desenhar_eixo_central(self, elev=22, azim=-55, salvar=None, mostrar=False, fechar=True):
        if not self.tem_eixo_central:
            print("[Este sistema não possui eixo central -- figura não gerada.]")
            return None

        extras = [self.P0, self.P0 + self.dir_eixo, self.P0 - self.dir_eixo]
        mn, mx = self._bbox(extras)
        escala = self._escala_setas(mn, mx)
        fF = 0.30 * escala
        fM = 0.30 * escala
        fEixo = 0.75 * escala

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        self._desenhar_caixa(ax, mn, mx)
        self._eixos_Oxyz(ax, mn, mx)

        for i in range(self.n):
            P = self.P[i]
            F = self.F[i]
            ax.scatter(*P, color="black", s=25)
            ax.text(*(P + 0.02 * escala), f"P{i+1}", fontsize=9)
            nF = np.linalg.norm(F)
            if nF > EPS_ABS:
                Fd = F / nF * fF
                ax.quiver(*P, *Fd, color="orange", lw=2, arrow_length_ratio=0.2)
                ax.text(*(P + Fd * 1.08), f"F{i+1}", color="darkorange", fontsize=9)

        ax.scatter(*self.Q, color="purple", s=45, marker="s")
        ax.text(*(self.Q + 0.02 * escala), "Q", color="purple", fontsize=11, fontweight="bold")
        if self.normMQ > self.tolM:
            Md = unit(self.M_Q) * fM
            ax.quiver(*self.Q, *Md, color="blue", lw=2.4, arrow_length_ratio=0.22, linestyle="dashed")
            ax.text(*(self.Q + Md * 1.15), "M_Q", color="blue", fontsize=11, fontweight="bold")

        e1 = self.P0 - self.dir_eixo * fEixo
        e2 = self.P0 + self.dir_eixo * fEixo
        ax.plot(*zip(e1, e2), color="crimson", lw=2)
        ax.text(*(e2 + 0.02 * escala), "eixo central", color="crimson", fontsize=10)
        ax.scatter(*self.P0, color="crimson", s=40)
        ax.text(*(self.P0 + 0.02 * escala), "P0", color="crimson", fontsize=9)

        Rd = unit(self.R) * fM * 1.3
        ax.quiver(*self.P0, *Rd, color="darkred", lw=3, arrow_length_ratio=0.18)
        ax.text(*(self.P0 + Rd * 1.1), "R", color="darkred", fontsize=12, fontweight="bold")

        normME = np.linalg.norm(self.M_min)
        if normME > self.tolM:
            Md = unit(self.M_min) * fM
            offset = self.P0 + 0.18 * escala * unit(np.cross(self.dir_eixo, [1, 0, 0]) + 1e-9)
            ax.quiver(*offset, *Md, color="magenta", lw=2.4, arrow_length_ratio=0.22, linestyle="dashed")
            ax.text(*(offset + Md * 1.15), "M_E (mínimo)", color="magenta", fontsize=10, fontweight="bold")
        else:
            ax.text(*(self.P0 - 0.12 * escala * self.dir_eixo), "M_E = 0", color="magenta", fontsize=10)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{self.nome}\nEixo central: resultante R e momento mínimo M_E")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        fig.tight_layout()
        if salvar:
            fig.savefig(salvar, dpi=140)
            print(f"[figura salva em: {salvar}]")
        if mostrar:
            plt.show()
        if fechar:
            plt.close(fig)
        return fig
