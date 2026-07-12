import streamlit as st
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------
st.set_page_config(page_title="Hill 방정식 수치적분 시뮬레이터", layout="wide")

st.title("💊 Hill 방정식의 수치적 적분(Numerical Integration) 시뮬레이션")
st.markdown(
    """
    고교 미적분에서는 Hill 계수 **n이 정수(특히 n=2)**일 때 삼각치환 등으로 부정적분을 구할 수 있지만,
    실제 약리학에서는 n = 2.8, n = 1.5 처럼 **실수(비정수) Hill 계수**가 흔합니다.
    이 경우 원시함수를 대수적으로 구하기 어렵기 때문에, **구분구적법의 응용인 사다리꼴 공식·심슨 공식**을 이용해
    정적분 값을 근사적으로 계산합니다.

    (그래프 안의 글자는 폰트 호환성을 위해 영어로 표기됩니다.)
    """
)

# ------------------------------------------------------------
# Hill 방정식 정의
#   E(C) = Vmax * C^n / (Km^n + C^n)
# ------------------------------------------------------------
def hill_function(C, Vmax, Km, n):
    C = np.asarray(C, dtype=float)
    return Vmax * (C ** n) / (Km ** n + C ** n)


# ------------------------------------------------------------
# 사이드바 - 파라미터 입력
# ------------------------------------------------------------
st.sidebar.header("⚙️ 파라미터 설정")

Vmax = st.sidebar.slider("Vmax (최대 반응/효과)", min_value=1.0, max_value=200.0, value=100.0, step=1.0)
Km = st.sidebar.slider("Km (반최대 반응 농도, EC50)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
n = st.sidebar.slider("n (Hill 계수)", min_value=0.5, max_value=5.0, value=2.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("적분 구간 및 분할 수")
a = st.sidebar.number_input("적분 시작 농도 a", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
b = st.sidebar.number_input("적분 끝 농도 b", min_value=0.1, max_value=200.0, value=30.0, step=1.0)
N = st.sidebar.slider("분할 수 N (구간을 나누는 개수)", min_value=4, max_value=1000, value=20, step=2)

if N % 2 != 0:
    st.sidebar.warning("심슨 공식은 짝수 개의 구간이 필요해 N을 자동으로 +1 합니다.")
    N += 1

if a >= b:
    st.error("적분 시작값 a는 끝값 b보다 작아야 합니다.")
    st.stop()

# ------------------------------------------------------------
# 수치적분 함수 직접 구현 (구분구적법 원리 적용)
# ------------------------------------------------------------
def trapezoidal_rule(f, a, b, N, **kwargs):
    x = np.linspace(a, b, N + 1)
    y = f(x, **kwargs)
    h = (b - a) / N
    result = h * (y[0] / 2 + y[-1] / 2 + np.sum(y[1:-1]))
    return result, x, y

def simpsons_rule(f, a, b, N, **kwargs):
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")
    x = np.linspace(a, b, N + 1)
    y = f(x, **kwargs)
    h = (b - a) / N
    result = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
    return result, x, y

# ------------------------------------------------------------
# 계산 실행
# ------------------------------------------------------------
trap_result, x_trap, y_trap = trapezoidal_rule(hill_function, a, b, N, Vmax=Vmax, Km=Km, n=n)
simp_result, x_simp, y_simp = simpsons_rule(hill_function, a, b, N, Vmax=Vmax, Km=Km, n=n)

# scipy를 이용한 고정밀 기준값 (quad는 적응형 가우스 구적법 사용)
scipy_result, scipy_err = integrate.quad(
    lambda C: hill_function(C, Vmax, Km, n), a, b
)

# ------------------------------------------------------------
# n=2일 때 해석적(analytic) 해와 비교 (삼각치환으로 유도 가능한 특수 케이스)
#   ∫ C^2/(Km^2+C^2) dC = C - Km*arctan(C/Km) + const
# ------------------------------------------------------------
show_analytic = abs(n - 2.0) < 1e-9
analytic_result = None
if show_analytic:
    def antiderivative(C):
        return Vmax * (C - Km * np.arctan(C / Km))
    analytic_result = antiderivative(b) - antiderivative(a)

# ------------------------------------------------------------
# 결과 표시
# ------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📊 정적분 결과 비교")

    data = {
        "방법": ["사다리꼴 공식 (Trapezoidal)", "심슨 공식 (Simpson's)", "scipy.integrate.quad (기준값)"],
        "적분값": [trap_result, simp_result, scipy_result],
        "scipy 대비 오차": [
            abs(trap_result - scipy_result),
            abs(simp_result - scipy_result),
            0.0,
        ],
    }
    if show_analytic:
        data["방법"].insert(0, "해석적 해 (n=2, 삼각치환)")
        data["적분값"].insert(0, analytic_result)
        data["scipy 대비 오차"].insert(0, abs(analytic_result - scipy_result))

    df = pd.DataFrame(data)
    st.dataframe(df.style.format({"적분값": "{:.6f}", "scipy 대비 오차": "{:.2e}"}), use_container_width=True)

    if show_analytic:
        st.success(
            f"n=2일 때는 삼각치환으로 구한 해석적 해와 수치적분 결과가 거의 일치합니다. "
            f"(오차 {abs(analytic_result - scipy_result):.2e})"
        )
    else:
        st.info(
            f"n={n}은 정수가 아니므로 고교 과정의 대수적 방법으로는 부정적분을 구하기 어렵습니다. "
            f"사다리꼴/심슨 공식과 scipy 기준값이 근사적으로 일치하는지 확인해 보세요."
        )

    st.metric("사다리꼴 공식 오차", f"{abs(trap_result - scipy_result):.2e}")
    st.metric("심슨 공식 오차", f"{abs(simp_result - scipy_result):.2e}")

with col2:
    st.subheader("📈 Hill 함수 그래프와 적분 영역")

    C_full = np.linspace(0, max(b * 1.3, Km * 2), 400)
    y_full = hill_function(C_full, Vmax, Km, n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(C_full, y_full, color="black", linewidth=2, label=f"E(C), n={n}")
    ax.fill_between(C_full, y_full, where=(C_full >= a) & (C_full <= b),
                     color="skyblue", alpha=0.6, label=f"Integral area [{a:.0f}, {b:.0f}]")
    ax.axvline(Km, color="red", linestyle="--", linewidth=1, label=f"Km = {Km:.1f}")
    ax.set_xlabel("Concentration C")
    ax.set_ylabel("Effect E(C)")
    ax.set_title(f"Hill Equation Dose-Response Curve (n={n})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ------------------------------------------------------------
# 수렴성 분석: N에 따른 오차 변화
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 분할 수 N에 따른 수렴성(오차 감소) 분석")
st.markdown("N(분할 수)이 커질수록 사다리꼴/심슨 공식의 오차가 어떻게 줄어드는지 비교합니다.")

N_list = [4, 8, 16, 32, 64, 128, 256, 512]
trap_errors = []
simp_errors = []

for Ni in N_list:
    t_res, _, _ = trapezoidal_rule(hill_function, a, b, Ni, Vmax=Vmax, Km=Km, n=n)
    s_res, _, _ = simpsons_rule(hill_function, a, b, Ni, Vmax=Vmax, Km=Km, n=n)
    trap_errors.append(abs(t_res - scipy_result))
    simp_errors.append(abs(s_res - scipy_result))

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.plot(N_list, trap_errors, marker="o", label="Trapezoidal Rule Error")
ax2.plot(N_list, simp_errors, marker="s", label="Simpson's Rule Error")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Number of Subintervals N (log scale)")
ax2.set_ylabel("Error vs scipy reference (log scale)")
ax2.set_title("Convergence of Numerical Integration Error")
ax2.legend()
ax2.grid(alpha=0.3, which="both")
st.pyplot(fig2)

st.caption(
    "이론적으로 사다리꼴 공식의 오차는 O(1/N²)로, 심슨 공식은 O(1/N⁴)로 감소합니다. "
    "그래프의 기울기를 비교해 이 이론적 수렴 속도를 직접 확인해 보세요."
)

# ------------------------------------------------------------
# 하단 설명
# ------------------------------------------------------------
with st.expander("📖 Hill 방정식과 수치적분 원리 설명 보기"):
    st.markdown(
        r"""
        **Hill 방정식**

        $$ E(C) = \dfrac{V_{max} \cdot C^n}{K_m^n + C^n} $$

        - $C$: 약물 농도
        - $V_{max}$: 최대 반응(효과)
        - $K_m$: 반최대 반응(효과)을 나타내는 농도 (EC50)
        - $n$: Hill 계수 (협동성의 정도를 나타냄)

        **왜 수치적분이 필요한가?**

        $n$이 정수, 특히 $n=2$인 경우 삼각치환($C = K_m \tan\theta$)을 이용해
        $\int \dfrac{C^2}{K_m^2+C^2}\,dC = C - K_m \arctan\left(\dfrac{C}{K_m}\right) + C_0$
        와 같은 원시함수를 구할 수 있습니다.

        그러나 실제 헤모글로빈의 산소 결합(n≈2.8)이나 여러 약물 수용체 결합(n=1.5, n=3.2 등)처럼
        $n$이 정수가 아닌 경우, 이런 대수적 방법으로는 원시함수를 구할 수 없습니다.

        이때 **구분구적법**의 아이디어를 확장한

        - **사다리꼴 공식**: 각 구간을 사다리꼴로 근사 → 오차 $O(1/N^2)$
        - **심슨 공식**: 각 구간을 포물선(2차함수)으로 근사 → 오차 $O(1/N^4)$

        을 이용하면 임의의 $n$ 값에 대해서도 정적분(= AUC, 총 반응량)을 매우 정밀하게 근사할 수 있습니다.
        """
    )
