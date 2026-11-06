import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import bisect

st.set_page_config(page_title='원관 층류 유동 · 관경 역설계', page_icon='💧', layout='wide')
st.title('💧 원관 내 층류 유동 및 관경 역설계 프로그램')
st.caption('화공유체역학 팀 프로젝트 | 정방향 계산 · Parameter Sweep · 역설계 · 독립 검증')

with st.sidebar:
    st.header('① 사용자 입력')
    rho = st.number_input('밀도 ρ (kg/m³)', min_value=0.000001, value=1000.0, format='%.4f')
    mu = st.number_input('점도 μ (Pa·s)', min_value=0.000000001, value=0.001, format='%.6f')
    L = st.number_input('관 길이 L (m)', min_value=0.000001, value=10.0)
    D_mm = st.number_input('관경 D (mm)', min_value=0.000001, value=7.99)
    delta_P = st.number_input('압력강하 ΔP (Pa)', min_value=0.000001, value=500.0)
    Q_ml = st.number_input('목표유량 (mL/s)', min_value=0.000001, value=5.0)
    P_allow = st.number_input('허용 압력강하 (Pa)', min_value=0.000001, value=500.0)
    Re_limit = st.number_input('층류 Reynolds 수 기준', min_value=1.0, value=2100.0)
    st.divider()
    run = st.button('▶ 전체 계산 및 검증 실행', type='primary', use_container_width=True)

st.info('왼쪽에서 조건을 입력하고 **전체 계산 및 검증 실행**을 누르세요. 결과는 순서대로 표시됩니다.')
if not run:
    st.stop()

D = D_mm / 1000
Q_target = Q_ml * 1e-6


def flow(d, dp=delta_P, density=rho, viscosity=mu):
    q = np.pi * dp * d**4 / (128 * viscosity * L)
    v = 4 * q / (np.pi * d**2)
    re = density * v * d / viscosity
    tau = dp * d / (4 * L)
    return q, v, re, tau


def pressure_for_flow(q, d):
    return 128 * mu * L * q / (np.pi * d**4)


def lineplot(x, y, xlabel, ylabel, title, limit=None):
    # English-only Matplotlib labels prevent missing Korean glyphs on cloud servers.
    labels = {
        '반지름 r (mm)': 'Radius r (mm)',
        '속도 u (m/s)': 'Velocity u (m/s)',
        '속도분포': 'Velocity Profile',
        '|τ| (Pa)': 'Shear Stress (Pa)',
        '전단응력분포': 'Shear Stress Profile',
        '관경 (mm)': 'Diameter (mm)',
        '유량 (mL/s)': 'Flow Rate (mL/s)',
        '관경–유량': 'Diameter vs Flow Rate',
        '관경–Re': 'Diameter vs Reynolds Number',
        '벽전단응력 (Pa)': 'Wall Shear Stress (Pa)',
        '관경–전단응력': 'Diameter vs Wall Shear Stress',
        '필요 압력강하 (Pa)': 'Required Pressure Drop (Pa)',
        '목표유량 유지 시 필요 압력강하': 'Required Pressure Drop vs Diameter',
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    if limit is None:
        ax.plot(x, y, linewidth=2)
    else:
        valid = x <= limit
        ax.plot(x[valid], y[valid], linewidth=2, label='Laminar Flow Region')
        ax.plot(x[~valid], y[~valid], linestyle='--', color='gray', label='Laminar Model Extrapolation')
        ax.axvline(limit, linestyle=':', color='red', label='Reynolds Number Limit')
        ax.legend(fontsize=8)
    ax.set(xlabel=labels.get(xlabel, xlabel), ylabel=labels.get(ylabel, ylabel), title=labels.get(title, title))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


st.header('② 정방향 계산')
Q, V, Re, tau_w = flow(D)
R = D / 2
r = np.linspace(0, R, 250)
u = delta_P * (R**2 - r**2) / (4 * mu * L)
tau = delta_P * r / (2 * L)
a, b, c, d = st.columns(4)
a.metric('유량 Q', f'{Q*1e6:.6f} mL/s')
b.metric('평균유속', f'{V:.6f} m/s')
c.metric('Reynolds 수', f'{Re:.2f}')
d.metric('벽면 전단응력', f'{tau_w:.6f} Pa')
st.write(f'최대유속: **{u[0]:.6f} m/s** · 층류 판정: **{"만족" if Re < Re_limit else "불만족 — 모델 적용 주의"}**')
a, b = st.columns(2)
with a:
    st.pyplot(lineplot(r*1000, u, '반지름 r (mm)', '속도 u (m/s)', '속도분포'))
with b:
    st.pyplot(lineplot(r*1000, tau, '반지름 r (mm)', '|τ| (Pa)', '전단응력분포'))

st.header('③ Parameter Sweep (조건 반복)')
Ds = np.unique(np.sort(np.r_[np.linspace(0.001, 0.030, 21), D]))
Qs, Vs, Res, Taus = np.array([flow(x) for x in Ds]).T
D_limit = (32 * mu**2 * L * Re_limit / (rho * delta_P))**(1/3)
st.write(f'자동 반복: **{len(Ds)}개 조건** · Re={Re_limit:g} 기준 한계 관경: **{D_limit*1000:.4f} mm**')
sweep = pd.DataFrame({'관경 (mm)': Ds*1000, '유량 (mL/s)': Qs*1e6, '평균유속 (m/s)': Vs, 'Re': Res, '벽전단응력 (Pa)': Taus, '층류식 적용': Res < Re_limit})
st.dataframe(sweep, hide_index=True, use_container_width=True)
columns = st.columns(3)
for col, (ys, ylabel, title) in enumerate([(Qs*1e6, '유량 (mL/s)', '관경–유량'), (Res, 'Re', '관경–Re'), (Taus, '벽전단응력 (Pa)', '관경–전단응력')]):
    with columns[col]:
        st.pyplot(lineplot(Ds*1000, ys, '관경 (mm)', ylabel, title, D_limit*1000))
st.caption('점선 구간은 Hagen–Poiseuille 층류식의 수학적 외삽이며 실제 난류 유량을 나타내지 않습니다.')

st.header('④ 역설계 — 최소 관경 탐색')
D_exact = (128 * mu * L * Q_target / (np.pi * P_allow))**0.25
f = lambda x: np.pi * P_allow * x**4 / (128 * mu * L) - Q_target
low = max(D_exact*0.25, 1e-12)
high = D_exact*2
D_num, root_info = bisect(f, low, high, xtol=1e-12, full_output=True)
D_design = np.ceil(D_num*1e5 - 1e-10) / 1e5  # Round up to 0.01 mm
p_design = pressure_for_flow(Q_target, D_design)
re_design = 4 * rho * Q_target / (np.pi * mu * D_design)
a, b, c, d = st.columns(4)
a.metric('이론 최소 관경', f'{D_exact*1000:.6f} mm')
b.metric('이분법 계산 관경', f'{D_num*1000:.6f} mm')
c.metric('설계 관경 (0.01 mm 올림)', f'{D_design*1000:.3f} mm')
d.metric('반복 횟수', f'{root_info.iterations}회')
st.write(f'설계 관경에서 목표유량을 위한 필요 압력강하: **{p_design:.6f} Pa** · Re: **{re_design:.3f}**')
if p_design <= P_allow and re_design < Re_limit:
    st.success('허용 압력강하 및 층류 조건 모두 만족')
else:
    st.warning('설계조건을 모두 만족하지 않습니다. 입력 조건 또는 설계 기준을 조정하세요.')
d_plot = np.linspace(D_exact*0.8, D_exact*1.4, 250)
fig = lineplot(d_plot*1000, np.array([pressure_for_flow(Q_target, x) for x in d_plot]), '관경 (mm)', '필요 압력강하 (Pa)', '목표유량 유지 시 필요 압력강하')
fig.axes[0].axhline(P_allow, color='red', linestyle='--', label='Allowable Pressure Drop')
fig.axes[0].axvline(D_design*1000, color='green', linestyle=':', label='Selected Diameter')
fig.axes[0].legend()
st.pyplot(fig)

st.header('⑤ 검증 및 해석')
Q_int = quad(lambda rr: 2*np.pi*rr*delta_P*(R**2-rr**2)/(4*mu*L), 0, R, epsabs=1e-16)[0]
err_Q = abs(Q_int-Q)/Q*100
err_D = abs(D_num-D_exact)/D_exact*100
st.subheader('정방향 계산 및 역설계 검증')
st.dataframe(pd.DataFrame({'검증 항목': ['이론 유량 (mL/s)', '수치적분 유량 (mL/s)', '유량 상대오차 (%)', '최대/평균유속 비', '이분법 관경 상대오차 (%)', '이분법 유량 잔차 (m³/s)'], '결과': [f'{Q*1e6:.9f}', f'{Q_int*1e6:.9f}', f'{err_Q:.3e}', f'{u[0]/V:.9f}', f'{err_D:.3e}', f'{abs(f(D_num)):.3e}']}), hide_index=True, use_container_width=True)
with st.expander('이전 Parameter Sweep 보고서 수치와 독립 재계산 비교'):
    st.caption('보고서 물성 ρ=998.2 kg/m³, μ=0.001002 Pa·s 사용. 관 길이와 압력강하는 현재 입력값을 사용합니다. 기본값 L=10 m, ΔP=500 Pa에서 원래 보고서 조건과 일치합니다.')
    report = {1: 0.012, 3: 0.995, 5: 7.680, 8: 50.961, 15: 590.160}
    rows = []
    for mm, reported in report.items():
        q, _, re, _ = flow(mm/1000, density=998.2, viscosity=0.001002)
        rows.append({'관경 (mm)': mm, '기존 보고서 유량 (mL/s)': reported, '재계산 유량 (mL/s)': round(q*1e6, 5), '상대차이 (%)': round(abs(reported-q*1e6)/(q*1e6)*100, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    limit_report = (32*0.001002**2*L*Re_limit/(998.2*delta_P))**(1/3)
    st.write(f'보고서 물성 기준 Re 한계 관경 재계산: **{limit_report*1000:.4f} mm**')
st.caption('가정: 정상·비압축성·Newtonian·축대칭·완전발달 원관 층류, 벽면 no-slip. 층류 범위 밖에서는 다른 모델이 필요합니다.')
