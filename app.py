import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o modelo e os encoders
try:
    model = joblib.load('modelo_turnover.pkl')
    le_cargo = joblib.load('le_cargo.pkl')
    le_feedback = joblib.load('le_feedback.pkl')
except FileNotFoundError:
    st.error("Arquivos do modelo ou encoders não encontrados. Certifique-se de que 'modelo_turnover.pkl', 'le_cargo.pkl' e 'le_feedback.pkl' estão na mesma pasta.")
    st.stop() # Para a execução do Streamlit se os arquivos não forem encontrados

# Título da aplicação
st.title('Previsão de Turnover de Colaboradores de TI')
st.write('Insira os dados do colaborador para prever o risco de turnover nos próximos 6 meses.')

# Criar campos de entrada para os dados do colaborador
st.sidebar.header('Dados do Colaborador')

idade = st.sidebar.slider('Idade', 20, 60, 30)
tempo_empresa_meses = st.sidebar.slider('Tempo de Empresa (meses)', 1, 180, 24)
cargo = st.sidebar.selectbox('Cargo', le_cargo.classes_)
salario_bruto = st.sidebar.slider('Salário Bruto', 3000, 20000, 7000)
performance_avaliacao = st.sidebar.slider('Performance (1-5)', 1, 5, 3)
ult_promocao_meses = st.sidebar.slider('Meses desde a Última Promoção', 0, 60, 12)
num_projetos_ult_ano = st.sidebar.slider('Número de Projetos no Último Ano', 0, 10, 2)
satisfacao_clima = st.sidebar.slider('Satisfação no Clima (1-5)', 1, 5, 3)
horas_extras_ult_mes = st.sidebar.slider('Horas Extras no Último Mês', 0, 80, 5)
distancia_escritorio_km = st.sidebar.slider('Distância do Escritório (km)', 0, 150, 10)
feedback_lideranca = st.sidebar.selectbox('Feedback da Liderança', le_feedback.classes_)

# Botão para fazer a previsão
if st.sidebar.button('Prever Turnover'):
    # Codificar as variáveis categóricas para a previsão
    cargo_encoded = le_cargo.transform([cargo])[0]
    feedback_lideranca_encoded = le_feedback.transform([feedback_lideranca])[0]
# Criar um DataFrame com os dados de entrada
input_data = pd.DataFrame([[
    idade, tempo_empresa_meses, salario_bruto, performance_avaliacao,
    ult_promocao_meses, num_projetos_ult_ano, satisfacao_clima,
    horas_extras_ult_mes, distancia_escritorio_km,
    cargo_encoded, feedback_lideranca_encoded
]], columns=[
    'idade', 'tempo_empresa_meses', 'salario_bruto', 'performance_avaliacao',
    'ult_promocao_meses', 'num_projetos_ult_ano', 'satisfacao_clima',
    'horas_extras_ult_mes', 'distancia_escritorio_km',
    'cargo_encoded', 'feedback_lideranca_encoded'
])

# Fazer a previsão
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0] # Probabilidade de cada classe

st.subheader('Resultado da Previsão:')
if prediction == 1:
    st.error(f'**ALTO RISCO DE TURNOVER!** (Probabilidade: {prediction_proba[1]*100:.2f}%)')
    st.write('Recomenda-se uma intervenção proativa (ex: conversa com o líder, plano de desenvolvimento, ajuste de carga de trabalho).')
else:
    st.success(f'**BAIXO RISCO DE TURNOVER.** (Probabilidade: {prediction_proba[0]*100:.2f}%)')
    st.write('O colaborador parece engajado. Continue monitorando e oferecendo suporte.')

st.write('---')
st.subheader('Fatores Considerados (Exemplo de Explicabilidade):')
st.write('Esta seção é um exemplo simplificado. Em um modelo real, técnicas como SHAP ou LIME seriam usadas para detalhar a contribuição de cada fator.')
st.write(f'- Idade: {idade} anos')
st.write(f'- Tempo de Empresa: {tempo_empresa_meses} meses')
st.write(f'- Cargo: {cargo}')
st.write(f'- Satisfação no Clima: {satisfacao_clima}/5')
st.write(f'- Horas Extras: {horas_extras_ult_mes}h/mês')
