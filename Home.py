import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import io
import base64
import plotly.graph_objects as go

st.set_page_config(layout="wide")

@st.cache_data
def load_and_preprocess_data():
    try:
        # Caminho do arquivo CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Corrected: use __file__
        csv_file_path = os.path.join(script_dir, 'forecast_Fevereiro(Previsoes).csv')

        # Leitura do CSV
        df = pd.read_csv(csv_file_path, encoding="latin1", sep=";")

        # Renomeação das colunas para remover acentos e espaços indesejados
        df.rename(columns={
            "y": "total",
            "cat_informação": "cat_informacao",
            "cat_reclamação": "cat_reclamacao",  # renomeia com acento para sem acento
            "cat_pré-venda": "cat_pre_venda",
            "cat_solicitação": "cat_solicitacao"
        }, inplace=True)

        # Conversão de datas
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['ds_normalized'] = df['ds'].dt.normalize()
        df['year'] = df['ds'].dt.year
        df.loc[:, 'year'] = df['year'].replace({np.nan: None})
        df['month'] = df['ds'].dt.month

        # Seleciona apenas o mês de fevereiro (mês == 2)
        df = df[df['month'] == 2]

        # Lista de colunas que precisamos converter e somar
        cols_to_sum = [
            "cat_cancelamento",
            "cat_informacao",
            "cat_reclamacao",
            "cat_troca",
            "cat_preventiva",
            "cat_pre_venda",
            "cat_solicitacao"
        ]
        # Verifica se cada coluna existe, convertendo para numérico ou criando com zeros
        for col in cols_to_sum:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0

        # Converte a coluna total para numérico
        df['count'] = df[cols_to_sum].sum(axis=1)
        df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0)

        # Ajuste: Reescala as categorias para que a soma por linha seja igual ao total
        df['sum_categories'] = df[cols_to_sum].sum(axis=1)
        mask = (df['sum_categories'] > 0) & (~np.isclose(df['sum_categories'], df['total']))
        df.loc[mask, cols_to_sum] = (
            df.loc[mask, cols_to_sum]
            .multiply(df.loc[mask, 'total'], axis=0)
            .div(df.loc[mask, 'sum_categories'], axis=0)
        )
        df.drop(columns='sum_categories', inplace=True)

        return df
    except FileNotFoundError:
        st.error("Erro: 'forecast_Fevereiro(Previsoes).csv' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
        st.stop()


df = load_and_preprocess_data()

# -------------------- Filtros na Sidebar --------------------

st.sidebar.header("Filtros")

# Filtro de Data
st.sidebar.subheader("Data")
min_date = df['ds_normalized'].min().date()
max_date = df['ds_normalized'].max().date()
selected_dates = st.sidebar.date_input("Selecione o intervalo de dias", [min_date, max_date])

# Filtro de Categoria
st.sidebar.subheader("Categorias")
available_categories = [col for col in df.columns if col.startswith('cat_')]
selected_categories = st.sidebar.multiselect("Selecione as Categorias", available_categories, default=available_categories)

# Filtro de Ano (se houver dados de mais de um ano)
if df['year'].nunique() > 1:
    st.sidebar.subheader("Ano")
    available_years = sorted(df['year'].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect("Selecione os Anos", available_years, default=available_years)
else:
    selected_years = [df['year'].iloc[0]]  # Garante que selected_years esteja sempre definido

# -------------------- Filtragem dos Dados --------------------

filtered_df = df.copy()

# Construir a condição para o filtro
filter_condition = pd.Series(True, index=filtered_df.index)

# Aplicar o filtro de Data
if isinstance(selected_dates, list) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    filter_condition &= filtered_df['ds_normalized'].dt.date.between(start_date, end_date)

# Aplicar o filtro de Categoria
if selected_categories:
    category_filter = filtered_df[selected_categories].any(axis=1)
    filter_condition &= category_filter

# Aplicar o filtro de Ano
if 'year' in filtered_df.columns and selected_years:
    filter_condition &= filtered_df['year'].isin(selected_years)

# Aplicar o filtro combinado
filtered_df = filtered_df[filter_condition]

# ------ CSS Styling ------

CSS = """
<style>
    /* Background da aplicação em um cinza escuro moderno */
    [data-testid="stAppViewContainer"] {
        background-color: #2b2b2b;
    }
.data-panel {
    text-align: center;
    background-color: #2b2b2b; /* Cor de fundo desejada */
    color: white;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3), 0 0 10px rgba(0, 0, 0, 0.1) inset; /* Sombra interna e externa */

}
.data-panel-title {
    font-size: 0.9em;
    text-transform: uppercase; /* Converte todos os títulos para maiúsculas */
}
.data-panel-value {
    font-size: 2em;
    font-weight: bold;
}
.data-panel-total {
    font-size: 0.8em;
}
.chart-container {
    border-bottom: 1px solid #666; /* Adiciona a borda inferior */
    margin-bottom: 20px; /* Espaçamento abaixo da borda */
    padding-bottom: 20px; /* Espaçamento entre o gráfico e a borda */
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2); /* Sombra sutil nos gráficos */
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- Função para exibir os painéis de dados --------------------

def display_data_panels(num_panels, data_dict=None):
    cols = st.columns(num_panels)
    with st.container():
        st.markdown('<div class="panel-row">', unsafe_allow_html=True)
        for i, col in enumerate(cols):
            with col:
                panel_num = i + 1
                if data_dict and panel_num in data_dict:
                    title, value, subtext = data_dict[panel_num]
                    # Formata o valor, removendo decimais e separando por milhar
                    if isinstance(value, (int, float)):
                        if value >= 1000:
                            formatted_value = f"{value:,.0f}".replace(",", ".")
                        else:
                            formatted_value = f"{int(value)}"
                    else:
                        formatted_value = value
                    st.markdown(f"""
                    <div class="data-panel">
                        <div class="data-panel-title">{title}</div>
                        <div class="data-panel-value">{formatted_value}</div>
                        <div class="data-panel-total">{subtext}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="data-panel"><h3>Painel {panel_num}</h3><h1>{panel_num}</h1></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Funções de Gráficos --------------------

def create_total_sum_bar_chart(data, categories):
    if data.empty:
        fig = px.bar(title="Sem dados para exibir com os filtros selecionados")
        fig.update_layout(yaxis_range=[0, 1])  # Define um intervalo mínimo para o eixo Y
    else:
        try:
            for col in categories:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            category_sums = data[categories].sum().reset_index()
            category_sums.rename(columns={'index': 'category'}, inplace=True)
            category_sums['category'] = category_sums['category'].str.replace('cat_', '')
            category_sums.columns = ['category', 'total']
            fig = px.bar(category_sums, x='category', y='total',
                        title='Total de Casos por Categoria',
                        color_discrete_sequence=['#CD9A33'],
                        text_auto=False)  # Desativa o text_auto padrão
            fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont=dict(color='white'))  # Formatação: sem decimais, separador de milhar
            fig.update_layout(
                height=350,
                title_x=0.1,
                margin=dict(b=100),
                title_font=dict(color='white'),
                font=dict(size=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                xaxis_title=None, yaxis_title=None,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),  # Escala dinâmica
            )
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar o gráfico: {e}")
            fig = px.bar(title="Erro ao gerar o gráfico")  # Crie um gráfico de erro
            fig.update_layout(
                height=350,
                title_x=0.1,
                margin=dict(b=100),
                title_font=dict(color='white'),
                font=dict(size=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                xaxis_title=None, yaxis_title=None,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        return fig

def create_daily_sum_bar_charts(data, categories):
    if data.empty:
        fig = px.bar(title="Sem dados para exibir com os filtros selecionados")
        fig.update_layout(yaxis_range=[0, 1])  # Define um intervalo mínimo para o eixo Y
    else:
        df_temp = data.copy()
        today = pd.to_datetime('today').normalize()
        last_7_days = pd.date_range(start=today, periods=7, freq='D')
        df_temp = df_temp[df_temp['ds_normalized'].isin(last_7_days)]
        if df_temp.empty:
            fig = px.bar(title="Sem dados para exibir nos próximos 7 dias")
        else:
            try:
                for col in categories:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                daily_counts = df_temp.groupby('ds_normalized')[categories].sum().reset_index()
                daily_counts['daily_sum'] = daily_counts[categories].sum(axis=1)
                daily_df = daily_counts[['ds_normalized', 'daily_sum']].copy()
                daily_df.rename(columns={'ds_normalized': 'Dia'}, inplace=True)
                fig = px.bar(daily_df, x='Dia', y='daily_sum',
                            title='Contagem Diária de Casos (Próximos 7 Dias)',
                            color_discrete_sequence=['#CD9A33'],
                            text_auto=False)  # Desativa o text_auto padrão
                fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont=dict(color='white'))  # Formatação: sem decimais, separador de milhar
                fig.update_layout(
                    height=350,
                    title_x=0.1,
                    margin=dict(b=100),
                    title_font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                    xaxis_title=None, yaxis_title=None,
                    xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True), # Escala dinâmica
                )
            except Exception as e:
                st.error(f"Ocorreu um erro ao gerar o gráfico: {e}")
                fig = px.bar(title="Erro ao gerar o gráfico")  # Crie um gráfico de erro
                fig.update_layout(
                    height=350,
                    title_x=0.1,
                    margin=dict(b=100),
                    title_font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                    xaxis_title=None, yaxis_title=None,
                    xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
            return fig

def create_big_bar_chart(data):
    if data.empty:
        fig = px.bar(title="Sem dados para exibir com os filtros selecionados")
        fig.update_layout(yaxis_range=[0, 1])  # Define um intervalo mínimo para o eixo Y
    else:
        df_temp = data.copy()
        max_date = df_temp['ds_normalized'].max()
        first_day_month = max_date.replace(day=1)
        df_temp = df_temp[df_temp['ds_normalized'] >= first_day_month].copy()
        if df_temp.empty:
            fig = px.bar(title="Sem dados para exibir no último mês")
        else:
            try:
                pie_categories = ['cat_cancelamento', 'cat_informacao', 'cat_reclamacao',
                                'cat_troca', 'cat_preventiva', 'cat_pre_venda', 'cat_solicitacao']
                for col in pie_categories:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                daily_counts = df_temp.groupby('ds_normalized')[pie_categories].sum().reset_index()
                daily_counts['monthly_sum'] = daily_counts[pie_categories].sum(axis=1)
                daily_df = daily_counts[['ds_normalized', 'monthly_sum']].copy()
                daily_df.rename(columns={'ds_normalized': 'Dia'}, inplace=True)
                fig = px.bar(daily_df, x='Dia', y='monthly_sum',
                            title='Contagem Diária de Casos (1 Mês)',
                            color_discrete_sequence=['#CD9A33'],
                            text_auto=False)  # Desativa o text_auto padrão
                fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont=dict(color='white'))  # Formatação: sem decimais, separador de milhar
                fig.update_layout(
                    height=350,
                    title_x=0.1,
                    margin=dict(b=100),
                    title_font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                    xaxis_title=None, yaxis_title=None,
                    xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True) # Escala dinâmica
                )
            except Exception as e:
                st.error(f"Ocorreu um erro ao gerar o gráfico: {e}")
                fig = px.bar(title="Erro ao gerar o gráfico")  # Crie um gráfico de erro
                fig.update_layout(
                    height=350,
                    title_x=0.1,
                    margin=dict(b=100),
                    title_font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                    xaxis_title=None, yaxis_title=None,
                    xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
            return fig

# Nova função: gráfico de linha suavizado para os últimos 30 dias

def create_30_day_category_smoothed_line_chart(data):
    df_temp = data.copy()

    # Renomear a coluna 'ds_normalized' para 'Data' para melhor exibição
    if 'ds_normalized' in df_temp.columns:
        df_temp = df_temp.rename(columns={'ds_normalized': 'Data'})
    else:
        fig = px.line(title="Sem dados para exibir nos últimos 30 dias", height=350)
        return fig

    max_date = df_temp['Data'].max()
    if pd.isna(max_date):
        fig = px.line(title="Sem dados para exibir nos últimos 30 dias", height=350)
        return fig

    last_30_days = pd.date_range(end=max_date, periods=30, freq='D')
    df_temp = df_temp[df_temp['Data'].isin(last_30_days)]

    if df_temp.empty:
        fig = px.line(title="Sem dados para exibir nos últimos 30 dias", height=350)
    else:
        try:
            category_cols = [
                "cat_cancelamento",
                "cat_informacao",
                "cat_reclamacao",
                "cat_troca",
                "cat_preventiva",
                "cat_pre_venda",
                "cat_solicitacao"
            ]
            for col in category_cols:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)

            daily_counts = df_temp.groupby('Data')[category_cols].sum().reset_index()
            daily_counts = daily_counts.melt(id_vars='Data', var_name='category', value_name='count')
            daily_counts['category'] = daily_counts['category'].str.replace('cat_', '')

            fig = px.line(
                daily_counts,
                x='Data',
                y='count',
                color='category',
                title='Contagem Diária de Casos por Categoria (Últimos 30 Dias)',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hover_data={"Data": "|%b %d", "count": ":,.0f", "category": True}  # Formata o texto ao passar o mouse
            )

            fig.update_traces(line_shape='spline', mode='lines+markers')
            fig.update_layout(
                height=350,
                title_x=0.1,
                margin=dict(b=100),
                title_font=dict(color='white'),
                font=dict(size=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                xaxis_title=None, yaxis_title=None,
                xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend_title=None,
            )
            return fig
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar o gráfico: {e}")
            fig = px.line(title="Erro ao gerar o gráfico")  # Crie um gráfico de erro
            fig.update_layout(
                height=350,
                title_x=0.1,
                margin=dict(b=100),
                title_font=dict(color='white'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin_t=50, margin_b=50, margin_l=0, margin_r=0,
                xaxis_title=None, yaxis_title=None,
                xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %d', showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            return fig

# -------------------- Função para exibir os painéis de dados --------------------

def display_data_panels(num_panels, data_dict=None):
    cols = st.columns(num_panels)
    with st.container():
        st.markdown('<div class="panel-row">', unsafe_allow_html=True)
        for i, col in enumerate(cols):
            with col:
                panel_num = i + 1
                if data_dict and panel_num in data_dict:
                    title, value, subtext = data_dict[panel_num]
                    # Formata o valor, removendo decimais e separando por milhar
                    if isinstance(value, (int, float)):
                        if value >= 1000:
                            formatted_value = f"{value:,.0f}".replace(",", ".")
                        else:
                            formatted_value = f"{int(value)}"
                    else:
                        formatted_value = value
                    st.markdown(f"""
                    <div class="data-panel">
                        <div class="data-panel-title">{title}</div>
                        <div class="data-panel-value">{formatted_value}</div>
                        <div class="data-panel-total">{subtext}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="data-panel"><h3>Painel {panel_num}</h3><h1>{panel_num}</h1></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Cálculo das Métricas e Painéis --------------------

categories = [
    'cat_cancelamento', 'cat_informacao', 'cat_reclamacao',
    'cat_troca', 'cat_preventiva', 'cat_pre_venda', 'cat_solicitacao'
]
first_7_categories = categories[:7]

if not filtered_df.empty:
    category_totals = {cat: filtered_df[cat].sum() if cat in filtered_df.columns else 0 for cat in first_7_categories}
    total_cases = filtered_df['total'].sum()

   # total_days = filtered_df['Data'].nunique() if 'Data' in filtered_df.columns else 0 #linha corrigida

    total_days = filtered_df['ds_normalized'].nunique() if 'ds_normalized' in filtered_df.columns else 0 #Linha original

    average_cases_per_day = total_cases / total_days if total_days != 0 else 0
    panel_data = {
        1: (first_7_categories[0].replace("cat_", ""), category_totals.get(first_7_categories[0], 0), "Total"),
        2: (first_7_categories[1].replace("cat_", ""), category_totals.get(first_7_categories[1], 0), "Total"),
        3: (first_7_categories[2].replace("cat_", ""), category_totals.get(first_7_categories[2], 0), "Total"),
        4: (first_7_categories[3].replace("cat_", ""), category_totals.get(first_7_categories[3], 0), "Total"),
        5: (first_7_categories[4].replace("cat_", ""), category_totals.get(first_7_categories[4], 0), "Total"),
        6: (first_7_categories[5].replace("cat_", ""), category_totals.get(first_7_categories[5], 0), "Total"),
        7: (first_7_categories[6].replace("cat_", ""), category_totals.get(first_7_categories[6], 0), "Total"),
        8: ("Total de Casos", total_cases, ""),
        9: ("Dias", total_days, ""),
        10: ("Média de casos", average_cases_per_day, "")
    }
else:
    # Define valores padrão para os painéis de dados
    panel_data = {
        1: ("N/A", 0, "Sem Dados"),
        2: ("N/A", 0, "Sem Dados"),
        3: ("N/A", 0, "Sem Dados"),
        4: ("N/A", 0, "Sem Dados"),
        5: ("N/A", 0, "Sem Dados"),
        6: ("N/A", 0, "Sem Dados"),
        7: ("N/A", 0, "Sem Dados"),
        8: ("Total de Casos", 0, ""),
        9: ("Dias", 0, ""),
        10: ("Média de casos", 0, "")
    }

# -------------------- Exibição dos Painéis --------------------

upper_panels = {i: panel_data[i] for i in range(1, 8)}
display_data_panels(7, upper_panels)

lower_panels = {i+1: v for i, v in enumerate(list(panel_data.values())[7:10])}
display_data_panels(3, lower_panels)

# -------------------- Exibição dos Gráficos --------------------

cols4 = st.columns(2)

def display_chart(col, chart_func, data, categories, title="Sem dados para exibir"):
    with col:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div>", unsafe_allow_html=True)
        st.plotly_chart(chart_func(data, categories), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

display_chart(cols4[0], create_total_sum_bar_chart, filtered_df, first_7_categories)
display_chart(cols4[1], create_daily_sum_bar_charts, filtered_df, first_7_categories)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<div>", unsafe_allow_html=True)
st.plotly_chart(create_big_bar_chart(filtered_df), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Gráfico de linha suavizado para os últimos 30 dias

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<div>", unsafe_allow_html=True)
st.plotly_chart(create_30_day_category_smoothed_line_chart(filtered_df), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Exportação de Dados --------------------

st.markdown("### Exportar Dados")
download_format = st.radio("Selecione o formato:", ["csv", "xlsx"], horizontal=True)
if st.button("Gerar Link de Download"):
    if not filtered_df.empty:
        def generate_download_link(data, file_format):
            if file_format == 'csv':
                output = io.StringIO()
                data.to_csv(output, index=False, encoding='utf-8')
                csv_data = output.getvalue()
                b64 = base64.b64encode(csv_data.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="dashboard_data.csv">Baixar CSV</a>'
            elif file_format == 'xlsx':
                output = io.BytesIO()
                data_export = data.copy()
                data_export = data_export.drop(columns=['year', 'month', 'ds_normalized'])
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data_export.to_excel(writer, index=False, sheet_name='Dashboard Data')
                    for column in data_export.columns:
                        column_width = max(data_export[column].astype(str).map(len).max(), len(column))
                        col_idx = data_export.columns.get_loc(column)
                        writer.sheets['Dashboard Data'].set_column(col_idx, col_idx, column_width + 2)
                xlsx_data = output.getvalue()
                b64 = base64.b64encode(xlsx_data).decode()
                return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="dashboard_data.xlsx">Baixar XLSX</a>'
            else:
                return ""
        download_link = generate_download_link(filtered_df, download_format)
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.warning("Nenhum dado para exportar.")