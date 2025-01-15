import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class ObtenerDatosMercado:
    def __init__(self):
        # Índices bursátiles de diferentes países
        self.indices_mercado = {
            'China': '000001.SS',  # Índice compuesto de Shanghái
            'EE. UU.': '^GSPC',     # S&P 500
            'Alemania': '^GDAXI',   # DAX
            'España': '^IBEX'       # IBEX 35
        }
        
        # Índices sectoriales (ETFs)
        self.indices_sectoriales = {
            'Tecnología': 'XLK',
            'Finanzas': 'XLF',
            'Energía': 'XLE',
            'Materiales': 'XLB',
            'Industrial': 'XLI'
        }

    @staticmethod
    @st.cache_data(ttl=3600)  # Almacena en caché los datos durante 1 hora (3600 segundos)
    def obtener_datos_mercado(fecha_inicio, fecha_fin, indices):
        """Obtiene datos de índices bursátiles."""
        datos_mercado = {}
        fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
        fecha_fin_str = fecha_fin.strftime('%Y-%m-%d')
        
        try:
            # Descarga datos de Yahoo Finance
            datos = yf.download(list(indices.values()), start=fecha_inicio_str, end=fecha_fin_str, progress=False)
            for mercado, ticker in indices.items():
                try:
                    if isinstance(datos, pd.Series):
                        # Si solo se ha descargado un ticker
                        precios = datos
                    elif isinstance(datos['Close'], pd.Series):
                        # Si no se ha podido descargar un ticker especifico
                        precios = datos['Close']
                    else:
                        # Si se han descargado varios tickers
                        precios = datos['Close'][ticker]
                        
                    if not precios.empty and not precios.isna().all():
                        # Limpia los datos, elimina los valores nulos
                        precios = precios.dropna()
                        if len(precios) > 0:
                            # Normaliza los precios respecto al primer valor
                            primer_valor = precios.iloc[0]
                            datos_mercado[mercado] = (precios / primer_valor) * 100
                except Exception as e:
                    st.warning(f"Error al procesar los datos de {mercado}: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"Error al obtener datos del mercado: {str(e)}")
            return pd.DataFrame()
            
        # Rellena los valores faltantes con el método 'ffill' (rellenar hacia adelante) y 'bfill' (rellenar hacia atrás)
        return pd.DataFrame(datos_mercado).fillna(method='ffill').fillna(method='bfill')

    def get_datos_mercado(self, fecha_inicio, fecha_fin):
        # Devuelve los datos del mercado
        return self.obtener_datos_mercado(fecha_inicio, fecha_fin, self.indices_mercado)

    @staticmethod
    @st.cache_data(ttl=3600)  # Almacena en caché los datos durante 1 hora
    def obtener_datos_sectoriales(fecha_inicio, fecha_fin, indices):
        """Obtiene datos de rendimiento sectorial."""
        datos_sectoriales = {}
        fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
        fecha_fin_str = fecha_fin.strftime('%Y-%m-%d')
        
        try:
            # Descarga datos de Yahoo Finance
            datos = yf.download(list(indices.values()), start=fecha_inicio_str, end=fecha_fin_str, progress=False)
            for sector, ticker in indices.items():
                try:
                    if isinstance(datos, pd.Series):
                        # Si solo se ha descargado un ticker
                        precios = datos
                    elif isinstance(datos['Close'], pd.Series):
                        # Si no se ha podido descargar un ticker especifico
                        precios = datos['Close']
                    else:
                        # Si se han descargado varios tickers
                        precios = datos['Close'][ticker]
                        
                    if not precios.empty and not precios.isna().all():
                        # Limpia los datos, elimina los valores nulos
                        precios = precios.dropna()
                        if len(precios) > 0:
                            # Normaliza los precios respecto al primer valor
                            primer_valor = precios.iloc[0]
                            datos_sectoriales[sector] = (precios / primer_valor) * 100
                except Exception as e:
                    st.warning(f"Error al procesar los datos de {sector}: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"Error al obtener datos sectoriales: {str(e)}")
            return pd.DataFrame()
            
        # Rellena los valores faltantes con el método 'ffill' y 'bfill'
        return pd.DataFrame(datos_sectoriales).fillna(method='ffill').fillna(method='bfill')

    def get_datos_sectoriales(self, fecha_inicio, fecha_fin):
        # Devuelve los datos sectoriales
        return self.obtener_datos_sectoriales(fecha_inicio, fecha_fin, self.indices_sectoriales)

    def calcular_puntaje_vulnerabilidad(self, datos_sectoriales):
        """Calcula los puntajes de vulnerabilidad sectorial."""
        if datos_sectoriales.empty:
            return pd.Series()
            
        # Calcula la volatilidad (desviación estándar móvil de 20 días)
        volatilidad = datos_sectoriales.pct_change().rolling(window=20).std()
        
        # Calcula el momentum (retornos de 20 días)
        momentum = datos_sectoriales.pct_change(periods=20)
        
        # Calcula la correlación con el mercado general
        correlacion_mercado = datos_sectoriales.corrwith(datos_sectoriales.mean(axis=1))
        
        # Combina las métricas
        ultima_volatilidad = volatilidad.iloc[-1]
        ultimo_momentum = momentum.iloc[-1]
        
        # Calcula el puntaje combinado
        vulnerabilidad = (0.4 * ultima_volatilidad + 
                       0.3 * abs(correlacion_mercado) + 
                       0.3 * abs(ultimo_momentum))
        
        # Normaliza al rango 0-1
        if len(vulnerabilidad) > 0:
            vulnerabilidad = (vulnerabilidad - vulnerabilidad.min()) / (vulnerabilidad.max() - vulnerabilidad.min())
        
        return vulnerabilidad.sort_values(ascending=False)

    def calcular_velocidad_mercado(self, datos_sectoriales, ventana=20):
        """Calcula la velocidad del mercado (tasa de cambio de precio)."""
        if datos_sectoriales.empty:
            return pd.DataFrame()
        return datos_sectoriales.pct_change().rolling(window=ventana).mean()

class PanelAnalisisContagio:
    def __init__(self):
        self.obtentor_datos = ObtenerDatosMercado()
        self.configurar_pagina()
        
    def configurar_pagina(self):
        # Configuración de la página
        st.set_page_config(
            page_title="Análisis de Contagio del Mercado Global",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Análisis de Contagio del Mercado Global")
        st.markdown("""
        Este panel analiza los posibles efectos de contagio de los mercados de valores chinos a los mercados globales,
        centrándose en las correlaciones del mercado, las vulnerabilidades sectoriales y los patrones de transmisión de shocks.
        """)

    def mostrar_panel_mercado(self, datos_mercado):
        # Panel de rendimiento del mercado
        st.header("Rendimiento de los Mercados Bursátiles Mundiales")
        
        if datos_mercado.empty:
            st.error("No hay datos de mercado disponibles")
            return
            
        fig = go.Figure()
        
        # Añade una línea por cada mercado
        for mercado in datos_mercado.columns:
            fig.add_trace(
                go.Scatter(
                    x=datos_mercado.index,
                    y=datos_mercado[mercado],
                    name=mercado,
                    mode='lines'
                )
            )
        
        # Configuración del gráfico
        fig.update_layout(
            height=500,
            template="plotly_dark",
            xaxis_title="Fecha",
            yaxis_title="Precio Normalizado (Base=100)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Muestra el rendimiento reciente
        if len(datos_mercado) >= 2:
            st.subheader("Rendimiento Reciente")
            columnas = st.columns(len(datos_mercado.columns))
            for i, mercado in enumerate(datos_mercado.columns):
                ultimo = datos_mercado[mercado].iloc[-1]
                cambio = ((datos_mercado[mercado].iloc[-1] / datos_mercado[mercado].iloc[-2]) - 1) * 100
                with columnas[i]:
                    st.metric(
                        mercado,
                        f"{ultimo:.2f}",
                        f"{cambio:+.2f}%"
                    )

    def mostrar_analisis_sectorial(self, datos_sectoriales, pais_seleccionado):
        # Análisis sectorial
        st.header(f"Análisis Sectorial - {pais_seleccionado}")
        
        if datos_sectoriales.empty:
            st.error("No hay datos sectoriales disponibles")
            return
            
        # Calcula las métricas
        puntajes_vulnerabilidad = self.obtentor_datos.calcular_puntaje_vulnerabilidad(datos_sectoriales)
        correlaciones = datos_sectoriales.corr()
        velocidad_mercado = self.obtentor_datos.calcular_velocidad_mercado(datos_sectoriales)
        
        # Crea una figura con subgráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rendimiento Sectorial',
                'Puntajes de Vulnerabilidad',
                'Matriz de Correlación',
                'Velocidad del Mercado'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.15
        )
        
        # 1. Rendimiento sectorial con su propia leyenda
        for sector in datos_sectoriales.columns:
            fig.add_trace(
                go.Scatter(
                    x=datos_sectoriales.index,
                    y=datos_sectoriales[sector],
                    name=sector,
                    mode='lines',
                    legendgroup="rendimiento",
                    legendgrouptitle_text="Rendimiento por Sector"
                ),
                row=1, col=1
            )
        
        # 2. Puntajes de vulnerabilidad con su propia leyenda
        if not puntajes_vulnerabilidad.empty:
            fig.add_trace(
                go.Bar(
                    x=puntajes_vulnerabilidad.index,
                    y=puntajes_vulnerabilidad.values,
                    name='Puntaje de Vulnerabilidad',
                    legendgroup="vulnerabilidad",
                    legendgrouptitle_text="Vulnerabilidad",
                    marker_color='cyan'
                ),
                row=1, col=2
            )
        
        # 3. Matriz de correlación (sin leyenda)
        fig.add_trace(
            go.Heatmap(
                z=correlaciones.values,
                x=correlaciones.columns,
                y=correlaciones.columns,
                colorscale='RdBu',
                showscale=False,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Velocidad del mercado con su propia leyenda
        if not velocidad_mercado.empty:
            for sector in velocidad_mercado.columns:
                fig.add_trace(
                    go.Scatter(
                        x=velocidad_mercado.index,
                        y=velocidad_mercado[sector],
                        name=sector,
                        mode='lines',
                        legendgroup="velocidad",
                        legendgrouptitle_text="Velocidad del Mercado"
                    ),
                    row=2, col=2
                )
        
        # Actualiza el diseño con leyendas separadas
        fig.update_layout(
            height=800,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                tracegroupgap=30,
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                groupclick="toggleitem"
            ),
            margin=dict(l=40, r=150, t=40, b=40)
        )
        
        # Actualiza las etiquetas de los ejes
        fig.update_xaxes(title_text="Fecha", row=1, col=1)
        fig.update_xaxes(title_text="Sectores", row=1, col=2)
        fig.update_xaxes(title_text="Fecha", row=2, col=2)
        
        fig.update_yaxes(title_text="Precio Normalizado", row=1, col=1)
        fig.update_yaxes(title_text="Puntaje", row=1, col=2)
        fig.update_yaxes(title_text="Velocidad", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de riesgo
        if not puntajes_vulnerabilidad.empty:
            sectores_alto_riesgo = puntajes_vulnerabilidad[puntajes_vulnerabilidad > 0.7].index.tolist()
            if sectores_alto_riesgo:
                st.warning(f"Sectores de alto riesgo: {', '.join(sectores_alto_riesgo)}")

def main():
    # Función principal
    panel = PanelAnalisisContagio()
    
    # Controles en la barra lateral
    st.sidebar.header("Controles")
    
    # Selector de rango de fechas
    hoy = datetime.now()
    inicio_predeterminado = hoy - timedelta(days=365)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        fecha_inicio = st.date_input(
            "Fecha de Inicio",
            value=inicio_predeterminado,
            key='fecha_inicio'
        )
        
    with col2:
        fecha_fin = st.date_input(
            "Fecha de Fin",
            value=hoy,
            key='fecha_fin'
        )
    
    # Valida que la fecha de inicio sea anterior a la fecha de fin
    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin")
        return
        
    # Selector de mercados
    mercados_seleccionados = st.sidebar.multiselect(
        "Seleccione los Mercados para Comparar con China",
        ["EE. UU.", "Alemania", "España"],
        default=["EE. UU.", "Alemania"]
    )
    
    # Siempre incluye China y agrega los mercados seleccionados
    mercados = ["China"] + mercados_seleccionados
    
    try:
        with st.spinner("Obteniendo datos del mercado..."):
            # Obtiene los datos del mercado
            datos_mercado = panel.obtentor_datos.get_datos_mercado(
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin
            )
            
            if not datos_mercado.empty:
                # Filtra los datos por los mercados seleccionados
                mercados_disponibles = [m for m in mercados if m in datos_mercado.columns]
                if mercados_disponibles:
                    datos_mercado = datos_mercado[mercados_disponibles]
                    panel.mostrar_panel_mercado(datos_mercado)
                    
                    # Selector de país para el análisis sectorial
                    pais_seleccionado = st.selectbox(
                        "Seleccione un País para el Análisis Sectorial",
                        mercados_disponibles
                    )
                    
                    # Obtiene los datos sectoriales
                    datos_sectoriales = panel.obtentor_datos.get_datos_sectoriales(
                        fecha_inicio=fecha_inicio,
                        fecha_fin=fecha_fin
                    )
                    
                    if not datos_sectoriales.empty:
                        panel.mostrar_analisis_sectorial(datos_sectoriales, pais_seleccionado)
                    else:
                        st.error("No hay datos sectoriales disponibles para los parámetros seleccionados")
                else:
                    st.error("No hay datos disponibles para los mercados seleccionados")
            else:
                st.error("No hay datos de mercado disponibles")
                
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
        st.error("Por favor, intente ajustar su selección o inténtelo de nuevo más tarde")

if __name__ == "__main__":
    main()