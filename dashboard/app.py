"""
Signal Research Pipeline - Dashboard
Real-time updates with modern UI design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import PriceFetcher
from src.signals import get_signal, SIGNAL_REGISTRY
from src.validation.backtester import Backtester
from src.monitoring.decay_tracker import DecayTracker, HealthScore

# Page config
st.set_page_config(
    page_title="Signal Research Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .metric-card h1 {
        font-size: 2.5em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card p {
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    
    .signal-long {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        border-radius: 12px;
        padding: 15px;
        color: white;
        margin: 5px 0;
    }
    .signal-short {
        background: linear-gradient(135deg, #ff5252 0%, #ff1744 100%);
        border-radius: 12px;
        padding: 15px;
        color: white;
        margin: 5px 0;
    }
    
    .status-healthy { 
        color: #00e676; 
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
    }
    .status-good { color: #69f0ae; }
    .status-warning { 
        color: #ffd600; 
        text-shadow: 0 0 10px rgba(255, 214, 0, 0.5);
    }
    .status-critical { 
        color: #ff5252; 
        text-shadow: 0 0 10px rgba(255, 82, 82, 0.5);
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 230, 118, 0.2);
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .live-dot {
        width: 10px;
        height: 10px;
        background: #00e676;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .ticker-up { color: #00e676; }
    .ticker-down { color: #ff5252; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


def load_price_data(symbols: list, days: int = 180):
    """Load price data."""
    fetcher = PriceFetcher(source='binance')
    return fetcher.fetch(symbols, days=days)


def compute_signal(signal_name: str, price_df: pd.DataFrame, **kwargs):
    """Compute signal."""
    signal = get_signal(signal_name, **kwargs)
    return signal.compute(price_df)


def create_price_chart(price_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create candlestick chart with volume."""
    df = price_df[price_df['symbol'] == symbol].copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#00e676',
            decreasing_line_color='#ff5252'
        ),
        row=1, col=1
    )
    
    colors = ['#00e676' if c >= o else '#ff5252' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='Volume', opacity=0.5),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=450,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_signal_gauge(value: float, name: str) -> go.Figure:
    """Create a gauge chart for signal value."""
    color = '#00e676' if value > 0 else '#ff5252'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': name, 'font': {'color': 'white', 'size': 16}},
        number={'font': {'color': 'white', 'size': 24}},
        gauge={
            'axis': {'range': [-2, 2], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 0,
            'steps': [
                {'range': [-2, -0.5], 'color': 'rgba(255, 82, 82, 0.3)'},
                {'range': [-0.5, 0.5], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [0.5, 2], 'color': 'rgba(0, 230, 118, 0.3)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=180,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def main():
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("# Signal Research Pipeline")
        st.markdown("*ML-Powered Crypto Signal Discovery & Monitoring*")
    with col2:
        st.markdown(f"""
        <div class="live-indicator">
            <div class="live-dot"></div>
            <span style="color: #00e676; font-weight: 600;">LIVE</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p style='color: #888; text-align: right;'>Updated: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        available_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK']
        selected_symbols = st.multiselect(
            "Select Assets",
            available_symbols,
            default=['BTC', 'ETH', 'SOL']
        )
        
        days = st.slider("Data Range (days)", 30, 365, 180)
        
        price_based_signals = [s for s in SIGNAL_REGISTRY.keys() if 'funding' not in s]
        selected_signals = st.multiselect(
            "Select Signals",
            price_based_signals,
            default=['momentum', 'mean_reversion', 'volatility']
        )
        
        st.divider()
        
        auto_refresh = st.toggle("Auto-Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 60)
            st.info(f"Refreshing every {refresh_interval}s")
        
        st.divider()
        
        load_clicked = st.button("Load Data", type="primary", use_container_width=True)
    
    # Validation
    if not selected_symbols:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    if not selected_signals:
        st.warning("Please select at least one signal from the sidebar.")
        return
    
    # Load data
    with st.spinner("Loading price data from Binance..."):
        try:
            price_df = load_price_data(selected_symbols, days)
            if len(price_df) == 0:
                st.error("No data received. Please try again.")
                return
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return
    
    # Live price ticker
    st.markdown("### Live Prices")
    price_cols = st.columns(len(selected_symbols))
    
    for i, symbol in enumerate(selected_symbols):
        symbol_data = price_df[price_df['symbol'] == symbol]
        if len(symbol_data) >= 2:
            latest_price = symbol_data['close'].iloc[-1]
            prev_price = symbol_data['close'].iloc[-2]
            change = (latest_price - prev_price) / prev_price * 100
            
            with price_cols[i]:
                color_class = "ticker-up" if change >= 0 else "ticker-down"
                arrow = "+" if change >= 0 else ""
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h3 style="margin: 0; color: #888;">{symbol}</h3>
                    <h2 style="margin: 5px 0; color: white;">${latest_price:,.2f}</h2>
                    <p class="{color_class}" style="font-size: 1.2em; margin: 0;">
                        {arrow}{change:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main tabs - Combined Signal Analysis with Charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Analysis",
        "Backtest Results",
        "ML Discovery",
        "Monitoring"
    ])
    
    # Tab 1: Combined Price Charts + Signal Analysis
    with tab1:
        # Asset selector at top
        selected_asset = st.selectbox("Select Asset", selected_symbols, key="main_asset")
        
        # Price Chart Section
        st.markdown("### Price Chart")
        fig = create_price_chart(price_df, selected_asset)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Compute signals
        signal_results = {}
        for sig_name in selected_signals:
            try:
                signal_df = compute_signal(sig_name, price_df)
                signal_results[sig_name] = signal_df
            except Exception as e:
                st.warning(f"Failed to compute {sig_name}: {e}")
        
        # Signal Analysis Section
        st.markdown("### Signal Analysis")
        
        if signal_results:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Signal heatmap
                st.markdown("#### Signal Heatmap")
                latest_signals = {}
                for sig_name, sig_df in signal_results.items():
                    latest = sig_df.groupby('symbol')['signal'].last()
                    latest_signals[sig_name] = latest
                
                heatmap_df = pd.DataFrame(latest_signals)
                
                fig = px.imshow(
                    heatmap_df.T,
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    labels={'color': 'Signal Strength'},
                    aspect='auto'
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Top Signals")
                
                for sig_name, sig_df in signal_results.items():
                    latest = sig_df.groupby('symbol')['signal'].last()
                    top = latest.nlargest(1)
                    bottom = latest.nsmallest(1)
                    
                    st.markdown(f"**{sig_name.replace('_', ' ').title()}**")
                    
                    for symbol, val in top.items():
                        st.markdown(f"""
                        <div class="signal-long">
                            Long: <strong>{symbol}</strong> ({val:.3f})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for symbol, val in bottom.items():
                        st.markdown(f"""
                        <div class="signal-short">
                            Short: <strong>{symbol}</strong> ({val:.3f})
                        </div>
                        """, unsafe_allow_html=True)
            
            # Signal gauges for selected asset
            st.markdown("#### Signal Gauges")
            
            gauge_cols = st.columns(len(signal_results))
            for i, (sig_name, sig_df) in enumerate(signal_results.items()):
                asset_signal = sig_df[sig_df['symbol'] == selected_asset]['signal'].iloc[-1] if len(sig_df[sig_df['symbol'] == selected_asset]) > 0 else 0
                with gauge_cols[i]:
                    fig = create_signal_gauge(asset_signal, sig_name)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Signal time series overlay
            st.markdown("#### Signal Time Series")
            overlay_signals = st.multiselect("Select signals to overlay", list(signal_results.keys()), default=list(signal_results.keys())[:2])
            
            if overlay_signals:
                fig = go.Figure()
                colors = ['#667eea', '#00e676', '#ff6d00', '#ff5252', '#ffd600']
                
                for i, sig_name in enumerate(overlay_signals):
                    sig_df = signal_results[sig_name]
                    asset_sig = sig_df[sig_df['symbol'] == selected_asset]
                    fig.add_trace(go.Scatter(
                        x=asset_sig['date'],
                        y=asset_sig['signal'],
                        name=sig_name,
                        mode='lines',
                        line=dict(width=2, color=colors[i % len(colors)])
                    ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Backtest Results
    with tab2:
        st.markdown("## Backtest Results")
        
        if signal_results:
            backtester = Backtester()
            backtest_results = {}
            
            progress = st.progress(0, text="Running backtests...")
            for i, (sig_name, sig_df) in enumerate(signal_results.items()):
                try:
                    result = backtester.run(sig_df, price_df, signal_name=sig_name)
                    backtest_results[sig_name] = result
                except Exception as e:
                    pass
                progress.progress((i + 1) / len(signal_results))
            progress.empty()
            
            if backtest_results:
                st.markdown("### Performance Metrics")
                
                for name, result in backtest_results.items():
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3 style="color: white; margin-bottom: 15px;">{name.replace('_', ' ').title()}</h3>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                            <div class="metric-card">
                                <h1>{result.ic_mean:.4f}</h1>
                                <p>IC Mean</p>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #00c853 0%, #00e676 100%);">
                                <h1>{result.ic_ir:.2f}</h1>
                                <p>IC IR</p>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #ff6d00 0%, #ff9100 100%);">
                                <h1>{result.sharpe_ratio:.2f}</h1>
                                <p>Sharpe</p>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #ff5252 0%, #ff1744 100%);">
                                <h1>{result.max_drawdown:.1%}</h1>
                                <p>Max DD</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### IC Time Series")
                fig = go.Figure()
                colors = ['#667eea', '#00e676', '#ff6d00', '#ff5252', '#ffd600']
                
                for i, (name, result) in enumerate(backtest_results.items()):
                    if result.ic_series is not None and len(result.ic_series) > 0:
                        ic_rolling = result.ic_series.rolling(20, min_periods=1).mean()
                        fig.add_trace(go.Scatter(
                            x=ic_rolling.index,
                            y=ic_rolling.values,
                            name=name,
                            mode='lines',
                            line=dict(width=3, color=colors[i % len(colors)])
                        ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Rolling IC (20d)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: ML Discovery
    with tab3:
        st.markdown("## ML-Based Signal Discovery")
        st.markdown("Use machine learning to discover which technical features best predict returns.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Run ML Discovery", type="primary", use_container_width=True):
                with st.spinner("Running XGBoost analysis..."):
                    try:
                        from src.ml.feature_selection import SignalDiscovery
                        
                        discovery = SignalDiscovery()
                        results = discovery.discover(price_df, top_n=10)
                        st.session_state['ml_results'] = results
                        st.success("Discovery complete!")
                    except Exception as e:
                        st.error(f"Discovery failed: {e}")
        
        if 'ml_results' in st.session_state:
            results = st.session_state['ml_results']
            
            st.markdown("### Feature Importance")
            if 'importance' in results:
                importance = results['importance'].head(10)
                
                fig = go.Figure(go.Bar(
                    x=importance.values,
                    y=importance.index,
                    orientation='h',
                    marker=dict(
                        color=importance.values,
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis_title="Importance Score",
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Monitoring
    with tab4:
        st.markdown("## Signal Health Monitoring")
        
        if signal_results and 'backtest_results' in dir() and backtest_results:
            health_scorer = HealthScore()
            
            monitoring_cols = st.columns(len(backtest_results))
            
            for i, (sig_name, result) in enumerate(backtest_results.items()):
                with monitoring_cols[i]:
                    if result.ic_series is not None and len(result.ic_series) > 0:
                        health = health_scorer.calculate(result.ic_series)
                        
                        status_class = f"status-{health['status']}"
                        
                        st.markdown(f"""
                        <div class="glass-card" style="text-align: center;">
                            <h3 style="color: white;">{sig_name}</h3>
                            <h1 class="{status_class}" style="font-size: 3em; margin: 10px 0;">
                                {health['health_score']:.0f}
                            </h1>
                            <p class="{status_class}" style="font-size: 1.2em; font-weight: 600;">
                                {health['status'].upper()}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for comp, val in health['components'].items():
                            st.progress(val / 100, text=f"{comp}: {val:.0f}")
        else:
            st.info("Run backtests first to see monitoring data.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
