import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import io

# Set page configuration
st.set_page_config(
    page_title="Kenya GDP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

print(sys.executable)

# Function to calculate real GDP based on growth rates
def calculate_real_gdp(data):
    # Create a copy to avoid modifying the original
    df = data.copy()

    # Convert columns to numeric
    df["Nominal GDP (Ksh Million)"] = pd.to_numeric(
        df["Nominal GDP (Ksh Million)"], errors="coerce"
    )
    df["Annual GDP Growth (%)"] = pd.to_numeric(
        df["Annual GDP Growth (%)"], errors="coerce"
    )
    df["Real GDP (Ksh Million)"] = pd.to_numeric(
        df["Real GDP (Ksh Million)"], errors="coerce"
    )
    df["Year"] = pd.to_numeric(df["Year"])

    # Create a new column for calculated real GDP
    df["Calculated Real GDP"] = np.nan

    # Set the base year (2000) real GDP
    base_year = 2000
    if base_year in df["Year"].values:
        df.loc[df["Year"] == base_year, "Calculated Real GDP"] = df.loc[
            df["Year"] == base_year, "Real GDP (Ksh Million)"
        ].values[0]
    else:
        raise ValueError("Base year 2000 is not in the DataFrame")

    # Determine the minimum year in the DataFrame
    min_year = df["Year"].min()

    # Calculate backward only for years present in the DataFrame
    for year in range(base_year - 1, min_year - 1, -1):
        if year in df["Year"].values and (year + 1) in df["Year"].values:
            current_year_index = df[df["Year"] == year].index[0]
            next_year_index = df[df["Year"] == year + 1].index[0]
            growth_rate = df.loc[current_year_index, "Annual GDP Growth (%)"] / 100
            next_year_real_gdp = df.loc[next_year_index, "Calculated Real GDP"]
            current_year_real_gdp = next_year_real_gdp / (1 + growth_rate)
            df.loc[current_year_index, "Calculated Real GDP"] = current_year_real_gdp

    # For years after the base year, use the actual real GDP values
    df.loc[df["Year"] > base_year, "Calculated Real GDP"] = df.loc[
        df["Year"] > base_year, "Real GDP (Ksh Million)"
    ]

    # Calculate the GDP deflator
    df["GDP Deflator"] = (
        df["Nominal GDP (Ksh Million)"] / df["Calculated Real GDP"]
    ) * 100

    # Fill in the Real GDP column with calculated values where it's NaN
    df.loc[df["Real GDP (Ksh Million)"].isna(), "Real GDP (Ksh Million)"] = df.loc[
        df["Real GDP (Ksh Million)"].isna(), "Calculated Real GDP"
    ]

    return df


# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown(
    "<h1 class='main-header'>Kenya GDP Analysis (1960-2023)</h1>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg", width=200
)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select a page:",
    [
        "Overview",
        "Historical Analysis",
        "Growth Trends",
        "GDP Components",
        "Data Explorer",
    ],
)


# Function to load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(
            r"C:\Users\Ricky\Desktop\For Fun Projects\kenya-gdp-analysis\data\Kenya_GDP_Complete.csv"
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
data = load_data()

if data is not None:
    # Process data
    processed_data = calculate_real_gdp(data)

    # Overview page
    if page == "Overview":
        st.markdown(
            "<h2 class='sub-header'>Kenya's Economic Journey</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create Plotly figure for GDP overview
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=processed_data["Year"],
                    y=processed_data["Nominal GDP (Ksh Million)"],
                    name="Nominal GDP",
                    line=dict(color="#1E88E5", width=2),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=processed_data["Year"],
                    y=processed_data["Real GDP (Ksh Million)"],
                    name="Real GDP",
                    line=dict(color="#FFC107", width=2),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Bar(
                    x=processed_data["Year"],
                    y=processed_data["Annual GDP Growth (%)"],
                    name="Annual Growth (%)",
                    marker_color="#4CAF50",
                    opacity=0.7,
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title_text="Kenya GDP and Growth Rate (1960-2023)",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=500,
            )
            fig.update_xaxes(title_text="Year", tickangle=45)
            fig.update_yaxes(title_text="GDP (Ksh Million)", secondary_y=False)
            fig.update_yaxes(title_text="Growth Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("### Key Indicators")
            latest_year = processed_data["Year"].max()
            latest_data = processed_data[processed_data["Year"] == latest_year].iloc[0]
            st.metric(
                "Latest Nominal GDP (2023)",
                f"{latest_data['Nominal GDP (Ksh Million)']:,.0f} Ksh Million",
            )
            st.metric(
                "Latest Real GDP (2023)",
                f"{latest_data['Real GDP (Ksh Million)']:,.0f} Ksh Million",
            )
            st.metric(
                "Latest Growth Rate", f"{latest_data['Annual GDP Growth (%)']:.1f}%"
            )
            # Calculate CAGR for different periods
            current_gdp = latest_data["Real GDP (Ksh Million)"]
            # 10-year CAGR
            ten_years_ago = processed_data[
                processed_data["Year"] == latest_year - 10
            ].iloc[0]["Real GDP (Ksh Million)"]
            cagr_10 = (current_gdp / ten_years_ago) ** (1 / 10) - 1
            # 20-year CAGR
            twenty_years_ago = processed_data[
                processed_data["Year"] == latest_year - 20
            ].iloc[0]["Real GDP (Ksh Million)"]
            cagr_20 = (current_gdp / twenty_years_ago) ** (1 / 20) - 1
            st.metric("10-Year CAGR", f"{cagr_10*100:.1f}%")
            st.metric("20-Year CAGR", f"{cagr_20*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("### About This Dashboard")
        st.info(
            """
            This interactive dashboard presents Kenya's economic data from 1960 to 2023, focusing on GDP trends and growth patterns.
            The real GDP values for 1960-1999 have been calculated using historical growth rates and the 2000 base year GDP figure.
            This methodology ensures consistency with official economic growth statistics while providing a complete historical picture.
            Use the sidebar to navigate to different analysis pages.
            """
        )

    # Historical Analysis page
    elif page == "Historical Analysis":
        st.markdown(
            "<h2 class='sub-header'>Historical GDP Analysis</h2>",
            unsafe_allow_html=True,
        )
        # Create periods for analysis
        processed_data["Decade"] = (processed_data["Year"] // 10) * 10
        processed_data["Period"] = pd.cut(
            processed_data["Year"],
            bins=[1959, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
            labels=["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"],
        )
        col1, col2 = st.columns(2)
        with col1:
            # GDP Deflator chart
            fig_deflator = px.line(
                processed_data,
                x="Year",
                y="GDP Deflator",
                title="GDP Deflator Over Time (Price Level Changes)",
                labels={"GDP Deflator": "GDP Deflator (2000=100)", "Year": ""},
                color_discrete_sequence=["#E91E63"],
            )
            fig_deflator.update_layout(height=400)
            st.plotly_chart(fig_deflator, use_container_width=True)
        with col2:
            # Average growth by decade
            growth_by_decade = (
                processed_data.groupby("Period")["Annual GDP Growth (%)"]
                .mean()
                .reset_index()
            )
            fig_decade_growth = px.bar(
                growth_by_decade,
                x="Period",
                y="Annual GDP Growth (%)",
                title="Average GDP Growth by Decade",
                color="Annual GDP Growth (%)",
                color_continuous_scale="RdYlGn",
                text_auto=".1f",
            )
            fig_decade_growth.update_layout(height=400)
            st.plotly_chart(fig_decade_growth, use_container_width=True)
        # GDP comparison with base year normalization
        st.markdown("### GDP Comparison with Normalized Base Year")
        base_year_options = sorted(processed_data["Year"].unique())
        selected_base_year = st.selectbox(
            "Select base year for normalization:",
            base_year_options,
            index=base_year_options.index(2000) if 2000 in base_year_options else 0,
        )
        # Normalize GDP values based on selected year
        base_nominal = processed_data[processed_data["Year"] == selected_base_year][
            "Nominal GDP (Ksh Million)"
        ].values[0]
        base_real = processed_data[processed_data["Year"] == selected_base_year][
            "Real GDP (Ksh Million)"
        ].values[0]
        processed_data["Normalized Nominal GDP"] = (
            processed_data["Nominal GDP (Ksh Million)"] / base_nominal * 100
        )
        processed_data["Normalized Real GDP"] = (
            processed_data["Real GDP (Ksh Million)"] / base_real * 100
        )
        fig_normalized = px.line(
            processed_data,
            x="Year",
            y=["Normalized Nominal GDP", "Normalized Real GDP"],
            title=f"Normalized GDP (Base Year {selected_base_year} = 100)",
            labels={"value": "Index Value", "Year": "", "variable": "Measure"},
            color_discrete_map={
                "Normalized Nominal GDP": "#1E88E5",
                "Normalized Real GDP": "#FFC107",
            },
        )
        fig_normalized.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_normalized, use_container_width=True)

    # Growth Trends page
    elif page == "Growth Trends":
        st.markdown(
            "<h2 class='sub-header'>Growth Trends Analysis</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            # Growth rate distribution
            fig_growth_hist = px.histogram(
                processed_data,
                x="Annual GDP Growth (%)",
                nbins=20,
                title="Distribution of Annual GDP Growth Rates (1960-2023)",
                color_discrete_sequence=["#4CAF50"],
                marginal="box",
            )
            fig_growth_hist.update_layout(height=400)
            st.plotly_chart(fig_growth_hist, use_container_width=True)
        with col2:
            # Growth rate over time with recession highlighting
            fig_growth = px.line(
                processed_data,
                x="Year",
                y="Annual GDP Growth (%)",
                title="Annual GDP Growth Rate (1960-2023)",
                labels={"Annual GDP Growth (%)": "Growth Rate (%)", "Year": ""},
                color_discrete_sequence=["#4CAF50"],
            )
            # Add a reference line at 0% growth
            fig_growth.add_hline(
                y=0, line_dash="dash", line_color="red", annotation_text="Zero Growth"
            )
            # Highlight recession periods (negative growth)
            for i, row in processed_data[
                processed_data["Annual GDP Growth (%)"] < 0
            ].iterrows():
                fig_growth.add_vrect(
                    x0=row["Year"] - 0.5,
                    x1=row["Year"] + 0.5,
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            fig_growth.update_layout(height=400)
            st.plotly_chart(fig_growth, use_container_width=True)
        # Growth trends by time period
        st.markdown("### Growth Trends by Period")
        # Create custom periods for analysis
        custom_periods = [
            (1960, 1970, "Early Independence"),
            (1970, 1980, "Oil Crisis & Recovery"),
            (1980, 1990, "Structural Adjustment"),
            (1990, 2000, "Liberalization"),
            (2000, 2010, "New Constitution Era"),
            (2010, 2020, "Vision 2030 Period"),
            (2020, 2023, "COVID & Recovery"),
        ]
        period_data = []
        for start, end, name in custom_periods:
            period_df = processed_data[
                (processed_data["Year"] >= start) & (processed_data["Year"] <= end)
            ]
            avg_growth = period_df["Annual GDP Growth (%)"].mean()
            min_growth = period_df["Annual GDP Growth (%)"].min()
            max_growth = period_df["Annual GDP Growth (%)"].max()
            volatility = period_df["Annual GDP Growth (%)"].std()
            period_data.append(
                {
                    "Period": name,
                    "Years": f"{start}-{end}",
                    "Average Growth": avg_growth,
                    "Min Growth": min_growth,
                    "Max Growth": max_growth,
                    "Volatility": volatility,
                }
            )
        period_df = pd.DataFrame(period_data)
        # Create a radar chart comparing periods
        fig_radar = go.Figure()
        for i, row in period_df.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[
                        row["Average Growth"],
                        max(0, row["Min Growth"]),
                        row["Max Growth"],
                        row["Volatility"],
                    ],
                    theta=["Average Growth", "Min Growth", "Max Growth", "Volatility"],
                    fill="toself",
                    name=f"{row['Period']} ({row['Years']})",
                )
            )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[
                        0,
                        max(
                            period_df["Max Growth"].max(),
                            period_df["Volatility"].max(),
                        )
                        * 1.1,
                    ],
                ),
            ),
            showlegend=True,
            title="Economic Performance by Period",
            height=600,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        # Show period data in a table
        st.markdown("### Economic Performance Metrics by Period")
        st.dataframe(
            period_df.style.format(
                {
                    "Average Growth": "{:.2f}%",
                    "Min Growth": "{:.2f}%",
                    "Max Growth": "{:.2f}%",
                    "Volatility": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    # GDP Components page (placeholder)
    elif page == "GDP Components":
        st.markdown(
            "<h2 class='sub-header'>GDP Components Analysis</h2>",
            unsafe_allow_html=True,
        )
        st.info(
            """
            This section would typically contain a breakdown of GDP by sector and components.
            Components that could be analyzed include:
            - Agriculture, Industry, and Services contribution
            - Government vs. Private consumption
            - Investments and Capital Formation
            - Exports and Imports
            **Note:** The current dataset does not include component-level data. To enable this analysis, additional sectoral data would need to be imported.
            """
        )
        # Placeholder visualization
        fig = go.Figure()
        sectors = ["Agriculture", "Manufacturing", "Services", "Other"]
        values_2023 = [33, 16, 43, 8]
        values_2000 = [40, 15, 38, 7]
        values_1980 = [45, 12, 35, 8]
        fig.add_trace(
            go.Bar(name="1980", x=sectors, y=values_1980, marker_color="#4CAF50")
        )
        fig.add_trace(
            go.Bar(name="2000", x=sectors, y=values_2000, marker_color="#FFC107")
        )
        fig.add_trace(
            go.Bar(name="2023", x=sectors, y=values_2023, marker_color="#1E88E5")
        )
        fig.update_layout(
            title="GDP Composition by Sector (Sample Data)",
            xaxis_title="Sector",
            yaxis_title="Percentage of GDP",
            barmode="group",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.warning(
            """
            The above chart uses sample data for illustrative purposes only.
            For accurate sectoral analysis, additional data sources would be required.
            """
        )

    # Data Explorer page
    elif page == "Data Explorer":
        st.markdown("<h2 class='sub-header'>Data Explorer</h2>", unsafe_allow_html=True)
        st.info(
            "This page allows you to explore the complete dataset and create custom visualizations."
        )
        # Show the data table
        st.markdown("### Complete Dataset")
        st.dataframe(
            processed_data.style.highlight_max(
                axis=0, subset=["Annual GDP Growth (%)"]
            ),
            use_container_width=True,
        )
        # Custom visualization options
        st.markdown("### Create Custom Visualization")
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"],
            )
            date_range = st.slider(
                "Select year range:",
                min_value=int(processed_data["Year"].min()),
                max_value=int(processed_data["Year"].max()),
                value=(
                    int(processed_data["Year"].min()),
                    int(processed_data["Year"].max()),
                ),
            )
        with col2:
            y_options = [
                "Nominal GDP (Ksh Million)",
                "Real GDP (Ksh Million)",
                "Annual GDP Growth (%)",
                "GDP Deflator",
            ]
            y_axis = st.multiselect(
                "Select data to visualize:",
                y_options,
                default=["Nominal GDP (Ksh Million)", "Real GDP (Ksh Million)"],
            )
            log_scale = st.checkbox("Use logarithmic scale for Y-axis")
        # Filter data based on selection
        filtered_data = processed_data[
            (processed_data["Year"] >= date_range[0])
            & (processed_data["Year"] <= date_range[1])
        ]
        # Create the selected chart
        if y_axis:
            if chart_type == "Line Chart":
                fig = px.line(
                    filtered_data,
                    x="Year",
                    y=y_axis,
                    title=f"Kenya GDP Metrics ({date_range[0]}-{date_range[1]})",
                    log_y=log_scale,
                )
            elif chart_type == "Bar Chart":
                fig = px.bar(
                    filtered_data,
                    x="Year",
                    y=y_axis,
                    title=f"Kenya GDP Metrics ({date_range[0]}-{date_range[1]})",
                    log_y=log_scale,
                    barmode="group",
                )
            elif chart_type == "Scatter Plot":
                fig = px.scatter(
                    filtered_data,
                    x="Year",
                    y=y_axis,
                    title=f"Kenya GDP Metrics ({date_range[0]}-{date_range[1]})",
                    log_y=log_scale,
                    size="Nominal GDP (Ksh Million)",
                    size_max=15,
                )
            elif chart_type == "Area Chart":
                fig = px.area(
                    filtered_data,
                    x="Year",
                    y=y_axis,
                    title=f"Kenya GDP Metrics ({date_range[0]}-{date_range[1]})",
                    log_y=log_scale,
                )
            fig.update_layout(height=600, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one metric to visualize.")
        # Download options
        st.markdown("### Download Processed Data")
        download_format = st.radio("Select download format:", ["CSV", "Excel", "JSON"])
        if download_format == "CSV":
            csv = processed_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="kenya_gdp_data.csv",
                mime="text/csv",
            )
        elif download_format == "Excel":
            buffer = io.BytesIO()
            processed_data.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name="kenya_gdp_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        elif download_format == "JSON":
            json_data = processed_data.to_json(orient="records")
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="kenya_gdp_data.json",
                mime="application/json",
            )
else:
    st.error("Failed to load data. Please check if the data file is available.")

