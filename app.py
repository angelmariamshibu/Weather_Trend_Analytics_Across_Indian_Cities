import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import calendar
import os
st.set_page_config(page_title="Indian Weather Trend Analytics", layout="wide")
st.title("🌦 Weather Trend Analytics Across Indian Cities")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

yearly_data = load_data("data/yearly_weather.csv")
monthly_data = load_data("data/monthly_weather.csv")
seasonal_data = load_data("data/seasonal_weather.csv")
all_cities = load_data("data/all_cities_weather.csv")  
all_cities['time'] = pd.to_datetime(all_cities['time'])
all_cities['year'] = all_cities['time'].dt.year
all_cities['month'] = all_cities['time'].dt.month
st.sidebar.header("Filters")
city = st.sidebar.selectbox("Select City", sorted(all_cities['city'].unique()))
metric = st.sidebar.selectbox("Select Parameter", [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "relative_humidity_2m_mean", "wind_speed_10m_max", "pressure_msl_mean","wind_direction_10m_dominant"
])
year_range = st.sidebar.slider(
    "Select Year Range",
    2015,
    2024,
    (2015, 2024)
)

# YEARLY TREND
st.subheader("Yearly Trend")
city_yearly = yearly_data[
    (yearly_data['city'] == city) & 
    (yearly_data['year'] >= year_range[0]) & 
    (yearly_data['year'] <= year_range[1])
]

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x='year', y=metric, data=city_yearly, marker='o', ax=ax)
ax.set_title(f"{metric} Trend for {city}")
ax.set_ylabel(metric)
ax.set_xticks(range(2015, 2025))         
ax.set_xticklabels(range(2015, 2025))
st.pyplot(fig)

# MONTHLY PATTERN
st.subheader("Monthly Pattern")
city_monthly = monthly_data[monthly_data['city'] == city]
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.lineplot(x='month', y=metric, data=city_monthly, marker='o', ax=ax2)
ax2.set_xticks(range(1,13))
ax2.set_xticklabels([calendar.month_abbr[i] for i in range(1,13)])
ax2.set_title(f"Monthly {metric} - {city}")
st.pyplot(fig2)

# SEASONAL ANALYSIS
st.subheader("Seasonal Analysis")
def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Summer'
    elif month in [6,7,8,9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

monthly_data['season'] = monthly_data['month'].astype(int).apply(get_season)
seasonal_avg = monthly_data.groupby(['city','season'])[metric].mean().reset_index()
city_seasonal = seasonal_avg[seasonal_avg['city'] == city]
season_order = ['Winter','Summer','Monsoon','Post-Monsoon']

fig3, ax3 = plt.subplots(figsize=(10,5))
sns.barplot(
    x='season', 
    y=metric, 
    data=city_seasonal.set_index('season').reindex(season_order).reset_index(),
    ax=ax3
)
ax3.set_title(f"Seasonal {metric} - {city}")
st.pyplot(fig3)

# MONTHLY HEATMAP
st.subheader("Monthly Heatmap Across Cities")
monthly_data['month_name'] = monthly_data['month'].apply(lambda x: calendar.month_abbr[int(x)])
month_order = [calendar.month_abbr[i] for i in range(1,13)]
heatmap_data = monthly_data.pivot_table(index='month_name', columns='city', values=metric, aggfunc='mean').reindex(month_order)

fig4, ax4 = plt.subplots(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".1f", ax=ax4)
ax4.set_title(f"Monthly {metric} Heatmap (2015-2024)")
st.pyplot(fig4)

# YEARLY HEATMAP
st.subheader("Yearly Heatmap Across Cities")
yearly_heatmap_data = yearly_data.pivot_table(index='year', columns='city', values=metric, aggfunc='mean')
fig5, ax5 = plt.subplots(figsize=(12,6))
sns.heatmap(yearly_heatmap_data, annot=True, cmap='YlOrRd', fmt=".1f", ax=ax5)
ax5.set_title(f"Yearly {metric} Heatmap (2015-2024)")
st.pyplot(fig5)

#  CITY-WISE MAX TEMPERATURE
st.subheader("City-wise Max Temperature Comparison")
yearly_city = all_cities.groupby(['city','year']).agg({
    'temperature_2m_max':'mean',
    'temperature_2m_min':'mean',
    'precipitation_sum':'sum',
    'relative_humidity_2m_mean':'mean',
    'wind_speed_10m_max':'mean',
    'pressure_msl_mean':'mean'
}).reset_index()

cities = yearly_city['city'].unique()
fig_comp, ax_comp = plt.subplots(figsize=(12,6))
for c in cities:
    city_data_plot = yearly_city[yearly_city['city']==c]
    ax_comp.plot(city_data_plot['year'], city_data_plot['temperature_2m_max'], marker='o', label=c)
ax_comp.set_title("City-wise Max Temperature Comparison (Yearly Avg)")
ax_comp.set_xlabel("Year")
ax_comp.set_ylabel("Max Temperature (°C)")
ax_comp.set_xticks(range(2015, 2025))      
ax_comp.set_xticklabels(range(2015, 2025))
ax_comp.legend()
ax_comp.grid(True)
st.pyplot(fig_comp)

# HIGHEST STATISTICS
city_max_temp = yearly_city.groupby('city')['temperature_2m_max'].mean()
highest_temp_city = city_max_temp.idxmax()
highest_temp_value = city_max_temp.max()

city_total_rain = yearly_city.groupby('city')['precipitation_sum'].sum()
highest_rain_city = city_total_rain.idxmax()
highest_rain_value = city_total_rain.max()

city_avg_humidity = yearly_city.groupby('city')['relative_humidity_2m_mean'].mean()
highest_humidity_city = city_avg_humidity.idxmax()
highest_humidity_value = city_avg_humidity.max()

st.write(f"Highest Average Max Temperature:** {highest_temp_city} ({highest_temp_value:.2f}°C)")
st.write(f"Highest Total Rainfall:** {highest_rain_city} ({highest_rain_value:.2f} mm)")
st.write(f"Highest Average Humidity:** {highest_humidity_city} ({highest_humidity_value:.2f}%)")

# EXTREME WEATHER ANALYSIS
st.subheader("Extreme Weather Analysis")
hottest_year = all_cities.groupby('year')['temperature_2m_max'].mean().idxmax()
wettest_year = all_cities.groupby('year')['precipitation_sum'].sum().idxmax()
st.write(f"Hottest Year (Avg Max Temp): {hottest_year}")
st.write(f"Wettest Year (Total Rainfall): {wettest_year}")

# TEMPERATURE VS RAINFALL
st.subheader(" Temperature vs Rainfall")
city_data = yearly_data[yearly_data['city'] == city]
fig6, ax6 = plt.subplots(figsize=(10,5))
ax6.plot(city_data['year'], city_data['temperature_2m_max'], marker='o', color='red', label='Max Temp')
ax6.set_xlabel('Year')
ax6.set_ylabel('Max Temp (°C)', color='red')
ax6.tick_params(axis='y', labelcolor='red')
ax7 = ax6.twinx()
ax7.plot(city_data['year'], city_data['precipitation_sum'], marker='s', color='blue', label='Rainfall')
ax7.set_ylabel('Rainfall (mm)', color='blue')
ax7.tick_params(axis='y', labelcolor='blue')
fig6.tight_layout()
st.pyplot(fig6)

# FORECASTING
st.subheader("Forecasting")
city_data = all_cities[all_cities['city'] == city].groupby('year').agg({
    'temperature_2m_max':'mean',
    'precipitation_sum':'sum'
}).reset_index()

# Temperature forecast
X_temp = city_data[['year']]
y_temp = city_data['temperature_2m_max']
model_temp = LinearRegression().fit(X_temp, y_temp)
city_data['pred_temp'] = model_temp.predict(X_temp)
next_year = pd.DataFrame({'year':[city_data['year'].max()+1]})
predicted_temp_next = model_temp.predict(next_year)[0]

# Rainfall forecast
X_rain = city_data[['year']]
y_rain = city_data['precipitation_sum']
model_rain = LinearRegression().fit(X_rain, y_rain)
city_data['pred_rain'] = model_rain.predict(X_rain)
predicted_rain_next = model_rain.predict(next_year)[0]

# Temperature forecast plot
fig7, ax7 = plt.subplots(figsize=(10,5))
ax7.plot(city_data['year'], city_data['temperature_2m_max'], marker='o', label='Actual Temp')
ax7.plot(city_data['year'], city_data['pred_temp'], linestyle='--', label='Predicted Temp')
ax7.scatter(next_year['year'], predicted_temp_next, color='red', label='Next Year Temp Prediction')
ax7.set_title(f"{city} Max Temperature Forecast")
ax7.set_xlabel('Year')
ax7.set_ylabel('Temperature (°C)')
ax7.legend()
st.pyplot(fig7)

# Rainfall forecast plot
fig8, ax8 = plt.subplots(figsize=(10,5))
ax8.plot(city_data['year'], city_data['precipitation_sum'], marker='o', label='Actual Rainfall', color='blue')
ax8.plot(city_data['year'], city_data['pred_rain'], linestyle='--', label='Predicted Rainfall', color='orange')
ax8.scatter(next_year['year'], predicted_rain_next, color='red', label='Next Year Rain Prediction')
ax8.set_title(f"{city} Total Rainfall Forecast")
ax8.set_xlabel('Year')
ax8.set_ylabel('Rainfall (mm)')
ax8.legend()
st.pyplot(fig8)

st.write(f"Predicted Max Temperature for {city} in {next_year['year'][0]}: {predicted_temp_next:.2f}°C")
st.write(f"Predicted Total Rainfall for {city} in {next_year['year'][0]}: {predicted_rain_next:.2f} mm")

