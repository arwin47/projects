# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.seasonal import STL

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,  external_stylesheets=external_stylesheets)



# Data

# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv(r"series_data_moving_added.csv")

df["SUV_prospects_30"]=df["SUV_prospects"].rolling(30).mean()
df["Sedan_prospects_30"]=df["Sedan_prospects"].rolling(30).mean()
df["Sedan_Sales_30"]=df["Sedan_Sales"].rolling(30).mean()
df["Sedan_Revenue_30"]=df["Sedan_Revenue"].rolling(30).mean()
df["Sedan_Revenue_per_sale_30"]=df["Sedan_Revenue_per_sale"].rolling(30).mean()
df["SUV_Sales_30"]=df["SUV_Sales"].rolling(30).mean()
df["SUV_Revenue_30"]=df["SUV_Revenue"].rolling(30).mean()


df["SUV_Revenue_per_sale_30"]=df["SUV_Revenue_per_sale"].rolling(30).mean()
df["Sales_30"]=df["Sales"].rolling(30).mean()
df["Revenue_30"]=df["Revenue"].rolling(30).mean()
df["Revenue_per_sale_30"]=df["Revenue_per_sale"].rolling(30).mean()

df["Model_1_sales_30"]=df["Model_1_sales"].rolling(30).mean()
df["Model_1_Revenue_30"]=df["Model_1_Revenue"].rolling(30).mean()
df["Model_1_Revenue_per_sale_30"]=df["Model_1_Revenue_per_sale"].rolling(30).mean()
df["Model_2_sales_30"]=df["Model_2_sales"].rolling(30).mean()
df["Model_2_Revenue_30"]=df["Model_2_Revenue"].rolling(30).mean()
df["Model_2_Revenue_per_sale_30"]=df["Model_2_Revenue_per_sale"].rolling(30).mean()
df["Model_3_sales_30"]=df["Model_3_sales"].rolling(30).mean()
df["Model_3_Revenue_30"]=df["Model_3_Revenue"].rolling(30).mean()
df["Model_3_Revenue_per_sale_30"]=df["Model_3_Revenue_per_sale"].rolling(30).mean()


df["SUV_prospects_7"]=df["SUV_prospects"].rolling(7).mean()
df["Sedan_prospects_7"]=df["Sedan_prospects"].rolling(7).mean()
df["Sedan_Sales_7"]=df["Sedan_Sales"].rolling(7).mean()
df["Sedan_Revenue_7"]=df["Sedan_Revenue"].rolling(7).mean()
df["Sedan_Revenue_per_sale_7"]=df["Sedan_Revenue_per_sale"].rolling(7).mean()
df["SUV_Sales_7"]=df["SUV_Sales"].rolling(7).mean()
df["SUV_Revenue_7"]=df["SUV_Revenue"].rolling(7).mean()


df["SUV_Revenue_per_sale_7"]=df["SUV_Revenue_per_sale"].rolling(7).mean()
df["Sales_7"]=df["Sales"].rolling(7).mean()
df["Revenue_7"]=df["Revenue"].rolling(7).mean()
df["Revenue_per_sale_7"]=df["Revenue_per_sale"].rolling(7).mean()

df["Model_1_sales_7"]=df["Model_1_sales"].rolling(7).mean()
df["Model_1_Revenue_7"]=df["Model_1_Revenue"].rolling(7).mean()
df["Model_1_Revenue_per_sale_7"]=df["Model_1_Revenue_per_sale"].rolling(7).mean()
df["Model_2_sales_7"]=df["Model_2_sales"].rolling(7).mean()
df["Model_2_Revenue_7"]=df["Model_2_Revenue"].rolling(7).mean()
df["Model_2_Revenue_per_sale_7"]=df["Model_2_Revenue_per_sale"].rolling(7).mean()
df["Model_3_sales_7"]=df["Model_3_sales"].rolling(7).mean()
df["Model_3_Revenue_7"]=df["Model_3_Revenue"].rolling(7).mean()
df["Model_3_Revenue_per_sale_7"]=df["Model_3_Revenue_per_sale"].rolling(7).mean()

df["Overall_Sedan_visitors_7"]=df["Overall_Sedan_visitors"].rolling(7).mean()
df["Overall_SUV_visitors_7"]=df["Overall_SUV_visitors"].rolling(7).mean()
df["Overall_Sedan_visitors_30"]=df["Overall_Sedan_visitors"].rolling(30).mean()
df["Overall_SUV_visitors_30"]=df["Overall_SUV_visitors"].rolling(30).mean()




df = df.fillna(method='ffill')
index = pd.date_range(df['date'][0], periods=len(df['date']), freq='D')
df.index = index

# Figure Layouts

figure_height = 9500
Vertical_gap = 0.05

# Figure Sedan

fig_Sedan = make_subplots(rows=5, cols=1, subplot_titles=("Visitors","Prospects", "Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)

fig_Sedan.add_trace(go.Scatter(x=df['date'], y=df['Overall_Sedan_visitors'],mode='lines',name = "Visitors"), 
              row=1, col=1)

fig_Sedan.add_trace(go.Scatter(x=df['date'], y=df['Sedan_prospects'],mode='lines',line=dict(color="#FF8040"),name = "Prospects"), 
              row=2, col=1)

fig_Sedan.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Sales'],mode='lines',name = "Sales"), 
              row=3, col=1)

fig_Sedan.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue'],mode='lines',name = "Revenue"), 
              row=4, col=1)

fig_Sedan.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"), 
              row=5, col=1)              
              
#fig.update_xaxes(rangeslider_visible=True)
fig_Sedan.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])
fig_Sedan.update_layout(showlegend=False)
fig_Sedan.update_yaxes(rangemode="tozero") 
#fig_SUV.update_layout(height=figure_height, 
                  #title_text="Stacked Subplots with Shared X-Axes"
#                  )             
  
  
# Figure Sedan Model Model_1

fig_Sedan_model_Model_1 = make_subplots(rows=3, cols=1, subplot_titles=("Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)



fig_Sedan_model_Model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_sales'],mode='lines',name = "Sales"), 
              row=1, col=1)

fig_Sedan_model_Model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue'],mode='lines',line=dict(color="#FF8040"),name = "Revenue"), 
              row=2, col=1)

fig_Sedan_model_Model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"), 
              row=3, col=1)              
              
#fig.update_xaxes(rangeslider_visible=True)
fig_Sedan_model_Model_1.update_layout(showlegend=False)
fig_Sedan_model_Model_1.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])
fig_Sedan_model_Model_1.update_yaxes(rangemode="tozero") 
#fig_SUV.update_layout(height=figure_height, 
                  #title_text="Stacked Subplots with Shared X-Axes"
#                  ) 

# Figure Sedan Model Model_2

fig_Sedan_model_Model_2 = make_subplots(rows=3, cols=1, subplot_titles=("Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)



fig_Sedan_model_Model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_sales'],mode='lines',name = "Sales"), 
              row=1, col=1)

fig_Sedan_model_Model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue'],mode='lines',line=dict(color="#FF8040"),name = "Revenue"), 
              row=2, col=1)

fig_Sedan_model_Model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"), 
              row=3, col=1)              
              
#fig.update_xaxes(rangeslider_visible=True)
fig_Sedan_model_Model_2.update_layout(showlegend=False)
fig_Sedan_model_Model_2.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])
fig_Sedan_model_Model_2.update_yaxes(rangemode="tozero") 
#fig_SUV.update_layout(height=figure_height, 
                  #title_text="Stacked Subplots with Shared X-Axes"
#                  ) 
 
# Figure Sedan Model Model_3

fig_Sedan_model_Model_3 = make_subplots(rows=3, cols=1, subplot_titles=("Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)



fig_Sedan_model_Model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_sales'],mode='lines',name = "Sales"), 
              row=1, col=1)

fig_Sedan_model_Model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue'],mode='lines',line=dict(color="#FF8040"),name = "Revenue"), 
              row=2, col=1)

fig_Sedan_model_Model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"), 
              row=3, col=1)              
              
#fig.update_xaxes(rangeslider_visible=True)
fig_Sedan_model_Model_3.update_layout(showlegend=False)
fig_Sedan_model_Model_3.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])
fig_Sedan_model_Model_3.update_yaxes(rangemode="tozero") 
#fig_SUV.update_layout(height=figure_height, 
                  #title_text="Stacked Subplots with Shared X-Axes"
#                  ) 

 
# Figure SUV

fig_SUV = make_subplots(rows=5, cols=1, subplot_titles=("Visitors","Prospects", "Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)

fig_SUV.add_trace(go.Scatter(x=df['date'], y=df['Overall_SUV_visitors'],mode='lines',name = "Visitors"), 
              row=1, col=1)

fig_SUV.add_trace(go.Scatter(x=df['date'], y=df['SUV_prospects'],mode='lines',line=dict(color="#FF8040"),name = "Prospects"), 
              row=2, col=1)

fig_SUV.add_trace(go.Scatter(x=df['date'], y=df['SUV_Sales'],mode='lines',name = "Sales"), 
              row=3, col=1)

fig_SUV.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue'],mode='lines',name = "Revenue"), 
              row=4, col=1)

fig_SUV.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"), 
              row=5, col=1)              
fig_SUV.update_yaxes(rangemode="tozero")  
fig_SUV.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])           
#fig.update_xaxes(rangeslider_visible=True)
fig_SUV.update_layout(showlegend=False)
#fig_SUV.update_layout(height=figure_height, 
                  #title_text="Stacked Subplots with Shared X-Axes"
#                  )             



# Anomaly Detection Algorithm


# Sedan Visitors

#Fit model
stl_Sedan_visitors = STL(df['Overall_Sedan_visitors'])
result_Sedan_visitors = stl_Sedan_visitors.fit()
# Extract the decomposition
seasonal_Sedan_visitors, trend_Sedan_visitors, resid_Sedan_visitors = result_Sedan_visitors.seasonal, result_Sedan_visitors.trend, result_Sedan_visitors.resid
estimated_Sedan_visitors = trend_Sedan_visitors + seasonal_Sedan_visitors
#Setting and finding Anomaly
resid_mu_Sedan_visitors = resid_Sedan_visitors.mean()
resid_dev_Sedan_visitors = resid_Sedan_visitors.std()
lower_Sedan_visitors = resid_mu_Sedan_visitors - 1.5*resid_dev_Sedan_visitors   # lower bound of anomaly
upper_Sedan_visitors = resid_mu_Sedan_visitors + 1.5*resid_dev_Sedan_visitors   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_visitors = pd.DataFrame(df[['Overall_Sedan_visitors','date']][(resid_Sedan_visitors < lower_Sedan_visitors) | (resid_Sedan_visitors > upper_Sedan_visitors)])

# Sedan Prospects

#Fit model
stl_Sedan_leads = STL(df['Sedan_prospects'])
result_Sedan_leads = stl_Sedan_leads.fit()
# Extract the decomposition
seasonal_Sedan_leads, trend_Sedan_leads, resid_Sedan_leads = result_Sedan_leads.seasonal, result_Sedan_leads.trend, result_Sedan_leads.resid
estimated_Sedan_leads = trend_Sedan_leads + seasonal_Sedan_leads
#Setting and finding Anomaly
resid_mu_Sedan_leads = resid_Sedan_leads.mean()
resid_dev_Sedan_leads = resid_Sedan_leads.std()
lower_Sedan_leads = resid_mu_Sedan_leads - 1.5*resid_dev_Sedan_leads   # lower bound of anomaly
upper_Sedan_leads = resid_mu_Sedan_leads + 1.5*resid_dev_Sedan_leads   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_leads = pd.DataFrame(df[['Sedan_prospects','date']][(resid_Sedan_leads < lower_Sedan_leads) | (resid_Sedan_leads > upper_Sedan_leads)])


# Sedan SALES

#Fit model
stl_Sedan_sales = STL(df['Sedan_Sales'])
result_Sedan_sales = stl_Sedan_sales.fit()
# Extract the decomposition
seasonal_Sedan_sales, trend_Sedan_sales, resid_Sedan_sales = result_Sedan_sales.seasonal, result_Sedan_sales.trend, result_Sedan_sales.resid
estimated_Sedan_sales = trend_Sedan_sales + seasonal_Sedan_sales
#Setting and finding Anomaly
resid_mu_Sedan_sales = resid_Sedan_sales.mean()
resid_dev_Sedan_sales = resid_Sedan_sales.std()
lower_Sedan_sales = resid_mu_Sedan_sales - 1.5*resid_dev_Sedan_sales   # lower bound of anomaly
upper_Sedan_sales = resid_mu_Sedan_sales + 1.5*resid_dev_Sedan_sales   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_sales = pd.DataFrame(df[['Sedan_Sales','date']][(resid_Sedan_sales < lower_Sedan_sales) | (resid_Sedan_sales > upper_Sedan_sales)])


# Sedan Revenue

#Fit model
stl_Sedan_Revenue = STL(df['Sedan_Revenue'])
result_Sedan_Revenue = stl_Sedan_Revenue.fit()
# Extract the decomposition
seasonal_Sedan_Revenue, trend_Sedan_Revenue, resid_Sedan_Revenue = result_Sedan_Revenue.seasonal, result_Sedan_Revenue.trend, result_Sedan_Revenue.resid
estimated_Sedan_Revenue = trend_Sedan_Revenue + seasonal_Sedan_Revenue
#Setting and finding Anomaly
resid_mu_Sedan_Revenue = resid_Sedan_Revenue.mean()
resid_dev_Sedan_Revenue = resid_Sedan_Revenue.std()
lower_Sedan_Revenue = resid_mu_Sedan_Revenue - 1.5*resid_dev_Sedan_Revenue   # lower bound of anomaly
upper_Sedan_Revenue = resid_mu_Sedan_Revenue + 1.5*resid_dev_Sedan_Revenue   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Revenue = pd.DataFrame(df[['Sedan_Revenue','date']][(resid_Sedan_Revenue < lower_Sedan_Revenue) | (resid_Sedan_Revenue > upper_Sedan_Revenue)])


# Sedan Revenue_per_sale

#Fit model
stl_Sedan_Revenue_per_sale = STL(df['Sedan_Revenue_per_sale'])
result_Sedan_Revenue_per_sale = stl_Sedan_Revenue_per_sale.fit()
# Extract the decomposition
seasonal_Sedan_Revenue_per_sale, trend_Sedan_Revenue_per_sale, resid_Sedan_Revenue_per_sale = result_Sedan_Revenue_per_sale.seasonal, result_Sedan_Revenue_per_sale.trend, result_Sedan_Revenue_per_sale.resid
estimated_Sedan_Revenue_per_sale = trend_Sedan_Revenue_per_sale + seasonal_Sedan_Revenue_per_sale
#Setting and finding Anomaly
resid_mu_Sedan_Revenue_per_sale = resid_Sedan_Revenue_per_sale.mean()
resid_dev_Sedan_Revenue_per_sale = resid_Sedan_Revenue_per_sale.std()
lower_Sedan_Revenue_per_sale = resid_mu_Sedan_Revenue_per_sale - 1.5*resid_dev_Sedan_Revenue_per_sale   # lower bound of anomaly
upper_Sedan_Revenue_per_sale = resid_mu_Sedan_Revenue_per_sale + 1.5*resid_dev_Sedan_Revenue_per_sale   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Revenue_per_sale = pd.DataFrame(df[['Sedan_Revenue_per_sale','date']][(resid_Sedan_Revenue_per_sale < lower_Sedan_Revenue_per_sale) | (resid_Sedan_Revenue_per_sale > upper_Sedan_Revenue_per_sale)])


# SUV Visitors

#Fit model
stl_SUV_visitors = STL(df['Overall_SUV_visitors'])
result_SUV_visitors = stl_SUV_visitors.fit()
# Extract the decomposition
seasonal_SUV_visitors, trend_SUV_visitors, resid_SUV_visitors = result_SUV_visitors.seasonal, result_SUV_visitors.trend, result_SUV_visitors.resid
estimated_SUV_visitors = trend_SUV_visitors + seasonal_SUV_visitors
#Setting and finding Anomaly
resid_mu_SUV_visitors = resid_SUV_visitors.mean()
resid_dev_SUV_visitors = resid_SUV_visitors.std()
lower_SUV_visitors = resid_mu_SUV_visitors - 1.5*resid_dev_SUV_visitors   # lower bound of anomaly
upper_SUV_visitors = resid_mu_SUV_visitors + 1.5*resid_dev_SUV_visitors   # upper bound of anomaly
# Storing the anomalies
anomalies_SUV_visitors = pd.DataFrame(df[['Overall_SUV_visitors','date']][(resid_SUV_visitors < lower_SUV_visitors) | (resid_SUV_visitors > upper_SUV_visitors)])

# SUV Prospects

#Fit model
stl_SUV_leads = STL(df['SUV_prospects'])
result_SUV_leads = stl_SUV_leads.fit()
# Extract the decomposition
seasonal_SUV_leads, trend_SUV_leads, resid_SUV_leads = result_SUV_leads.seasonal, result_SUV_leads.trend, result_SUV_leads.resid
estimated_SUV_leads = trend_SUV_leads + seasonal_SUV_leads
#Setting and finding Anomaly
resid_mu_SUV_leads = resid_SUV_leads.mean()
resid_dev_SUV_leads = resid_SUV_leads.std()
lower_SUV_leads = resid_mu_SUV_leads - 1.5*resid_dev_SUV_leads   # lower bound of anomaly
upper_SUV_leads = resid_mu_SUV_leads + 1.5*resid_dev_SUV_leads   # upper bound of anomaly
# Storing the anomalies
anomalies_SUV_leads = pd.DataFrame(df[['SUV_prospects','date']][(resid_SUV_leads < lower_SUV_leads) | (resid_SUV_leads > upper_SUV_leads)])


# SUV SALES

#Fit model
stl_SUV_sales = STL(df['SUV_Sales'])
result_SUV_sales = stl_SUV_sales.fit()
# Extract the decomposition
seasonal_SUV_sales, trend_SUV_sales, resid_SUV_sales = result_SUV_sales.seasonal, result_SUV_sales.trend, result_SUV_sales.resid
estimated_SUV_sales = trend_SUV_sales + seasonal_SUV_sales
#Setting and finding Anomaly
resid_mu_SUV_sales = resid_SUV_sales.mean()
resid_dev_SUV_sales = resid_SUV_sales.std()
lower_SUV_sales = resid_mu_SUV_sales - 1.5*resid_dev_SUV_sales   # lower bound of anomaly
upper_SUV_sales = resid_mu_SUV_sales + 1.5*resid_dev_SUV_sales   # upper bound of anomaly
# Storing the anomalies
anomalies_SUV_sales = pd.DataFrame(df[['SUV_Sales','date']][(resid_SUV_sales < lower_SUV_sales) | (resid_SUV_sales > upper_SUV_sales)])


# SUV Revenue

#Fit model
stl_SUV_Revenue = STL(df['SUV_Revenue'])
result_SUV_Revenue = stl_SUV_Revenue.fit()
# Extract the decomposition
seasonal_SUV_Revenue, trend_SUV_Revenue, resid_SUV_Revenue = result_SUV_Revenue.seasonal, result_SUV_Revenue.trend, result_SUV_Revenue.resid
estimated_SUV_Revenue = trend_SUV_Revenue + seasonal_SUV_Revenue
#Setting and finding Anomaly
resid_mu_SUV_Revenue = resid_SUV_Revenue.mean()
resid_dev_SUV_Revenue = resid_SUV_Revenue.std()
lower_SUV_Revenue = resid_mu_SUV_Revenue - 1.5*resid_dev_SUV_Revenue   # lower bound of anomaly
upper_SUV_Revenue = resid_mu_SUV_Revenue + 1.5*resid_dev_SUV_Revenue   # upper bound of anomaly
# Storing the anomalies
anomalies_SUV_Revenue = pd.DataFrame(df[['SUV_Revenue','date']][(resid_SUV_Revenue < lower_SUV_Revenue) | (resid_SUV_Revenue > upper_SUV_Revenue)])


# SUV Revenue_per_sale

#Fit model
stl_SUV_Revenue_per_sale = STL(df['SUV_Revenue_per_sale'])
result_SUV_Revenue_per_sale = stl_SUV_Revenue_per_sale.fit()
# Extract the decomposition
seasonal_SUV_Revenue_per_sale, trend_SUV_Revenue_per_sale, resid_SUV_Revenue_per_sale = result_SUV_Revenue_per_sale.seasonal, result_SUV_Revenue_per_sale.trend, result_SUV_Revenue_per_sale.resid
estimated_SUV_Revenue_per_sale = trend_SUV_Revenue_per_sale + seasonal_SUV_Revenue_per_sale
#Setting and finding Anomaly
resid_mu_SUV_Revenue_per_sale = resid_SUV_Revenue_per_sale.mean()
resid_dev_SUV_Revenue_per_sale = resid_SUV_Revenue_per_sale.std()
lower_SUV_Revenue_per_sale = resid_mu_SUV_Revenue_per_sale - 1.5*resid_dev_SUV_Revenue_per_sale   # lower bound of anomaly
upper_SUV_Revenue_per_sale = resid_mu_SUV_Revenue_per_sale + 1.5*resid_dev_SUV_Revenue_per_sale   # upper bound of anomaly
# Storing the anomalies
anomalies_SUV_Revenue_per_sale = pd.DataFrame(df[['SUV_Revenue_per_sale','date']][(resid_SUV_Revenue_per_sale < lower_SUV_Revenue_per_sale) | (resid_SUV_Revenue_per_sale > upper_SUV_Revenue_per_sale)])

# Model_1 SALES

#Fit model
stl_Sedan_Model_1_sales = STL(df['Model_1_sales'])
result_Sedan_Model_1_sales = stl_Sedan_Model_1_sales.fit()
# Extract the decomposition
seasonal_Sedan_Model_1_sales, trend_Sedan_Model_1_sales, resid_Sedan_Model_1_sales = result_Sedan_Model_1_sales.seasonal, result_Sedan_Model_1_sales.trend, result_Sedan_Model_1_sales.resid
estimated_Sedan_Model_1_sales = trend_Sedan_Model_1_sales + seasonal_Sedan_Model_1_sales
#Setting and finding Anomaly
resid_mu_Sedan_Model_1_sales = resid_Sedan_Model_1_sales.mean()
resid_dev_Sedan_Model_1_sales = resid_Sedan_Model_1_sales.std()
lower_Sedan_Model_1_sales = resid_mu_Sedan_Model_1_sales - 1.5*resid_dev_Sedan_Model_1_sales   # lower bound of anomaly
upper_Sedan_Model_1_sales = resid_mu_Sedan_Model_1_sales + 1.5*resid_dev_Sedan_Model_1_sales   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_1_sales = pd.DataFrame(df[['Model_1_sales','date']][(resid_Sedan_Model_1_sales < lower_Sedan_Model_1_sales) | (resid_Sedan_Model_1_sales > upper_Sedan_Model_1_sales)])


# Model_1 Revenue

#Fit model
stl_Sedan_Model_1_Revenue = STL(df['Model_1_Revenue'])
result_Sedan_Model_1_Revenue = stl_Sedan_Model_1_Revenue.fit()
# Extract the decomposition
seasonal_Sedan_Model_1_Revenue, trend_Sedan_Model_1_Revenue, resid_Sedan_Model_1_Revenue = result_Sedan_Model_1_Revenue.seasonal, result_Sedan_Model_1_Revenue.trend, result_Sedan_Model_1_Revenue.resid
estimated_Sedan_Model_1_Revenue = trend_Sedan_Model_1_Revenue + seasonal_Sedan_Model_1_Revenue
#Setting and finding Anomaly
resid_mu_Sedan_Model_1_Revenue = resid_Sedan_Model_1_Revenue.mean()
resid_dev_Sedan_Model_1_Revenue = resid_Sedan_Model_1_Revenue.std()
lower_Sedan_Model_1_Revenue = resid_mu_Sedan_Model_1_Revenue - 1.5*resid_dev_Sedan_Model_1_Revenue   # lower bound of anomaly
upper_Sedan_Model_1_Revenue = resid_mu_Sedan_Model_1_Revenue + 1.5*resid_dev_Sedan_Model_1_Revenue   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_1_Revenue = pd.DataFrame(df[['Model_1_Revenue','date']][(resid_Sedan_Model_1_Revenue < lower_Sedan_Model_1_Revenue) | (resid_Sedan_Model_1_Revenue > upper_Sedan_Model_1_Revenue)])

# Model_1 Revenue_per_sale

#Fit model
stl_Sedan_Model_1_Revenue_per_sale = STL(df['Model_1_Revenue_per_sale'])
result_Sedan_Model_1_Revenue_per_sale = stl_Sedan_Model_1_Revenue_per_sale.fit()
# Extract the decomposition
seasonal_Sedan_Model_1_Revenue_per_sale, trend_Sedan_Model_1_Revenue_per_sale, resid_Sedan_Model_1_Revenue_per_sale = result_Sedan_Model_1_Revenue_per_sale.seasonal, result_Sedan_Model_1_Revenue_per_sale.trend, result_Sedan_Model_1_Revenue_per_sale.resid
estimated_Sedan_Model_1_Revenue_per_sale = trend_Sedan_Model_1_Revenue_per_sale + seasonal_Sedan_Model_1_Revenue_per_sale
#Setting and finding Anomaly
resid_mu_Sedan_Model_1_Revenue_per_sale = resid_Sedan_Model_1_Revenue_per_sale.mean()
resid_dev_Sedan_Model_1_Revenue_per_sale = resid_Sedan_Model_1_Revenue_per_sale.std()
lower_Sedan_Model_1_Revenue_per_sale = resid_mu_Sedan_Model_1_Revenue_per_sale - 1.5*resid_dev_Sedan_Model_1_Revenue_per_sale   # lower bound of anomaly
upper_Sedan_Model_1_Revenue_per_sale = resid_mu_Sedan_Model_1_Revenue_per_sale + 1.5*resid_dev_Sedan_Model_1_Revenue_per_sale   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_1_Revenue_per_sale = pd.DataFrame(df[['Model_1_Revenue_per_sale','date']][(resid_Sedan_Model_1_Revenue_per_sale < lower_Sedan_Model_1_Revenue_per_sale) | (resid_Sedan_Model_1_Revenue_per_sale > upper_Sedan_Model_1_Revenue_per_sale)])

# Model_2 SALES

#Fit model
stl_Sedan_Model_2_sales = STL(df['Model_2_sales'])
result_Sedan_Model_2_sales = stl_Sedan_Model_2_sales.fit()
# Extract the decomposition
seasonal_Sedan_Model_2_sales, trend_Sedan_Model_2_sales, resid_Sedan_Model_2_sales = result_Sedan_Model_2_sales.seasonal, result_Sedan_Model_2_sales.trend, result_Sedan_Model_2_sales.resid
estimated_Sedan_Model_2_sales = trend_Sedan_Model_2_sales + seasonal_Sedan_Model_2_sales
#Setting and finding Anomaly
resid_mu_Sedan_Model_2_sales = resid_Sedan_Model_2_sales.mean()
resid_dev_Sedan_Model_2_sales = resid_Sedan_Model_2_sales.std()
lower_Sedan_Model_2_sales = resid_mu_Sedan_Model_2_sales - 1.5*resid_dev_Sedan_Model_2_sales   # lower bound of anomaly
upper_Sedan_Model_2_sales = resid_mu_Sedan_Model_2_sales + 1.5*resid_dev_Sedan_Model_2_sales   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_2_sales = pd.DataFrame(df[['Model_2_sales','date']][(resid_Sedan_Model_2_sales < lower_Sedan_Model_2_sales) | (resid_Sedan_Model_2_sales > upper_Sedan_Model_2_sales)])


# Model_2 Revenue

#Fit model
stl_Sedan_Model_2_Revenue = STL(df['Model_2_Revenue'])
result_Sedan_Model_2_Revenue = stl_Sedan_Model_2_Revenue.fit()
# Extract the decomposition
seasonal_Sedan_Model_2_Revenue, trend_Sedan_Model_2_Revenue, resid_Sedan_Model_2_Revenue = result_Sedan_Model_2_Revenue.seasonal, result_Sedan_Model_2_Revenue.trend, result_Sedan_Model_2_Revenue.resid
estimated_Sedan_Model_2_Revenue = trend_Sedan_Model_2_Revenue + seasonal_Sedan_Model_2_Revenue
#Setting and finding Anomaly
resid_mu_Sedan_Model_2_Revenue = resid_Sedan_Model_2_Revenue.mean()
resid_dev_Sedan_Model_2_Revenue = resid_Sedan_Model_2_Revenue.std()
lower_Sedan_Model_2_Revenue = resid_mu_Sedan_Model_2_Revenue - 1.5*resid_dev_Sedan_Model_2_Revenue   # lower bound of anomaly
upper_Sedan_Model_2_Revenue = resid_mu_Sedan_Model_2_Revenue + 1.5*resid_dev_Sedan_Model_2_Revenue   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_2_Revenue = pd.DataFrame(df[['Model_2_Revenue','date']][(resid_Sedan_Model_2_Revenue < lower_Sedan_Model_2_Revenue) | (resid_Sedan_Model_2_Revenue > upper_Sedan_Model_2_Revenue)])

# Model_2 Revenue_per_sale

#Fit model
stl_Sedan_Model_2_Revenue_per_sale = STL(df['Model_2_Revenue_per_sale'])
result_Sedan_Model_2_Revenue_per_sale = stl_Sedan_Model_2_Revenue_per_sale.fit()
# Extract the decomposition
seasonal_Sedan_Model_2_Revenue_per_sale, trend_Sedan_Model_2_Revenue_per_sale, resid_Sedan_Model_2_Revenue_per_sale = result_Sedan_Model_2_Revenue_per_sale.seasonal, result_Sedan_Model_2_Revenue_per_sale.trend, result_Sedan_Model_2_Revenue_per_sale.resid
estimated_Sedan_Model_2_Revenue_per_sale = trend_Sedan_Model_2_Revenue_per_sale + seasonal_Sedan_Model_2_Revenue_per_sale
#Setting and finding Anomaly
resid_mu_Sedan_Model_2_Revenue_per_sale = resid_Sedan_Model_2_Revenue_per_sale.mean()
resid_dev_Sedan_Model_2_Revenue_per_sale = resid_Sedan_Model_2_Revenue_per_sale.std()
lower_Sedan_Model_2_Revenue_per_sale = resid_mu_Sedan_Model_2_Revenue_per_sale - 1.5*resid_dev_Sedan_Model_2_Revenue_per_sale   # lower bound of anomaly
upper_Sedan_Model_2_Revenue_per_sale = resid_mu_Sedan_Model_2_Revenue_per_sale + 1.5*resid_dev_Sedan_Model_2_Revenue_per_sale   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_2_Revenue_per_sale = pd.DataFrame(df[['Model_2_Revenue_per_sale','date']][(resid_Sedan_Model_2_Revenue_per_sale < lower_Sedan_Model_2_Revenue_per_sale) | (resid_Sedan_Model_2_Revenue_per_sale > upper_Sedan_Model_2_Revenue_per_sale)])

# Model_3 SALES

#Fit model
stl_Sedan_Model_3_sales = STL(df['Model_3_sales'])
result_Sedan_Model_3_sales = stl_Sedan_Model_3_sales.fit()
# Extract the decomposition
seasonal_Sedan_Model_3_sales, trend_Sedan_Model_3_sales, resid_Sedan_Model_3_sales = result_Sedan_Model_3_sales.seasonal, result_Sedan_Model_3_sales.trend, result_Sedan_Model_3_sales.resid
estimated_Sedan_Model_3_sales = trend_Sedan_Model_3_sales + seasonal_Sedan_Model_3_sales
#Setting and finding Anomaly
resid_mu_Sedan_Model_3_sales = resid_Sedan_Model_3_sales.mean()
resid_dev_Sedan_Model_3_sales = resid_Sedan_Model_3_sales.std()
lower_Sedan_Model_3_sales = resid_mu_Sedan_Model_3_sales - 1.5*resid_dev_Sedan_Model_3_sales   # lower bound of anomaly
upper_Sedan_Model_3_sales = resid_mu_Sedan_Model_3_sales + 1.5*resid_dev_Sedan_Model_3_sales   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_3_sales = pd.DataFrame(df[['Model_3_sales','date']][(resid_Sedan_Model_3_sales < lower_Sedan_Model_3_sales) | (resid_Sedan_Model_3_sales > upper_Sedan_Model_3_sales)])


# Model_3 Revenue

#Fit model
stl_Sedan_Model_3_Revenue = STL(df['Model_3_Revenue'])
result_Sedan_Model_3_Revenue = stl_Sedan_Model_3_Revenue.fit()
# Extract the decomposition
seasonal_Sedan_Model_3_Revenue, trend_Sedan_Model_3_Revenue, resid_Sedan_Model_3_Revenue = result_Sedan_Model_3_Revenue.seasonal, result_Sedan_Model_3_Revenue.trend, result_Sedan_Model_3_Revenue.resid
estimated_Sedan_Model_3_Revenue = trend_Sedan_Model_3_Revenue + seasonal_Sedan_Model_3_Revenue
#Setting and finding Anomaly
resid_mu_Sedan_Model_3_Revenue = resid_Sedan_Model_3_Revenue.mean()
resid_dev_Sedan_Model_3_Revenue = resid_Sedan_Model_3_Revenue.std()
lower_Sedan_Model_3_Revenue = resid_mu_Sedan_Model_3_Revenue - 1.5*resid_dev_Sedan_Model_3_Revenue   # lower bound of anomaly
upper_Sedan_Model_3_Revenue = resid_mu_Sedan_Model_3_Revenue + 1.5*resid_dev_Sedan_Model_3_Revenue   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_3_Revenue = pd.DataFrame(df[['Model_3_Revenue','date']][(resid_Sedan_Model_3_Revenue < lower_Sedan_Model_3_Revenue) | (resid_Sedan_Model_3_Revenue > upper_Sedan_Model_3_Revenue)])

# Model_3 Revenue_per_sale

#Fit model
stl_Sedan_Model_3_Revenue_per_sale = STL(df['Model_3_Revenue_per_sale'])
result_Sedan_Model_3_Revenue_per_sale = stl_Sedan_Model_3_Revenue_per_sale.fit()
# Extract the decomposition
seasonal_Sedan_Model_3_Revenue_per_sale, trend_Sedan_Model_3_Revenue_per_sale, resid_Sedan_Model_3_Revenue_per_sale = result_Sedan_Model_3_Revenue_per_sale.seasonal, result_Sedan_Model_3_Revenue_per_sale.trend, result_Sedan_Model_3_Revenue_per_sale.resid
estimated_Sedan_Model_3_Revenue_per_sale = trend_Sedan_Model_3_Revenue_per_sale + seasonal_Sedan_Model_3_Revenue_per_sale
#Setting and finding Anomaly
resid_mu_Sedan_Model_3_Revenue_per_sale = resid_Sedan_Model_3_Revenue_per_sale.mean()
resid_dev_Sedan_Model_3_Revenue_per_sale = resid_Sedan_Model_3_Revenue_per_sale.std()
lower_Sedan_Model_3_Revenue_per_sale = resid_mu_Sedan_Model_3_Revenue_per_sale - 1.5*resid_dev_Sedan_Model_3_Revenue_per_sale   # lower bound of anomaly
upper_Sedan_Model_3_Revenue_per_sale = resid_mu_Sedan_Model_3_Revenue_per_sale + 1.5*resid_dev_Sedan_Model_3_Revenue_per_sale   # upper bound of anomaly
# Storing the anomalies
anomalies_Sedan_Model_3_Revenue_per_sale = pd.DataFrame(df[['Model_3_Revenue_per_sale','date']][(resid_Sedan_Model_3_Revenue_per_sale < lower_Sedan_Model_3_Revenue_per_sale) | (resid_Sedan_Model_3_Revenue_per_sale > upper_Sedan_Model_3_Revenue_per_sale)])


#Layout

app.layout =  html.Div([ html.Div([html.H1(children='Car Sales Funnel - Anomaly Detection', style={
               'textAlign': 'center'
            
               })]),
    dcc.Tabs([
        dcc.Tab(label='Sedan', children=[
            html.Div(children=[
               

               html.H2(children='''
               Sedan
               ''', style={
                   'textAlign': 'center'
                   
               }),
               
               html.Br(),
               
               html.Div(  dcc.Checklist( id ='Sedan_checkbox',
               options=[
                   {'label': 'Trend', 'value': 'trend' ,'disabled':True},
                   {'label': '7 days Moving  ', 'value': 'moving_7'},
                   {'label': '30 days Moving  ', 'value': 'moving_30'},
                   {'label': 'Anomalies  ', 'value': 'anomalies'}
               ],
               value=['trend'] , labelStyle={'display': 'inline-block'}
               ),  style={
                   'textAlign': 'center'
                
               }),
        
               html.Br(),
              
               dcc.Graph(
               id='Sedan_graph', figure=fig_Sedan, style={'width': '200vh', 'height': '2300px'}
               
               )
              ])
        ]),
        dcc.Tab(label='Sedan Models', children=[
            html.Div(children=[
               

               html.H2(children='''
               Sedan Models
               ''', style={
                   'textAlign': 'center'
                   
               }),
               
               html.Br(),
               
                 
            
            html.Div( dcc.Checklist( id ='Sedan_model_checkbox',
               options=[
                   {'label': 'Trend', 'value': 'trend' ,'disabled':True},
                   {'label': '7 days Moving  ', 'value': 'moving_7'},
                   {'label': '30 days Moving  ', 'value': 'moving_30'},
                   {'label': 'Anomalies  ', 'value': 'anomalies'}
               ],
               value=['trend'] , labelStyle={'display': 'inline-block'}
               ),  style={
                   'textAlign': 'center'
                
               }) ,
            
            html.Br(),
            
               html.Div([
               dcc.Dropdown(
                id='Sedan_model_dropdown',
                options=[
                
                {'label': 'Model 1', 'value': 'Model_1'},
                {'label': 'Model 2', 'value': 'Model_2'},
                {'label': 'Model 3', 'value': 'Model_3'},
                ],
                value='Model_1', clearable=False
                ),
                html.Div(id='dd-output-container')
            ],style={"width": "25%", 'display': 'inline-block'}
            
            ),
               
        
               html.Br(),
              
               dcc.Graph(
               id='Sedan_model_graph', figure=fig_Sedan_model_Model_1, style={'width': '200vh', 'height': '1380px'}
               
               )
              ])
        ]),
        dcc.Tab(label='SUV', children=[
            html.Div(children=[
               

               html.H2(children='''
               SUV
               ''', style={
                   'textAlign': 'center'
                   
               }),
               
               html.Br(),
               
               html.Div(  dcc.Checklist( id ='SUV_checkbox',
               options=[
                   {'label': 'Trend', 'value': 'trend' ,'disabled':True},
                   {'label': '7 days Moving', 'value': 'moving_7'},
                   {'label': '30 days Moving', 'value': 'moving_30'},
                   {'label': 'Anomalies', 'value': 'anomalies'}
               ],
               value=['trend'] , labelStyle={'display': 'inline-block'}
               ),  style={
                   'textAlign': 'center'
                   
               }),
        
               html.Br(),
              
               dcc.Graph(
               id='SUV_graph', figure=fig_SUV, style={'width': '200vh', 'height': '2300px'}
               
               )
              ])
        ]),
       
    ])
])


# Checkbox callback Sedan

@app.callback(Output('Sedan_graph', 'figure'),
              Input('Sedan_checkbox', 'value'),
            )


# Checbox Function Sedan

def update_graph_Sedan(var_Sedan_checkbox):

        fig_Sedan_30 = make_subplots(rows=5, cols=1, subplot_titles=("Visitors","Prospects", "Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)
        
        fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_Sedan_visitors'],mode='lines',name = "Visitors"), row=1, col=1)
        
        fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_prospects'],mode='lines',line=dict(color="#FF8040"),name = "Prospects"), row=2, col=1)
        
        fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Sales'],mode='lines',name = "Sales"),  row=3, col=1)
        
        fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue'],mode='lines',name = "Revenue"),   row=4, col=1)
        
        fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"),  row=5, col=1)
        

        if ('moving_7' in var_Sedan_checkbox):
       
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_Sedan_visitors_7'],mode='lines',line=dict(color="#000000"),name = "Visitors"),  row=1, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_prospects_7'],mode='lines',line=dict(color="#000000"),name = "Prospects"),  row=2, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Sales_7'],mode='lines',line=dict(color="#000000"),name = "Sales"),  row=3, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_7'],mode='lines',line=dict(color="#000000"),name = "Revenue"),  row=4, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_per_sale_7'],mode='lines',line=dict(color="#000000"),name = "Revenue_per_sale"),  row=5, col=1)

            
        if ('moving_30' in var_Sedan_checkbox):
       
        
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_Sedan_visitors_30'],mode='lines',line=dict(color="#000000"),name = "Visitors"),   row=1, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_prospects_30'],mode='lines',line=dict(color="#000000"),name = "Prospects"),  row=2, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Sales_30'],mode='lines',line=dict(color="#000000"),name = "Sales"),   row=3, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_30'],mode='lines',line=dict(color="#000000"),name = "Revenue"),   row=4, col=1)
        
            fig_Sedan_30.add_trace(go.Scatter(x=df['date'], y=df['Sedan_Revenue_per_sale_30'],mode='lines',line=dict(color="#000000"),name = "Revenue_per_sale"),  row=5, col=1)
        
        if ('anomalies' in var_Sedan_checkbox):
       
        
            fig_Sedan_30.add_trace(go.Scatter(x=anomalies_Sedan_visitors['date'], y=anomalies_Sedan_visitors['Overall_Sedan_visitors'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Visitors", mode='markers'),  row=1, col=1)
            
            fig_Sedan_30.add_trace(go.Scatter(x=anomalies_Sedan_leads['date'], y=anomalies_Sedan_leads['Sedan_prospects'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Prospects", mode='markers'),  row=2, col=1)  
           
            fig_Sedan_30.add_trace(go.Scatter(x=anomalies_Sedan_sales['date'], y=anomalies_Sedan_sales['Sedan_Sales'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Sales", mode='markers'),  row=3, col=1)
            
            fig_Sedan_30.add_trace(go.Scatter(x=anomalies_Sedan_Revenue['date'], y=anomalies_Sedan_Revenue['Sedan_Revenue'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue", mode='markers'),  row=4, col=1)
            
            fig_Sedan_30.add_trace(go.Scatter(x=anomalies_Sedan_Revenue_per_sale['date'], y=anomalies_Sedan_Revenue_per_sale['Sedan_Revenue_per_sale'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue_per_sale", mode='markers'),  row=5, col=1)
            
        #fig.update_xaxes(rangeslider_visible=True)
        fig_Sedan_30.update_layout(showlegend=False)
        fig_Sedan_30.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]]) 
        fig_Sedan_30.update_yaxes(rangemode="tozero")         


        return (fig_Sedan_30)
    
    
# Checkbox callback SUV

@app.callback(Output('SUV_graph', 'figure'),
              Input('SUV_checkbox', 'value'),
            )


# Checkbox Function SUV

def update_graph_SUV(var_SUV_checkbox):

        fig_SUV_30 = make_subplots(rows=5, cols=1, subplot_titles=("Visitors","Prospects", "Sales","Revenue", "Revenue_per_sale"), vertical_spacing = Vertical_gap)
        
        fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_SUV_visitors'],mode='lines',name = "Visitors"), row=1, col=1)
        
        fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_prospects'],mode='lines',line=dict(color="#FF8040"),name = "Prospects"),  row=2, col=1)
        
        fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Sales'],mode='lines',name = "Sales"),  row=3, col=1)
        
        fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue'],mode='lines',name = "Revenue"), row=4, col=1)
        
        fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_per_sale'],mode='lines',name = "Revenue_per_sale"),  row=5, col=1)
        

        if ('moving_7' in var_SUV_checkbox):
       
        
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_SUV_visitors_7'], mode='lines',line=dict(color="#000000"),name = "Visitors"),  row=1, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_prospects_7'],mode='lines',line=dict(color="#000000"),name = "Prospects"),  row=2, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Sales_7'],mode='lines',line=dict(color="#000000"),name = "Sales"),  row=3, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_7'],mode='lines',line=dict(color="#000000"),name = "Revenue"),  row=4, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_per_sale_7'],mode='lines',line=dict(color="#000000"),name = "Revenue_per_sale"),   row=5, col=1)
            
        
        if ('moving_30' in var_SUV_checkbox):
       
        
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['Overall_SUV_visitors_30'], mode='lines',line=dict(color="#000000"),name = "Visitors"), row=1, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_prospects_30'],mode='lines',line=dict(color="#000000"),name = "Prospects"),  row=2, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Sales_30'],mode='lines',line=dict(color="#000000"),name = "Sales"),  row=3, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_30'],mode='lines',line=dict(color="#000000"),name = "Revenue"),  row=4, col=1)
        
            fig_SUV_30.add_trace(go.Scatter(x=df['date'], y=df['SUV_Revenue_per_sale_30'],mode='lines',line=dict(color="#000000"),name = "Revenue_per_sale"),  row=5, col=1)
            
        if ('anomalies' in var_SUV_checkbox):
       
        
            fig_SUV_30.add_trace(go.Scatter(x=anomalies_SUV_visitors['date'], y=anomalies_SUV_visitors['Overall_SUV_visitors'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Visitors", mode='markers'),  row=1, col=1)
              
            fig_SUV_30.add_trace(go.Scatter(x=anomalies_SUV_leads['date'], y=anomalies_SUV_leads['SUV_prospects'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Prospects", mode='markers'),  row=2, col=1)  
           
            fig_SUV_30.add_trace(go.Scatter(x=anomalies_SUV_sales['date'], y=anomalies_SUV_sales['SUV_Sales'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Sales", mode='markers'),  row=3, col=1)
            
            fig_SUV_30.add_trace(go.Scatter(x=anomalies_SUV_Revenue['date'], y=anomalies_SUV_Revenue['SUV_Revenue'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue", mode='markers'),  row=4, col=1)
            
            fig_SUV_30.add_trace(go.Scatter(x=anomalies_SUV_Revenue_per_sale['date'], y=anomalies_SUV_Revenue_per_sale['SUV_Revenue_per_sale'],marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue_per_sale", mode='markers'),  row=5, col=1)  
        #fig.update_xaxes(rangeslider_visible=True)
        fig_SUV_30.update_layout(showlegend=False)
        fig_SUV_30.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]]) 
        fig_SUV_30.update_yaxes(rangemode="tozero")           


        return (fig_SUV_30)   


# Checkbox and Dropdown callback Sedan Models

@app.callback(Output('Sedan_model_graph', 'figure'),
              Input('Sedan_model_checkbox', 'value'),
              Input('Sedan_model_dropdown', 'value'),
			  
            )


# Checkbox and Dropdown Function Sedan Models

def update_graph_Sedan_Models(var_Sedan_model_checkbox,var_Sedan_model_dropdown):
        
        
        
        if(var_Sedan_model_dropdown == 'Model_1'):
        
            fig_model_1 = go.Figure(fig_Sedan_model_Model_1)
        
            if('moving_7' in var_Sedan_model_checkbox):
           
            
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_sales_7'],mode='lines',line=dict(color="#000000"), name = "Sales"),row=1, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue_7'],mode='lines',line=dict(color="#000000"), name = "Revenue"),row=2, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue_per_sale_7'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),row=3, col=1)
                
                
                
            if('moving_30' in var_Sedan_model_checkbox):    
                
                
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_sales_30'],mode='lines',line=dict(color="#000000"), name = "Sales"), row=1, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue_30'],mode='lines',line=dict(color="#000000"), name = "Revenue"),  row=2, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=df['date'], y=df['Model_1_Revenue_per_sale_30'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),  row=3, col=1)
                
                
                
            if('anomalies' in var_Sedan_model_checkbox):
            
                fig_model_1.add_trace(go.Scatter(x=anomalies_Sedan_Model_1_sales['date'], y=anomalies_Sedan_Model_1_sales['Model_1_sales'],mode='markers', marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Sales"), row=1, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=anomalies_Sedan_Model_1_Revenue['date'], y=anomalies_Sedan_Model_1_Revenue['Model_1_Revenue'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue"), row=2, col=1)
                
                fig_model_1.add_trace(go.Scatter(x=anomalies_Sedan_Model_1_Revenue_per_sale['date'], y=anomalies_Sedan_Model_1_Revenue_per_sale['Model_1_Revenue_per_sale'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue_per_sale"), row=3, col=1)
            
                
            
            fig_model = fig_model_1

        if(var_Sedan_model_dropdown == 'Model_2'):
        
            fig_model_2 = go.Figure(fig_Sedan_model_Model_2)
            
        
            if('moving_7' in var_Sedan_model_checkbox):
         
            
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_sales_7'],mode='lines',line=dict(color="#000000"), name = "Sales"),row=1, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue_7'],mode='lines',line=dict(color="#000000"), name = "Revenue"),row=2, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue_per_sale_7'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),row=3, col=1)
                
                
                
            if('moving_30' in var_Sedan_model_checkbox):    
                
                
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_sales_30'],mode='lines',line=dict(color="#000000"), name = "Sales"), row=1, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue_30'],mode='lines',line=dict(color="#000000"), name = "Revenue"),  row=2, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=df['date'], y=df['Model_2_Revenue_per_sale_30'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),  row=3, col=1)
                
                
                
            if('anomalies' in var_Sedan_model_checkbox):
            
                fig_model_2.add_trace(go.Scatter(x=anomalies_Sedan_Model_2_sales['date'], y=anomalies_Sedan_Model_2_sales['Model_2_sales'],mode='markers', marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Sales"), row=1, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=anomalies_Sedan_Model_2_Revenue['date'], y=anomalies_Sedan_Model_2_Revenue['Model_2_Revenue'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue"), row=2, col=1)
                
                fig_model_2.add_trace(go.Scatter(x=anomalies_Sedan_Model_2_Revenue_per_sale['date'], y=anomalies_Sedan_Model_2_Revenue_per_sale['Model_2_Revenue_per_sale'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue_per_sale"), row=3, col=1)
            
                

            fig_model = fig_model_2

        if(var_Sedan_model_dropdown == 'Model_3'):
        
            
            
            fig_model_3 = go.Figure(fig_Sedan_model_Model_3)
        
            if('moving_7' in var_Sedan_model_checkbox):
         
            
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_sales_7'],mode='lines',line=dict(color="#000000"), name = "Sales"),row=1, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue_7'],mode='lines',line=dict(color="#000000"), name = "Revenue"),row=2, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue_per_sale_7'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),row=3, col=1)
                
                
                
            if('moving_30' in var_Sedan_model_checkbox):    
                
                
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_sales_30'],mode='lines',line=dict(color="#000000"), name = "Sales"), row=1, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue_30'],mode='lines',line=dict(color="#000000"), name = "Revenue"),  row=2, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=df['date'], y=df['Model_3_Revenue_per_sale_30'],mode='lines',line=dict(color="#000000"), name = "Revenue_per_sale"),  row=3, col=1)
                
                
                
            if('anomalies' in var_Sedan_model_checkbox):
            
                fig_model_3.add_trace(go.Scatter(x=anomalies_Sedan_Model_3_sales['date'], y=anomalies_Sedan_Model_3_sales['Model_3_sales'],mode='markers', marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Sales"), row=1, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=anomalies_Sedan_Model_3_Revenue['date'], y=anomalies_Sedan_Model_3_Revenue['Model_3_Revenue'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue"), row=2, col=1)
                
                fig_model_3.add_trace(go.Scatter(x=anomalies_Sedan_Model_3_Revenue_per_sale['date'], y=anomalies_Sedan_Model_3_Revenue_per_sale['Model_3_Revenue_per_sale'],mode='markers',marker=dict(color='rgba(255, 255, 255, 0.05)',line=dict(width=2, color='Red'), size=10),name = "Revenue_per_sale"), row=3, col=1)
            
                
                
            fig_model = fig_model_3    

        fig_model.update_layout(showlegend=False)
        fig_model.update_layout(xaxis_range=[df['date'].iloc[0],df['date'].iloc[-1]])
        fig_model.update_yaxes(rangemode="tozero")
        return (fig_model)     
        
        
   
#Run Program

if __name__ == '__main__':
    app.run_server(debug=True)