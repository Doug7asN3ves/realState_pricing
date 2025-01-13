import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fastapi import FastAPI
import joblib
import uvicorn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  

from fastapi.middleware.cors import CORSMiddleware

# Carregando os dados para o modelo
properties = pd.read_csv('property_data.csv')
transactions = pd.read_csv('transactions.csv')
economics = pd.read_csv('economics.csv')

# Bases adicionais
mortgage = pd.read_csv('mortgage.csv')
inflation = pd.read_csv('inflation.csv')
rental_vacancy = pd.read_csv('rental_vacancy.csv')

# Inicialmente, observa-se que o property_id das tabelas não correspondem e entregam propriedades diferentes, portanto as bases "properties" e "transactions" não podem ser unidas diretamente.

# A base de propriedades é a base principal, e será enriquecida posteriormente com possíveis novas variáveis

# Análise exploratória das variáveis da base de propriedades
properties.info()

# A base não possui valores nulos
properties.isna().sum()

# Análise inicial estatística das variáveis
properties.describe().T

# Plotando os histogramas das variáveis numéricas para embasar o feature selection do modelo
cols = ['size_sqft', 'bedrooms', 'bathrooms', 'property_size', 'year_built', 'listing_price', 'num_bedrooms', 'num_bathrooms', 'garage_spaces']

fig, axes = plt.subplots(nrows=(len(cols) + 1) // 2, ncols=2, figsize=(15, 30))
for i, col in enumerate(cols):
    sns.histplot(properties[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
plt.tight_layout()
plt.show()

# Scatter Plot
var = ['size_sqft', 'bedrooms', 'bathrooms', 'property_size', 'year_built', 'num_bedrooms', 'garage_spaces']
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
fig.suptitle('Scatter Plots com listing_price', fontsize=16)
axes = axes.flatten()
for i, col in enumerate(var):
    sns.scatterplot(data=properties, x=col, y='listing_price', ax=axes[i])
    axes[i].set_title(f'{col} vs listing_price')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Matriz de correlação
var = ['listing_price','size_sqft', 'bedrooms', 'bathrooms', 'property_size', 'year_built', 'num_bedrooms', 'garage_spaces']
correlation_matrix = properties[var].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
print(correlation_matrix)

# Seleção das variáveis da base de propriedades
var_modelo = ['property_id', 'bedrooms', 'bathrooms', 'property_type', 'property_size', 'year_built', 'listing_price', 'neighborhood', 'garage_spaces', 'has_pool']
properties_modelo = properties[var_modelo]

# Separando a base entre treino e teste
X = properties_modelo.drop(columns=['listing_price'])
y = properties_modelo['listing_price']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar os modelos
model = LinearRegression()
model_RF = RandomForestRegressor()
model_XGB = XGBRegressor()

# Treinar os modelos
model.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_XGB.fit(X_train, y_train)

# Avaliar a performance dos modelos
models = [model, model_RF, model_XGB]
model_names = ['Linear Regression', 'Random Forest', 'XGBoost']
for i in range(len(models)):
    print(f'{model_names[i]}:')
    y_train_pred = models[i].predict(X_train)
    y_test_pred = models[i].predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print('Training MSE:', mse_train)
    print('Validation MSE:', mse_test)
    print('Training MAE:', mae_train)
    print('Validation MAE:', mae_test)
    print('Training R²:', r2_train)
    print('Validation R²:', r2_test)
    print()

# Salvando os modelos treinados
pickle.dump(model, open("model.pkl", "wb"))
