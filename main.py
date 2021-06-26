from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importar base de dados
df = pd.read_csv("advertising.csv")
display(df)

# Análise Exploratória
# - Visualizar como as informações de cada item estão distribuídas
# - Ver a correlação entre cada um dos itens
sns.heatmap(df.corr(), annot =True, cmap="Wistia")
plt.show()
sns.pairplot(df)
plt.show()

# Preparação dos dados para treinamento do Modelo de Machine Learning
x = df.drop('Vendas', axis=1)
y = df['Vendas']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

# Problema de regressão - Vamos escolher os modelos que vamos usar:
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# Treino AI
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

# Teste da AI e Avaliação do Melhor Modelo
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece
# - Também vamos olhar o MSE (Erro Quadrático Médio) -> diz o quanto o nosso modelo "erra" quando tenta fazer uma previsão

# Teste AI
test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
mse_lin = metrics.mean_squared_error(y_test, test_pred_lin)
print(f'R² da Regressão Linear: {r2_lin}')
print(f'MSE da Regressão Linear: {mse_lin}')
r2_rf= metrics.r2_score(y_test, test_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, test_pred_rf)
print(f'R² do Random Forest: {r2_rf}')
print(f'MSE do Random Forest: {mse_rf}')

# Visualização Gráfica
df_resultado = pd.DataFrame()
# df_resultado.index = x_test
df_resultado['y_teste'] = y_test
df_resultado['y_previsao_rf'] = test_pred_rf
df_resultado['y_previsao_lin'] = test_pred_lin
# display(df_resultado)
# df_resultado = df_resultado.reset_index(drop=True)
plt.figure(figsize=(15, 5))
sns.lineplot(data=df_resultado)
plt.show()
display(df_resultado)

# importância de cada variável para as vendas
# importancia_features = pd.DataFrame(rf_reg.feature_importances_, x_train.columns)
# plt.figure(figsize=(15, 5))
sns.barplot(x=x_train.columns, y=rf_reg.feature_importances_)
plt.show()

print(df[["Radio", "Jornal"]].sum())