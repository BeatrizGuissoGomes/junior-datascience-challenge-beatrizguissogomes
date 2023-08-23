"""Qualidade em Processo de Mineração

Projeto com dados reais de uma Indústria de Mineração disponibilizado no kaggle:  https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process

1. Inicializando o projeto
Olhada inicial dos dados, ver o tamanho e buscar por valores ausentes, ou nulos.
"""

# Importando pacotes gerais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Leitura do dataset para primeiras impressões
df = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv", decimal =",", parse_dates = True, index_col = 'date')
df.head()

# Analisando o tamanho do dataset
df.shape

# Analisando data inicial e final
print(df.index.min())
print(df.index.max())

# Analisando o tipo de dado e também a presença de valores nulos.
df.info()

# Estatística Descritiva
df.describe().round(2)

"""**Conclusão:**
*  Não foram encontrados dados faltantes.
*  Os dados estão no formato adequado, não há presença de categóricos para serem tratados por exemplo com one-hot-encoding
*  Inicialmente pecebe-se que nas colunas "% Iron Feed", "% Silica Feed",	"% Iron Concentrate" e "% Silica Concentrate" os dados se repetem, vamos, a seguir, analisar qual a frequência de amostragem das features

2. Analisando frequência de amostragem
"""

# Agrupando data de acordo com as horas e vendo quantas observações tem para cada hora
horas_obs = df.groupby(pd.Grouper(freq='H')).count()
horas_obs

# Contando os valores observação por hora
horas_obs['% Iron Feed'].value_counts()

"""Maioria das horas tem 180 observações, uma hora tem 60 minutos e 3600 segundos

Logo, dividindo 3600 seg/180 obs = 20 seg/1 obs.

1 observação a cada 20 segundos
"""

# Analisando horas que não tem 180 observações
horas_obs[horas_obs["% Iron Feed"] != 180]

# Analisando horas que não tem observações
horas_obs[horas_obs["% Iron Feed"] == 0]

"""320 horas não tem 180 observações:

*  **10/03/2027 01:00:00**: faltam 6 observações, pode ser devido ao fato ser a  primeira hora inserida no dataset, 2 min (6 obs * 20seg) de dados dessa primeira hora não foram inseridos. Esta data será desconsiderada do dataset.

*  **16/03/2017 06:00:00 até 29/03/2017 11:00:00**: sem observações. Com base na documentação do Kaggle, isso foi causado por uma parada na produção indústrial. Logo os dias anteriores à parada de operação serão desconsiderados.

*  **10/04/2017 00:00:00**: falta 1 observação, como é pouco comparado com o total de 179, será realizada a média por hora desconsiderando que falta essa observação.
"""

# Definindo tema e uma paleta de cores para os gráficos
sns.set_theme(style="dark")
paleta = sns.color_palette('crest')

# Plotando gráfico para vusualizar a parada de operação
plt.figure(figsize=(15,5))
plt.plot(df['% Silica Concentrate']['2017-03-15 00:00:00':'2017-03-30 00:00:00'],'s', label='% Sílica no concentrado', color=paleta[2])
plt.title('Parada de Operação', fontsize=14)
plt.ylabel('Sílica Concentrado (%)', fontsize=12)
plt.grid(color='white')
plt.legend(fontsize=10)
plt.show()

# Removendo as datas de antes da parada
df = df.loc["2017-03-29 12:00:00":]
df.head()

# Porcentagem de dados perdidos
print(f"Retirada dos dados anteriores à parada do processo ocasionou redução de apenas {round((100 - (df.shape[0]/737453 * 100)), 2)}% das linhas")

# Valores únicos por hora
unicos_hora = df.groupby(pd.Grouper(freq='H')).nunique().mean()

# Plotando gráfico para melhor visualização
plt.figure(figsize=(14,6))
sns.lineplot(x = unicos_hora.index, y = unicos_hora.values, color='gray')
ax = sns.barplot(x = unicos_hora.index, y = unicos_hora.values, palette="crest")
plt.title('Número Médio de Valores Únicos por Hora', fontsize=14)
plt.ylabel('Contagem Média', fontsize=12)
plt.xticks(rotation=90);

for i in ax.containers:
        labels = [f"{round(h,0):.0f}" if (h := v.get_height()) > 0 else '' for v in i]
        ax.bar_label(i, labels=labels, label_type='edge')

"""Pelo gráfico acima comprova-se o que o autor do dataset no kaggle informou que algumas features foram amostradas a cada 20 segundos e outras a cada hora.

*  As colunas 3 a 21 são as amostradas a cada 20 segundos. O valor não ser exatamente igual a 180 é normal, já que podem existir valores que se repetiram e aqui só são considerados os valores únicos.

*  A colunas das variáveis de entrada "% Iron Feed" e "% Silica Feed" estão coerentes com terem amostras únicas a cada hora.

*  Já as variáveis de saída "% Iron Concentrate" e "% Silica Concentrate" paracem serem amostradas 11 e 15 vezes respectivamente a cada hora, vamos analisar mais a fundo os valores únicos para essas variáveis
"""

# Valores únicos por hora Ferro concentrado
unicos_hora_ferro_conc = df.groupby(pd.Grouper(freq='H'))['% Iron Concentrate'].nunique().value_counts()
unicos_hora_ferro_conc

# Valores únicos por hora Sílica concentrado
unicos_hora_silica_conc = df.groupby(pd.Grouper(freq='H'))['% Silica Concentrate'].nunique().value_counts()
unicos_hora_silica_conc

# Porcentagem de horas com valor único
print(f"{round(((unicos_hora_ferro_conc[1]/unicos_hora_ferro_conc.sum()) * 100), 2)}% do total de horas do dataset tem 1 valor único de amostra de Ferro concentrado")
print(f"{round(((unicos_hora_silica_conc[1]/unicos_hora_silica_conc.sum()) * 100), 2)}% do total de horas do dataset tem 1 valor único de amostra de Sílica concentrado")

# Analisando horas que não tem valores únicos
unico_hora = df.groupby(pd.Grouper(freq='H'))['% Silica Concentrate'].nunique()
unico_hora[unico_hora != 1].index

# Analisando um dia/hora específica com 180 observações
hora_mais_obs = df['% Silica Concentrate'].loc[(df.index.day==30) & (df.index.month==3) & (df.index.hour >= 19) & (df.index.hour <= 21)]
hora_mais_obs

# Trocando index para frequência de 20seg
novo_index = pd.Series(pd.date_range(start='2017-03-30 19:00:00', end='2017-03-30 21:59:40', freq='20S'))
hora_mais_obs.index = novo_index.values
hora_mais_obs

# Plotando um dos horários divergentes para verificação

plt.figure(figsize=(14,5))
hora_mais_obs.plot(label='%  Silica no concentrado', color=paleta[2])
plt.title('Intervalo com Variação de Amostragem', fontsize=14)
plt.ylabel('Concentrado (%)', fontsize=12)
plt.legend()
plt.grid(color='white')
plt.show()

"""Há uma linha reta contínua entre os pontos de dados, sem saltos ou variações abruptas nos valores ao longo do tempo, é provável que os valores tenham sido interpolados.
Talvez esses dados verdadeiros não puderam ser coletados e analisados, por isso foi feita a interpolação.
Não há problema em seguir com eles no dataset
"""

# Valores únicos por dia
unicos_hora = df.groupby(pd.Grouper(freq='D')).nunique().mean()

# Plotando gráfico para melhor visualização
sns.set_theme(style="dark")
plt.figure(figsize=(14,6))
sns.lineplot(x = unicos_hora.index, y=unicos_hora.values, color='gray')
ax = sns.barplot(x = unicos_hora.index, y=unicos_hora.values,palette='crest')
plt.title('Número Médio de Valores Únicos por Dia')
plt.ylabel('Contagem Média')
plt.xticks(rotation=90);

for i in ax.containers:
        labels = [f"{round(h,0):.0f}" if (h := v.get_height()) > 0 else '' for v in i]
        ax.bar_label(i, labels=labels, label_type='edge')
plt.show()

"""As colunas que se referem à qualidade do material de alimentação (% Silica Feed e % Iron Feed) são atualizadas em média 3 vezes por dia, o que equivale a uma vez a cada 8 horas.

Entretanto, tendo em vista a amostragem da variável de predição % Silica Concentrate ser a cada uma hora e considerando que o resultado final de qualidade do minério após o processo de flotação depende do comportamento de toda operação, os dados serão considerados com frequência de uma hora
"""

# Agrupando a data de acordo com as horas e pegando a média dos valores
df = df.groupby(pd.Grouper(freq='H')).mean()
df

""" 3. Analisando as correlações"""

# Distribuição de frequencia todas as variáveis

plt.figure(figsize=(20,15),dpi=200)

for i , n in enumerate(df.columns.to_list()):
    plt.subplot(6,4,i+1)
    ax = sns.histplot(data=df,x=n, kde=False, bins=20, color=paleta[4])
    plt.title(f"Histograma {n}", fontdict={"fontsize":14})
    plt.xlabel("")
    plt.ylabel(ax.get_ylabel(), fontdict={"fontsize":12})
    if i not in [0,4,8,12,16,20,24]:
        plt.ylabel("")
plt.tight_layout();

# Criando uma figura com 2 linhas e 1 coluna de subplots
plt.figure(figsize=(15, 8))
paleta = sns.color_palette('crest')

# Primeiro subplot: Concentração de Ferro
plt.subplot(2, 1, 1)
plt.plot(df['% Iron Concentrate'], color = paleta[5])
plt.ylabel('%')
plt.title('Concentração de Ferro ao final do processo em %')

# Segundo subplot: Concentração de Sílica
plt.subplot(2, 1, 2)
plt.plot(df['% Silica Concentrate'],color = paleta[1])
plt.xlabel('Date')
plt.ylabel('%')
plt.title('Concentração de Sílica ao final do processo em %')

plt.tight_layout()
plt.show()

"""Correlação negativa entre Sílica e Ferro"""

# Análise inicial de Correlação entre as Features
plt.figure(figsize=(19, 8))
p = sns.heatmap(df.corr(), annot=True)
plt.title("Heatmap de Correlações")
plt.show();

# Correlação entre as Features em ordem descrescente
correlacao = df.corr(method='pearson')
correlacao['% Silica Concentrate'].abs().sort_values(ascending=False)

# Plotando material do minério antes e após o processo
content = ['% Iron Concentrate','% Iron Feed', '% Silica Feed','% Silica Concentrate']
palette = [paleta[5], "#d74b53", paleta[1], "#ff8877"]


fig, ax = plt.subplots(figsize=(18,6))
for pct, color in zip(content, palette):
    ax.plot(df.index.values, pct, data=df, color=color)
ax.set_title('Conteúdo do Mínério na Alimentação e no Concentrado')
ax.set_ylabel('% Minério')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='center left')
plt.show()

"""4. Analisando features agrupadas por tempo"""

# Criando novas features para agrupamento
df_tempo = df.copy()
df_tempo['Hour'] = pd.Series(df_tempo.index).dt.hour.values
df_tempo['Day'] = pd.Series(df_tempo.index).dt.day.values
df_tempo['Day of week'] = pd.Series(df_tempo.index).dt.dayofweek.values
df_tempo['Month'] = pd.Series(df_tempo.index).dt.month.values
df_tempo

# Qual são as horas do dia que em média tem melhores resultados para qualidade de minério (menos impureza)?

# Agrupando por hora
agrupado_hora = df_tempo.groupby('Hour').mean()

# Plotando os grupos horas do dia
ax = sns.barplot(data=agrupado_hora, x=agrupado_hora.index, y='% Silica Concentrate', color = paleta[2])
ax.figure.set_size_inches(15, 8)
ax.set_title('Média da % de Impureza Final por Hora do Dia')
ax.set_xlabel('Horas do dia')
ax.set_ylabel('% Sílica no Concentrado')

# Definindo as cores para as menores colunas
cores = [paleta[2] if index in agrupado_hora.nsmallest(3, '% Silica Concentrate').index else paleta[3] for index in agrupado_hora.index]

# Adicionando valores nas 3 menores colunas
for i in agrupado_hora.nsmallest(3, '% Silica Concentrate').index:
    ax.text(i, agrupado_hora.loc[i, '% Silica Concentrate'] + 0.1, f'{agrupado_hora.loc[i, "% Silica Concentrate"]:.2f}', ha='center')

# Defina as cores das barras
for bar, cor in zip(ax.patches, cores):
    bar.set_facecolor(cor)

plt.show();

# Analisando as horas com menores impurezas
agrupado_hora['% Silica Concentrate'].abs().sort_values().head(5)

# Qual é o dia da semana que em média tem melhores resultados para qualidade de minério (menos impureza)?

# Agrupando por dia da semana
agrupado_dia_sem = df_tempo.groupby('Day of week').mean()

# plotando os grupos dia da semana
ax = sns.barplot(data=agrupado_dia_sem,x=agrupado_dia_sem.index, y='% Silica Concentrate', color = paleta[2])
ax.figure.set_size_inches(15, 8)
ax.set_title('Média da % de Impureza Final por Dia da Semana')
ax.set_xlabel('Dias da Semana')
ax.set_ylabel('% Sílica no Concentrado')
plt.xticks(ticks=agrupado_dia_sem.index, labels=['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'])

# Definindo as cores para menores colunas
cores = [paleta[2] if index in agrupado_dia_sem.nsmallest(1, '% Silica Concentrate').index else paleta[3] for index in agrupado_dia_sem.index]

# Definindo as cores das barras
for bar, cor in zip(ax.patches, cores):
    bar.set_facecolor(cor)

plt.show();

# Analisando dias da Semana com menos impureza
agrupado_dia_sem['% Silica Concentrate'].abs().sort_values().head(5)

# Qual é o dia do mês que em média tem melhores resultados para qualidade de minério (menos impureza)?

# Agrupando por dia do mês
agrupado_dia_mes = df_tempo.groupby('Day').mean()

# plotando os grupos dia do mês
ax = sns.barplot(data=agrupado_dia_mes,x=agrupado_dia_mes.index, y='% Silica Concentrate', color = paleta[2])
ax.figure.set_size_inches(15, 8)
ax.set_title('Média da % de Impureza Final por Dia do Mês')
ax.set_xlabel('Dias do Mês')
ax.set_ylabel('% Sílica no Concentrado')

# Definindo as cores para menores colunas
cores = [paleta[2] if index in agrupado_dia_mes.nsmallest(3, '% Silica Concentrate').index else paleta[3] for index in agrupado_dia_mes.index]

# Definindo as cores das barras
for bar, cor in zip(ax.patches, cores):
    bar.set_facecolor(cor)

plt.show();

# Analisando dias do mês com menos impureza
agrupado_dia_mes['% Silica Concentrate'].abs().sort_values().head(5)

# Qual é o mês que em média tem melhores resultados para qualidade de minério (menos impureza)?

# Agrupando por mês
agrupado_mes = df_tempo.groupby('Month').mean()

# plotando os grupos por mês
ax = sns.barplot(data=agrupado_mes,x=agrupado_mes.index, y='% Silica Concentrate', color = paleta[2])
ax.figure.set_size_inches(10, 8)
ax.set_title('Média da % de Impureza Final por Mês')
ax.set_xlabel('Meses')
ax.set_ylabel('% Sílica no Concentrado')

# Definindo as cores para as menores colunas
cores = [paleta[2] if index in agrupado_mes.nsmallest(3, '% Silica Concentrate').index else paleta[3] for index in agrupado_mes.index]

# Definindo as cores das barras
for bar, cor in zip(ax.patches, cores):
    bar.set_facecolor(cor)

plt.show();

# Analisando dias do mês com menos impureza
agrupado_mes['% Silica Concentrate'].abs().sort_values().head(5)

# Meses com melhor qualidade de minério na alimentação coincide com meses com melhores resultados para qualidade pós processo (menos impureza) ?

# Qual é o mês que em média tem melhores resultados para qualidade de minério (menos impureza)?

# plotando os grupos do mês para % ferro na alimentação
ax = sns.barplot(data=agrupado_mes,x=agrupado_mes.index, y='% Iron Feed', color = paleta[2])
ax.figure.set_size_inches(10, 8)
ax.set_title('Média da % de Ferro na Alimentação por Mês')
ax.set_xlabel('Meses')
ax.set_ylabel('% Ferro na Alimentação')

# Definindo as cores para as menores colunas
cores = [paleta[2] if index in agrupado_mes.nlargest(3, '% Iron Feed').index else paleta[3] for index in agrupado_mes.index]

# Definindo as cores das barras
for bar, cor in zip(ax.patches, cores):
    bar.set_facecolor(cor)

plt.show();

# Analisando meses com melhor qualidade minério na alimentação
agrupado_mes['% Iron Feed'].abs().sort_values().head(5)

"""Só mês de junho coincidiu

Análise Temporal

Decomposição da variável target em seus componentes de tendência, sazonalidade e ruído.
Observamos que ela não possui sazonalidade nem mudanças de tendência
"""

# Decompondo série temporal da variável target

resultado = seasonal_decompose(df["% Silica Concentrate"])

plt.figure(figsize=(15,10))
plt.suptitle('Decomposição Temporal da % Sílica no Concentrado')
plt.subplot(311)
plt.plot(resultado.trend, label='Tendência', color = paleta[3])
plt.legend()
plt.subplot(312)
plt.plot(resultado.seasonal,label='Sazonalidade', color = paleta[3])
plt.legend()
plt.subplot(313)
plt.plot(resultado.resid, label='Ruído', color = paleta[3])
plt.legend()
plt.tight_layout()
plt.show();

"""Dados bem ruidosos e sem uma tendência clara. A sasionalidade parace haver em um espaço de tempo menor. A seguir serão analisadas apenas os primeiros 4 dias"""

# Sazonalidade

ax = sns.lineplot(data=resultado.seasonal[:96], color = paleta[3])
ax.figure.set_size_inches(25, 10)
ax.set_title('Sazonalidade diária da % Sílica no Concentrado', fontsize = 20)
ax.set_ylabel('')
ax.set_xlabel('Horas')

ax.plot()

""" Modelagem

Divisão - Treino e Teste

Nesse momento, faremos a divisão do dataset, para que possamos submetê-lo ao treinamento e teste, sem que o nosso resultado seja contaminado com informações indesejadas. A % Ferro Concetrado foi retirada nas features porque está muito correlacionado com a % Sílica no Concentrado. Será dividido 80 % para treino e 20% para teste
"""

# Definindo o nosso X e y a ser trabalhado
X = df.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])
y = df['% Silica Concentrate']

# Separando os dados para treino e teste para X e y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

""" Feature Scaling

Uma observação importante é que é necessáriofazer o Scaling no projeto após o dataset ter sido dividido. Isso porque no contrário, pode haver vazamento de dados de X para o nosso target, o que tornaria a análise viciada.
"""

# Aplicando o scaler
scaler_train = StandardScaler()
X_train_std = scaler_train.fit_transform(X_train)
X_test_std = scaler_train.fit_transform(X_test)

""" Uso da Árvore de Decisão Regressão

Como estamos tratando de uma saída contínua, trabalharemos com um algorítmo de REGGRESSÃO. Será usada uma ÁRVORE DE DECISÃO.
"""

# Criando o modelo
modelo = DecisionTreeRegressor(random_state = 42)

# Fazendo o Fit dos dados no Modelo
modelo.fit(X_train_std, y_train)

# Criando a predição do modelo
y_pred_train = modelo.predict(X_train_std)
y_pred_test = modelo.predict(X_test_std)

# Analisando o resultado
print('Erro médio absoluto (MAE) treino:', round(mean_absolute_error(y_train, y_pred_train),3))
print('Erro médio absoluto (MAE) teste:', round(mean_absolute_error(y_test, y_pred_test),3))
print('Erro quadrático médio (MSE) treino:', round(mean_squared_error(y_train, y_pred_train),3))
print('Erro quadrático médio (MSE) teste:', round(mean_squared_error(y_test, y_pred_test),3))
print('Raiz quadrada do erro quadrático médio (RMSE) treino:', round(mean_squared_error(y_train, y_pred_train, squared = False),3))
print('Raiz quadrada do erro quadrático médio (RMSE) teste:', round(mean_squared_error(y_test, y_pred_test, squared= False),3))

np.mean(y_train)

np.mean(y_test)

# Preparando dados para fazer uma curva de aprendizado
def aprendizado_profundidade(range_profundidade, X_train, X_test, y_train, y_test):
    erros_train = []
    erros_test = []
    for profundidade in range_profundidade:
        modelo = DecisionTreeRegressor(random_state=42, max_depth=profundidade)

        X_train_temp = X_train
        X_test_temp = X_test

        modelo.fit(X_train_temp, y_train)

        y_pred_train = modelo.predict(X_train_temp)
        y_pred_test = modelo.predict(X_test_temp)

        erros_train.append(mean_absolute_error(y_train, y_pred_train))
        erros_test.append(mean_absolute_error(y_test, y_pred_test))

    return range_profundidade, erros_train, erros_test

# Retornando valores para construção do gráfico
range_profundidade, erros_train, erros_test = aprendizado_profundidade(list(range(1, 30)), X_train_std, X_test_std, y_train, y_test)

# Plotando o gráfico que demonstra os erros para treino e teste
plt.figure(figsize=(10, 6))
plt.plot(range_profundidade, erros_test, 'o--', color='r', label='Erro de Teste')
plt.plot(range_profundidade, erros_train, 'o--', color='g', label='Erro de Treino')
plt.title('Curva de Aprendizado', fontsize=14)
plt.ylabel('Erro', fontsize=12)
plt.xlabel('Profundidade', fontsize=12)
plt.grid(color='white')
plt.legend(fontsize=6)
plt.show;

""" Otimização de Hiperparâmetro"""

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DecisionTreeRegressor())
])

parametros = {
    'model__min_samples_leaf':[2, 3, 4, 5, 6],
    'model__max_depth': [None, 3, 4, 6, 8, 10, 12],
    'model__max_leaf_nodes':[3, 4, 5, 6],
}

grid = GridSearchCV(estimator=pipe, param_grid=parametros, scoring='neg_root_mean_squared_error', cv=4)

grid.fit(X_train_std, y_train)

grid.best_params_

modelo_otimizado = DecisionTreeRegressor(criterion = 'squared_error', max_depth = 3, max_leaf_nodes = 6, min_samples_leaf = 2)

modelo_otimizado.fit(X_train_std, y_train)

# Criando a predição do modelo
y_pred_train = modelo_otimizado.predict(X_train_std)
y_pred_test = modelo_otimizado.predict(X_test_std)

# Analisando o resultado
print('Erro médio absoluto (MAE) treino:', round(mean_absolute_error(y_train, y_pred_train),3))
print('Erro médio absoluto (MAE) teste:', round(mean_absolute_error(y_test, y_pred_test),3))
print('Erro quadrático médio (MSE) treino:', round(mean_squared_error(y_train, y_pred_train),3))
print('Erro quadrático médio (MSE) teste:', round(mean_squared_error(y_test, y_pred_test),3))
print('Raiz quadrada do erro quadrático médio (RMSE) treino:', round(mean_squared_error(y_train, y_pred_train, squared = False),3))
print('Raiz quadrada do erro quadrático médio (RMSE) teste:', round(mean_squared_error(y_test, y_pred_test, squared= False),3))

def aprendizado_profundidade_otimizado(range_profundidade, X_train, X_test, y_train, y_test):
    erros_train = []
    erros_test = []
    for profundidade in range_profundidade:
        modelo_otimizadog = DecisionTreeRegressor(random_state=42, max_depth = profundidade, max_leaf_nodes = 6, min_samples_leaf = 2)

        X_train_temp = X_train
        X_test_temp = X_test

        modelo_otimizadog.fit(X_train_temp, y_train)

        y_pred_train = modelo_otimizadog.predict(X_train_temp)
        y_pred_test = modelo_otimizadog.predict(X_test_temp)

        erros_train.append(mean_absolute_error(y_train, y_pred_train))
        erros_test.append(mean_absolute_error(y_test, y_pred_test))

    return range_profundidade, erros_train, erros_test

# Retornando valores para construção do gráfico
range_profundidade, erros_train, erros_test = aprendizado_profundidade_otimizado(list(range(1, 30)), X_train_std, X_test_std, y_train, y_test)

# Plotando o gráfico que demonstra os erros para treino e teste
plt.figure(figsize=(10, 6))
plt.plot(range_profundidade, erros_test, 'o--', color='r', label='Erro de Teste')
plt.plot(range_profundidade, erros_train, 'o--', color='g', label='Erro de Treino')
plt.title('Curva de Aprendizado após Otimização de Hiperpâmetros', fontsize=14)
plt.ylabel('Erro', fontsize=12)
plt.xlabel('Profundidade', fontsize=12)
plt.grid(color='white')
plt.legend(fontsize=6)
plt.show();

""" Random Forest"""

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

parametro_rf = {'n_estimators':[20,50,100],
         'max_depth':[3, 4, 5],
         'min_samples_leaf':[2, 3, 4]}

grid = GridSearchCV(estimator = rf_model, param_grid = parametro_rf, cv = 5, n_jobs = -1) #

grid.fit(X_train_std, y_train)

grid.best_params_

rf_model_otimizado = RandomForestRegressor(max_depth = 5, min_samples_leaf = 3, n_estimators = 100, random_state=42)

rf_model_otimizado.fit(X_train_std, y_train)

# Criando a predição do modelo
y_pred_train = rf_model_otimizado.predict(X_train_std)
y_pred_test = rf_model_otimizado.predict(X_test_std)

# Analisando o resultado
print('Erro médio absoluto (MAE) treino:', round(mean_absolute_error(y_train, y_pred_train),3))
print('Erro médio absoluto (MAE) teste:', round(mean_absolute_error(y_test, y_pred_test),3))
print('Erro quadrático médio (MSE) treino:', round(mean_squared_error(y_train, y_pred_train),3))
print('Erro quadrático médio (MSE) teste:', round(mean_squared_error(y_test, y_pred_test),3))
print('Raiz quadrada do erro quadrático médio (RMSE) treino:', round(mean_squared_error(y_train, y_pred_train, squared = False),3))
print('Raiz quadrada do erro quadrático médio (RMSE) teste:', round(mean_squared_error(y_test, y_pred_test, squared= False),3))