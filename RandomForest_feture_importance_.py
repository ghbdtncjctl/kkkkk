
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression

# зафиксируем значение генератора случайных чисел для воспроизводимости 
SEED = 1

# Функции, которые в дальнейшем понадобятся
def plot_features_scores(model, data, target, column_names):
    '''Функция для визуализации важности признаков'''
    
    model.fit(data, target)
    
    pd.DataFrame(data={'score': model['rf'].feature_importances_}, 
                      index=column_names).sort_values(by='score').plot(kind='barh', grid=True,
                                               figsize=(10,6), legend=False)

def grid_search(model, gs_params):
    '''Функция для подбора гиперпараметров с помощью перекрёстной проверки'''
     
    gs = GridSearchCV(estimator=model, param_grid=gs_params, refit=True,
                      scoring='roc_auc', n_jobs=-1, cv=skf, verbose=0)
    gs.fit(X, y)
    scores = [gs.cv_results_[f'split{i}_test_score'][gs.best_index_] for i in range(5)]
    print('scores = {}, \nmean score = {:.5f} +/- {:.5f} \
           \nbest params = {}'.format(scores,
                                      gs.cv_results_['mean_test_score'][gs.best_index_],
                                      gs.cv_results_['std_test_score'][gs.best_index_],
                                      gs.best_params_))
    return gs
        
# загрузим данные        
df = pd.read_csv(r'..\adult.dataa.csv',sep=';')

# датасет, с которым будем работать
# оставим только численые признаки
X = df.select_dtypes(exclude=['object']).copy()
# преобразуем целевую переменную
print(df)
y = df['salary'].map({' <=50K':0, ' >50K':1}).values
print(X.head(10))
rf = Pipeline([('rf', RandomForestClassifier(n_jobs=-1, 
                                             class_weight='balanced', 
                                             random_state=SEED))])

# параметры кросс-валидации (стратифицированная 5-фолдовая с перемешиванием) 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scores = cross_val_score(estimator=rf, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1)
print('scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(scores, scores.mean(), scores.std()))

# важность признаков
plot_features_scores(model=rf, data=X, target=y, column_names=X.columns)
#plt.show()
np.random.seed(SEED)
fix, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14,5))
ax1.set_title("normal distribution")
ax2.set_title("uniform distribution")
ax3.set_title("laplace distribution")
for i in range(4):
    X.loc[:, f'norm_{i}'] = np.random.normal(loc=np.random.randint(low=0, high=10), 
                                             scale=np.random.randint(low=1, high=10), 
                                             size=(X.shape[0], 1))
    
    X.loc[:, f'unif_{i}'] = np.random.uniform(low=np.random.randint(low=1, high=4), 
                                              high=np.random.randint(low=5, high=10), 
                                              size=(X.shape[0], 1))
    X.loc[:, f'lapl_{i}'] = np.random.laplace(loc=np.random.randint(low=0, high=10), 
                                              scale=np.random.randint(low=1, high=10), 
                                              size=(X.shape[0], 1))
    # визуализирукем распределения признаков
    sns.kdeplot(X[f'norm_{i}'], ax=ax1)
    sns.kdeplot(X[f'unif_{i}'], ax=ax2)
    sns.kdeplot(X[f'lapl_{i}'], ax=ax3)

# итоговый датасет
X.head()
#plt.show()
scores = cross_val_score(estimator=rf, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1)
print('scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(scores, scores.mean(), scores.std()))
plot_features_scores(model=rf, data=X, target=y, column_names=X.columns)
#plt.show()

selector = GenericUnivariateSelect(score_func=mutual_info_classif, 
                                   mode='k_best', 
                                   param=6)
# применим к нашему датасету
selector.fit(X, y)
# метод transform вернёт массив с отобранными признаками
# выведем результаты оценки каждого признака 
print(pd.DataFrame(data={'score':selector.scores_,
                   'support':selector.get_support()}, 
             index=X.columns).sort_values(by='score',ascending=False))

selector = ('selector', GenericUnivariateSelect(score_func=mutual_info_classif, 
                                                mode='k_best'))
rf.steps.insert(0, selector)

    
#grid search
rf_params = {'selector__param': np.arange(4,6),
            'rf__max_depth': np.arange(10, 14, 2),
            'rf__max_features': np.arange(0.1, 0.4, 0.1)}
print('grid search results for rf')
rf_grid = grid_search(model = rf, gs_params= rf_params)
# выведем признаки, отобранные селектором
selected_features = [X.columns[i] for i, support
                     in enumerate(rf_grid.best_estimator_['selector'].get_support()) if support]
plot_features_scores(model=rf_grid.best_estimator_, 
                     data=X, target=y, column_names=selected_features)
print(pd.DataFrame(data={'score':rf_grid.best_estimator_['rf'].feature_importances_,
                       'support':True},
                 index=selected_features).sort_values(by='score',ascending=False))

plt.show()

