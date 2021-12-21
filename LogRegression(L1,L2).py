import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression

# зафиксируем значение генератора случайных чисел для воспроизводимости 
SEED = 1

# Функции, которые в дальнейшем понадобятся
def plot_features_scores(model, data, target, column_names):
    '''Функция для визуализации важности признаков'''
    
    model.fit(data, target)

    pd.DataFrame(data={'score': model['lr'].coef_[0]/sum(model['lr'].coef_[0])},
                 index=column_names) .sort_values(by='score').plot(kind='barh',grid=True,figsize=(10,6), legend=False)
        

def grid_search(model, gs_params):
    '''Функция для подбора гиперпараметров с помощью перекрёстной проверки'''
     
    gs = GridSearchCV(estimator=model, param_grid=gs_params, refit=True,
                      scoring='roc_auc', n_jobs=-1, cv=skf, verbose=0)
    gs.fit(X, y)
    scores = [gs.cv_results_[f'split{i}_test_score'][gs.best_index_] for i in range(skf.n_splits)]
    print('best params = {}'.format(gs.best_params_))
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

lr1 = Pipeline([('p_trans', PowerTransformer(method='yeo-johnson', standardize=True)),
               ('lr', LogisticRegression(solver='liblinear',
                                         penalty='l1',
                                         max_iter=200,
                                         class_weight='balanced',
                                         random_state=SEED)
               )])
# параметры кросс-валидации (стратифицированная 5-фолдовая с перемешиванием) 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

dict = cross_validate(estimator=lr1, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))
# важность признаков
plot_features_scores(model=lr1, data=X, target=y, column_names=X.columns)
print(pd.DataFrame(data={'score': lr1['lr'].coef_[0]/sum(lr1['lr'].coef_[0])},
                 index=X.columns) .sort_values(by='score',ascending = False))
plt.show()
lr2 = Pipeline([('p_trans', PowerTransformer(method='yeo-johnson', standardize=True)),
               ('lr', LogisticRegression(solver='liblinear',
                                         penalty='l2',
                                         max_iter=200,
                                         class_weight='balanced',
                                         random_state=SEED)
               )])
dict = cross_validate(estimator=lr2, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))

plot_features_scores(model=lr2, data=X, target=y, column_names=X.columns)
print(pd.DataFrame(data={'score': lr2['lr'].coef_[0]/sum(lr2['lr'].coef_[0])},
                 index=X.columns) .sort_values(by='score',ascending = False))
plt.show()
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
plt.show()
dict = cross_validate(estimator=lr1, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))

plot_features_scores(model=lr1, data=X, target=y, column_names=X.columns)
print(pd.DataFrame(data={'score': lr1['lr'].coef_[0]/sum(lr1['lr'].coef_[0])},
                 index=X.columns) .sort_values(by='score',ascending = False))
plt.show()
dict = cross_validate(estimator=lr2, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))


plot_features_scores(model=lr2, data=X, target=y, column_names=X.columns)
print(pd.DataFrame(data={'score': lr2['lr'].coef_[0]/sum(lr2['lr'].coef_[0])},
                 index=X.columns) .sort_values(by='score',ascending = False))
plt.show()

lr1_params = {'lr__C': np.arange(0.001,0.02,0.001)}
lr2_params = {'lr__C': np.arange(0.001,0.02,0.0001)}
             
print('grid search results for lr')
lr1_grid = grid_search(model=lr1, gs_params=lr1_params)
lr2_grid = grid_search(model=lr2, gs_params=lr2_params)

dict = cross_validate(estimator=lr1_grid, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))

dict = cross_validate(estimator=lr2_grid, X=X, y=y, 
                         cv=skf, scoring='roc_auc', n_jobs=-1,return_train_score=True)
test_score = dict['test_score'];
train_score = dict['train_score'];
print('train scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(train_score, train_score.mean(), train_score.std()))
print('test score = {} \nmean score = {:.5f} +/- {:.5f}'.format(test_score, test_score.mean(), test_score.std()))

plot_features_scores(model=lr1_grid.best_estimator_, 
                     data=X, target=y, column_names=X.columns)
plt.show()
plot_features_scores(model=lr2_grid.best_estimator_, 
                     data=X, target=y, column_names=X.columns)
plt.show()
lr1_selector = SelectFromModel(estimator=lr1_grid.best_estimator_['lr'], prefit=True, threshold=0.1)

# посмотрим выбранные признаки
print(pd.DataFrame(data={'score':lr1_selector.estimator.coef_[0]/sum(lr1_selector.estimator.coef_[0]),
                   'support':lr1_selector.get_support()}, 
             index=X.columns).sort_values(by='score',ascending=False))

lr2_selector = SelectFromModel(estimator=lr2_grid.best_estimator_['lr'], prefit=True, threshold=0.1)

# посмотрим выбранные признаки
print(pd.DataFrame(data={'score':lr2_selector.estimator.coef_[0]/sum(lr2_selector.estimator.coef_[0]),
                   'support':lr2_selector.get_support()}, 
             index=X.columns).sort_values(by='score',ascending=False))
