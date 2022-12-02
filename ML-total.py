import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('descriptor.csv')

print(df)

X = df[['Temperature','Pressure','Number','Group','Atomic_weight','e-aff-ev','Pauling_e-neg',
        'Atomic_radius','Covalent_radius','Ion-e_1st','Ion-e_2nd','Melting_point','Density','Molar_volume',
        'Thermal_conductivity',
        ]].values
y = df['TOF'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, random_state = 91)
                                          

svm = svm.SVR(C=1000.0, gamma=0.01)
rf = RandomForestRegressor(max_depth=15, n_estimators=128, random_state=1)
dtr = tree.DecisionTreeRegressor(max_depth=55)
etr = tree.ExtraTreeRegressor(max_depth=108)
gbr = GradientBoostingRegressor(max_depth=6, n_estimators=188)
knn = KNeighborsRegressor(n_neighbors=1)
lasso = linear_model.Lasso(alpha=0.1)
lr = LinearRegression()
pls = PLSRegression(n_components=10)

def model_train(model, X_train, X_test, y_train, y_test,title):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(model)
    print('training R2 =' + str(round(model.score(X_train,y_train), 3)))
    print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true = y_train,
                                                          y_pred = model.predict(X_train))))

    rmse_tr_model = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_te_model = mean_squared_error(y_test, y_pred_test, squared=False)
    print('RMSE(training)%.3f' % rmse_tr_model)
    print('RMSE(test)%.3f' % rmse_te_model)
    with open(title+'-train.txt', 'w') as f:
        f.write('training R2 =' + str(round(model.score(X_train,y_train), 3)))
        f.write('\n')
        f.write('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true = y_train,
                                                              y_pred = model.predict(X_train))))
        f.write('\n')
        f.write('RMSE(training)%.3f' % rmse_tr_model)
        f.write('\n')
        f.write('RMSE(test)%.3f' % rmse_te_model)    
    with open('RMSE-test.txt', 'a') as f:
        f.write(title+'%.3f' % rmse_te_model)
        f.write('\n')    

    plt.figure(figsize = (5,5))
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16
    plt.scatter(y_train, y_pred_train,s =90, alpha=0.8, color = 'hotpink', label = 'training')
    plt.scatter(y_test, y_pred_test, s =90, alpha=0.8, color = '#88c999', marker = 'v',label = 'test')
    plt.plot([-16,-3], [-16,-3], 'r--', color = 'black')
    plt.legend() 
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.minorticks_on()
    plt.tick_params(which='major', width=1.5, length=2)
    #plt.tick_params(which='minor', width=1.5, length=2)
    #plt.xlabel('DFT')
    #plt.ylabel('prediction')
    plt.show()
    plt.savefig(title+'.tif')
    plt.close()
    return y_pred_test, y_pred_train, rmse_tr_model, rmse_te_model

def cv_test(model,X,y,title):
    crossvalidation = KFold(n_splits = 10, shuffle = True)
    r2_scores_model = cross_val_score(model, X, y, scoring = 'r2', cv = crossvalidation)
    rmse_scores_model = cross_val_score(model, X, y, scoring = 'neg_root_mean_squared_error',
                                        cv = crossvalidation)

    print('Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(r2_scores_model), r2_scores_model.mean()))
    print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores_model), -rmse_scores_model.mean()))
    print('\n')
    with open(title+'-cross-validation.txt', 'w') as f:
        f.write('Cross-validation results:')
        f.write('\n')
        f.write('Folds: %i, mean R2: %.3f' % (len(r2_scores_model), r2_scores_model.mean()))
        f.write('\n')
        f.write('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores_model), -rmse_scores_model.mean()))
    with open('CV-R2.txt', 'a') as f:
        f.write(title+'%.3f' %  r2_scores_model.mean())
        f.write('\n') 
    with open('CV-RMSE.txt', 'a') as f:
        f.write(title+' %.3f' % -rmse_scores_model.mean())
        f.write('\n')    
    
    y_cv = cross_val_predict(model, X, y, cv=crossvalidation)
    plt.figure(figsize=(5,5))
    plt.plot([-16.5,-2.5],[-16.5,-2.5],'r--')
    plt.scatter(y, y_cv, s = 90, c = None, edgecolor = 'k', alpha = 0.7)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.show()
    plt.savefig(title+'-cv.tif',dpi=300)
    plt.close()
    return r2_scores_model, rmse_scores_model

#svm
model_train(svm, X_train, X_test, y_train, y_test,'svm')
cv_test(svm,X,y,'svm')

#rf
model_train(rf, X_train, X_test, y_train, y_test,'rf')
cv_test(rf,X,y,'rf')

#gbr
model_train(gbr, X_train, X_test, y_train, y_test,'gbr')
cv_test(gbr,X,y,'gbr')

#dtr
model_train(dtr, X_train, X_test, y_train, y_test,'dtr')
cv_test(rf,X,y,'dtr')

#etr
model_train(etr, X_train, X_test, y_train, y_test,'etr')
cv_test(etr,X,y,'etr')

#knn
model_train(knn, X_train, X_test, y_train, y_test,'knn')
cv_test(rf,X,y,'knn')

#lasso
model_train(lasso, X_train, X_test, y_train, y_test,'lasso')
cv_test(lasso,X,y,'lasso')

#lr
model_train(lr, X_train, X_test, y_train, y_test,'lr')
cv_test(lr,X,y,'lr')

#pls
model_train(pls, X_train, X_test, y_train, y_test,'pls')
cv_test(pls,X,y,'pls')
