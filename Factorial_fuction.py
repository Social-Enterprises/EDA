import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg

def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()+2
        ax.annotate(percentage, (x, y), size = 18)

def without_hueH(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.0f}'.format(p.get_width())
        x = p.get_x() + p.get_width()+0.3
        y = p.get_y() + p.get_height()/2+0.4
        ax.annotate(percentage, (x, y), size = 12)

def with_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.0f}'.format(p.get_height())
        x = p.get_x() + p.get_width() / 2 -0.05
        y = p.get_y() + p.get_height()+2
        ax.annotate(percentage, (x, y), size = 24)

def v_kramer(var1,var2):
  data=np.array(pd.crosstab(var1,var2))
  chi=stats.chi2_contingency(data, correction=False)[0]
  n=np.sum(data)
  minD=min(data.shape)-1
  V=np.sqrt((chi/n)/minD)
  return V
  
def Coding(data,scala):
  for i in data.columns:
    encoder = OrdinalEncoder(categories=[scala])
    encoder.fit((data[[i]]))
    data[i+"-encoder"]=encoder.transform(data[[i]])

def Nfactores (data):
  data=data.select_dtypes(include = "float64")
  fa = FactorAnalyzer()
  fa.fit(data)
  # Check Eigenvalues
  ev, v = fa.get_eigenvalues()
  # Create scree plot using matplotlib
  plt.scatter(range(1,data.shape[1]+1),ev)
  plt.plot(range(1,data.shape[1]+1),ev)
  plt.axhline(y=1, color='r', linestyle='--')
  plt.text(1.5,ev.max()-0.05, 'Number of factors:'+str(np.sum(ev>1)), fontsize=20, color='black')
  plt.title('Scree Plot')
  plt.xlabel('Factors')
  plt.ylabel('Eigenvalue')
  plt.grid()
  plt.show()

def validez(data):
  data=data.select_dtypes(include = "float64")
  #Alfa de cronbanch
  print('Validez de criterio: ')
  print("--------------------------------------")
  print('Alfa de cronbanch: '+str(pg.cronbach_alpha(data=data, ci=.99)[0]))
  chi_square_value,p_value=calculate_bartlett_sphericity(data)
  print("   ")
  print('Validez de constructo: ')
  print("--------------------------------------")
  # Bartlett’s Test
  print("Bartlett test: "+str(chi_square_value)+'  P-Value: '+str(p_value))
  #Kaiser-Meyer-Olkin Test
  kmo_all,kmo_model=calculate_kmo(data)
  print('KMO: '+str(kmo_model))