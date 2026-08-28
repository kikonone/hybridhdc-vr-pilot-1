from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import scipy, sklearn, joblib, threadpoolctl, matplotlib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'experiments/phase_04a_traditional_classification_baselines/configs/phase04a_environment.json'
X=np.array([[0.,1.],[1.,0.],[0.,0.],[1.,1.],[.2,.8],[.8,.2]]); y=np.array([0,1,0,1,0,1])
pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('variance',VarianceThreshold()),('scaler',StandardScaler()),('selector',SelectKBest(k='all')),('model',LogisticRegression(max_iter=1000,random_state=42))])
pipe.fit(X,y); pred=pipe.predict(X)
result={'python_executable':sys.executable,'python_version':sys.version,'PYTHONNOUSERSITE':__import__('os').environ.get('PYTHONNOUSERSITE'),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__,'matplotlib':matplotlib.__version__,'jupyter_kernel':'python3 / Python 3 (ipykernel); launch with PYTHONNOUSERSITE=1','environment_recovery_reason':'USER_SITE_PACKAGE_OVERLAY_CONFLICT','package_changes':'NO PACKAGE INSTALLATION OR UPGRADE PERFORMED','import_validation':'PASS','software_smoke_test':'PASS','synthetic_macro_f1':f1_score(y,pred,average='macro'),'synthetic_balanced_accuracy':balanced_accuracy_score(y,pred),'note':'Synthetic software validation only; not Phase 04A thesis evidence.'}
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
