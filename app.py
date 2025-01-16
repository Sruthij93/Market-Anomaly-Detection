import pandas as pd
import numpy as np
import pickle
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

# load all pickle files
with open('/Users/sruthi/Documents/My projects/HEADSTARTER/Anomaly Detection/trained_models/isoforest_model.pkl', 'rb') as file:
  isoforest_model = pickle.load(file)
with open('/Users/sruthi/Documents/My projects/HEADSTARTER/Anomaly Detection/trained_models/svm_model.pkl', 'rb') as file:
  svm_model = pickle.load(file)
with open('/Users/sruthi/Documents/My projects/HEADSTARTER/Anomaly Detection/trained_models/voting_clf_model.pkl', 'rb') as file:
  voting_clf = pickle.load(file)
with open('/Users/sruthi/Documents/My projects/HEADSTARTER/Anomaly Detection/data_preprocessors/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
