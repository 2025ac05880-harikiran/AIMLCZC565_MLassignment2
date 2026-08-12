import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title='Customer Personality Analysis Classification', page_icon='📊', layout='wide')

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODELS = {
    'Logistic Regression': ['logistic_regression.pkl','logistic_regression.joblib','logistic_regression_model.pkl','logistic.pkl','lr.pkl'],
    'Decision Tree': ['decision_tree.pkl','decision_tree.joblib','decision_tree_model.pkl','decisiontree.pkl','dt.pkl'],
    'K-Nearest Neighbor (KNN)': ['knn.pkl','knn.joblib','knn_model.pkl','k_nearest_neighbor.pkl','knearestneighbor.pkl'],
    'Gaussian Naive Bayes': ['naive_bayes.pkl','naive_bayes.joblib','naive_bayes_model.pkl','gaussian_naive_bayes.pkl','gaussian_nb.pkl','gnb.pkl','nb.pkl'],
    'Random Forest (Ensemble)': ['random_forest.pkl','random_forest.joblib','random_forest_model.pkl','random_forest_classifier.pkl','random_forest_ensemble.pkl','randomforest.pkl','rf.pkl']
}

@st.cache_resource
def load_obj(path):
    return joblib.load(path)

def find_model(name):
    if not MODEL_DIR.exists(): return None
    for f in MODELS[name]:
        p = MODEL_DIR / f
        if p.is_file(): return p
    files = [p for p in MODEL_DIR.rglob('*') if p.is_file() and p.suffix.lower() in ('.pkl','.joblib')]
    for p in files:
        if p.name.lower() in {x.lower() for x in MODELS[name]}: return p
    keys = {
        'Logistic Regression':['logistic'], 'Decision Tree':['decision_tree','decisiontree'],
        'K-Nearest Neighbor (KNN)':['knn','nearest'], 'Gaussian Naive Bayes':['naive','gaussian','gnb'],
        'Random Forest (Ensemble)':['random_forest','randomforest','rf']
    }
    for p in files:
        s=p.stem.lower().replace('-','_')
        if any(k in s for k in keys[name]): return p
    return None

def find_aux(names):
    if not MODEL_DIR.exists(): return None
    files=[p for p in MODEL_DIR.rglob('*') if p.is_file() and p.suffix.lower() in ('.pkl','.joblib')]
    for n in names:
        for p in files:
            if p.name.lower()==n.lower() or Path(n).stem.lower() in p.stem.lower(): return p
    return None

def prepare_x(df, model):
    X=df.drop(columns=['Response'], errors='ignore').copy()
    X=X.drop(columns=[c for c in X.columns if str(c).lower() in ('unnamed: 0','index')], errors='ignore')
    # A saved sklearn Pipeline should perform its own preprocessing.
    if hasattr(model,'steps') or hasattr(model,'named_steps'):
        return X
    # Prefer the exact feature list saved by train_models.py.
    fp=find_aux(['feature_columns.pkl','feature_columns.joblib','features.pkl','features.joblib','columns.pkl'])
    feature_cols=None
    if fp:
        try:
            obj=load_obj(str(fp))
            if isinstance(obj,dict): obj=obj.get('feature_columns',obj.get('features',obj.get('columns')))
            if isinstance(obj,(list,tuple,np.ndarray,pd.Index)): feature_cols=list(obj)
        except Exception: pass
    cat=X.select_dtypes(include=['object','category','bool']).columns.tolist()
    if cat: X=pd.get_dummies(X, columns=cat, drop_first=True, dtype=float)
    if feature_cols: X=X.reindex(columns=feature_cols, fill_value=0)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors='coerce')
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0)
    sp=find_aux(['scaler.pkl','scaler.joblib','standard_scaler.pkl','standard_scaler.joblib','feature_scaler.pkl'])
    if sp:
        scaler=load_obj(str(sp)); X=scaler.transform(X)
    return X

def evaluate(name, df):
    path=find_model(name)
    if path is None: raise FileNotFoundError(f'No trained model file found for {name} in {MODEL_DIR}.')
    model=load_obj(str(path))
    X=prepare_x(df,model)
    y=pd.to_numeric(df['Response'],errors='coerce')
    valid=y.notna().to_numpy(); y=y[valid].astype(int).to_numpy()
    if hasattr(X,'iloc'): X=X.iloc[valid]
    else: X=X[valid]
    pred=np.asarray(model.predict(X)).astype(int)
    score=None
    if hasattr(model,'predict_proba'):
        try:
            p=model.predict_proba(X); score=p[:,1] if p.ndim==2 and p.shape[1]>1 else None
        except Exception: pass
    if score is None and hasattr(model,'decision_function'):
        try: score=model.decision_function(X)
        except Exception: pass
    auc=np.nan
    if score is not None and len(np.unique(y))==2:
        auc=roc_auc_score(y,score)
    return dict(model=name,path=path,accuracy=accuracy_score(y,pred),auc=auc,precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),mcc=matthews_corrcoef(y,pred),pred=pred,score=score,y=y,cm=confusion_matrix(y,pred,labels=[0,1]))

def observation(r,winner=False):
    s=(f"{r['ML Model Name']} achieved Accuracy {r['Accuracy']:.3f}, AUC "
       f"{r['AUC Score']:.3f}, Precision {r['Precision']:.3f}, Recall {r['Recall']:.3f}, "
       f"F1 Score {r['F1 Score']:.3f}, and MCC {r['MCC Score']:.3f}. ")
    if r['Precision']>r['Recall']+0.10: s+='Precision is notably higher than Recall, indicating a more conservative positive prediction pattern.'
    elif r['Recall']>r['Precision']+0.10: s+='Recall is notably higher than Precision, indicating stronger detection of positive cases with more false positives.'
    else: s+='Precision and Recall are relatively balanced.'
    if winner: s+=' This model has the highest F1 Score among the successfully evaluated models.'
    return s

st.title('📊 Customer Personality Analysis Classification')
st.subheader('ML Assignment 2')
st.write("This application predicts whether a customer will accept a company's marketing campaign offer using five classification models.")
st.info('**Target Variable:** `Response` — 1 = accepted the campaign offer; 0 = did not accept the campaign offer.')

st.sidebar.header('1. Upload Test Data')
up=st.sidebar.file_uploader('Upload test_data.csv',type=['csv'])
st.sidebar.header('2. Select Model')
selected=st.sidebar.selectbox('Choose a Classification Model',list(MODELS.keys()))
st.sidebar.divider(); st.sidebar.subheader('Expected CSV Format'); st.sidebar.write('Upload a CSV with the predictor columns used during training and a `Response` column containing 0 and 1.')

if up is None:
    st.markdown('## 👈 Please upload `test_data.csv`')
    st.write('Performance metrics, observations, winner selection and predictions are generated only after the CSV is uploaded. No metric values are hardcoded.')
    with st.expander('🔧 Model File Status', expanded=True):
        for n in MODELS:
            p=find_model(n)
            (st.success if p else st.error)(f"{'✅' if p else '❌'} {n}: {p.relative_to(MODEL_DIR) if p else 'model file not found'}")
    st.stop()

try: df=pd.read_csv(up)
except Exception as e: st.error(f'Could not read the CSV: {e}'); st.stop()
if 'Response' not in df.columns: st.error("The uploaded CSV must contain a `Response` column."); st.stop()
y=pd.to_numeric(df['Response'],errors='coerce')
if y.isna().any() or not set(y.astype(int).unique()).issubset({0,1}): st.error('`Response` must contain only 0 and 1.'); st.stop()

st.success(f'Loaded `{up.name}` — {df.shape[0]} rows × {df.shape[1]} columns.')
st.subheader('📄 Uploaded Test Data Preview'); st.dataframe(df.head(10),use_container_width=True); st.write(f'**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns')

with st.expander('🔧 Model File Status', expanded=False):
    for n in MODELS:
        p=find_model(n)
        (st.success if p else st.error)(f"{'✅' if p else '❌'} {n}: {p.relative_to(MODEL_DIR) if p else 'model file not found'}")

st.subheader('📈 Model Performance Summary')
rows=[]; results={}; errors={}
for n in MODELS:
    try:
        r=evaluate(n,df); results[n]=r
        rows.append({'ML Model Name':n,'Accuracy':r['accuracy'],'AUC Score':r['auc'],'Precision':r['precision'],'Recall':r['recall'],'F1 Score':r['f1'],'MCC Score':r['mcc']})
    except Exception as e: errors[n]=str(e); st.warning(f'⚠️ **{n}** could not be evaluated: {e}')
if not rows: st.error('None of the five models could be evaluated. Check the model folder and preprocessing files.'); st.stop()
res=pd.DataFrame(rows); metrics=['Accuracy','AUC Score','Precision','Recall','F1 Score','MCC Score']; st.dataframe(res.style.format({c:'{:.4f}' for c in metrics},na_rep='N/A'),use_container_width=True)

st.subheader('📝 Observations')
winner=res.loc[res['F1 Score'].idxmax(),'ML Model Name']
obs=pd.DataFrame([{'ML Model Name':r['ML Model Name'],'Observation about model performance':observation(r,r['ML Model Name']==winner)} for _,r in res.iterrows()])
st.dataframe(obs,use_container_width=True,hide_index=True)
st.success(f'🏆 **Overall Winner based on the highest F1 Score: {winner}**')

st.subheader(f'🤖 Evaluation: {selected}')
if selected not in results:
    st.error(f'No prediction results are available for **{selected}**.')
    if selected in errors: st.code(errors[selected])
    st.stop()
r=results[selected]
cols=st.columns(6)
for c,label,key in zip(cols,['Accuracy','AUC Score','Precision','Recall','F1 Score','MCC Score'],['accuracy','auc','precision','recall','f1','mcc']): c.metric(label, 'N/A' if pd.isna(r[key]) else f'{r[key]:.4f}')
st.subheader('📌 Confusion Matrix'); st.dataframe(pd.DataFrame(r['cm'],index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']),use_container_width=True)
st.subheader('🔍 Prediction Results')
out=df.copy(); out['Predicted_Response']=r['pred'];
if r['score'] is not None: out['Positive_Class_Probability']=r['score']
st.dataframe(out.head(100),use_container_width=True)
st.download_button('⬇️ Download Prediction Results',out.to_csv(index=False).encode('utf-8'),file_name=f"{selected.lower().replace(' ','_').replace('(','').replace(')','').replace('-','_')}_predictions.csv",mime='text/csv')

if errors:
    with st.expander('⚠️ Model Evaluation Details'):
        for n,e in errors.items(): st.write(f'**{n}:**'); st.code(e)
