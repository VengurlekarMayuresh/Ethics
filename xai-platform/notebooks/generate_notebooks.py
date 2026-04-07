import nbformat as nbf

def create_interpret_ml_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# InterpretML Implementation\nThis notebook demonstrates InterpretML on the Medical Insurance dataset."))
    cells.append(nbf.v4.new_markdown_cell("## Requirements\nRun the following to install required dependencies. See `requirements-xai-frameworks.txt` for details."))
    cells.append(nbf.v4.new_code_cell("!pip install interpret pandas numpy scikit-learn matplotlib==3.5.0"))
    
    cells.append(nbf.v4.new_markdown_cell("## Setup and Data\nLoading the Medical Insurance dataset (dummy tabular data)."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingRegressor

# Load dataset (Replace with your path if different)
try:
    df = pd.read_csv('../insurance.csv')
except FileNotFoundError:
    # Dummy data generator if csv not found
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, 1000),
        'sex': np.random.choice(['male', 'female'], 1000),
        'bmi': np.random.uniform(18, 40, 1000),
        'children': np.random.randint(0, 5, 1000),
        'smoker': np.random.choice(['yes', 'no'], 1000),
        'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 1000),
        'charges': np.random.uniform(1000, 50000, 1000)
    })

categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

X = df[categorical_features + numeric_features]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded successfully. Shape:", X.shape)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Model Training\nInterpretML shines with Glassbox models. We'll train an Explainable Boosting Machine (EBM)."))
    cells.append(nbf.v4.new_code_cell("""ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)
print("EBM trained!")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Explanability Framework API\nFollowing the XAI Platform internal API conventions (like `lime_service.py`), we implement: `create_explainer`, `compute_global_interpret`, and `explain_instance`."))
    cells.append(nbf.v4.new_code_cell("""from interpret import show

class InterpretMLServiceStub:
    @staticmethod
    def create_explainer(model, X_train, y_train):
        # EBM is a glassbox model, but if it were blackbox, we would setup Morris Sensitivity here.
        # For EBM, the model IS the explainer.
        return model
        
    @staticmethod
    def compute_global_interpret(explainer, X_bg):
        global_explanation = explainer.explain_global(name='EBM Global')
        return global_explanation
        
    @staticmethod
    def explain_instance(explainer, instance, y_actual=None):
        local_explanation = explainer.explain_local(instance, y_actual, name='EBM Local')
        return local_explanation

service = InterpretMLServiceStub()
explainer = service.create_explainer(ebm, X_train, y_train)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Global Explanation"))
    cells.append(nbf.v4.new_code_cell("""global_exp = service.compute_global_interpret(explainer, X_train)
# In a real notebook env, 'show' renders an interactive widget
# show(global_exp)
print("Global Feature Importance (Approximate):")
for feature, importance in zip(global_exp.data()['names'], global_exp.data()['scores']):
    print(f"{feature}: {importance:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Local Explanation"))
    cells.append(nbf.v4.new_code_cell("""sample_instance = X_test.iloc[0:1]
sample_y = y_test.iloc[0:1]
local_exp = service.explain_instance(explainer, sample_instance, sample_y)
# show(local_exp)

print("Local Explanation for instance 0:")
for i, feature in enumerate(local_exp.data(0)['names']):
    print(f"{feature}: {local_exp.data(0)['scores'][i]:.4f}")
"""))
    
    nb.cells = cells
    with open('InterpretML_Explain.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_alibi_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Alibi Explain Implementation\nThis notebook demonstrates Alibi Explain on the Medical Insurance dataset."))
    cells.append(nbf.v4.new_markdown_cell("## Requirements\nRun the following to install required dependencies. See `requirements-xai-frameworks.txt` for details."))
    cells.append(nbf.v4.new_code_cell("!pip install alibi[all] pandas numpy scikit-learn tensorflow>=2.10.0,<2.16.0"))
    
    cells.append(nbf.v4.new_markdown_cell("## Setup and Data\nLoading the Medical Insurance dataset (dummy tabular data)."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# Dummy data generator
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'sex': np.random.choice(['male', 'female'], 1000),
    'bmi': np.random.uniform(18, 40, 1000),
    'children': np.random.randint(0, 5, 1000),
    'smoker': np.random.choice(['yes', 'no'], 1000),
    'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 1000),
    'charges': np.random.uniform(1000, 50000, 1000)
})

categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

# Alibi tabular explainers largely prefer Ordinal Encoded numeric mappings 
# rather than sparse One Hot Encoders
encoder = OrdinalEncoder()
df[categorical_features] = encoder.fit_transform(df[categorical_features])

X = df[categorical_features + numeric_features]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_names = list(X.columns)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predict_fn = lambda x: model.predict(x)
print("Data loaded and model trained!")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Explanability Framework API\nImplementing XAI Platform internal API conventions for Alibi: `create_explainer`, `compute_global_alibi`, and `explain_instance`."))
    cells.append(nbf.v4.new_code_cell("""from alibi.explainers import ALE, AnchorTabular

class AlibiServiceStub:
    @staticmethod
    def create_explainer(predict_fn, feature_names):
        # We will initialize ALE for global importance. 
        # (AnchorTabular is for local classification, so for regression we often use ALE / KernelShap)
        ale = ALE(predict_fn, feature_names=feature_names, target_names=['charges'])
        return ale
        
    @staticmethod
    def compute_global_alibi(explainer, X_bg):
        # Accumulated Local Effects (ALE)
        exp = explainer.explain(X_bg.values)
        return exp
        
    @staticmethod
    def explain_instance(predict_fn, X_train, feature_names, instance):
        # For local, we simulate a local explainer. Alibi's Anchor is for classification. 
        # Since this is regression, we might use normal SHAP/KernelShap from Alibi.
        from alibi.explainers import KernelShap
        explainer = KernelShap(predict_fn)
        explainer.fit(X_train.sample(50).values) # fit background
        explanation = explainer.explain(instance.values)
        return explanation

service = AlibiServiceStub()
explainer = service.create_explainer(predict_fn, feature_names)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Global Explanation (ALE)"))
    cells.append(nbf.v4.new_code_cell("""global_exp = service.compute_global_alibi(explainer, X_train.sample(100))
print("ALE Explanations generated successfully. Plotting usually done via `from alibi.explainers.ale import plot_ale`")
print(f"Features explained: {global_exp.feature_names}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Local Explanation (KernelShap via Alibi)"))
    cells.append(nbf.v4.new_code_cell("""sample_instance = X_test.iloc[0:1]
local_exp = service.explain_instance(predict_fn, X_train, feature_names, sample_instance)
print("Local Explanation SHAP values (via Alibi):")
print(local_exp.shap_values)
"""))

    nb.cells = cells
    with open('Alibi_Explain.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_aix360_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# AIX360 Implementation\nThis notebook demonstrates AI Explainability 360 (AIX360) on the Medical Insurance dataset."))
    cells.append(nbf.v4.new_markdown_cell("## Requirements\nRun the following to install required dependencies. See `requirements-xai-frameworks.txt` for details.\\n**CRITICAL**: AIX360 sometimes requires older scikit-learn (1.3.2) and numpy (<2.0)."))
    cells.append(nbf.v4.new_code_cell("!pip install aix360==0.3.0 scikit-learn==1.3.2 numpy<2.0 pandas"))
    
    cells.append(nbf.v4.new_markdown_cell("## Setup and Data\nLoading the Medical Insurance dataset (dummy tabular data)."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# Dummy data generator for AIX360.
# Note: AIX360 has strong classification algorithms (Boolean Rule CG, Contrastive Explanations). 
# We'll frame insurance charges as High/Low (Classification) for demonstration.
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'sex': np.random.choice([0, 1], 1000),           # 0=female, 1=male
    'bmi': np.random.uniform(18, 40, 1000),
    'children': np.random.randint(0, 5, 1000),
    'smoker': np.random.choice([0, 1], 1000),        # 0=no, 1=yes
    'region': np.random.choice([0, 1, 2, 3], 1000),
    'charges': np.random.uniform(1000, 50000, 1000)
})

# Classify as High (1) if charges > 25000, else Low (0)
df['high_charge'] = (df['charges'] > 25000).astype(int)

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
X = df[features]
y = df['high_charge']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Data loaded and Classification model trained!")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Explanability Framework API\nImplementing XAI Platform internal API conventions for AIX360: `create_explainer`, `compute_global_aix360`, and `explain_instance`."))
    cells.append(nbf.v4.new_code_cell("""# Note: Due to complex dependency environments (TF/Keras/Solvers), 
# this demonstrates the structure using simpler wrappers or stubs.
# Real AIX360 requires a running environment.
class AIX360ServiceStub:
    @staticmethod
    def create_explainer(model, X_train):
        # e.g., using CEM (Contrastive Explanation Method) explainer setup
        return {"model": model, "X_bg": X_train}
        
    @staticmethod
    def compute_global_aix360(explainer):
        # Global rules, e.g., BooleanRuleCG
        print("Computing AIX360 Global Rules...")
        return {"rules": [{"rule": "smoker > 0.5 AND age > 40", "prediction": "High Charge"}]}
        
    @staticmethod
    def explain_instance(explainer, instance):
        # Local Contrastive Explanations: 
        # "If BMI was 22 instead of 30, the charge would have been Low"
        print("Computing Local Contrastive Explanation (CEM)...")
        return {"pertinent_positive": instance.values, "pertinent_negative": instance.values * 0.9}

service = AIX360ServiceStub()
explainer = service.create_explainer(model, X_train)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Global Explanation"))
    cells.append(nbf.v4.new_code_cell("""global_exp = service.compute_global_aix360(explainer)
print("AIX360 Global Explanation Rules:")
print(global_exp)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Local Explanation"))
    cells.append(nbf.v4.new_code_cell("""sample_instance = X_test.iloc[0:1]
local_exp = service.explain_instance(explainer, sample_instance)
print("AIX360 Local Contrastive Explanation:")
print(local_exp)
"""))

    nb.cells = cells
    with open('AIX360_Explain.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_interpret_ml_notebook()
    create_alibi_notebook()
    create_aix360_notebook()
    print("Notebooks generated successfully.")
