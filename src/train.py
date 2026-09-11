from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocessing import preprocessor
import joblib

default_of_credit_card_clients = fetch_ucirepo(id=350)

X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets.squeeze()

feature_map = {
    'X1': 'LIMIT_BAL',
    'X2': 'SEX',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',
    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',
    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6'
}
X = X.rename(columns=feature_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', FixedThresholdClassifier(estimator=HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.1,
        l2_regularization=100,
        max_bins=127,
        max_depth=4,
        min_samples_leaf=1,
    ), threshold=0.3689292523278752))
])

model.fit(X_train, y_train)

joblib.dump(model, '../model/model.joblib')

preds = model.predict(X_test)
print(classification_report(y_test, preds, zero_division=0))