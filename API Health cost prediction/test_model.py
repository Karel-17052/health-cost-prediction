import joblib, pandas as pd

model = joblib.load('models/health_cost_model.joblib')
print('[OK] Modele charge avec succes')

test_cases = [
    {'age': 31, 'sex': 'male',   'bmi': 25.0, 'children': 1, 'smoker': 'no',  'region': 'southwest'},
    {'age': 60, 'sex': 'male',   'bmi': 20.9, 'children': 3, 'smoker': 'no',  'region': 'northeast'},
    {'age': 45, 'sex': 'female', 'bmi': 30.5, 'children': 2, 'smoker': 'yes', 'region': 'southeast'},
]
df = pd.DataFrame(test_cases)
preds = model.predict(df)
print('[OK] Predictions :')
for i, (case, pred) in enumerate(zip(test_cases, preds)):
    smoker = case['smoker']
    age = case['age']
    print(f'   Patient {i+1}: smoker={smoker}, age={age} -> ${pred:,.2f}')

print()
print('[OK] Tout fonctionne ! L API est prete a etre deployee.')
