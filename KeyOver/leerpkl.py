import joblib

feature_names = [
    "user_id",
    "element_id",
    "entity_id",
    "action_id",
    "hour",
    "minute",
    "day_of_week"
]

model = joblib.load("activity_model.pkl")

print("Modelo cargado correctamente")
print("Tipo:", type(model))
print("Número de árboles:", model.n_estimators)
print("Número de variables:", model.n_features_in_)

print("\nImportancia de variables:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance * 100:.4f}%")

''' 
Aqui el programa da un porcentaje de las variables que mas importancia tiene
a la hora de elegir y decidir una anomalia
'''