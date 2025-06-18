import numpy as np

def preprocess_input(data):
    geography = {"France": 0, "Spain": 1, "Germany": 2}
    gender = {"Male": 0, "Female": 1}

    features = [
        int(data["credit_score"]),
        geography[data["country"]],
        gender[data["gender"]],
        int(data["age"]),
        int(data["tenure"]),
        float(data["balance"]),
        int(data["products"]),
        int(data["credit_card"]),
        int(data["active_member"]),
        float(data["salary"])
    ]

    return np.array([features])
