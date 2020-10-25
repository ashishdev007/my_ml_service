import joblib
import pandas as pd
from server.settings import BASE_DIR
import os


class ExtraTreesClassifier:
    def __init__(self) -> None:
        path_to_artifacts = os.path.join(BASE_DIR, "research")
        self.values_fill_missing = joblib.load(
            path_to_artifacts + "/train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts + "/encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "/extra_trees.joblib")

    def preprocessing(self, input_data):

        input_data = pd.DataFrame(input_data, index=[0])
        input_data = input_data.fillna(self.values_fill_missing)

        for column in [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.transform(
                input_data[column])

        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = ">50K" if input_data[0][1] > 0.5 else "<=50K"

        return ({"probability": input_data[0][1], "label": label, "status": "OK"})

    def compute_prediction(self, input_data):

        try:
            input_data = self.preprocessing(input_data)
            probalility = self.predict(input_data)
            result = self.postprocessing(probalility)

            return result

        except Exception as e:
            return {"status": "Error", "message": str(e)}
