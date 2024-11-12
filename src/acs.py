import pandas as pd
from folktables import ACSDataSource
from sklearn.preprocessing import StandardScaler

import src.utils.helper as helper

"""
Some categorias are similiar to the ones in the employment dataset
ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}
"""


def group_race(x):
    if x == 3.0 or x == 4.0 or x == 5.0 or x == 7.0:
        return 4.0  # America Native and Alaska Native
    if x == 6.0:
        return 3.0  # Asian
    if x == 8.0:
        return 5.0  # Some Other
    if x == 9.0:
        return 6.0  # Two or More
    else:
        return x  # 1. White, 2. Black, 3. Asian, 4. America Native and Alaska Native, 5. Some Other, 6. Two or More


def _group_race(x):
    if x == 3.0 or x == 4.0 or x == 5.0 or x == 7.0 or x == 8.0 or x == 9.0:
        return 4.0
    if x == 6.0:
        return 3.0  # Asian
    else:
        return x  # 1. White, 2. Black, 3. Asian, 4.Others


class ACSDataset:
    def __init__(self, survey_year="2017", US_states=["LA", "MI"], horizon="1-Year", survey="person"):
        self.survey_year = survey_year
        self.horizon = horizon
        self.survey = survey
        self.states = US_states

    def task_task(self, task_name: str):
        if task_name == "employment":
            from folktables import ACSEmploymentFiltered

            return ACSEmploymentFiltered
        elif task_name == "income":
            from folktables import ACSIncome

            return ACSIncome
        elif task_name == "public_coverage":
            from folktables import ACSPublicCoverage

            return ACSPublicCoverage
        else:
            raise AttributeError("Attribute not found")

    def get_data(self, download=True, task_name="employment", return_type="csv"):
        data_source = ACSDataSource(survey_year=self.survey_year, horizon=self.horizon, survey=self.survey)
        states_data = data_source.get_data(states=self.states, download=download)

        acs_task = self.task_task(task_name)
        features, labels, _ = acs_task.df_to_pandas(states_data)

        # 1. White, 2. Black, 3. Asian, 4. America Native and Alaska Native, 5. Some Other, 6. Two or More
        features["RACE"] = features["RAC1P"].apply(lambda x: group_race(x))
        features = features.drop(columns=["RAC1P"])

        if task_name == "employment":
            # keep only the features with RACE as black and white:
            features = features[features["RACE"].isin([1.0, 2.0])]

        # raw labels are boolean, we need to convert them to int
        labels = labels.astype(int)
        features["LABELS"] = labels

        # check if there is duplications and remove them
        features = features.drop_duplicates()

        if return_type == "dataframe":
            return features
        elif return_type == "csv":
            features_obj = features.to_csv(index=False).encode("utf-8")
            return features_obj
        else:
            return "return_type not found"

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, stratify=None, dtype=None):
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

        if dtype == "csv":
            return train.to_csv(index=False).encode("utf-8"), test.to_csv(index=False).encode("utf-8")
        else:
            return train, test

    def preprocess_data(self, df: pd.DataFrame, categorical_features: list = [], dtype=None):
        features_list_not_labels = df.columns.to_list()[:-1]

        # check if there is duplications and remove them
        df = df.drop_duplicates()

        if categorical_features and not set(categorical_features).issubset(set(features_list_not_labels)):
            raise ValueError("Categorical features not found in the dataset")

        if categorical_features:
            df = helper.one_hot_encode(df, categorical_features)
            df = df.drop(columns=categorical_features)

        continuous_features = []
        for feature in features_list_not_labels:
            if feature not in categorical_features:
                continuous_features.append(feature)

        scale = StandardScaler()
        df[continuous_features] = scale.fit_transform(df[continuous_features])

        if dtype == "csv":
            return df.to_csv(index=False).encode("utf-8")
        else:
            return df
