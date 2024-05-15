import joblib
import pandas as pd


def _get_model_and_data_local(selected_model: str, fold_num: int, local_data_path: str = None):

    model_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}.pkl"
    model = joblib.load(model_filename)

    val_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_val_oh.csv"
    df_val_oh = pd.read_csv(val_filename)
    df_val_oh = df_val_oh.drop("y_pred", axis=1)

    train_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_train_oh.csv"
    df_train_oh = pd.read_csv(train_filename)

    return df_train_oh, df_val_oh, model
