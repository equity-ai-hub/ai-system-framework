import joblib
import pandas as pd
import dill


def _get_model_and_data_local(selected_model: str, fold_num: int, local_data_path: str = None):
    model_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}.pkl"

    try:
        # Attempt to load with joblib
        model = joblib.load(model_filename)
        print("Loaded successfully using joblib.")
    except ModuleNotFoundError as e:
        if "dill" in str(e):
            # If the error is related to dill, try loading with dill
            print("Module 'dill' not found, trying to load with dill.")
            with open(model_filename, "rb") as file:
                model = dill.load(file)
                print("Loaded successfully using dill.")
        else:
            raise e
    except Exception as e:
        print(f"An error occurred: {e}")

    val_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_val_oh.csv"
    df_val_oh = pd.read_csv(val_filename)
    df_val_oh = df_val_oh.drop("y_pred", axis=1)

    train_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_train_oh.csv"
    df_train_oh = pd.read_csv(train_filename)

    return df_train_oh, df_val_oh, model
