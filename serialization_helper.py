
# --- Serialization ---
import pickle

def save_model(filepath, components, P_final, metrics, report_str):
    """
    Saves the fitted model to a file.
    """
    model_data = {
        'components': components,
        'P_final': P_final,
        'metrics': metrics,
        'report': report_str
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a fitted model from a file.
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data
