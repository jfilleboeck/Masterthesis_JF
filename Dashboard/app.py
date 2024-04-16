import math
import numpy
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import CubicSpline
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_preprocess_data
from model_adapter import ModelAdapter

# Load and split the data, determine if classification/regression
X_train, X_val, y_train, y_val, task = load_and_preprocess_data()
# Necessary placeholder for the instance view table
rows = None

# Initializes a model using the ModelAdapter class
adapter = ModelAdapter(task)

adapter.fit(X_train, y_train)

# Setup
app = Flask(__name__)


def load_data():
    """
    Loads initial data from the model and prepares it for the web application.
    Populates global variables with shape functions, feature history (deprecated), and current state data.
    """
    global shape_functions_dict, feature_history, feature_current_state
    shape_functions_dict = adapter.get_shape_functions_as_dict()
    feature_history = {}
    # Feature current state has previously been used to allow the user to 'undo' steps. Due to ongoing bugs,
    # this was deprecated, but remains in the code (without effect) for further development.

    feature_current_state = {}

    for feature in shape_functions_dict:
        name = feature['name']
        y_value = feature['y']

        feature_history[name] = [y_value]
        feature_current_state[name] = y_value

# Call load_data() to initially load data from the model upon starting the web application.
load_data()


@app.route('/')
def index():
    """
    The main route that renders the index page of the web application, displaying the initial feature data.

    Returns:
    - A rendered template of 'index.html' with necessary data for the initial view.
    """
    X_names = X_train.columns.tolist()

    feature_name, x_data, y_data, is_numeric_feature, hist_data, bin_edges = next(
        (feature['name'], feature['x'].astype(float).tolist(),
         feature['y'].astype(float).tolist(), feature['datatype'],
         feature['hist'].hist.tolist(), feature['hist'].bin_edges.tolist())
        for feature in shape_functions_dict
    )


    return render_template('index.html', feature_names=X_names, x_data=x_data,
                           y_data=y_data, displayed_feature=feature_name, is_numeric_feature=is_numeric_feature,
                           hist_data=hist_data, bin_edges=bin_edges)


@app.route('/feature_data', methods=['POST'])
def feature_data():
    """
    Provides data for a specific feature requested. The data is required to display the shape functions in the frontend.

    Returns:
    - JSON response containing the feature data or an error message if the feature is not found.
    """
    data = request.json
    displayed_feature = data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if feature_data:
        if feature_data['datatype'] == 'numerical':
            x_data = feature_data['x'].tolist()
            y_data = feature_current_state[displayed_feature].tolist()
            hist_data = feature_data['hist'].hist.tolist()
            bin_edges = feature_data['hist'].bin_edges.tolist()
            return jsonify({'is_numeric': True, 'x': x_data, 'y': y_data,
                            'displayed_feature': displayed_feature,
                            'hist_data': hist_data, 'bin_edges': bin_edges})
        else:
            x_data = feature_data['x']
            encoded_x_data = encode_categorical_data(x_data)
            y_data = feature_current_state[displayed_feature]
            y_data = [float(y) if isinstance(y, np.float32) else y for y in y_data]
            hist_data = to_float_list(feature_data['hist'][0])
            bin_edges = encode_categorical_data(feature_data['hist'][1])
            return jsonify({'is_numeric': False, 'original_values': x_data,
                            'x': encoded_x_data, 'y': y_data,
                            'hist_data': hist_data, 'bin_edges': bin_edges,
                            'displayed_feature': displayed_feature})

    else:
        return jsonify({'error': 'Feature not found'}), 404



@app.route('/setConstantValue', methods=['POST'])
def setConstantValue():
    """
    Sets a constant value for a specified range of x values for a displayed feature.
    The range and new value are specified in the request.

    Returns:
    - JSON response containing the updated y values for the feature.
    """
    data = request.json
    x1, x2, new_y, displayed_feature = data['x1'], data['x2'], float(data['new_y']), data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404

    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])

    for i, x in enumerate(x_data):
        if x1 <= x <= x2:
            y_data[i] = new_y

    feature_history[displayed_feature].append(y_data)
    feature_current_state[displayed_feature] = y_data

    return jsonify({'y': [float(y) for y in y_data] if feature_data['datatype'] != 'numerical' else [float(y) for y in
                                                                                                     y_data.tolist()]})


@app.route('/setLinear', methods=['POST'])
def setLinear():
    """
    Applies a linear transformation to a specified range of x values for
    a displayed feature based on the provided start and end points in the request.

    Returns:
    - JSON response containing the updated y values for the feature.
    """
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)

    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404

    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])

    # Find indices for x1 and x2
    index_x1 = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x1))
    index_x2 = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x2))

    # Ensure indices are in the correct order
    index_start, index_end = sorted([index_x1, index_x2])

    slope = (y_data[index_end] - y_data[index_start]) / (x_data[index_end] - x_data[index_start])

    # Update y values along the line
    for i in range(index_start, index_end + 1):
        y_data[i] = y_data[index_start] + slope * (x_data[i] - x_data[index_start])

    feature_history[displayed_feature].append(y_data)
    feature_current_state[displayed_feature] = y_data

    return jsonify({'y': y_data if feature_data['datatype'] != 'numerical' else y_data.tolist()})



@app.route('/monotonic_increase', methods=['POST'])
def monotonic_increase():
    """
    Applies monotonic increasing transformation to a specified range of x values for a displayed feature.

    Returns:
    - JSON response containing the updated y values for the feature.
    """
    data = request.json
    displayed_feature = data['displayed_feature']
    x1, x2 = data['x1'], data['x2']
    y_data_full = feature_current_state[displayed_feature].copy()

    selected_item = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
    # Numpy arrays are required for the IsotonicRegression package
    x_data = np.array(selected_item['x'])
    hist_data = np.array(selected_item['hist'].hist)
    bin_edges = np.array(selected_item['hist'].bin_edges)

    indices = np.where((x_data >= x1) & (x_data <= x2))[0]
    y_pred_subset = weighted_isotonic_regression(
        x_data[indices], y_data_full[indices], hist_data, bin_edges, increasing=True)

    y_data_full[indices] = y_pred_subset

    # Update feature history and current state
    feature_history[displayed_feature].append(feature_current_state[displayed_feature])
    feature_current_state[displayed_feature] = y_data_full

    return jsonify({'y': y_data_full.tolist()})


@app.route('/monotonic_decrease', methods=['POST'])
def monotonic_decrease():
    """
    Applies monotonic decreasing transformation to a specified range of x values for a displayed feature.

    Returns:
    - JSON response containing the updated y values for the feature.
    """
    data = request.json
    displayed_feature = data['displayed_feature']
    x1, x2 = data['x1'], data['x2']
    y_data_full = feature_current_state[displayed_feature].copy()

    selected_item = next(item for item in shape_functions_dict if item['name'] == displayed_feature)
    # Numpy arrays are required for the IsotonicRegression package
    x_data = np.array(selected_item['x'])
    hist_data = np.array(selected_item['hist'].hist)
    bin_edges = np.array(selected_item['hist'].bin_edges)

    indices = np.where((x_data >= x1) & (x_data <= x2))[0]
    y_pred_subset = weighted_isotonic_regression(
        x_data[indices], y_data_full[indices], hist_data, bin_edges, increasing=False)

    y_data_full[indices] = y_pred_subset

    feature_history[displayed_feature].append(feature_current_state[displayed_feature])
    feature_current_state[displayed_feature] = y_data_full

    return jsonify({'y': y_data_full.tolist()})

@app.route('/setSmooth', methods=['POST'])
def setSmooth():
    """
    Applies a smoothing operation to a specified range of x values for a displayed feature
    using a simple moving average.

    Returns:
    - JSON response containing the smoothed y values for the feature.
    """
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    window_size = 5
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    data = request.json
    x1, x2, displayed_feature = data['x1'], data['x2'], data['displayed_feature']
    feature_data = next((item for item in shape_functions_dict if item['name'] == displayed_feature), None)
    if not feature_data:
        return jsonify({'error': 'Feature not found'}), 404
    y_data = feature_current_state[displayed_feature].copy()

    if feature_data['datatype'] == 'numerical':
        x_data = feature_data['x']
    else:
        x_data = encode_categorical_data(feature_data['x'])
    # Find indices for x1 and x2
    index_start = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x1))
    index_end = min(range(len(x_data)), key=lambda i: abs(x_data[i]-x2))

    # Simple Moving Average
    smoothed_y = y_data.copy()
    for i in range(index_start, index_end + 1):
        window_indices = range(max(i - window_size // 2, 0), min(i + window_size // 2 + 1, len(y_data)))
        smoothed_y[i] = sum(y_data[j] for j in window_indices) / len(window_indices)

    feature_history[displayed_feature].append(smoothed_y)
    feature_current_state[displayed_feature] = smoothed_y

    return jsonify({'y': smoothed_y if feature_data['datatype'] != 'numerical' else smoothed_y.tolist()})


@app.route('/cubic_spline_interpolate', methods=['POST'])
def cubic_spline_interpolate():
    """
    Performs cubic spline interpolation on the selected features' data.

    Updates the adapter with the interpolated data (based on the user changes in the frontend)
    and returns the interpolated data for the displayed feature.

    Returns:
    - JSON response containing the interpolated x and y values for the displayed feature.
    """
    data = request.json
    selectedFeatures = data['selectedFeatures']
    displayed_feature = data['displayed_feature']

    # Generate the training data for retraining process
    updated_data = {}
    for feature in shape_functions_dict:
        name = feature['name']
        x_values = feature['x']
        if name in selectedFeatures:
            y_values = np.array(feature_current_state[name])
        else:
            y_values = feature['y']
        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}

    # Adapt the model
    adapter.adapt(selectedFeatures, updated_data, "spline_interpolation")

    # Generate new shape functions
    displayed_feature_data = next((item for item in adapter.get_shape_functions_as_dict() if item['name'] == displayed_feature), None)

    x_data = displayed_feature_data['x']
    y_data = displayed_feature_data['y']
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return if type(x) != str]
    y_data_to_return = [float(y) for y in y_data_to_return if type(y) != str]


    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})



@app.route('/retrain_feature', methods=['POST'])
def retrain_feature():
    """
    Retrains the model for selected features.

    Updates the adapter with the retrained data and returns the retrained data for the displayed feature.

    Returns:
    - JSON response containing the retrained x and y values for the displayed feature.
    """
    data = request.json
    selectedFeatures = data['selectedFeatures']
    displayed_feature = data['displayed_feature']
    # Hyperparameters
    elmScale = float(data['elmScale'])
    elmAlpha = float(data['elmAlpha'])
    nrSyntheticDataPoints = int(data['nrSyntheticDataPoints'])


    # Generate the training data for retraining process
    updated_data = {}
    for feature in shape_functions_dict:
        name = feature['name']
        x_values = feature['x']
        if name in selectedFeatures:
            y_values = np.array(feature_current_state[name])
        else:
            y_values = feature['y']
        if feature['datatype'] == 'numerical':
            updated_data[name] = {'x': x_values, 'y': y_values.tolist(), 'datatype': 'numerical'}
        else:
            updated_data[name] = {'x': x_values, 'y': y_values, 'datatype': 'categorical'}

    # Adapt the model
    adapter.adapt(selectedFeatures, updated_data, "feature_retraining", (elmScale, elmAlpha, nrSyntheticDataPoints))

    # Generate new shape functions
    displayed_feature_data = next(
        (item for item in adapter.get_shape_functions_as_dict() if item['name'] == displayed_feature), None)

    x_data = displayed_feature_data['x']
    y_data = displayed_feature_data['y']
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return if not isinstance(x, str)]
    y_data_to_return = [float(y) for y in y_data_to_return if not isinstance(y, str)]

    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})




@app.route('/predict_and_get_metrics', methods=['GET'])
def predict_and_get_metrics():
    """
    Makes predictions using the trained model and calculates performance metrics based on the task type

    Returns:
    - JSON response containing the training and validation scores along with the task type.
    """
    y_train_pred = adapter.predict(X_train)
    y_val_pred = adapter.predict(X_val)

    if task == "regression":
        train_score = mean_squared_error(y_train, y_train_pred)
        val_score = mean_squared_error(y_val, y_val_pred)
    else:
        train_score = f1_score(y_train, y_train_pred, average='weighted')
        val_score = f1_score(y_val, y_val_pred, average='weighted')

    return jsonify({'train_score': train_score, 'val_score': val_score, 'task': task})



@app.route('/get_original_data', methods=['POST'])
def get_original_data():
    """
    Resets the displayed feature to its original data as per the initial model state.

    Returns:
    - JSON response containing the original x and y values for the displayed feature.
    """
    global feature_history, feature_current_state, feature_spline_state, adapter

    data = request.json
    displayed_feature = data['displayed_feature']

    # Obtain the original model
    adapter = ModelAdapter(task)
    adapter.fit(X_train, y_train)
    load_data()
    original_data = next(item for item in adapter.get_shape_functions_as_dict() if item['name'] == displayed_feature)
    original_y = original_data['y']

    # Reset the current state and history for the selected feature
    feature_current_state[displayed_feature] = original_y
    feature_history[displayed_feature] = [original_y.copy()]  # Reset history with the original state


    x_data = original_data['x']
    y_data = original_y
    x_data_to_return = x_data.tolist() if not isinstance(x_data, list) else x_data
    y_data_to_return = y_data.tolist() if not isinstance(y_data, list) else y_data
    x_data_to_return = [float(x) for x in x_data_to_return if not isinstance(x, str)]
    y_data_to_return = [float(y) for y in y_data_to_return if not isinstance(y, str)]


    return jsonify({'x': x_data_to_return, 'y': y_data_to_return})


# deprecated
@app.route('/undo_last_change', methods=['POST'])
def undo_last_change():
    """
    Undoes the last change made to the displayed feature's data, reverting it to its previous state.

    Returns:
    - JSON response containing the reverted y values for the feature or an error message if no changes can be undone.
    """
    data = request.json
    displayed_feature = data['displayed_feature']

    if displayed_feature in feature_history and len(feature_history[displayed_feature]) > 1:
        feature_current_state[displayed_feature] = feature_history[
            displayed_feature].pop()  # Revert to the previous state and remove the last change
        y_data = feature_current_state[displayed_feature]  # Update y_data to the reverted state
        return jsonify({'y': y_data.tolist()})
    else:
        return jsonify({'error': 'No more changes to undo for feature ' + displayed_feature}), 400


@app.route('/load_data_grid_instances', methods=['POST'])
def load_data_grid_instances():
    """
    Prepares and returns the data for displaying in the validation data in a table in the frontend.

    Returns:
    - JSON response containing the data and column information for the table.
    """
    X_val_preprocessed_df = X_val
    X_val_preprocessed_df = X_val_preprocessed_df.round(3)
    X_val_preprocessed_df = X_val_preprocessed_df.reset_index(drop=True)
    if not isinstance(y_val, pd.DataFrame):
         y_val_df = pd.DataFrame(y_val, columns=["target"])
    else:
        y_val_df = y_val
    y_val_reset = y_val_df.reset_index(drop=True)
    prediction = adapter.predict(X_val_preprocessed_df)

    combined_data = pd.concat([X_val_preprocessed_df, y_val_reset], axis=1)
    combined_data.insert(len(combined_data.columns), 'prediction', prediction)

    # Ensure that the target variable is rounded as well
    combined_data = combined_data.round(3)

    combined_data.insert(0, 'ID', combined_data.index)
    global rows
    # Convert DataFrame to dictionary
    rows = combined_data.to_dict(orient='records')
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 3)  # Ensure rounding persists in the dictionary

    response = {
        'data': rows,
        'columns': ['ID'] + combined_data.columns.tolist()  # Assuming 'target' is the name in y_val_reset
    }

    return jsonify(response)


@app.route('/instance_explanation', methods=['POST'])
def instance_explanation():
    """
    Provides the contributions of each feature the model's prediction on a selected instance

    Returns:
    - JSON response containing the predicted contribution of each feature, feature names, intercept, target, and prediction for the selected instance.
    """
    selectedRowId = request.json['selectedRow_ID']

    intercept = adapter.model.init_classifier.intercept_
    global rows
    row_data = get_row_by_id(int(selectedRowId)) if selectedRowId is not None else None

    target = row_data['target']
    prediction = row_data['prediction']
    intercept = round(adapter.model.init_classifier.intercept_, 3)
    feature_names = adapter.feature_names
    values = [0] * len(feature_names)

    if row_data is not None:
        values = [row_data.get(feature, 0) for feature in feature_names if feature and feature != "target"]
    pred_list = []

    for position, name in enumerate(feature_names):
        i = feature_names.index(name)
        x = values[position]
        pred = torch.tensor([0], dtype=torch.float)
        for regressor, boost_rate in zip(adapter.model.regressors, adapter.model.boosting_rates):
            pred += (
                    boost_rate
                    * regressor.predict_single((torch.tensor([x], dtype=torch.float)).reshape(-1, 1),
                                               i).squeeze()
            ).cpu()
        pred_list.append(pred)
    pred_scalar = [t.item() for t in pred_list]
    return jsonify({"pred_instance": pred_scalar, "feature_names": feature_names,
                    "intercept": intercept, "target": target, "prediction": prediction})


@app.route('/path/to/order_by_nearest', methods=['POST'])
def order_by_nearest():
    """
    Orders the provided dataset (for numerical features) by nearest to a selected row based on Euclidean distance.

    The request must contain the dataset and the ID of the selected row.

    Returns:
    - JSON response containing the ordered dataset.
    """
    data = request.json['data']
    selectedRowId = request.json['selectedRowId']
    selectedRow = next(item for item in data if item['ID'] == selectedRowId)

    # Calculate distances to the selected row and sort
    for item in data:
        item['distance'] = calculate_distance(item, selectedRow)
    orderedData = sorted(data, key=lambda x: x['distance'])

    return jsonify(orderedData)

@app.route('/plot_all_shape_functions')
def plot_all_shape_functions():
    """
    Generates an image depicting all shape functions used by the model.

    Returns:
    - The image containing the plot of all shape functions.
    """
    matplotlib.use('Agg')
    adapter.model.plot_single(show_n=15, max_cat_plotted=15)
    fig = plt.gcf()
    fig.savefig('static/images/allShapeFunctions.png')
    plt.close(fig)

    return send_file('static/images/allShapeFunctions.png', mimetype='image/png')


@app.route('/plot_correlation_matrix')
def plot_correlation_matrix():
    """
    Generates an image of the correlation matrix for the training dataset.

    Returns:
    - The image containing the correlation matrix plot.
    """
    encoded_df = X_train.copy()
    # Identify and encode each categorical column
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':
            categories = encoded_df[column].unique()
            category_to_code = {category: index for index, category in enumerate(categories)}
            # Replace each categorical value with its encoded value
            encoded_df[column] = encoded_df[column].map(category_to_code)

    plt.figure(figsize=(10, 8))
    sns.heatmap(encoded_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')

    image_path = 'static/images/correlationMatrix.png'
    plt.savefig(image_path)
    plt.close()

    return send_file(image_path, mimetype='image/png')


""" Helper functions """
def weighted_isotonic_regression(x_data, y_data, hist_counts, bin_edges, increasing=True):
    """
    Performs weighted isotonic regression on the given data.

    Parameters:
    - x_data (numpy array): The x values of the data points.
    - y_data (numpy array): The y values of the data points.
    - hist_counts (numpy array): The histogram counts for weighting.
    - bin_edges (numpy array): The bin edges used for weighting.
    - increasing (bool): Whether the isotonic regression should be increasing or not.

    Returns:
    - numpy array: The predicted y values after isotonic regression.
    """
    # Determine the bin index for each x_data point
    bin_indices = np.digitize(x_data, bin_edges, right=True).astype(int)
    bin_indices = np.clip(bin_indices, 1, len(hist_counts))

    weights = np.array([hist_counts[index - 1] for index in bin_indices])

    iso_reg = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    iso_reg.fit(x_data, y_data, sample_weight=weights)
    y_pred = iso_reg.predict(x_data)

    # In some cases, isotonic regression fits the data in a way, that the
    # starting point is lower than initially. This might be confusing for the
    # user of the dashboard. Therefore, this effect is removed in the following:
    if increasing:
        if y_pred[0] < y_data[0]:
            y_pred[0] = y_data[0]
            # Ensure the rest of the predictions maintain monotonicity
            for i in range(1, len(y_pred)):
                y_pred[i] = max(y_pred[i], y_pred[i-1])
    else:
        if y_pred[0] < y_data[0]:
            y_pred[0] = y_data[0]
            # Ensure the rest of the predictions maintain monotonicity
            for i in range(1, len(y_pred)):
                y_pred[i] = min(y_pred[i], y_pred[i-1])
    return y_pred

def get_row_by_id(row_id):
    """
    Retrieves a row from the global 'rows' dataset by its ID.

    Parameters:
    - row_id (int): The ID of the row to retrieve.

    Returns:
    - dict: The row corresponding to the given ID, or None if not found.
    """
    global rows
    print(rows)
    for row in rows:
        if row["ID"] == row_id:
            return row
    return None

def calculate_distance(row1, row2):
    """
    Calculates the Euclidean distance between two rows, excluding the 'ID' and target columns.

    Parameters:
    - row1 (dict): The first row in the comparison
    - row2 (dict): The second row in the comparison

    Returns:
    - float: The Euclidean distance between the two rows.
    """
    distance = 0.0
    keys = list(row1.keys())
    # Exclude the first ('ID') and last columns from the keys
    for key in keys[1:-1]:
        if key not in adapter.model.dummy_encodings.values() and key in row2:
            print(key)
            distance += (row1[key] - row2[key]) ** 2
    return math.sqrt(distance)

def encode_categorical_data(categories):
    """
    Encodes categorical data into numerical indices.

    Parameters:
    - categories (list): The list of categorical values to encode.

    Returns:
    - list: The list of indices representing the encoded categorical values.
    """
    encoded_values = [index for index, category in enumerate(categories)]

    return encoded_values

def to_float_list(lst):
    """
    Converts a list of mixed data types to a list of floats. Tensors in the list are converted to their numerical values.

    Parameters:
    - lst (list): The list containing elements of various types including tensors.

    Returns:
    - list: A list of floats.
    """
    float_list = []
    for item in lst:
        if torch.is_tensor(item):
            float_list.append(item.item())
        elif isinstance(item, list):
            for sub_item in item:
                if torch.is_tensor(sub_item):
                    float_list.append(sub_item.item())
    return float_list



if __name__ == '__main__':
    app.run(debug=True)