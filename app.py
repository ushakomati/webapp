from flask import Flask, render_template, request, Response
import pandas as pd
import io
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')
total_features_per_month = pd.DataFrame()
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        train_size = float(request.form['train_size'])
        test_size = float(request.form['test_size'])
        dataset = request.form['dataset']
        if dataset == 'zoom':
            filename = 'Zoom-features-2022.xlsx'
            sheets_to_extract = ['Jan-2022','Feb-2022', 'March-2022', 'April-2022' ,'May-2022', 'June-2022', 'July-2022', 'Aug-2022', 'Sept-2022', 'Nov-2022', 'Dec-2022']
        elif dataset == 'webex':
            filename = 'Webex-features-2022.xlsx'
            sheets_to_extract = ['Webex-Jan-2022','Webex-Feb-2022', 'Webex-March-2022', 'Webex-April-2022' ,'Webex-May-2022', 'Webex-June-2022', 'Webex-July-2022', 'Webex-Aug-2022', 'Webex-Sept-2022', 'Webex-Nov-2022', 'Webex-Dec-2022']
        #sheets_to_extract = ['Jan-2022','Feb-2022', 'March-2022', 'April-2022' ,'May-2022', 'June-2022', 'July-2022', 'Aug-2022', 'Sept-2022', 'Nov-2022', 'Dec-2022']
        sheets_dict = pd.read_excel(filename, sheet_name=sheets_to_extract)
        df = pd.concat(sheets_dict.values(), ignore_index=True)

# CHANGES

        df_list = []

        # Iterate through the sheets and append each sheet as a DataFrame to the list
        for sheet_name in sheets_to_extract:
            df = pd.read_excel(filename, sheet_name=sheet_name)
            df['Month-Year'] = sheet_name
            df_list.append(df)

        # Concatenate the list of DataFrames into a single DataFrame
        df = pd.concat(df_list, ignore_index=True)
        global total_features_per_month
        total_features_per_month = df.groupby("Month-Year", sort=False)["Feature Title"].count()

# CHANGES
        feature_desc = []
        df['Combined'] = df['Feature Title'].astype(str) + df['Feature Description'].astype(str)
        feature_desc = df["Combined"].tolist()
        feature_desc_line = []
        for each_line in feature_desc:
            trimmed_each_line = each_line.strip()
            if trimmed_each_line:
                feature_desc_line.append(trimmed_each_line.lower())
        X = feature_desc_line
        vector = CountVectorizer()
        vector.fit(X)
        vector.vocabulary_
        document_term_matrix = vector.transform(X)
        document_term_matrix.shape
        feature_arr = document_term_matrix.toarray()
        target = [0] * len(feature_arr)
        k = int(len(feature_arr) * train_size)
        while k < len(feature_arr):
            target[k] = 1
            k = k + 1
        X_train, X_test, y_train, y_test = train_test_split(feature_arr,target, test_size=test_size, random_state=42)
        dtclf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=0)
        dtclf_gini.fit(X_train, y_train)
        y_pred_gini = dtclf_gini.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_gini)
        plt.figure(figsize=(12, 6))
        tree.plot_tree(dtclf_gini,max_depth=4,class_names=True,filled=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        # sheets_to_extract = ['Jan-2022','Feb-2022', 'March-2022', 'April-2022' ,'May-2022', 'June-2022', 'July-2022', 'Aug-2022', 'Sept-2022', 'Nov-2022', 'Dec-2022']


        return render_template('result.html', accuracy=accuracy, img_data=img_data)
    return render_template('index.html')


@app.route('/getGraph')
def index():
    # Generate some data to plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a Matplotlib figure and plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)

    # Render the plot to a canvas
    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)

    # Return the canvas as an image response
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/plot")
def plot():
    # Generate the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(total_features_per_month.index, total_features_per_month, alpha=0.5, color='orange')
    ax.set_title("Total Number of Features per Month/Release")
    ax.set_xlabel("Month/Release")
    ax.set_ylabel("Total Number of Features")

    # Save the plot as a PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the PNG image as base64 string
    image_b64 = base64.b64encode(image_png).decode("utf-8")

    # Render the HTML page with the plot image
    return render_template("plot.html", plot_image=image_b64)
    # global total_features_per_month
    # total_features_per_month.plot(kind='line', rot=45, alpha=0.5, color='orange', label='Features')
    # plt.title("Total Number of Features per Month/Release")
    # plt.xlabel("Month/Release")
    # plt.ylabel("Total Number of Features")
    # # Save the plot as a PNG image
    # buffer = io.BytesIO()
    # plt.savefig(buffer, format="png")
    # buffer.seek(0)
    # image_png = buffer.getvalue()
    # buffer.close()

    # # Encode the PNG image as base64 string
    # image_b64 = base64.b64encode(image_png).decode("utf-8")

    # # Render the HTML page with the plot image
    # return render_template("plot.html", plot_image=image_b64)


if __name__ == '__main__':
    app.run(debug=True)

