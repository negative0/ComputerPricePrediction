import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

csv_file = ""

def clean_data(data):
    data.loc[data.cd == "yes", "cd"] = 1
    data.loc[data.cd == "no", "cd"] = 0

    data.loc[data.premium == "yes", "premium"] = 1
    data.loc[data.premium == "no", "premium"] = 0

    data.loc[pd.isna(data.speed), "speed"] = np.median(data["speed"])
    return data


def load_file(file_name):
    data = pd.read_csv(file_name)

    data = clean_data(data)

    # Set of Independent Variables
    X = pd.DataFrame(data, columns=["speed", "ram", "hd", "screen", "cd", "premium"])

    # Set of Dependent Variables
    y = pd.DataFrame(data, columns=["price"])

    return X, y



if __name__ == "__main__":
    X, y = load_file("computers.csv")
    # speed = sm.add_constant(speed)

    # Scatter plot
    # plot = plt.boxplot(X, y)
    plt.show()
    # Train the model using statsmodels api with
    # OLS (Ordinary Least Squares)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    print(model.summary())

