from sklearn import linear_model
import pandas as pd

csv_file = "computers.csv"


def load_file(file_name):
    data = pd.read_csv(file_name, na_values=False)

    data.loc[data.cd == "yes", "cd"] = 1
    data.loc[data.cd == "no", "cd"] = 0

    data.loc[data.premium == "yes", "premium"] = 1
    data.loc[data.premium == "no", "premium"] = 0

    # Set of Independent Variables
    X = pd.DataFrame(data, columns=["speed", "ram", "hd", "screen", "cd", "premium"])

    # Set of Dependent Variables
    y = pd.DataFrame(data, columns=["price"])

    print(y)

    return X, y


if __name__ == "__main__":
    X, y = load_file(csv_file)

    # Create a Linear Regression model from SKLearn
    lm = linear_model.LinearRegression()

    # Fit the model to the data
    model = lm.fit(X, y)

    # Get the predictions for Data
    predictions = lm.predict(X)
    print(predictions)

    print("Score is: %d %%" % (lm.score(X, y) * 100))

