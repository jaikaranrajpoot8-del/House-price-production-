{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "or5zyrdSF-3g"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#pip install scikit-learn pandas\n"
      ],
      "metadata": {
        "id": "QoPe-J3sGQcT"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import fetch_california_housing\n",
        "import pandas as pd\n",
        "housing = fetch_california_housing()\n",
        "df = pd.DataFrame(housing.data, columns=housing.feature_names)\n",
        "df['HousePrice'] = housing.target\n",
        "print(df.head())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oUAl_YDXGZJt",
        "outputId": "712bbd01-5e16-49de-850f-d7029218e77a"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \\\n",
            "0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   \n",
            "1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   \n",
            "2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   \n",
            "3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   \n",
            "4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   \n",
            "\n",
            "   Longitude  HousePrice  \n",
            "0    -122.23       4.526  \n",
            "1    -122.22       3.585  \n",
            "2    -122.24       3.521  \n",
            "3    -122.25       3.413  \n",
            "4    -122.25       3.422  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.dropna(inplace = True)"
      ],
      "metadata": {
        "id": "rX4YPmZNHsMD"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 366
        },
        "id": "un2EiZgRG3Nm",
        "outputId": "00b19c5b-d542-4459-cfd6-5198d7eebce1"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MedInc        0\n",
              "HouseAge      0\n",
              "AveRooms      0\n",
              "AveBedrms     0\n",
              "Population    0\n",
              "AveOccup      0\n",
              "Latitude      0\n",
              "Longitude     0\n",
              "HousePrice    0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>MedInc</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HouseAge</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>AveRooms</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>AveBedrms</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Population</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>AveOccup</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Latitude</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Longitude</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HousePrice</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.head(5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "ymL0Nu8tG8T-",
        "outputId": "024f5855-8c1f-4ef2-bfbd-7801c063b3f9"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \\\n",
              "0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   \n",
              "1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   \n",
              "2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   \n",
              "3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   \n",
              "4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   \n",
              "\n",
              "   Longitude  HousePrice  \n",
              "0    -122.23       4.526  \n",
              "1    -122.22       3.585  \n",
              "2    -122.24       3.521  \n",
              "3    -122.25       3.413  \n",
              "4    -122.25       3.422  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9fd738a5-97f4-4c37-8971-107c54783ae4\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>MedInc</th>\n",
              "      <th>HouseAge</th>\n",
              "      <th>AveRooms</th>\n",
              "      <th>AveBedrms</th>\n",
              "      <th>Population</th>\n",
              "      <th>AveOccup</th>\n",
              "      <th>Latitude</th>\n",
              "      <th>Longitude</th>\n",
              "      <th>HousePrice</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>8.3252</td>\n",
              "      <td>41.0</td>\n",
              "      <td>6.984127</td>\n",
              "      <td>1.023810</td>\n",
              "      <td>322.0</td>\n",
              "      <td>2.555556</td>\n",
              "      <td>37.88</td>\n",
              "      <td>-122.23</td>\n",
              "      <td>4.526</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>8.3014</td>\n",
              "      <td>21.0</td>\n",
              "      <td>6.238137</td>\n",
              "      <td>0.971880</td>\n",
              "      <td>2401.0</td>\n",
              "      <td>2.109842</td>\n",
              "      <td>37.86</td>\n",
              "      <td>-122.22</td>\n",
              "      <td>3.585</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>7.2574</td>\n",
              "      <td>52.0</td>\n",
              "      <td>8.288136</td>\n",
              "      <td>1.073446</td>\n",
              "      <td>496.0</td>\n",
              "      <td>2.802260</td>\n",
              "      <td>37.85</td>\n",
              "      <td>-122.24</td>\n",
              "      <td>3.521</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5.6431</td>\n",
              "      <td>52.0</td>\n",
              "      <td>5.817352</td>\n",
              "      <td>1.073059</td>\n",
              "      <td>558.0</td>\n",
              "      <td>2.547945</td>\n",
              "      <td>37.85</td>\n",
              "      <td>-122.25</td>\n",
              "      <td>3.413</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3.8462</td>\n",
              "      <td>52.0</td>\n",
              "      <td>6.281853</td>\n",
              "      <td>1.081081</td>\n",
              "      <td>565.0</td>\n",
              "      <td>2.181467</td>\n",
              "      <td>37.85</td>\n",
              "      <td>-122.25</td>\n",
              "      <td>3.422</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9fd738a5-97f4-4c37-8971-107c54783ae4')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-9fd738a5-97f4-4c37-8971-107c54783ae4 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-9fd738a5-97f4-4c37-8971-107c54783ae4');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 20640,\n  \"fields\": [\n    {\n      \"column\": \"MedInc\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.8998217179452732,\n        \"min\": 0.4999,\n        \"max\": 15.0001,\n        \"num_unique_values\": 12928,\n        \"samples\": [\n          5.0286,\n          2.0433,\n          6.1228\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"HouseAge\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 12.585557612111637,\n        \"min\": 1.0,\n        \"max\": 52.0,\n        \"num_unique_values\": 52,\n        \"samples\": [\n          35.0,\n          25.0,\n          7.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"AveRooms\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.4741731394243205,\n        \"min\": 0.8461538461538461,\n        \"max\": 141.9090909090909,\n        \"num_unique_values\": 19392,\n        \"samples\": [\n          6.111269614835948,\n          5.912820512820513,\n          5.7924528301886795\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"AveBedrms\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.47391085679546435,\n        \"min\": 0.3333333333333333,\n        \"max\": 34.06666666666667,\n        \"num_unique_values\": 14233,\n        \"samples\": [\n          0.9906542056074766,\n          1.112099644128114,\n          1.0398230088495575\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Population\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1132.4621217653375,\n        \"min\": 3.0,\n        \"max\": 35682.0,\n        \"num_unique_values\": 3888,\n        \"samples\": [\n          4169.0,\n          636.0,\n          3367.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"AveOccup\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10.386049562213591,\n        \"min\": 0.6923076923076923,\n        \"max\": 1243.3333333333333,\n        \"num_unique_values\": 18841,\n        \"samples\": [\n          2.6939799331103678,\n          3.559375,\n          3.297082228116711\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Latitude\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.1359523974571117,\n        \"min\": 32.54,\n        \"max\": 41.95,\n        \"num_unique_values\": 862,\n        \"samples\": [\n          33.7,\n          34.41,\n          38.24\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Longitude\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.003531723502581,\n        \"min\": -124.35,\n        \"max\": -114.31,\n        \"num_unique_values\": 844,\n        \"samples\": [\n          -118.63,\n          -119.86,\n          -121.26\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"HousePrice\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.1539561587441483,\n        \"min\": 0.14999,\n        \"max\": 5.00001,\n        \"num_unique_values\": 3842,\n        \"samples\": [\n          1.943,\n          3.79,\n          2.301\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Split data\n",
        "X = df.drop('HousePrice', axis=1)\n",
        "y = df['HousePrice']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)\n",
        "\n",
        "# Train model\n",
        "model = LinearRegression()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict\n",
        "prediction = model.predict(X_test)\n",
        "\n",
        "print(\"house price Prediction:\", prediction[0])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "75atDKZYGeTT",
        "outputId": "222a65e6-bd50-46c6-8b57-23a0f6e27518"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "house price Prediction: 0.7260490726242494\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import r2_score"
      ],
      "metadata": {
        "id": "gi2SF_QgGqeL"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "r2 = r2_score(y_test , prediction)"
      ],
      "metadata": {
        "id": "DusRskwiH1ow"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "r2"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eRytnpH6H16B",
        "outputId": "ae5bccc5-22f2-4890-9935-e3e3431cfdbf"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.595770232606166"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "m = model.coef_\n",
        "m"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "utvNYZTyH_zx",
        "outputId": "5b11ea92-a9b6-4f8a-8d0a-01d09835d750"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 4.45822565e-01,  9.68186799e-03, -1.22095112e-01,  7.78599557e-01,\n",
              "       -7.75740400e-07, -3.37002667e-03, -4.18536747e-01, -4.33687976e-01])"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "c = model.intercept_\n",
        "c"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4gW6pFdrIAuf",
        "outputId": "532b91ae-10b5-4953-a1da-260daf9b3ac0"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(-37.05624133152533)"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "m *4.0  + c"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QldPTxqdIDuc",
        "outputId": "6a2b2b4b-f01c-431e-c1a5-e1ba1b73ca18"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-35.27295107, -37.01751386, -37.54462178, -33.9418431 ,\n",
              "       -37.05624443, -37.06972144, -38.73038832, -38.79099324])"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "from sklearn.metrics import r2_score, mean_squared_error\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "r2 = r2_score(y_test, prediction)\n",
        "mse = mean_squared_error(y_test, prediction)\n",
        "\n",
        "print(\"R2 Score:\", r2)\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sABmpTiWIFo0",
        "outputId": "f96bdba6-d9e1-42f0-87fd-3b225e57a0cc"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "R2 Score: 0.595770232606166\n",
            "Mean Squared Error: 0.5305677824766758\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "plt.figure()\n",
        "plt.scatter(y_test, prediction)\n",
        "plt.xlabel(\"Actual House Prices\")\n",
        "plt.ylabel(\"Predicted House Prices\")\n",
        "plt.title(\"Actual vs Predicted House Prices\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "71urrYypJBkm",
        "outputId": "b9970d69-7ba2-421b-c08c-f50ad423b41d"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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iLwah5aIiIhIsZjIEBERkWIxkSEiIiLFYiJDREREisVEhoiIiBSLiQwREREpFhMZIiIiUiwmMkRERKRYTGSIiIhIsZjIEBERkWIxkSEiIiLFYiJDREREisVEhoiIiBSLiQwREREpFhMZIiIiUiwmMkRERKRYTGSIiIhIsZjIEBERkWIxkSEiIiLFYiJDREREisVEhoiIiBSLiQwREREpFhMZIiIiUiwmMkRERKRYTGSIiIhIsZjIEBERkWIxkSEiIiLFYiJDREREisVEhoiIiBTLrYnM/v370bdvX1StWhUqlQpff/210eWCIODNN99ETEwMgoKC0K1bN5w7d849wRIREZHHcWsik5eXh8aNG+Ojjz4ye/k777yD//73v1i6dCl+/fVXhISEoGfPnsjPz3dxpEREROSJ/Nx554888ggeeeQRs5cJgoBFixbhP//5D/r37w8A+Pzzz1GlShV8/fXXGDp0qCtDJSIiIg/ksXNk0tLSkJGRgW7duhmOaTQatGrVCklJSRavV1BQgOzsbKMfIiIi8k4em8hkZGQAAKpUqWJ0vEqVKobLzJk7dy40Go3hp0aNGk6Nk4iIiNzHYxMZe02dOhVZWVmGnytXrrg7JCIiInISj01koqOjAQDXrl0zOn7t2jXDZeao1WqEhYUZ/RAREZF38thEJj4+HtHR0dizZ4/hWHZ2Nn799Ve0bt3ajZERERGRp3DrqqXc3Fz8+eefhr/T0tKQkpKCiIgI1KxZEy+99BJmz56N2rVrIz4+HtOmTUPVqlUxYMAA9wVNREREHsOticzRo0fRuXNnw99TpkwBAIwcORKrV6/Gq6++iry8PDz77LO4c+cO2rVrh507dyIwMNBdIRMREZEHUQmCILg7CGfKzs6GRqNBVlYW58sQEREphNjzt8fOkSEiIiKyhYkMERERKRYTGSIiIlIsJjJERESkWExkiIiISLGYyBAREZFiMZEhIiIixWIiQ0RERIrFRIaIiIgUi4kMERERKRYTGSIiIlIsJjJERESkWExkiIiISLGYyBAREZFiMZEhIiIixWIiQ0RERIrFRIaIiIgUi4kMERERKRYTGSIiIlIsJjJERESkWExkiIiISLGYyBAREZFiMZEhIiIixWIiQ0RERIrFRIaIiIgUi4kMERERKRYTGSIiIlIsJjJERESkWExkiIiISLGYyBAREZFi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a75JdHQ0atWqZTaJEcPa65OYmAi1Wo3Lly+bxFWjRg1Du6ioKIwcORJr1qzBokWLsHz5crtiIVIy9sgQKcz27dtx+/ZtPP300yY1SR5//HF8+umnGD9+PKZPn46uXbsiISEBQ4cORXFxMb777ju89tprAErqyOzfvx9Dhw6FWq1GZGQkOnXqhBs3buCdd97BoEGDsHPnTuzYsQNhYWGG+6hduzb+97//oXnz5sjOzsYrr7xiV++PGP/617/QokULzJo1C0OGDEFSUhIWL16Mjz/+2NCmS5cuWLx4MVq3bg2tVovXXnsN/v7+hsvff/99xMTE4KGHHoKPjw+++uorREdHIzw8HN26dUPr1q0xYMAAvPPOO6hTpw6uXr1qmChta3jIEVOnTkXDhg3x/PPPY/z48QgICMC+ffvwxBNPIDIyEi+//DImT54MnU6Hdu3aISsrCwcPHkRYWBhGjhyJN998E82aNcODDz6IgoICbN++HfXr13davEQey71TdIhIqkcffVTo3bu32ct+/fVXAYBh0ummTZuEJk2aCAEBAUJkZKQwcOBAQ9ukpCShUaNGglqtFkp/FCxZskSoUaOGEBISIjz11FPCnDlzjCb7JicnC82bNxcCAwOF2rVrC1999ZXJxGGImOzbv39/k+NlJ8QKgiBs3LhRSExMFPz9/YWaNWsK7777rtF1/v77b6FHjx5CSEiIULt2beG7774zmuy7fPlyoUmTJkJISIgQFhYmdO3aVUhOTjZcPzs7W3jhhReEqlWrCv7+/kKNGjWE4cOHC5cvX7YYv63HZ+5yc4/txx9/FNq0aSOo1WohPDxc6Nmzp+FynU4nLFq0SKhbt67g7+8vREVFCT179hR++uknQRAEYdasWUL9+vWFoKAgISIiQujfv79w4cIFizEReSuVIJQZXCYiIiJSCM6RISIiIsViIkNERESKxUSGiIiIFIuJDBERESkWExkiIiJSLCYyREREpFhMZIiIiEixmMgQERGRYjGRISIiIsViIkNERESKxUSGiIiIFIuJDBERESnW/wNOJdrgsO83mwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Ua7wEtgpJftH"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
