# Potato Disease Classification

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Installation](#installation)
  * [Deploying the TF Model (.h5) on GCP](#Deploying the TF Model (.h5) on GCP)
  * [Directory Tree](#directory-tree)
  * [Technologies Used](#technologies-used)
  * [About Me](#about-me)
  * [Links](#links)
  * [Skills](#skills)
  * [License](#license)


## Demo
Link: [https://flight-price-prediction-webui.herokuapp.com/](https://flight-price-prediction-webui.herokuapp.com/)

[![](https://i.imgur.com/axcQn0y.png)](https://flight-price-prediction-webui.herokuapp.com/)

[![](https://i.imgur.com/BFERBOA.png)](https://flight-price-prediction-webui.herokuapp.com/)

## Overview
This is a potato disease classification app which predicts potato leaf disease.

## Installation(#installation)

### Setup for Python:

1. Install Python ([Setup instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install Python packages

```
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

3. Install Tensorflow Serving ([Setup instructions](https://www.tensorflow.org/tfx/serving/setup))

### Setup for ReactJS

1. Install Nodejs ([Setup instructions](https://nodejs.org/en/download/package-manager/))
2. Install NPM ([Setup instructions](https://www.npmjs.com/get-npm))
3. Install dependencies

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```

4. Change API url in `.env`.

### Setup for React-Native app

1. Go to the [React Native environment setup](https://reactnative.dev/docs/environment-setup), then select `React Native CLI Quickstart` tab.  

2. Install dependencies

```bash
cd mobile-app
yarn install
```

3. Change API url in `.env`.

### Training the Model

1. Download the data from [kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).
2. Only keep folders related to Potatoes.
3. Run Jupyter Notebook in Browser.

```bash
jupyter notebook
```

4. Open `training.ipynb` in Jupyter Notebook.
5. In cell #2, update the path to dataset.
6. Run all the Cells one by one.
7. Copy the model generated and save it with the version number in the `saved_models` folder.

### Running the API

#### Using FastAPI

1. Get inside `api` folder

```bash
cd api
```

2. Run the FastAPI Server using uvicorn

```bash
uvicorn main:app --reload --host 0.0.0.0
```

3. Your API is now running at `0.0.0.0:8000`

#### Using FastAPI & TF Serve

1. Get inside `api` folder

```bash
cd api
```

2. update the paths in file to get access latest model from saved_models directory
3. Run the TF Serve (Update config file path below)

```bash
docker run -it -v C:/potatodisease:/potatodisease -p 8501:8501 --entrypoint /bin/bash tensorflow/serving
tensorflow_model_server --rest_api_port=8501 --model_name=potata-diesease-classification --model_base_path=/potatodiesease/saved_models/

```

4. Run the FastAPI Server using uvicorn
   For this you can directly run it from your main.py or main-tf-serving.py
   OR you can run it from command prompt as shown below,

```bash
uvicorn main-tf-serving:app --reload --host 0.0.0.0
```

5. Your API is now running at `0.0.0.0:8000`

### Running the Frontend

1. Get inside `api` folder

```bash
cd frontend
```

2. In `.env` and update `REACT_APP_API_URL` to API URL if needed.
3. Run the frontend

```bash
npm run start
```

### Running the app

1. Get inside `mobile-app` folder

```bash
cd mobile-app
```

2. In `.env` and update `URL` to API URL if needed.

3. Run the app (android)

```bash
npm run android
```

4. Creating public ([signed APK](https://reactnative.dev/docs/signed-apk-android))

### Experimenting tracking
```bash
pip install wandb
```
```run the tracking with 
wandb.init(anonymous='allow', project="potato-diesease-classification", name= "convnet architecture")
```

## Deploying the TF Model (.h5) on GCP

1. Create a [GCP account](https://console.cloud.google.com/freetrial/signup/tos?_ga=2.25841725.1677013893.1627213171-706917375.1627193643&_gac=1.124122488.1627227734.Cj0KCQjwl_SHBhCQARIsAFIFRVVUZFV7wUg-DVxSlsnlIwSGWxib-owC-s9k6rjWVaF4y7kp1aUv5eQaAj2kEALw_wcB).
2. Create a [Project on GCP](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) (Keep note of the project id).
3. Create a [GCP bucket](https://console.cloud.google.com/storage/browser/).
4. Upload the tf .h5 model generate in the bucket in the path `models/potato-model.h5`.
5. Install Google Cloud SDK ([Setup instructions](https://cloud.google.com/sdk/docs/quickstarts)).
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id
```

8. Your model is now deployed.
9. Use Postman to test the GCF using the [Trigger URL](https://cloud.google.com/functions/docs/calling/http).

Inspiration: https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions

## Directory Tree 
```
├── api 
│   ├── main.py
    ├── main-tf-serving.py
    ├── requirements.txt
├── frontend
│   ├── node_modules directory
    ├── public directory
    ├── src directory
    ├── .env
├── gcp
|   ├── main.py
    ├── requirements.txt
├── mobile-app directory
├── saved_models directory
├── test_images
├── training
    ├── dataset
    ├── PlantVillage
    ├── wandb
    ├── models.config
    ├── potato-disease-classification-model-using-image-data-generator.ipynb
    ├── training.ipynb
├── wandb
├── potatoes.h5
├── potatoes1.h5
├── README.md
├── LICENSE
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.gstatic.com/devrel-devsite/prod/vdb149cdc08c87ab249cdebfec6395e8f073056d752ca9c2d285d3b8426fcfa32/tensorflow/images/lockup.svg](https://www.tensorflow.org/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/)




## 🚀 About Me
Machine Learning Expert | I wish to do connect & collaborate together with the best minds in AI & making a successful project that impacts millions of people....


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/hemantkumar0506)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


## 🛠 Skills
Python, TensorFlow, numpy, pandas, PyTorch, Flask, Docker, MLOps, Fastapi, Google Cloud Platform, scikit-learn, Weight&Bias, ApacheSpark, SQL,XGBoost ...


## License

[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)



Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


