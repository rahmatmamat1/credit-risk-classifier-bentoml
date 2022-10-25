# Deploy Credit Risk Classifier with BentoML

[BentoML](https://github.com/bentoml/BentoML) is an end-to-end solution for machine learning model serving. It facilitates Data Science teams to develop production-ready model serving endpoints, with DevOps best practices and performance optimization at every stage.

BentoML offers a flexible and performant framework to serve, manage, and deploy ML models in production. It simplifies the process of building production-ready model API endpoints through its standard and easy architecture. It empowers teams with a powerful dashboard to help organize models and monitor deployments centrally.

## Dataset
The raw dataset is in the file "CreditScoring.csv" which contains 4455 rows and 14 columns:
* Status: credit status
* Seniority: job seniority (years)
* Home: type of home ownership
* Time:time of requested loan
* Age: client's age
* Marital: marital status
* Records: existance of records
* Job: type of job
* Expenses: amount of expenses
* Income: amount of income
* Assets: amount of assets
* Debt: amount of debt
* Amount: amount requested of loan
* Price: price of good

## Files
* `README.md`: Description and explanation of the project
* `CreditScoring.csv`: CSV file contains Credit Scoring dataset
* `requirements.txt`: library requirement for this project
* `train.py`: python script that will train and save the model to bentoml
* `service.py`: python script for serving ML models using BentoML
* `bentofile.yaml`: file used for creating bento. for more information: https://docs.bentoml.org/en/latest/concepts/bento.html

## Steps
1. Install Dependencies

    ```
    pip install -r requirements.txt
    ```
2. Train model and save to bentoml

    ```
    python train.py
    ```
    This will save a new model in the BentoML local model store, a new version tag is automatically generated when the model is saved. You can see all model revisions from CLI via bentoml models commands:
    ```bash
    bentoml models get credit_risk_model:latest
    bentoml models list
    bentoml models --help
    ```
3. Serving the model

    ```
    bentoml serve service:svc --reload
    ```
    Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests.
    You may also send request with curl command or any HTTP client, e.g.:
    ```bash
    curl -X 'POST' \
    'http://localhost:3000/classify' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"seniority": 3,
    "home": "owner",
    "time": 36,
    "age": 26,
    "marital": "single",
    "records": "no",
    "job": "freelance",
    "expenses": 35,
    "income": 0.0,
    "assets": 60000.0,
    "debt": 3000.0,
    "amount": 800,
    "price": 1000
    }'
    ```
4. Building Bento

    Once the service definition is finalized, we can build the model and service into a bento. Bento is the distribution format for a service. It is a self-contained archive that contains all the source code, model files and dependency specifications required to run the service.

    To begin with building Bento, it need `bentofile.yaml` under your project directory:
    ```yaml
    service: "service:svc"  # Same as the argument passed to `bentoml serve`
    labels:
        owner: bentoml-team
        stage: dev
    include:
    - "*.py"  # A pattern for matching which files to include in the bento
    python:
        packages:  # Additional pip packages required by the service
        - scikit-learn
        - xgboost
    ```
    Next, run the `bentoml build` CLI command from the same directory:

    ```bash
    $ bentoml build
    Building BentoML service "credit_risk_classifier:cltplfkuk6qp5dhm" from build context "D:\Rahmatsyah Firdaus\Courses\DataTalksClub\machine-learning-zoomcamp\07-bentoml"
    Packing model "credit_risk_model:ibjwp4cs4k2xddhm"
    Locking PyPI package versions..

    ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
    ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
    ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
    ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
    ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
    ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

    Successfully built Bento(tag="credit_risk_classifier:cltplfkuk6qp5dhm")

    ```
    A new Bento is now built and saved to local Bento store. You can view and manage it via 
    `bentoml list`,`bentoml get` and `bentoml delete` CLI command.
    you can now serve it with the `bentoml serve` CLI command:
    ```bash
    bentoml serve credit_risk_classifier:latest --production
    ```

5. Containerize

    A docker image can be automatically generated from a Bento for production deployment, via the bentoml containerize CLI command:
    ```bash
    bentoml containerize credit_risk_classifier:latest
    ```
    ```bash
    $ docker images
    REPOSITORY                                TAG                IMAGE ID       CREATED         SIZE
    credit_risk_classifier                    cltplfkuk6qp5dhm   a8a98f72e8b6   10 seconds ago  1.1GB
    ```
    Run the docker image to start the BentoServer:
    ```bash
    docker run -it --rm -p 3000:3000 credit_risk_classifier:cltplfkuk6qp5dhm serve --production
    ```

6. Deployment on Google Cloud Run

    Cloud Run is Google Cloud's serverless solution for containers. With Cloud Run, you can develop and deploy highly scalable containerized applications on a fully managed serverless platform. Cloud Run is great for running small to medium models since you only pay for the compute you use and it is super scalable.

    you can deploy your service by using [bentoctl](https://github.com/bentoml/bentoctl). bentoctl supports AWS Lambda, AWS SageMaker, AWS EC2, Google Cloud Run, Google Compute Engine, Azure Container Instances, Heroku. you can follow the instruction [here](https://github.com/bentoml/google-cloud-run-deploy) to deploy your service to Google Cloud Run.

    Another way, you can also push your docker image that was created earlier to the Google Container Registry by following the documentation [here](https://cloud.google.com/container-registry/docs/pushing-and-pulling). and then deploy it to the Cloud Run.