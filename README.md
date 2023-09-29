## Fetch AI Assessment

Shallow Neural Net trained on receipt data from 2021 that makes predictions on receipt counts in the future. Predictions tend to be monotonically increasing which does seem to match the trend of a growing userbase pretty well. Tried a simple linear regression, as well as more complex LSTM, ARIMA, and ETS models for time series. Ultimately decided on making a neural net which uses lag values to train on its own predictions as well as the trends seem to be relatively simple and those other models don't seem to perform relaitvely well even with feature engineering. It'd Likely a good idea to audit the model in the future especially if the trends grow more complex with the introduction of seasonality. I would have also loved to add visualization, something I could very likely do within my frontend if given more time. There might be a performing model or hyperparameter tuning that performs better, but I don't have the time to do in-depth research currently.

### Built With

* [PyTorch] [https://pytorch.org/]
* [Astro][https://astro.build/]
* [Flask][https://flask.palletsprojects.com/en/2.3.x/]
* [node.js][https://nodejs.org/en]
* [Docker][https://www.docker.com/]

## Getting Started

### Prerequisites

Make sure you have nvm for easy package management.
* MacOS - Using HomeBrew
  ```sh
  brew install nvm
  ```
* Linux - 
  ```sh
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
  ```
   Note - Make sure to source nvm before you can use it
   ```sh
    source ~/.nvm/nvm.sh
   ```
* Install Node Version
  ```sh
    nvm install 18.17.1
    nvm use 18.17.1
  ```

### Setup

1. Clone the repo
   ```sh
   git clone https://github.com/dronavallipranav/FetchBackendAPI.git
   ```
2. build the docker image and run the container backend on port 5001
   ```sh
   docker build -t backend_model .
   docker run -it -p 5001:5001 backend_model
   ```
Once the server is started, open a new terminal instance
3. Install dependencies
   ```sh
   cd frontend/webApp
   npm install
   ```
4. Start the frontend on port 4321
    ```sh
   npm run dev
   ```

## Usage

Enter number of days in web app to get a table with all dates from number of days after 2022 and their corresponding receipt_count predictions

## Contact

Pranav Dronavalli - dronavallipranav@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
