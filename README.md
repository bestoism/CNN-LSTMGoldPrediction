# Indonesian Gold Price Prediction with Deep Learning

This project uses a Deep Learning model (LSTM + Conv1D) to predict gold prices in Indonesia based on historical data.

## Features
- Data preprocessing with normalization
- Sequence creation for time series
- Combined Conv1D and LSTM model
- Visualization of prediction results and training loss

## Folder Structure
```
.
├── ai.py
├── 1990-2021.csv
├── 1990-2021 copy.csv
├── .vscode/
└── README.md
```

## How to Run

1. **Clone this repository**
2. **Install dependencies**
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib
   ```
3. **Run the program**
   ```bash
   python ai.py
   ```

## Dataset
- `1990-2021.csv`: Indonesian monthly gold price data from 1990 to 2021.

## Results
- The model will display prediction vs actual graphs and training loss graphs.

## Contribution
Feel free to create a pull request or open an issue for improvements or feature additions.

