import streamlit as st
from datetime import datetime, date, timedelta

from plotting_utils import (
    plot_candlesticks,
)
from utils import get_sp500_tickers, get_nse_tickers
from data_processing import (
    check_if_today_starts_with_vertical_green_overlay,
)

# from models import train_model, test_model
from strategies.backtesting import backtest_strategy
from utils import setup_logger
from models.predictive_sma20_crossover_model import PredictiveSma20CrossoverModel
from models.predictive_macd_crossover_model import PredictiveMacdCrossoverModel
from models.bollinger_bands_metalabel import BollingerBandsMetalabel
from models.rolling_precision_recall_model import RollingPrecisionRecallModel
_ = PredictiveMacdCrossoverModel
_=PredictiveSma20CrossoverModel
_=BollingerBandsMetalabel
_=RollingPrecisionRecallModel


def test_date_input_handler():
    """
    Handles the input for the test date, initializes session state variables if not present, and provides buttons for moving the date forward or backward.

    Parameters:
    None

    Returns:
    None
    """
    # Initialize session state variables if not present
    if "curr_date" not in st.session_state:
        st.session_state.curr_date = datetime.now().date()
    if "update_flag" not in st.session_state:
        st.session_state.update_flag = False

    # Buttons for moving the date forward or backward
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("←", key="back"):
            st.session_state.curr_date -= timedelta(days=1)
            st.session_state.update_flag = not st.session_state.update_flag
    with col2:
        st.write("use these arrows to traverse the calendar")
    with col3:
        if st.button("→", key="forward"):
            st.session_state.curr_date += timedelta(days=1)
            st.session_state.update_flag = not st.session_state.update_flag


def main():
    train_until = date(2019, 1, 1)
    selected_date = st.date_input("Train Until: ", train_until)
    train_until = selected_date.strftime("%Y-%m-%d")

    if st.toggle("Indian ticker data"):
        options = get_nse_tickers()
        st.session_state["data_source"] = "nse"
    else:
        options = get_sp500_tickers()
        st.session_state["data_source"] = "yf"

    selected_option = st.selectbox(
        "Select a stock", options, index=0, key="my_selectbox"
    )

    # Drop down to select model/strategy
    model_names = [
        "RollingPrecisionRecallModel",
        'BollingerBandsMetalabel',
        "PredictiveMacdCrossoverModel",
        "PredictiveSma20CrossoverModel",
    ]
    selected_model_name = st.selectbox("Select a Model", model_names)
    if selected_model_name:
        selected_model_class = globals()[selected_model_name]

    st.write("You selected:", selected_option)

    if st.button("Train Model"):
        # model = PredictiveMacdCrossoverModel(selected_option, train_until)
        model = selected_model_class(selected_option, train_until)
        model_results = model.run_train()
        backtest_strategy(model_results["df_test"])

    with st.expander("Test Model"):
        last_n_days = st.text_input("Last N Days", "30")
        test_date_input_handler()

        # Display the date input with the current date from session state
        selected_end_date = st.date_input(
            "End Date",
            value=st.session_state.curr_date,
            key=st.session_state.update_flag,
        )

        st.write(f"End Date: {st.session_state.curr_date}")
        if st.button("Test Model", key="btn2"):
            # model = PredictiveMacdCrossoverModel(selected_option, train_until)
            model = selected_model_class(selected_option, train_until)
            # Adding 1 day to the end date because yfinance downloads data up to one day before the end date
            df_test = model.run_test(
                selected_option,
                last_n_days,
                (
                    datetime.combine(
                        selected_end_date + timedelta(days=1), datetime.min.time()
                    )
                ),
                data_source=st.session_state["data_source"],
            )
            plot_candlesticks(df_test)

    with st.expander("Scan all stocks where the model recommends a buy today"):
        last_n_days = st.text_input("Last N Days", "30", key="txt2")
        mlist = []
        if st.button("Scan Stocks", key="btn3"):
            for so in options:
                logger = setup_logger(so)

                try:
                    # model = PredictiveMacdCrossoverModel(so, train_until, data_source=st.session_state['data_source'])
                    model = selected_model_class(
                        so, train_until, data_source=st.session_state["data_source"]
                    )
                    model_results = model.run_train()

                    # Log conditions and decisions
                    logger.info(
                        f"Train Accuracy: {model_results['train_accuracy']}, Test Accuracy: {model_results['test_accuracy']}"
                    )
                    logger.info(
                        f"Train Precision: {model_results['train_precision']}, Test Precision: {model_results['test_precision']}"
                    )

                    if (
                        model_results["train_accuracy"] > 0.6
                        and model_results["test_accuracy"] > 0.6
                        and model_results["test_precision"] > 0.6
                        and model_results["train_precision"] > 0.6
                    ):
                        df_test = model.run_test(
                            so, last_n_days, data_source=st.session_state["data_source"]
                        )
                        if check_if_today_starts_with_vertical_green_overlay(df_test):
                            mlist.append(so)
                            logger.info(f"Buy recommendation for {so}")
                        else:
                            logger.info(f"No recommendation for {so}")
                except Exception as e:
                    logger.error(f"Failed to process {so} due to error: {str(e)}")
            st.write(mlist)


if __name__ == "__main__":
    main()
