import backtrader as bt

class CustomData(bt.feeds.PandasData):
    # Add a 'lines' definition for your custom data line
    lines = ('pred',)

    # add the parameter to the parameters inherited from the base class
    params = (('pred', -1),)