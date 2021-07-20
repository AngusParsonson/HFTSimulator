from abc import ABC, abstractmethod

class Trader(ABC):
    @abstractmethod
    def respond(self, tick):
        '''Called by the exchange whenever there is a new tick

        Args:
            tick: current top of limit order book (level 1)
                tick['Local time'] = time in seconds since start of trading day
                tick['Ask'] = current best ask price
                tick['Bid'] = current best bid price
                tick['AskVolume'] = current best ask volume
                tick['BidVolume'] = current best bid volume
                tick['DeltaT'] = time in seconds since last tick
        Returns:
            A list of dicts each representing a message to the order book 
            [{
                type: "BID" or "ASK",
                price: float,
                quantity: float
            }]
        '''
        pass
    
    @abstractmethod
    def filled_order(self, order):
        '''Called by the exchange whenever on of this trader's orders is filled

        Args:
            order: A dict representing the fulfilled order 
            {
                type: "BID" or "ASK",
                price: float,
                quantity: float
            }
        '''
        pass
