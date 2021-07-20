import time
import os
from utils.data_utils import TradingDataLoader
from traders.ode_trader import ODETrader

if __name__ == '__main__':
    days = os.listdir(__file__ + "\\..\\utils\\data\\")
    # days = ["WTB.GBGBX_Ticks_27.04.2021-27.04.2021.csv"]
    for day in days:
        print("trading day: " + str(day))
        data_loader = TradingDataLoader(__file__ + "\\..\\utils\\data\\" + day)
        traders = []
        orders = []
        traders.append(ODETrader(init_balance=4000, pred_horizon=50))
        curr_tick = data_loader.step()
        speed = float('inf')
        while(curr_tick[1]):
            market_price = ((curr_tick[0]['Bid'] * curr_tick[0]['AskVolume'] + 
                        curr_tick[0]['Ask'] * curr_tick[0]['BidVolume']) / 
                        (curr_tick[0]['AskVolume'] + curr_tick[0]['BidVolume']))
            # print(orders)
            # Main loop
            for i, t in enumerate(traders):
                t_orders = t.respond(curr_tick[0])
                if (t_orders == None): continue
                else:
                    for o in t_orders:
                        orders.append((i, o))
                
            unfilled = []
            while (len(orders) > 0):
                order = orders.pop(0)
                if (order[1]['type'] == 'BID'):
                    if (order[1]['price'] >= market_price):
                        order[1]['price'] = market_price
                        order[1]['quantity'] = 1
                        traders[order[0]].filled_order(order[1])
                    else:
                        unfilled.append(order)
                else:
                    if (order[1]['price'] <= market_price):
                        order[1]['price'] = market_price
                        order[1]['quantity'] = 1
                        traders[order[0]].filled_order(order[1])
                    else:
                        unfilled.append(order)
                    # print("received ask from trader: " + str(order[0]))
            orders = unfilled
            # print(orders)
            # Next tick
            if (speed != float('inf')):
                time.sleep(curr_tick[1]/speed)
            curr_tick = data_loader.step()

        results = []
        for t in traders:
            results.append(t.print_vals())
    print(results)
        