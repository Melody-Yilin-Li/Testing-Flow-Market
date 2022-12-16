import numpy as np 
import pandas as pd
import itertools 
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import seaborn as sns
import faulthandler; faulthandler.enable()
from functools import reduce                # Import reduce function
from sys import exit


# Replace NaN by empty dict
def replace_nans_with_dict(series):
    for idx in series[series.isnull()].index:
        series.at[idx] = {}
    return series

# Explodes list and dicts
def df_explosion(df, col_name:str):
    if df[col_name].isna().any():
        df[col_name] = replace_nans_with_dict(df[col_name])
    df.reset_index(drop=True, inplace=True)
    df1 = pd.DataFrame(df.loc[:,col_name].values.tolist())
    df = pd.concat([df,df1], axis=1)
    df.drop([col_name], axis=1, inplace=True)
    return df

plt.close()

# input session constants 
num_groups = 5
players_per_group = 12
prac_rounds = 2 
num_rounds = 14
round_length = 120
leave_out_seconds = 15
ce_price = [10, 10, 10, 7, 7, 7, 11, 11, 11, 7, 7, 7]
ce_quantity = [1200, 1200, 1200, 1200, 1200, 1200, 1500, 1500, 1500, 1100, 1100, 1100]
ce_profit = [10600, 10600, 10600, 11600, 11600, 11600, 3750, 3750, 3750, 10500, 10500, 10500]
if players_per_group == 12: 
    ce_quantity = [2 * v for v in ce_quantity]
    ce_profit = [2 * v for v in ce_profit]

# read in data 

# df for market prices and rates/quantities for all groups 
# create a list of dfs to be merged 
groups_mkt_flow = []
prices = []
rates = []
cum_quantities = []
colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'peachpuff', 'lightskyblue'] # add more colors with more than 6 groups

for g in range(1, num_groups + 1):
    name = 'group' + str(g)
    group = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = '/Users/YilinLi/Documents/UCSC/Flow Data/flow' + str(g) + '/' + str(r + 2) + '/1_market.json'
        rnd = pd.read_json(
            path,
        )
        rnd.fillna(0, inplace=True)
        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['before_transaction'] == False)]
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        if max(rnd['timestamp']) < 119:     # add missing timestamps back with all zeros
            makeup = pd.DataFrame(np.zeros((round_length - leave_out_seconds - rnd.shape[0], rnd.shape[1])), columns=rnd.columns)
            rnd = pd.concat([rnd, makeup], axis=0, ignore_index=True)
        rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        # print(rnd)
        group.append(rnd) 
    df = pd.concat(group, ignore_index=True, sort=False)
    price = 'clearing_price_' + str(g)
    prices.append(price)
    rate = 'clearing_rate_' + str(g)
    rates.append(rate)
    cumsum = 'cumulative_quantity' + str(g)
    cum_quantities.append(cumsum)
    df.columns = ['timestamp', price, rate, cumsum]
    df['timestamp'] = np.arange(1, len(df) + 1)
    groups_mkt_flow.append(df)
    
# merge the list of df's
data_groups_mkt_flow = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['timestamp']),
                     groups_mkt_flow)
data_groups_mkt_flow = data_groups_mkt_flow.replace(0, np.NaN)
data_groups_mkt_flow['mean_clearing_price'] = data_groups_mkt_flow[prices].mean(skipna=True, axis=1)
data_groups_mkt_flow['mean_clearing_rate'] = data_groups_mkt_flow[rates].mean(skipna=True, axis=1)
data_groups_mkt_flow['mean_cumulative_quantity'] = data_groups_mkt_flow[cum_quantities].mean(skipna=True, axis=1)
data_groups_mkt_flow = data_groups_mkt_flow.replace(np.NaN, 0)
data_groups_mkt_flow['ce_price'] = [p for p in ce_price for i in range(round_length - leave_out_seconds)]
data_groups_mkt_flow['ce_quantity'] = [q for q in ce_quantity for i in range(round_length - leave_out_seconds)]
data_groups_mkt_flow['ce_rate'] = data_groups_mkt_flow['ce_quantity'] / (round_length - leave_out_seconds) 

# plot clearing prices in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(prices)): 
    lab = 'group' + str(l + 1)
    plt.step(data=data_groups_mkt_flow, x='timestamp', y=prices[l], where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow, x='timestamp', y='mean_clearing_price', where='pre', c='green', label='Mean Price')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_price', where='pre', c='plum', label='CE Price')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.ylim(0, 20)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Flow Clearing Prices vs Time')
plt.savefig('groups_flow_price.png')
plt.close()

# plot clearing rates in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(rates)): 
    lab = 'group' + str(l + 1)
    plt.step(data=data_groups_mkt_flow, x='timestamp', y=rates[l], where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow, x='timestamp', y='mean_clearing_rate', where='pre', c='green', label='Mean Rate')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_rate', where='pre', c='plum', label='CE Rate')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.ylabel('Units/second')
plt.title('Flow Clearing Rates vs Time')
plt.savefig('groups_flow_rate.png')
plt.close()

# plot cumulative quantities in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(rates)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_mkt_flow['timestamp'], data_groups_mkt_flow[cum_quantities[l]], c=colors[l], label=lab)
plt.plot(data_groups_mkt_flow['timestamp'], data_groups_mkt_flow['mean_cumulative_quantity'], c='green', label='Mean Cumulative Quantity')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_quantity', where='pre', c='plum', label='CE Quantity')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.ylabel('Units')
plt.title('Flow Cumulative Quantity vs Time')
plt.savefig('groups_flow_cumsum.png')
plt.close()

# exit(0) # exit the script 

groups_par_flow = []

for g in range(1, num_groups + 1): 
    # dictionary for market prices and rates/quantities 
    # create a list of dataframes to be concatenated after groupby 
    data_mkt = []

    # each round X is denoted as 'mktX'
    market = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'mkt' + str(r)
        path = '/Users/YilinLi/Documents/UCSC/Flow Data/flow' + str(g) + '/' + str(r + 2) + '/1_market.json'
        market[name] = pd.read_json(
            path,
            )
        market[name].fillna(0, inplace=True)
        # print("market", market[name])
        market[name]['unit_weighted_price'] = market[name]['clearing_price'] * market[name]['clearing_rate']
        df = market[name][market[name]['before_transaction'] == False].groupby('id_in_subsession').aggregate({'clearing_price': 'mean', 'clearing_rate': 'sum', 'unit_weighted_price': 'sum'}).reset_index()
        df['unit_weighted_price'] = df['unit_weighted_price'] / df['clearing_rate']
        df['ce_price'] = ce_price[r - 1]
        df['round'] = r
        df.columns = ['group_id', 'time_weighted_price_' + str(g), 'quantity_' + str(g), 'unit_weighted_price_' + str(g), 'ce_price_' + str(g), 'round']
        df.fillna(0, inplace=True)
        data_mkt.append(df)
        # print(name, '\n', market[name])

    # print('MARKET', data_mkt)

    # dictionary for participant cash, inventories, and transaction rates if any
    # create a list of dataframes to be concatenated after groupby 
    data_par = []

    # each round X is denoted as 'parX'
    participant = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'par' + str(r)
        path = '/Users/YilinLi/Documents/UCSC/Flow Data/flow' + str(g) + '/' + str(r + 2) + '/1_participant.json' 
        participant[name] = pd.read_json(
            path,
            )
        participant[name].fillna(0, inplace=True)
        participant[name] = participant[name].explode('executed_contracts')
        participant[name].reset_index(drop=True, inplace=True)
        participant[name] = df_explosion(participant[name], 'executed_contracts')
        # tmp_df = participant[name][(participant[name]['before_transaction'] == False) & (participant[name]['timestamp'] == round_length - 1)]
        tmp_df = participant[name][(participant[name]['before_transaction'] == False) & (participant[name]['timestamp'] == max(participant[name]['timestamp']))]
        # print("TMP", tmp_df, tmp._df.columns)
        # print('round', r)
        if 'fill_quantity' not in tmp_df.columns: 
            tmp_df = tmp_df.explode('active_contracts')
            tmp_df.reset_index(drop=True, inplace=True)
            tmp_df = df_explosion(tmp_df, 'active_contracts')
            df = tmp_df[-players_per_group:].groupby('id_in_subsession').aggregate({'cash': 'sum', 'fill_quantity': 'sum', 'quantity': 'sum'}).reset_index()
            df['cash'] = 0
        else: 
            df = tmp_df.groupby('id_in_subsession').aggregate({'cash': 'sum', 'fill_quantity': 'sum', 'quantity': 'sum'}).reset_index()
        df['ce_profit'] = ce_profit[r - 1]
        df['ce_quantity'] = ce_quantity[r - 1] 
        df['payoff_percent'] = round(df['cash'] / df['ce_profit'], 4)
        df['contract_percent'] = round(df['fill_quantity'] / df['ce_quantity'] / 2, 4)
        df['round'] = r
        df.columns = ['group_id', 'payoff_' + str(g), 'fill_quantity_' + str(g), 'contract_quantity_' + str(g), 'ce_profit_' + str(g), 'ce_quantity_' + str(g), 'payoff_percent_' + str(g), 'contract_percent_' + str(g), 'round']
        df.fillna(0, inplace=True)
        # print("DF", df)
        data_par.append(df)
        # print(name, participant[name])
    # print("PARTICIPANT", data_par)
    # print(market, participant)

    ########## Within-period ##########
    # price 
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('Flow prices vs time')
    for r in range(1, num_rounds - prac_rounds + 1):
        sns.lineplot(ax=axes[(r - 1) // 3, (r - 1) % 3], 
        data=market['mkt'+str(r)], 
        x='timestamp', 
        y='clearing_price', 
        drawstyle='steps-pre', 
        hue='id_in_subsession',
        legend=False,)
        axes[(r - 1) // 3, (r - 1) % 3].set_xlim(0, round_length)
        axes[(r - 1) // 3, (r - 1) % 3].set_ylim(0, 18)
        axes[(r - 1) // 3, (r - 1) % 3].set_xticks(np.arange(0, round_length + 1, 15))
        axes[(r - 1) // 3, (r - 1) % 3].axhline(y=ce_price[r - 1], xmin=0, xmax=round_length + 1, c='green')
    plt.savefig('within_price_flo_' + str(g) + '.png')
    plt.close()

    # rate
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('Flow rates vs time')
    for r in range(1, num_rounds - prac_rounds + 1):
        sns.lineplot(ax=axes[(r - 1) // 3, (r - 1) % 3], 
        data=market['mkt'+str(r)], 
        x='timestamp', 
        y='clearing_rate', 
        drawstyle='steps-pre', 
        hue='id_in_subsession',
        legend=False,)
        axes[(r - 1) // 3, (r - 1) % 3].set_xlim(0, round_length)
        axes[(r - 1) // 3, (r - 1) % 3].set_ylim(0, 40)
        axes[(r - 1) // 3, (r - 1) % 3].set_xticks(np.arange(0, round_length + 1, 15))
    plt.savefig('within_rate_flo_' + str(g) + '.png')
    plt.close()


    ########## Between-period ##########
    between_df1 = pd.concat(data_mkt, ignore_index=True, sort=False)
    between_df2 = pd.concat(data_par, ignore_index=True, sort=False)
    between_df = pd.merge(between_df1, between_df2, on=['group_id', 'round'])
    between_df = between_df.drop(columns=['group_id'])
    groups_par_flow.append(between_df)

    # time-weighted mean price 
    sns.lineplot(data=between_df, x='round', y='time_weighted_price_' + str(g))
    plt.step(data=between_df, x='round', y='ce_price_' + str(g), where='pre', c='green', label='CE Price')
    plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left', 
        borderaxespad=0.5)
    plt.ylim(0, max(between_df['time_weighted_price_' + str(g)]) * 1.1)
    plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
    plt.title('Time-Weighted Price')
    plt.savefig('between_time_weighted_price_' + str(g) + '.png')
    plt.close()

    # unit-weighted price 
    sns.lineplot(data=between_df, x='round', y='unit_weighted_price_' + str(g))
    plt.step(data=between_df, x='round', y='ce_price_' + str(g), where='pre', c='green', label='CE Price')
    plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left', 
        borderaxespad=0.5)
    plt.ylim(0, max(between_df['unit_weighted_price_' + str(g)]) * 1.1)
    plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
    plt.title('Unit-Weighted Price')
    plt.savefig('between_unit_weighted_price_' + str(g) + '.png')
    plt.close()

    # volume
    sns.lineplot(data=between_df, x='round', y='quantity_' + str(g))
    plt.step(data=between_df, x='round', y='ce_quantity_' + str(g), where='pre', c='red', label='CE Quantity')
    plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left', 
        borderaxespad=0.5)
    plt.ylim(0, max(ce_quantity) * 1.1)
    plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
    plt.title('Quantity')
    plt.savefig('between_quantity_' + str(g) + '.png')
    plt.close()

    # total payoff
    sns.lineplot(data=between_df, x='round', y='payoff_percent_' + str(g))
    # plt.ylim(0, 1.1)
    plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
    plt.ylim(0, 2)
    plt.title('%CE Payoff')
    plt.savefig('between_payoff_' + str(g) + '.png')
    plt.close()

    # contract execution 
    sns.lineplot(data=between_df, x='round', y='contract_percent_' + str(g))
    plt.ylim(0, 1.5)
    plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
    plt.title('%Contract Execution')
    plt.savefig('between_contract_percent_' + str(g) + '.png')
    plt.close()


# merge the list of df's
data_groups_par_flow = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['round']),
                     groups_par_flow)         

data_groups_par_flow = data_groups_par_flow.replace(0, np.NaN)
payoffs = ['payoff_percent_' + str(g) for g in range(1, num_groups + 1)]
contracts = ['contract_percent_' + str(g) for g in range(1, num_groups + 1)]
time_weighted = ['time_weighted_price_' + str(g) for g in range(1, num_groups + 1)]
unit_weighted = ['unit_weighted_price_' + str(g) for g in range(1, num_groups + 1)]
quantities = ['quantity_' + str(g) for g in range(1, num_groups + 1)]
data_groups_par_flow['mean_realized_surplus'] = data_groups_par_flow[payoffs].mean(skipna=True, axis=1)
data_groups_par_flow['mean_contract_execution'] = data_groups_par_flow[contracts].mean(skipna=True, axis=1)
data_groups_par_flow['mean_time_weighted_price'] = data_groups_par_flow[time_weighted].mean(skipna=True, axis=1)
data_groups_par_flow['mean_unit_weighted_price'] = data_groups_par_flow[unit_weighted].mean(skipna=True, axis=1)
data_groups_par_flow['mean_quantity'] = data_groups_par_flow[quantities].mean(skipna=True, axis=1)
data_groups_par_flow = data_groups_par_flow.replace(np.NaN, 0)


# realized surplus for all groups
plt.figure(figsize=(10, 5))
for l in range(len(payoffs)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[payoffs[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[payoffs[l]] > 0][payoffs[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_realized_surplus'], linestyle='solid', c='green', label='Mean Realized Surplus')
plt.legend(loc='upper right')
plt.ylim(0, 2)
plt.xlabel('Round')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('%Realized Surplus')
plt.title('Realized Surplus vs Round')
plt.savefig('groups_flow_surplus.png')
plt.close()

# contract execution for all groups
plt.figure(figsize=(10, 5))
for l in range(len(contracts)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[contracts[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[contracts[l]] > 0][contracts[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_contract_execution'], linestyle='solid', c='green', label='Mean Filled Contract (Infra-marginal)')
plt.legend(loc='upper right')
plt.ylim(0, 1.5)
plt.xlabel('Round')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('%\Filled Contracts')
plt.title('Filled Contract vs Round')
plt.savefig('groups_flow_contract.png')
plt.close()

# traded volume for all groups
plt.figure(figsize=(10, 5))
for l in range(len(quantities)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[quantities[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[quantities[l]] > 0][quantities[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_quantity'], linestyle='solid', c='green', label='Mean Traded Volume')
plt.legend(loc='upper right')
plt.ylim(0, 2500)
plt.xlabel('Round')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Units')
plt.title('Traded Volume vs Round')
plt.savefig('groups_flow_quantity.png')
plt.close()

# time weighted price for all groups
plt.figure(figsize=(10, 5))
for l in range(len(time_weighted)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[time_weighted[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[time_weighted[l]] > 0][time_weighted[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_time_weighted_price'], linestyle='solid', c='green', label='Mean Time-Weighted Price')
plt.step(data=data_groups_par_flow, x='round', y='ce_price_1', where='mid', c='plum', label='CE Price')
plt.legend(loc='upper right')
plt.ylim(0, 20)
plt.xlabel('Round')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Time-Weighted Price vs Round')
plt.savefig('groups_flow_time_weighted_price.png')
plt.close()

# unit weighted price for all groups
plt.figure(figsize=(10, 5))
for l in range(len(unit_weighted)): 
    lab = 'group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[unit_weighted[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[unit_weighted[l]] > 0][unit_weighted[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_unit_weighted_price'], linestyle='solid', c='green', label='Mean Unit-Weighted Price')
plt.step(data=data_groups_par_flow, x='round', y='ce_price_1', where='mid', c='plum', label='CE Price')
plt.legend(loc='upper right')
plt.ylim(0, 20)
plt.xlabel('Round')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Round')
plt.savefig('groups_flow_unit_weighted_price.png')
plt.close()

