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
ce_rate = [2 * i / round_length for i in ce_quantity]
ce_profit = [10600, 10600, 10600, 11600, 11600, 11600, 3750, 3750, 3750, 10500, 10500, 10500]
if players_per_group == 12: 
    ce_quantity = [2 * v for v in ce_quantity]
    ce_profit = [2 * v for v in ce_profit]

# read in data 

# df for market clearing_prices and clearing_rates/quantities for all groups 
# create a list of dfs to be merged 
groups_par_flow = []
clearing_prices = []
clearing_rates = []
cum_quantities = []
rates = []
inventories = []
colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'peachpuff', 'lightskyblue'] # add more colors with more than 6 groups

for g in range(1, num_groups + 1):
    name = 'group' + str(g)
    group_mkt = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = '/Users/YilinLi/Documents/UCSC/Flow Data/flow' + str(g) + '/' + str(r + prac_rounds) + '/1_market.json'
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
        rnd['ce_rate'] =  ce_rate[r - 1]
        rnd['ce_quantity'] = ce_quantity[r - 1]
        rnd['ce_price'] = ce_price[r - 1]
        rnd['timestamp'] = np.arange(leave_out_seconds, round_length, 1)
        # print(rnd.columns)
        group_mkt.append(rnd) 

    group_par = []
    for r in range(1, num_rounds - prac_rounds + 1):
        path = '/Users/YilinLi/Documents/UCSC/Flow Data/flow' + str(g) + '/' + str(r + prac_rounds) + '/1_participant.json'
        rnd = pd.read_json(
            path,
        )
        rnd.fillna(0, inplace=True)
        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['before_transaction'] == False)]
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp') # attache clearing price and clearing rate 
        for ind, row in rnd.iterrows(): # determine which order is in the market given multiple 
            if len(row['active_orders']) > 1: 
                for order in row['active_orders']:
                    if order['min_price'] > row['clearing_price'] or order['max_price'] < row['clearing_price']:
                        row['active_orders'].remove(order)
        rnd = rnd.explode('active_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_contracts')
        if max(rnd.iloc[:, 0]) < 119:     # add missing timestamps back with all zeros
            makeup = pd.DataFrame(np.zeros((players_per_group * (round_length - leave_out_seconds) - rnd.shape[0], rnd.shape[1])), columns=rnd.columns)
            rnd = pd.concat([rnd, makeup], axis=0, ignore_index=True)
        rnd['id_in_group'] = [i for _ in range(round_length - leave_out_seconds) for i in np.arange(1, players_per_group + 1, 1)]
        direction = rnd.groupby('id_in_group', as_index=False)['direction'].first()
        del rnd['direction']
        rnd = pd.merge(rnd, direction, how='left', on=['id_in_group'])
        rnd = rnd[['timestamp', 'id_in_subsession', 'id_in_group', 'inventory', 'rate', 'direction', 'clearing_price', 'clearing_rate', 'cumulative_quantity', 'ce_rate', 'ce_quantity', 'ce_price']]
        rnd = rnd.drop(columns=['timestamp'])
        timestamp = [i for i in np.arange(leave_out_seconds, round_length, 1) for _ in range(players_per_group)] # create correct timestamps 
        rnd['timestamp'] = timestamp
        group_par.append(rnd)

    df = pd.concat(group_par, ignore_index=True, sort=False)
    df['ind_ce_rate'] = df['ce_rate'].div(players_per_group / 2)
    id_in_subsession = 'id_in_subsession_' + str(g)
    id_in_group = 'id_in_group_' + str(g) 
    inventory = 'inventory_' + str(g)
    inventories.append(inventory)
    rate = 'rate_' + str(g)
    rates.append(rate)
    direction = 'direction_' + str(g)
    clearing_price = 'clearing_price_' + str(g)
    clearing_prices.append(clearing_price)
    clearing_rate = 'clearing_rate_' + str(g)
    clearing_rates.append(clearing_rate)
    cumsum = 'cumulative_quantity_' + str(g)
    cum_quantities.append(cumsum)
    ind_ce_rate = 'ind_ce_rate_' + str(g)
    df.columns = [id_in_subsession, id_in_group, inventory, rate, direction, clearing_price, clearing_rate, cumsum, 'ce_rate', 'ce_quantity', 'ce_price', 'timestamp', ind_ce_rate]
    df['timestamp'] = df.groupby([id_in_subsession, id_in_group])[id_in_group].cumcount() + 1
    groups_par_flow.append(df)
    

    # plot indivdual transaction rates
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df[(0 < df[rate]) & (df[rate] < 20)], x='timestamp', y=rate, hue=id_in_group, style=direction, drawstyle='steps-pre', legend='full')
    plt.step(data=df.groupby('timestamp', as_index=False)[['timestamp', ind_ce_rate]].first(), x='timestamp', y=ind_ce_rate, where='pre', c='green')
    plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left',
        borderaxespad=.5)
    plt.ylim(0, 20)
    plt.xlabel('Time')
    plt.ylabel('Individual Transacting Rate')
    plt.title('Flow Individual Transaction Rate vs Time')
    plt.savefig('group_clearing_rate_{}.png'.format(str(g)))
    plt.show()

    # plot indivdual transaction quantity
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df[(df[rate] < 20)], x='timestamp', y=inventory, hue=id_in_group, style=direction, legend='full')
    plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left',
        borderaxespad=.5)
    plt.ylim(-1500, 1500)
    plt.xlabel('Time')
    plt.ylabel('Individual Transacted Qunantity')
    plt.title('Flow Individual Transacted Quantity vs Time')
    plt.savefig('group_transaction_{}.png'.format(str(g)))
    plt.show()

