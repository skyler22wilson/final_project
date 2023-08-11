def custom_distance_function(df, x, y):
    turnover_x = df.at[x, 'Turnover']
    EOQ_x = df.at[x, 'EOQ']
    demand_x = df.at[x, 'Demand']
    three_month_demand_x = df.loc[(df['Months No Sale'] < 3) & (df['Sales Last 3 Months'] > 1), 'Demand']

    turnover_y = df.at[y, 'Turnover']
    EOQ_y = df.at[y, 'EOQ']
    demand_y = df.at[y, 'Demand']
    three_month_demand_y = df.loc[(df['Months No Sale'] < 3) & (df['Sales Last 3 Months'] > 1), 'Demand']

    # Calculate the differences for each feature
    turnover_diff = turnover_x - turnover_y
    EOQ_diff = EOQ_x - EOQ_y
    demand_diff = demand_x - demand_y
    three_month_demand_diff = three_month_demand_x - three_month_demand_y

    custom_distance = (
        0.35 * turnover_diff + 0.30 * three_month_demand_diff + 0.20 * demand_diff + 0.15 * EOQ_diff
    )
    return custom_distance