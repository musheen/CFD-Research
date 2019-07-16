#------------------------------------------------------------------------------------------
#
# Portfolio Optimization Using Monte Carlo Simulation
# 
# Author: Oisin Tong
# Date: 07-16-2019
# Version: 1.0
# 
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#
# Preamble
#
#------------------------------------------------------------------------------------------
#
# This script is designed to grab data from Quandl for certain stocks and run a 
# Monte Carlo simulation in order to find the optimal portfolio. 
# The data then is plotted afterwards.
# It is set up for four stocks, but can easily be modified for more or less.
# 
# Requirements:
# Python 3.7 or higher
# Pandas
# Scipy
# Numpy
# Matplotlib
# Quandl
#
# To install the required packages on a mac, install pip first:
#
# sudo easy_install pip
#
# Then install the packages this way:
#
# sudo pip3 install python3
# sudo pip3 install pandas
# etc
#
# There may be some dependencies that you need to install to
# get all of these packages installed
#
# Quandl no longer provides free data, so it is limited to March 2018
#
# Quandl API key
# MZxJKMsodvsAjJUB9SdZ 
#
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#
# Import
#
#------------------------------------------------------------------------------------------

# Import Libraries
import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize as opt
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import quandl
from datetime import date

#------------------------------------------------------------------------------------------
#
# Script Parameters
#
#------------------------------------------------------------------------------------------

# Number of Monte Carlo points
num_iter = 5000

# Number of trading days a year
n_trade = 252

# Get today's date
today = str(date.today())

# Start and end date of the periods you want to analyze
start = '2014-1-1'
end = '2017-31-1'

# If you want to set your own risk free rate as a percentage 
rf = 5

#------------------------------------------------------------------------------------------
#
# Get Data From Quandl
#
#------------------------------------------------------------------------------------------

## Change this
# Fetch data from yahoo and save under DataFrame named 'data'
stock = ['TSLA','AMZN', 'AAPL', 'CAT']

# number of colums in stock
num_col = len(stock)

# Quandl API key
quandl.ApiConfig.api_key = "MZxJKMsodvsAjJUB9SdZ"
#
data_raw = quandl.get_table('WIKI/PRICES', ticker = stock,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': start, 'lte': end }, paginate=True)

# Reorganise data pulled by setting date as index with
# columns of tickers and their corresponding adjusted prices
clean = data_raw.set_index('date')
data = clean.pivot(columns='ticker')

# Compute stock returns and print the returns in percentage format
stock_return = data.pct_change()

#------------------------------------------------------------------------------------------
#
# Find Return and Covariance
#
#------------------------------------------------------------------------------------------

# Calculate the mean daily returns and covariances of all the stocks
mean_daily_return = stock_return.mean()
# Calculate the mean annual returns
mean_return = ((mean_daily_return+1)**n_trade-1)
# Calculate the covariances of all the stocks
cov_matrix = stock_return.cov()
# Convert daily covariance to annual
cov_matrix *= n_trade
# Print out annual eturns and covariance
print ("\n")
print ("Mean Returns")
print (mean_return)
print ("\n")
print ("Covariance Matrix")
print (cov_matrix)

#------------------------------------------------------------------------------------------
#
# Run Monte Carlo
#
#------------------------------------------------------------------------------------------

# Define an array to hold the simulation results; initially set to all zeros
simulation = np.zeros((4+len(stock)-1,num_iter))

# Set for loop for Monte Carlo
for i in range(num_iter):
    # Select random weights and normalize to set the sum to 1
    weights = np.array(np.random.random(num_col))
    weights /= np.sum(weights)
    # Calculate the return and standard deviation for every step
    portfolio_return = np.sum(mean_return*weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
    # Store all the results in a defined array
    simulation[0,i] = portfolio_return*100
    simulation[1,i] = portfolio_std_dev*100
    # Calculate Sharpe ratio and store it in the array
    simulation[2,i] = simulation[0,i] / simulation[1,i]
    # Save the weights in the array
    for j in range(len(weights)):
        simulation[j+3,i] = weights[j]

# Set up new data frame with new weighted calcualtions
simFrame = pd.DataFrame(simulation.T,columns=['Return','StdDev','Sharpe',stock[0],stock[1],stock[2],stock[3]])

#------------------------------------------------------------------------------------------
#
# Find Optimal Portfolios
#
#------------------------------------------------------------------------------------------

# Spot the position of the portfolio with highest Sharpe Ratio
max_sharpe = simFrame.iloc[simFrame['Sharpe'].idxmax()]
print ("\n")
print ("The portfolio with the maximum Sharpe Ratio:")
print (max_sharpe)

#------------------------------------------------------------------------------------------
#
# Find Plotting Ranges
#
#------------------------------------------------------------------------------------------

# Find x and y limits for plotting
x_max_lim = max(simFrame.StdDev)
x_min_lim = min(simFrame.StdDev)
y_max_lim = max(simFrame.Return)
y_min_lim = min(simFrame.Return)
# Find range
x_diff = x_max_lim - x_min_lim
y_diff = y_max_lim - y_min_lim
# Extend 10% of range to limits
x_max_lim += 0.1*x_diff
x_min_lim -= 0.1*x_diff
y_max_lim += 0.1*y_diff
y_min_lim -= 0.1*y_diff

#------------------------------------------------------------------------------------------
#
# Find Capital Market Line
#
#------------------------------------------------------------------------------------------

# Initialize and set to zero
CML_x = np.zeros((3))
CML_y = np.zeros((3))

# Fill in values
CML_y[0] = rf
CML_x[1] = max_sharpe[1]
CML_y[1] = max_sharpe[0]
CML_x[2] = x_max_lim
CML_y[2] = (max_sharpe[0]-rf)/max_sharpe[1]*CML_x[2] + rf

#------------------------------------------------------------------------------------------
#
# Find the True Efficent Front
#
#------------------------------------------------------------------------------------------

# Set up Frontier 
frontier_y = np.linspace(y_min_lim,y_max_lim,300)

# Function to return volatility and return for a given weight
def get_ret_vol_sr(f_weights):
    f_weights = np.array(f_weights)
    ret = np.sum(mean_return * f_weights)*100
    vol = np.sqrt(np.dot(f_weights.T, np.dot(cov_matrix, f_weights)))*100
    sr = ret/vol
    return np.array([ret, vol, sr])

# Send back the negative sharp ratio
def neg_sharpe(f_weights):
    # The number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(f_weights)[2]*(-1) 

# Add to cover precision error 
eps = 1E-12
# Check the sum of weights as a constraint
def check_sum(f_weights):
    check = sum(f_weights)
    if (check > 1+eps):
        weight_check = 1
    elif (check < 1-eps):
        weight_check = 1
    else:
        weight_check = 0
    return weight_check

# Return volatility
def minimize_volatility(f_weights):
   return get_ret_vol_sr(f_weights)[1]

# Contraint bounding
bounds = ((0,1),(0,1),(0,1),(0,1))
# Initial weighting guess
init_guess = [0.25,0.25,0.25,0.25]

# Set up constraints
cons = ({'type':'eq', 'fun':check_sum})
# Now find improved optimal weighting
opt_weights = opt.minimize(neg_sharpe,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
opt_results = get_ret_vol_sr(opt_weights.x)

# Difference between Monte Carlo and minimizing routine
print(' ')
print('Monte Carlo  Max Sharpe: ',max_sharpe[2])
print('Optimization Max Sharpe: ',opt_results[2])
print('Difference: ',opt_results[2] - max_sharpe[2])
print(' ')

frontier_x = []
# Loop through the specified point to minimize each one
for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = opt.minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])
 
#------------------------------------------------------------------------------------------
#
# Plot Efficient Frontier
#
#------------------------------------------------------------------------------------------

# Create a scatter plot coloured by various Sharpe Ratios with standard deviation on the x-axis and returns on the y-axis
plt.scatter(simFrame.StdDev,simFrame.Return,c=simFrame.Sharpe,cmap='viridis',zorder=3,s=5)
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Annual Return (%)')
plt.colorbar(label='Sharpe Ratio')
plt.ylim(y_min_lim,y_max_lim)
plt.xlim(x_min_lim,x_max_lim)
plt.grid(zorder=0)
# Plot CML
plt.plot(CML_x, CML_y, '-k', zorder = 3,linewidth=1.5,label='Capital Market Line')
# Plot a red star to highlight position of the portfolio with highest Sharpe Ratio
plt.scatter(opt_results[1],opt_results[0],marker=(5,1,0),color='r',s=50,zorder=4,label='Optimal Portfolio')
plt.title('Markowtiz Efficient Frontier')
# Plot the frontier
plt.plot(frontier_x, frontier_y, '--k', zorder = 3,linewidth=2.0,label='Efficient Frontier')
# Legend
plt.legend(loc='lower right')
plt.show()

#------------------------------------------------------------------------------------------
#
#
# End of Script
#
#
#------------------------------------------------------------------------------------------
