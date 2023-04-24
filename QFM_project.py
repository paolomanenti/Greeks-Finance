# installing necessary libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd

# define the preliminary ingredients for the Greeks
N = norm.cdf
n = norm.pdf

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) / sigma*np.sqrt(T)

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

# DELTA call and put
def delta_C(S, K, T, r, sigma):
    return N(d1(S, K, T, r, sigma))
    
def delta_P(S, K, T, r, sigma):
    return -N(-d1(S, K, T, r, sigma))

# GAMMA (call and put coincide)
def gamma(S, K, T, r, sigma):
    return n(d1(S, K, T, r, sigma))/(S*sigma*np.sqrt(T))

# THETA call and put
def theta_C(S, K, T, r, sigma):
    p1 = -S*n(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(d2(S, K, T, r, sigma)) 
    return p1 - p2

def theta_P(S, K, T, r, sigma):
    p1 = -S*n(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(-d2(S, K, T, r, sigma)) 
    return p1 + p2

# VEGA (call and put coincide)
def vega(S, K, T, r, sigma):
    return S*np.sqrt(T)*n(d1(S, K, T, r, sigma)) 

# RHO call and put
def rho_C(S, K, T, r, sigma):
    return K*T*np.exp(-r*T)*N(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return -K*T*np.exp(-r*T)*N(-d2(S, K, T, r, sigma))

# Application with real data - META prices
META = pd.read_csv("/Users/manenti_paolo/Desktop/MSc/AY #2/Quantitative Financial Modeling/Group Project/META.csv")['Adj Close'].dropna()
prices= META[-50:]
K = 110
r = 0.00
sigma = 0.25
time=np.linspace(0.1, 2)
X, Y = np.meshgrid(prices,time)

# DELTA PLOTS
# CALL
Deltas_C = []
Deltas_C = delta_C(X, K, Y, r, sigma) 

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Deltas_C, cmap="plasma", linewidth=0, antialiased=True, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Delta");
ax.set_title("Call")
ax.view_init(30, 60) # rotation
plt.show()

# PUT
Deltas_P = []
Deltas_P = delta_P(X, K, Y, r, sigma) 

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Deltas_P, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Delta");
ax.set_title("Put")
ax.view_init(30, 60)
plt.show()

# GAMMA PLOT
# REMARK: CALL and PUT coincide
Gammas = []
Gammas = gamma(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Gammas, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Gamma");
ax.set_title("Gamma")
ax.view_init(30, 60) 
plt.show()

# VEGA PLOT
# REMARK: CALL and PUT coincide
Vegas = []
Vegas = vega(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Vegas, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Vega");
ax.set_title("Vega")
ax.view_init(30, 60)
plt.show()

# THETA PLOTS
# CALL
Thetas_C = []
Thetas_C = theta_C(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Thetas_C, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Theta");
ax.set_title("Theta Call")
ax.view_init(30, 60)
plt.show()

# PUT
Thetas_P = []
Thetas_P = theta_P(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Thetas_P, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("Price")
ax.set_ylabel("Time")
ax.set_zlabel("Theta");
ax.set_title("Theta Put")
ax.view_init(30, 60)
plt.show()

# RHO PLOTS
# CALL
Rhos_C = []
Rhos_C = rho_C(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Rhos_C, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("price")
ax.set_ylabel("time")
ax.set_zlabel("rho");
ax.set_title("Rho Call")
ax.view_init(30, 60)
plt.show()

# PUT
Rhos_P = []
Rhos_P = rho_put(X, K, Y, r, sigma)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Rhos_P, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel("price")
ax.set_ylabel("time")
ax.set_zlabel("rho");
ax.set_title("Rho Put")
ax.view_init(30, 60)
plt.show()

S = 100
K = 100
T = 1
r = 0.00
sigma = 0.25

prices = np.arange(1, 250, 1)
deltas_c = delta_C(prices, K, T, r, sigma)
deltas_p = delta_P(prices, K, T, r, sigma)

plt.plot(prices, deltas_c, label='Delta Call')
plt.plot(prices, deltas_p, label='Delta Put')
plt.xlabel('$S_0$')
plt.ylabel('Delta')
plt.title('Stock Price Effect on Delta for Calls/Puts' )
plt.axvline(K, color='black', linestyle='dashed', linewidth=2,label="Strike")
plt.legend()
plt.show()
plt.savefig("Delta_CP.pgf")

plt.plot(prices, Rhos_C, label="Rho Call")
plt.plot(prices, Rhos_P, label = "Rho Put")
plt.xlabel('$S_0$')
plt.ylabel('Rho')
plt.title('Rho for Calls/Puts')
plt.legend()
plt.show()
