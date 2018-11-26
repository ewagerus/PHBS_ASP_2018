### Base Theory
“Let’s be Rational” by Peter Jackel, introduces an industrial solution to attaining implied volatility through option prices using minimal iterations and maximal precision. This “industrial solution” that Jackel’s updated paper is addressing is the recurrent appearance of unlikely input parameters in the use of the Black formula during everyday analytical transformations. While the Black-Scholes Model assumes constant volatility in practice, supply and demand from the market continuously alter the risk associated with securities, derivatives, indices, etc. For a more accurate price on a European option, volatility must be computed through a repetition of computations or iterations. Iterative root-finding methods, such as the Householder’s Method executed here, are required to convey implied volatility. Although, with computational effort under consideration, the optimization of this tradeoff between convergence order and number of iterations is the improvement Jackel has made from his previous paper “By Implication” (2006).

This issue could be resolved with the application of an altered configuration of asymptotics. The cumulative normal function and its inverse was selected over the Lambert W function in this new method, allowing for a practical initial guess to initialize optimization. To achieve maximum accuracy, the new method defines four segments for the initial guess function using rational approximations, while the highest and lowest segments are explained by nonlinear transformation. The method also stipulates three branches for the objective function and two iterations of Householder’s third order, a rational function with convergence order four.

### Equations
1.	The Black-Scholes-Merton Model: The Black formula depicts variations in prices over time of financial instruments assuming constant volatility, dividends, and interest rates. Implied volatility is iterated to create a riskless portfolio of options with the underlying asset assuming a lognormal distribution. A standardized Black formula is replaced by a normalized formula as it is convenient in bounded intervals.
2.	Rational Cubic Interpolation Method (Delbourgo-Gregory Interpolation): Given a set of existing data, the value of four coefficients are found using four interpolation zones. Therefore, the value of any point between the lowest and highest values of the function may be found maintaining monotonicity and convexity.
3.	Householder’s Method: Higher order derivatives of objective function were easier to compute then the objective function itself. For this reason, the Householder method was the iterative procedure chosen, achieving higher convergence order and fewer iterations.

### Partial Differential Equation (PDE)
The price variation of the underlying asset of an option over time modeled by the Black Scholes formula follows a geometric Brownian motion. A stochastic variable is included in the formula, which represents price uncertainty of the stock. This is reflected in the option price as a partial differential equation. 

### Strengths and Weaknesses
Strengths:
-	Fewer iterations necessary which leads to overall increase in speed
-	Industrially more efficient for everyday use
-	Faster and more accurate implementation
-	Applicable with high and low input prices
-	Less noisy Black function
-	Rational approximation which minimizes formulations with exponential functions

Weaknesses:
-	Higher convergence order of Householder’s Method requires more precision to measure effect
-	Significant number of coefficients
-	Each single iteration is slower than previous version
-	Speed decreased significantly with higher target precision

### Function Prototype and Arguments
     Module: rational.py
     Class: BetaFunction(forward, strike, disc_fac, cp_sign)
     Function: impl_vol(price)
     Returns implied volatility of the Black-Scholes Model


### Usage Examples
```
-   bsm = bsm.BsmModel(sigma=2, intr=intr, divr=divr)
-   vol = bsm.impvol_rational(price, strike, spot, texp, cp_sign)
```
