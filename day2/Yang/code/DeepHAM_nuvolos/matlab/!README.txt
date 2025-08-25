 



The folder “MMV_JEDC_34_2010” contains MATLAB software accompanying 
the article “Solving the Incomplete Markets Model with Aggregate 
Uncertainty Using the Krusell-Smith Algorithm” by Lilia Maliar, 
Serguei Maliar and Fernando Valli, (2010), Journal of Economic 
Dynamics and Control 34, 42-49. 

This version: December 25, 2008 (earlier versions: 2004, 2007)

The following items are provided: 

1. LICENSE AGREEMENT.

2. The article “Solving the Incomplete Markets Model with Aggregate 
   Uncertainty Using the Krusell-Smith Algorithm” by Lilia Maliar, 
   Serguei Maliar and Fernando Valli, (2010), Journal of Economic 
   Dynamics and Control 34, 42-49 (a working paper version). 

3. MATLAB FILES.

   a. "MAIN.m" computes a solution and stores the results in 
      "Solution";
   b. "SHOCKS.m" is a subroutine of MAIN.m generating the shocks;
   c. "INDIVIDUAL.m" is a subroutine of MAIN.m computing a solution 
       to the individual problem;
   d. "AGGREGATE_ST.m" is a subroutine of MAIN.m performing a 
       stochastic simulation;
   e. "AGGREGATE_NS.m" is a subroutine of MAIN.m performing a 
       non-stochastic simulation;
   f. "Inputs_for_test" contains initial distribution of capital and 
       10,000-period realizations of aggregate shock and idiosyncratic 
       shock for one agent provided in Den Haan, Judd and Juillard, 
       Journal of Economic Dynamics and Control, 2010, 34, pp. 1-3;
   g. "TEST.m" should be run after "MAIN.m"; it uses "Inputs_for_test" 
       and "Solution_to_model" for computing the statistics reported 
       in Den Haan's article, Journal of Economic Dynamics and Control, 
       2010, 34, pp. 4-27.    

For updates and other related software, please, check the authors' web 
pages. For additional information, please, contact the corresponding 
author: Serguei Maliar, T24, Hoover Institution, 434 Galvez Mall, 
Stanford University, Stanford, CA 94305-6010, USA, maliars@stanford.edu.

-------------------------------------------------------------------------
Copyright © 2008 by Lilia Maliar, Serguei Maliar and Fernando Valli. 
All rights reserved. The code may be used, modified and redistributed 
under the terms provided in the file "License_Agreement.txt".
-------------------------------------------------------------------------


warnings: 09/01/2019

> In interpn (line 151)
  In INDIVIDUAL (line 161)
  In MAIN (line 202) 
Warning: The 'cubic' method requires the grid to have a uniform spacing.
Switching the method from 'cubic' to 'spline' because this condition is not met. 
> In interpn (line 151)
  In INDIVIDUAL (line 136)
  In MAIN (line 202) 
Warning: The 'cubic' method requires the grid to have a uniform spacing.
Switching the method from 'cubic' to 'spline' because this condition is not met. 
> In interpn (line 151)
  In INDIVIDUAL (line 147)
  In MAIN (line 202) 
Warning: The 'cubic' method requires the grid to have a uniform spacing.
Switching the method from 'cubic' to 'spline' because this condition is not met. 
> In interpn (line 151)
  In INDIVIDUAL (line 154)
  In MAIN (line 202) 
Warning: The 'cubic' method requires the grid to have a uniform spacing.
Switching the method from 'cubic' to 'spline' because this condition is not met. 
> In interpn (line 151)
  In INDIVIDUAL (line 161)
  In MAIN (line 202) 

dif_B =

   8.3574e-09


iteration =

    34

Elapsed Time (in seconds):

et =

   1.1962e+03

Iterations

iteration =

    34

R^2 bad aggregate shock:

ans =

         0.999999345002395

R^2 good aggregare shock:

ans =

         0.999999718523611
