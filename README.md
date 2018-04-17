# /* About python---linear_regression_demo */

Summary:

This project is an extension of my fun lesson on linear regression for my statistics course in which gradient descent is discussed in more detail.  The initial lesson was designed to help students see the actual math behind the scenes.  This was done by starting with basic graphing concepts as well as fully writing out how the equations for linear regression are derived.  It could be out there, but I could not find any sources showing the detailed math behind Simple Linear Regression, nor its explanation in laymen's terms.  All of the sources I came across would show the advanced equations, but not how everything was derived.  I like knowing and showing where stuff comes from, so that's what also helped to open this investigation...

Process:

First, I decided we would start with a concept I knew my students could easily grasp, which is graphing the equation of a line, y = mx + b.  As shown on the first page of my written notes (above) in this repository, I did a quick calculation of an arbitrary line containing three common points including both the y-intercept as well as the x-intercept. These (three) points were plotted as shown in the graph below using a popular user-friendly platform for basic graphing (https://www.desmos.com).

![LR1](https://user-images.githubusercontent.com/7031463/38827244-49dbde1e-4180-11e8-98b0-4df70ff2d2a8.png)

This line containing all (three) initial points is shown in the graph below.  Since these points are all on the same line, the line of best fit is of course the equation of the line itself, y = 2x - 2, which is demonstrated in my code (above) as well.  And, as expected, this line relative to the points on the line has a perfect goodness of fit in which the correlation coefficient calculates to 100% (r = 1).

![LR2](https://user-images.githubusercontent.com/7031463/38827534-38abed68-4181-11e8-975e-fcedff3f7f72.png)

Once students had a basic understanding and a good comfort level with linear regression via a line containing all given data points, it was time to sprinkle a few random points around our line, as shown below, in order to visualize the typical scatter plot.  This ensures that our new set of data points will include our three initial points per the previous exact line equation (y = 2x - 2).

![LR3](https://user-images.githubusercontent.com/7031463/38827568-51f4ffee-4181-11e8-9612-2c6439b4b738.png)

So now we can remove the initial line, as shown below, and see all (six) of our data points on the scatter plot.  Clearly the line that best fits these points will slightly differ from our initial line.  At this point, it was time to show what's behind the curtains math-wise for linear regression, as shown in the remaining pages of my notes.  This includes how the partial derivatives of the SSE (Sum of Squares Error) relative to the slope and y-intercept are derived, as well as how setting these partial derivatives to zero leads to the calculation of the slope and y-intercept of the line that best fits a given linear type data set, i.e. linear regression.  Essentially the calculus and optimization methods, as shown in my notes step-by-step, show how gradient descent is the foundation behind linear regression.

![LR4](https://user-images.githubusercontent.com/7031463/38827589-63dbe132-4181-11e8-9011-64c901b2b7bf.png)

After deriving the equations needed to find the line of best fit, it was time to show how computer programming can be used to do the dirty work and quickly calculate the slope and the y-intercept of such a line.  In my code, not only do I show the linear regression calculation for both (1.) the three initial test data points and (2.) the six actual scattered data points, but I also show three different methods of calculation using three different programs.  All three programs yield the same equation for our new line that best fits the six scattered points, which is y = (4/3) - (4/3), as also graphed below.  And, as suspected, this line, which includes the additional randomly yet strategically placed scattered points, does NOT provide a perfect goodness of fit.  In fact, its correlation coefficient calculates to a poor rating of approximately 57% (r ~ .57).

![LR5](https://user-images.githubusercontent.com/7031463/38827610-7b93b49e-4181-11e8-8e59-307f2bea5233.png)

The final graph below simply shows our two lines graphed together.  Note how they both share the same x-intercept.  And, their slopes and y-intercepts are very close.

![LR6](https://user-images.githubusercontent.com/7031463/38827643-8eb9e3c2-4181-11e8-9368-2c57e256e110.png)

And, because I used the same info and consistent data all throughout my lesson, the students seemed to be able to grasp the concepts more quickly and easily.  I was given feedback that having visuals and showing different methods of solving the "class project" was very helpful.  Not only did these methods include math calculations and code, but I also demonstrated on a spreadsheet (above) the method of manually guessing the slope and y-intercept with the goal of reducing the error as much as possible.  This was of course the least advanced method, but it helped to visually show how everything was related in our linear regression lesson.
