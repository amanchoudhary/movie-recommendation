# movie-recommendation
Project By: Aman Choudhary
Topic: Recommendation system using Matrix Factorization Techniques

The various files are as follows:

Main_with_argument_input.cpp:
Takes input as parameters to the executable file with the following format:
<executable_file_path> <training_set_path> <test_set_path>
For example - ./a.out u1.base u1.test

Main_with_standard_input.cpp:
Takes input from standard input after the execution begins.
You will have to enter training set path and test set path in the beginning.

RMSE_Tests.png:
The indiviual and mean results of tests on the five files from movielens.

TimeVsRMSE_u5_test_Graph.png:
It is a time (seconds) vs. RMSE graph for the trade-off observed on test set u5. Increasing the MIN_TRIES constant in the code increases the computation time but reduces the RMSE. The graph is attached to showcase that the number of iterations are set to an optimal level, where increasing or decreasing the constant would increase the RMSE. But, to reduce the runtime, MIN_TRIES constant can be decreased with an increase in the error of about 0.1 to 0.2 points.

u1..u5.base:
All the training set files comprising of 80% of data.

u1..u5.test:
All the test set files comprising of respective 20% disjoint data.

Note: In the code, the '#define DEBUG' line may be commented to remove unnecessary output.
