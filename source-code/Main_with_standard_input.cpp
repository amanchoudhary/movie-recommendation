/* Written By: Aman Choudhary */
/* Recommendation System based on Matrix Factorization Techniques */

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>
#include <fstream>

using namespace std;

// My Macros
#define pb(x) push_back(x)
#define F(i,a,n) for(int i=(a);i<(n);++i)
#define FD(i,a,n) for(int i=(a);i>=(n);--i)
#define FE(it,x) for(it=x.begin();it!=x.end();++it)
#define debug(i,sz,x) F(i,0,sz){cout<<x[i]<<" ";}cout<<endl

/* Relevant code begins here */

#define MAX_USERS 1000
#define MAX_MOVIES 1700
#define MAX_FACTORS 30

#define BASE 0.1
#define EPSILON 0.0001
#define MIN_TRIES 200
#define LEARNING_CONST 0.001
#define REG 0.02

#define DEBUG //Enable output checks!

char training_file[50];
char test_file[50];

struct Information {
	int user_id;
	int movie_id;
	int rating;
	int timestamp;
	float save;
	Information() {
		save = 0.0;
	}
};

struct Movie {
	int total_rating_sum;
	int total_count;
	float mean;
	Movie() {
		total_rating_sum = 0;
		total_count = 0;
	}
};

struct User {
	float total_offset_sum;
	int total_count;
	float mean;
	User() {
		total_offset_sum = 0;
		total_count = 0;
	}
};

vector<Information> training_set;
vector<Movie> movie_set(MAX_MOVIES);
vector<User> user_set(MAX_USERS);

float FactorVsUser[MAX_FACTORS][MAX_USERS];
float FactorVsMovie[MAX_FACTORS][MAX_MOVIES];


void load_training_data() {
	
	ifstream fin(training_file);

	if ( !fin ) {
		cout << "File cannot be opened.." << endl;
		cout << "Please check the file path entered!" << endl;
		exit(0);
	}
	
	Information input;
	while ( fin >> input.user_id ) {

		fin >> input.movie_id;
		fin >> input.rating;
		fin >> input.timestamp;
		
		training_set.pb(input);

	}

	fin.close();

#ifdef DEBUG
	cout << "Total training set count : " << training_set.size() << endl;
#endif

}

void compute_movie_average() {

	// cout << "Reached compute_movie_average().." << endl;
	int global_rating_sum;
	int global_count;
	float global_mean;

	global_count = 0;
	global_rating_sum = 0;

	F(i,0,training_set.size()) {

		int thisuser = training_set[i].user_id;
		int thismovie = training_set[i].movie_id;
		int thisrating = training_set[i].rating;
		
		global_rating_sum += thisrating;
		global_count++;
		movie_set[thismovie].total_rating_sum += thisrating;
		movie_set[thismovie].total_count++;

	}

	global_mean = global_rating_sum / float(global_count);
	const float CONST = 15.0;
	// cout << global_mean << endl;

	F(i,0,MAX_MOVIES) {
		movie_set[i].mean = (global_mean*CONST + movie_set[i].total_rating_sum);
		movie_set[i].mean /= (CONST + movie_set[i].total_count);
	}

#ifdef DEBUG
	cout << "Movie Average Computed!" << endl;
#endif

}

void compute_user_offset_average() {

	float global_offset_sum;
	int global_offset_count;
	float global_offset_mean;

	global_offset_count = 0;
	global_offset_sum = 0;

	F(i,0,training_set.size()) {

		int thisuser = training_set[i].user_id;
		int thismovie = training_set[i].movie_id;
		int thisrating = training_set[i].rating;
		
		global_offset_sum += (thisrating - movie_set[thismovie].mean);
		global_offset_count++;
		user_set[thisuser].total_offset_sum += (thisrating - movie_set[thismovie].mean);
		user_set[thisuser].total_count++;

	}

	global_offset_mean = global_offset_sum / float(global_offset_count);
	const float CONST = 15.0;
	// cout << global_offset_mean << endl;

	F(i,0,MAX_USERS) {
		user_set[i].mean = (global_offset_mean*CONST + user_set[i].total_offset_sum);
		user_set[i].mean /= (CONST + user_set[i].total_count);
	}

#ifdef DEBUG
	cout << "User offset Average Computed!" << endl;
#endif

}

void initialise_matrices() {

	// cout << "Entering initialise_matrices().." << endl;
	F(i,0,MAX_FACTORS) {
		F(j,0,MAX_MOVIES) FactorVsMovie[i][j] = BASE;
	}
	F(i,0,MAX_FACTORS) {
		F(j,0,MAX_USERS) FactorVsUser[i][j] = BASE;
	}

#ifdef DEBUG
	cout << "Initialised Matrices!" << endl;
#endif

}

void check_range(float &value) {

	if ( value > 5.0 ) {
		value = 5.0;
	} else if ( value < 1.0 ) {
		value = 1.0;
	}

}

float get_saved_prediction(int movie_id,int user_id,float save,int factor) {

	float result = save;
	if ( result == 0.0 ) {
		result = movie_set[movie_id].mean + user_set[user_id].mean;
	}

	result += FactorVsMovie[factor][movie_id] * FactorVsUser[factor][user_id];
	check_range(result);

	return result;

}

float get_default_prediction(int factor) {

	int uncomputed_factors = (MAX_FACTORS-factor+1);
	float default_prediction = float(uncomputed_factors) * BASE * BASE;
	
	return default_prediction;

}

void decompose_matrix() {

#ifdef DEBUG
	cout << "Computing " << MAX_FACTORS << " factors:" << endl;
#endif

	F(i,0,MAX_FACTORS) {

		int tries = 0;
		float get_last_rmse;
		float rmse = 1e3;
		int thisuser,thismovie,thisrating;
		float thissave;

		do {
			
			float get_prediction,error;
			get_last_rmse = rmse;
			float squared_error = 0.0;

			F(j,0,training_set.size()) {

				thisuser = training_set[j].user_id;
				thismovie = training_set[j].movie_id;
				thisrating = training_set[j].rating;
				thissave = training_set[j].save;

				get_prediction = get_saved_prediction(thismovie,thisuser,thissave,i);
				get_prediction += get_default_prediction(i);
				check_range(get_prediction);
				error = float(thisrating)-get_prediction;
				squared_error += error * error;

				float temp_userfactor = FactorVsUser[i][thisuser];
				// cout << temp_userfactor << endl;

				float movie_error = error * FactorVsMovie[i][thismovie];
				float user_regularise = REG * temp_userfactor;
				float user_diff = movie_error - user_regularise;
				float user_delta = (float)LEARNING_CONST * user_diff;
				FactorVsUser[i][thisuser] += user_delta;

				float temp_moviefactor = FactorVsMovie[i][thismovie];
				// cout << temp_moviefactor << endl;

				float user_error = error * temp_userfactor;
				float movie_regularise = REG * temp_moviefactor;
				float movie_diff = user_error - movie_regularise;
				float movie_delta = (float)LEARNING_CONST * movie_diff;
				FactorVsMovie[i][thismovie] += movie_delta;

			}

			rmse = squared_error / float(training_set.size());
			rmse = sqrt(rmse);
			// cout << get_last_rmse << " " << rmse << endl;
			// getchar();
			tries++;

		} while ( ( tries < MIN_TRIES ) || ( rmse + float(EPSILON) <= get_last_rmse ) );

		F(j,0,training_set.size()) {

			thisuser = training_set[j].user_id;
			thismovie = training_set[j].movie_id;
			thisrating = training_set[j].rating;
			thissave = training_set[j].save;

			training_set[j].save = get_saved_prediction(thismovie,thisuser,thissave,i);

		}

#ifdef DEBUG
		cout << "Factor " << i+1 << " computed." << endl;
#endif

	}

}

float recommendation_value(int movie_id,int user_id) {

	float result = movie_set[movie_id].mean + user_set[user_id].mean;

	F(i,0,MAX_FACTORS) {
		result += FactorVsMovie[i][movie_id] * FactorVsUser[i][user_id];
		check_range(result);
	}

	return result;

}

void testRMSE() {

	// cout << "Testing begins now.." << endl;
	ifstream fin(test_file);

	if ( !fin ) {
		cout << "File cannot be opened.." << endl;
		cout << "Please check the file path entered!" << endl;
		exit(0);
	}

	float square = 0.0;
	int count = 0;
	
	Information input;
	while ( fin >> input.user_id ) {
		
		fin >> input.movie_id;
		fin >> input.rating;
		fin >> input.timestamp;

		float predicted = recommendation_value(input.movie_id,input.user_id);

		float error = (predicted-input.rating);
		square += error * error;
		count++;

	}

	float meansquare = square/float(count);
	float rmse = sqrt(meansquare);

	fin.close();
#ifdef DEBUG
	cout << "Total test set count : " << count << endl;
#endif
	cout << "The resulting RMSE is : " << rmse << endl;

}

int main(int argc,char **argv) {

	cout << "Enter name of training file : ";
	cin >> training_file;
	cout << "Enter name of test file : ";
	cin >> test_file;

	load_training_data();

	compute_movie_average();
	compute_user_offset_average();

	// cout << "Computed all the averages" << endl;
	initialise_matrices();
	decompose_matrix();

	// cout << "Reached RMSE" << endl;
	testRMSE();

	return 0;
}