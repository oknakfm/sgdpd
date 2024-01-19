// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>
#include <random>
const double PI = acos(-1);

// initialization of random generator
std::random_device rd;
std::mt19937 gen(rd());

// [[Rcpp::export]]
std::vector<double> lr_scheduler(const int n_itr=3000, const double lr0=0.1,
                                 const double gamma=0.8, const int decay_period=50, const int cyclic_period=200){
  std::vector<double> lr;
  double tmp_lr=lr0;
  int cyclic_count=0;
  for(int itr=0; itr<n_itr; itr++){
    if(itr % decay_period == 0) tmp_lr *= gamma;
    if(itr % cyclic_period == 0){
      tmp_lr = lr0 * pow(gamma, cyclic_count);
      cyclic_count++;
    }
    lr.push_back(tmp_lr);
  }
  return lr;
}

std::vector<int> batch_id_selector(const int n=100, const int N=1){
  //randomly choose N indices from {0,1,2,...,n-1} with repetition
  std::uniform_int_distribution<> d_unif(0, n-1);
  std::vector<int> selected;
  for (int i=0; i<N; i++) selected.push_back(d_unif(gen));
  return selected;
}

// [[Rcpp::export]]
double sigmoid(double z){
  return 1/(1+exp(-z));
}

std::vector<double> ArmaToStdVec(const arma::vec& armaVec) {
  std::vector<double> stdVec(armaVec.begin(), armaVec.end());
  return stdVec;
}

// [[Rcpp::export]]
std::vector<double> apply_f(Rcpp::Function f,
    arma::mat Z,
    arma::vec theta){
  const int n = Z.n_rows;
  const int q = Z.n_cols;
  arma::vec f_vals = arma::zeros(n);
  arma::vec z = arma::zeros(q);
  for(int i=0; i<n; i++){
    z = Z.row(i);
    f_vals(i) = Rcpp::as<double>(f(z, theta));
  }
  return ArmaToStdVec(f_vals);
}

// [[Rcpp::export]]
Rcpp::List sgdpd(Rcpp::Function f, // f=f(z,theta): parametric model to be optimized
                      arma::mat Z, // design matrix
                      arma::vec theta0, // initial parameter
                      arma::vec lr, // learning rate schedule
                      arma::vec positives = 0, // parameter indices (whose corresponding parameter should be positive, 0 if none)
                      arma::vec conditions = 0, // outcome indices (whose corresponding outcomes belong to the condition, 0 if none)
                      double exponent=0.1, // DPD parameter
                      const int N = 20, // [optional] N: subsampling size for the 1st term
                      const int M = 2, // [optional] M: subsampling size for the 2nd term
                      bool showProgress = true, // [optional] yes if you want to show the progress
                      const double h = 10^(-10), // [optional] interval for numerical differentiation
                      int log_interval = 10 // [optional] log_interval: interval to trace the parameter update. if 0, no trace.
){
  // regularization coefficient
  const double lambda1 = pow(10,-5), lambda2 = pow(10,-3);

  // normal random number generator
  std::normal_distribution<> r_normal(0,1);

  // check of the exponent
  bool isMLE = false;
  if(abs(exponent) < pow(10,-4)){
    isMLE = true;
    Rcpp::warning("KL-divergence is minimized instead, as the exponent is absolute value of exponent is too small (< 10e-4) or negative\n");
  }else if(abs(exponent) > 1){
    Rcpp::warning("Too small/large exponent (<-1 or >1) is not supported. Default exponent is 0.1\n");
  }

  // constants
  const int n = Z.n_rows;
  const int q = Z.n_cols;
  const int d = theta0.size();
  int n_itr_tmp = lr.size();

  // which of the variables are random (i.e., not included in conditions)
  int n_conditions;
  if(conditions(0) == 0){
    n_conditions = 0;
  }else{
    n_conditions = conditions.n_elem;
  }
  int n_random = q - n_conditions;
  arma::vec random_vars = arma::zeros(n_random);
  int count = 0;
  for(int l=1; l<=q; l++){
    if(!(arma::any(conditions == l))){
      random_vars(count) = l;
      count++;
    }
  }

  // learning rate
  if(n_itr_tmp == 1){
    lr = lr_scheduler();
    n_itr_tmp = lr.size();
  }
  const int n_itr = n_itr_tmp;

  // dataset summary
  arma::vec mu, sdev; mu = sdev = arma::zeros(q);
  double e1, e2;
  for(int l=0; l<q; l++){
    e1 = e2 = 0;
    for(int i=0; i<n; i++){
      e1 += Z(i,l)/n;
      e2 += pow(Z(i,l),2)/n;
    }
    mu(l) = e1;
    sdev(l) = sqrt(e2 - pow(e1,2));
  }

  // variables used for optimization
  arma::vec tmp_theta = theta0;
  arma::vec tp, tm; tp.set_size(d); tm.set_size(d);
  double fp, fm, fc, pd;

  if(n_itr < log_interval) log_interval = n_itr;

  // variables for monitoring optimization (if needed)
  int log_count = 0, n_log;
  if(log_interval>0){
    n_log = n_itr/log_interval;
  }else{
    n_log = 1;
  }
  arma::mat theta_log = arma::zeros(n_log,d);

  // optimization
  for(int itr=0; itr<n_itr; itr++){
    if(showProgress){
      // progress report
      Rcpp::Rcout << "[Iteration progress] " << 1+itr << " / " << n_itr << "\r";
      Rcpp::Rcout.flush();
    }

    std::vector<int> batch_id;
    arma::vec grad_1, grad_2, grad_f;
    arma::vec z, xi; z = xi = arma::zeros(q);
    grad_1 = grad_2 = grad_f = arma::zeros(d);

    if(isMLE){ //MLE
      // stochastic gradient for the first term
      batch_id = batch_id_selector(n,N);
      for(int i=0; i<N; i++){
        z = Z.row(batch_id[i]);
        for(int l=0; l<d; l++){
          tp = tm = tmp_theta;
          tp(l) += h; fp = Rcpp::as<double>(f(z, tp));
          tm(l) -= h; fm = Rcpp::as<double>(f(z, tm));
          fc = Rcpp::as<double>(f(z, tmp_theta));
          grad_f(l) -= ((fp-fm)/(2*h))/((fc+lambda1) * N);
        }
      }
    }else{ //DPD
      // stochastic gradient for the first term
      batch_id = batch_id_selector(n,N);
      for(int i=0; i<N; i++){
        z = Z.row(batch_id[i]);
        for(int l=0; l<d; l++){
          tp = tm = tmp_theta;
          tp(l) += h; fp = Rcpp::as<double>(f(z, tp));
          tm(l) -= h; fm = Rcpp::as<double>(f(z, tm));
          fc = Rcpp::as<double>(f(z, tmp_theta));
          grad_1(l) -= (pow(fc, exponent) * (fp-fm)/(2*h))/(fc*N + lambda1);
        }
      }
      batch_id = batch_id_selector(n,M); double e; int c_l;
      // stochastic gradient for the second term
      for(int j=0; j<M; j++){
        pd = 1;
        for(auto l : conditions){
          c_l = static_cast<int>(l)-1;
          if(c_l >= 0) xi(c_l) = Z(batch_id[j], c_l);
        }
        for(auto l : random_vars){
          c_l = static_cast<int>(l)-1;
          e = r_normal(gen);
          xi(c_l) = Z(batch_id[j], c_l) + e * sdev(c_l);
          pd *= exp(-pow(e,2)/2)/(sqrt(2*PI));
        }

        for(int l=0; l<d; l++){
          tp = tm = tmp_theta;
          tp(l) += h; fp = Rcpp::as<double>(f(xi, tp));
          tm(l) -= h; fm = Rcpp::as<double>(f(xi, tm));
          fc = Rcpp::as<double>(f(xi, tmp_theta));
          grad_2(l) += (pow(fc, exponent) * (fp-fm)/(2*h))/(pd*M + lambda1);
        }
      }
      // overall stochastic gradient
      grad_f = grad_1 + grad_2;
    }

    // 2-norm of the gradient
    double g_norm = 0;
    for(int l=0; l<d; l++){
      g_norm += pow(grad_f(l),2);
    }
    g_norm = sqrt(g_norm);

    // parameter update
    tmp_theta -= lr(itr) * grad_f / (g_norm + lambda2);

    // make some designated parameters positive (by reversing the sign)
    int c_itr;
    for(auto itr : positives){
      c_itr = static_cast<int>(itr)-1;
      if(c_itr >= 0) tmp_theta(c_itr) = abs(tmp_theta(c_itr));
    }

    // log
    if(log_interval>0){
      if(((itr+1) % log_interval == 0)){
        for(int l=0; l<d; l++){
          theta_log(log_count, l) = tmp_theta(l);
        }
        log_count++;
      }
    }
  }
  if(showProgress){
    Rcpp::Rcout << "[Iteration progress] " << n_itr << " / " << n_itr << " : completed!\n";
  }

  Rcpp::List setup;
  setup["positives"] = ArmaToStdVec(positives);
  setup["conditions"] = ArmaToStdVec(conditions);
  setup["exponent"] = exponent;
  setup["N"] = N;
  setup["M"] = M;
  setup["h"] = h;
  setup["lr"] = ArmaToStdVec(lr);
  if(log_interval>0) setup["log_interval"] = log_interval;

  Rcpp::List results;
  results["setup"] = setup;
  if(log_interval>0) results["log"] = theta_log;
  results["theta"] = ArmaToStdVec(tmp_theta);

  return results;
}
