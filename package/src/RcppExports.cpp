// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// lr_scheduler
std::vector<double> lr_scheduler(const int n_itr, const double lr0, const double gamma, const int decay_period, const int cyclic_period);
RcppExport SEXP _sgdpd_lr_scheduler(SEXP n_itrSEXP, SEXP lr0SEXP, SEXP gammaSEXP, SEXP decay_periodSEXP, SEXP cyclic_periodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int >::type n_itr(n_itrSEXP);
    Rcpp::traits::input_parameter< const double >::type lr0(lr0SEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const int >::type decay_period(decay_periodSEXP);
    Rcpp::traits::input_parameter< const int >::type cyclic_period(cyclic_periodSEXP);
    rcpp_result_gen = Rcpp::wrap(lr_scheduler(n_itr, lr0, gamma, decay_period, cyclic_period));
    return rcpp_result_gen;
END_RCPP
}
// sigmoid
double sigmoid(double z);
RcppExport SEXP _sgdpd_sigmoid(SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(sigmoid(z));
    return rcpp_result_gen;
END_RCPP
}
// apply_f
std::vector<double> apply_f(Rcpp::Function f, arma::mat Z, arma::vec theta);
RcppExport SEXP _sgdpd_apply_f(SEXP fSEXP, SEXP ZSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Function >::type f(fSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(apply_f(f, Z, theta));
    return rcpp_result_gen;
END_RCPP
}
// sgdpd
Rcpp::List sgdpd(Rcpp::Function f, arma::mat Z, arma::vec theta0, arma::vec lr, arma::vec positives, arma::vec conditions, double exponent, const int N, const int M, bool showProgress, const double h, int log_interval);
RcppExport SEXP _sgdpd_sgdpd(SEXP fSEXP, SEXP ZSEXP, SEXP theta0SEXP, SEXP lrSEXP, SEXP positivesSEXP, SEXP conditionsSEXP, SEXP exponentSEXP, SEXP NSEXP, SEXP MSEXP, SEXP showProgressSEXP, SEXP hSEXP, SEXP log_intervalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Function >::type f(fSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type positives(positivesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type conditions(conditionsSEXP);
    Rcpp::traits::input_parameter< double >::type exponent(exponentSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    Rcpp::traits::input_parameter< bool >::type showProgress(showProgressSEXP);
    Rcpp::traits::input_parameter< const double >::type h(hSEXP);
    Rcpp::traits::input_parameter< int >::type log_interval(log_intervalSEXP);
    rcpp_result_gen = Rcpp::wrap(sgdpd(f, Z, theta0, lr, positives, conditions, exponent, N, M, showProgress, h, log_interval));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sgdpd_lr_scheduler", (DL_FUNC) &_sgdpd_lr_scheduler, 5},
    {"_sgdpd_sigmoid", (DL_FUNC) &_sgdpd_sigmoid, 1},
    {"_sgdpd_apply_f", (DL_FUNC) &_sgdpd_apply_f, 3},
    {"_sgdpd_sgdpd", (DL_FUNC) &_sgdpd_sgdpd, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_sgdpd(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
