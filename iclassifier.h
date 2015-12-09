#ifndef ICLASSIFIER_H
#define ICLASSIFIER_H

#include <opencv2/core.hpp>

class IClassifier
{
public:
    virtual ~IClassifier(){}

    // http://dlib.net/dlib/statistics/lda_abstract.h.html#equal_error_rate
    /*
        ensures
            - This function finds a threshold T that best separates the elements of
              low_vals from high_vals by selecting the threshold with equal error rate.  In
              particular, we try to pick a threshold T such that:
                - for all valid i:
                    - high_vals[i] >= T
                - for all valid i:
                    - low_vals[i] < T
              Where the best T is determined such that the fraction of low_vals >= T is the
              same as the fraction of high_vals < T.
            - Let ERR == the equal error rate.  I.e. the fraction of times low_vals >= T
              and high_vals < T.  Note that 0 <= ERR <= 1.
            - returns make_pair(ERR,T)
    */
    inline std::pair<double,double> equal_error_rate (
            const std::vector<double>& low_vals,
            const std::vector<double>& high_vals)
        {
            std::vector<std::pair<double,int> > temp;
            temp.reserve(low_vals.size()+high_vals.size());
            for (unsigned long i = 0; i < low_vals.size(); ++i)
                temp.push_back(std::make_pair(low_vals[i], -1));
            for (unsigned long i = 0; i < high_vals.size(); ++i)
                temp.push_back(std::make_pair(high_vals[i], +1));

            std::sort(temp.begin(), temp.end());

            if (temp.size() == 0)
                return std::make_pair(0,0);

            double thresh = temp[0].first;

            unsigned long num_low_wrong = low_vals.size();
            unsigned long num_high_wrong = 0;
            double low_error = num_low_wrong/(double)low_vals.size();
            double high_error = num_high_wrong/(double)high_vals.size();
            for (unsigned long i = 0; i < temp.size() && high_error < low_error; ++i)
            {
                thresh = temp[i].first;
                if (temp[i].second > 0)
                {
                    num_high_wrong++;
                    high_error = num_high_wrong/(double)high_vals.size();
                }
                else
                {
                    num_low_wrong--;
                    low_error = num_low_wrong/(double)low_vals.size();
                }
            }

            return std::make_pair((low_error+high_error)/2, thresh);
        }

};

#endif // ICLASSIFIER_H
