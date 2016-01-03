#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _windowSize = size;
}

//Returns a vector of cv::Rect, 1 for every face detected
std::vector<cv::Rect> FaceClassifier::classify(const cv::Mat &image) {
    std::vector<std::pair<cv::Rect, std::pair<size_t, std::vector<float>>>> allCandidates;
    size_t iter_count = 0;
    cv::Mat1b gray = Preprocessor::gray(image);

    size_t desiredLayers = 14;
    std::vector<cv::Size> levelSearched;
    levelSearched.reserve(desiredLayers);
    // the scale should change in function of the image dimensions.
    // a bigger image is tested a lots of times with respect to a little one.
    // lets find out a scale factor that will make the number of levels the same for every image.


    auto imgArea = cv::Rect(0,0, image.cols, image.rows).area();
    std::cout << "image area: " << imgArea << "\n";
    auto windowArea = cv::Rect(0,0,19,19).area();
    std::cout << "sample area: " << windowArea << "\n";
    // we can define, with a good approximation the scale factor to obtaion the desired number of layer
    //std::cout << "[!] Desider layers: " << desiredLayers << std::endl;
    //float dynamicFactor = 1 + (float)std::log10(imgArea/windowArea)/desiredLayers;
    std::vector<float> factors = {1.25f, /*dynamicFactor, dynamicFactor + 0.25f*/};


    for(float _scaleFactor : factors) {
        std::cout << "[!] Scale factor: " << _scaleFactor << std::endl;
        // pyramid downsampling
        // from smaller to bigger.
        // avoid to search where a face in a lower scale is found < image, scale factor >
        std::vector<std::pair<cv::Mat1b, float>> pyramid;
        pyramid.reserve(desiredLayers);
        for(float factor = 1; ; factor *=_scaleFactor) {
            ++iter_count;
            // Size of the image scaled up
            cv::Size winSize(std::round(_windowSize.width*factor), std::round(_windowSize.height*factor));

            // Size of the image scaled down (from bigger to smaller)
            cv::Size sz(image.cols/factor, image.rows/factor);

            // Difference between sized of the scaled image and the original detection window
            cv::Size sz1(sz.width - _windowSize.width, sz.height - _windowSize.height);

            // if the actual scaled image is smaller than the origina detection window, break
            if(sz1.width < 0 || sz1.height < 0) {
                break;
            }

            cv::Mat1b level;
            cv::resize(gray,level,sz,0,0,cv::INTER_NEAREST);
            pyramid.push_back(std::pair<cv::Mat1b, float>(level, factor));
        }

        // from smaller to bigger, skipping level of same dimension (searched previously)
        size_t levHalf = pyramid.size()/2, levActual = 0;
        std::cout << "half: " << levHalf << std::endl;
        _step = 1;
        for(auto rit = pyramid.rbegin(); rit != pyramid.rend(); rit++) {
            cv::Mat1b level = (*rit).first;
            float factor = (*rit).second;
            if(std::find(levelSearched.begin(), levelSearched.end(), level.size()) != levelSearched.end()) {
                std::cout << "Skipping level: " << level.size() << ": " << factor << std::endl;
                continue;
            }
            std::cout << "Searching on: " << level.size() << ": " << factor << std::endl;
            if(levActual++ > levHalf) {
                _step = 2;
            }
            _slidingSearch(level,factor,allCandidates);
            levelSearched.push_back(level.size());
        }
    }

    std::vector<cv::Rect> ret;
    ret.reserve(allCandidates.size());
    float avg[4] = {0,0,0,0};
    size_t count = 0;
    // trovare il valore maximo delle costanti che hanno più di un hit e fare passare quelli single hit con valori maggiori di questi
    for(const std::pair<cv::Rect, std::pair<size_t, std::vector<float>>> &hits: allCandidates) {
        ret.push_back(hits.first);
        if(hits.second.first > 0) { // found on more than one scale
            // TODO
            for(auto i=0;i<4;++i) {
                avg[i] += hits.second.second[i];
            }
            ++count;
        }
    }
    /*
    if(count > 0) {
        for(auto i=0;i<4;++i) {
            avg[i] /= count;
            std::cout << "AVG: " << avg[i] << "\n";
        }
        for(const std::pair<cv::Rect, std::pair<size_t, std::vector<float>>> &hits: allCandidates) {
            if(hits.second.first == 0) {
                float c1 = hits.second.second[0],
                        c2 = hits.second.second[1],
                        c3 = hits.second.second[2],
                        c4 = hits.second.second[3];
                std::cout << "Compare: " << c1 << " " << c2 << " " << c3 << " " << c4 << std::endl;
                if(c2 >= avg[1] && c3 >= avg[2]) {
                    std::cout << "passed" << std::endl;
                    ret.push_back(hits.first);
                }
            }
        }
    }
    */
    return ret;
}

// allCandidates contains the previous positions, scaled to the original dimension of image of the found face
// thus we can exploit this thing to skip a big section of the level = the intersection of the candidates
// scaled by factor (thus is compatible with level) with level
void FaceClassifier::_slidingSearch(cv::Mat1b &level, float factor, std::vector<std::pair<cv::Rect, std::pair<size_t, std::vector<float>>>> &allCandidates) {
    cv::Size winSize(std::ceil(_windowSize.width*factor)+3, std::ceil(_windowSize.height*factor)+3);

    std::vector<std::pair<cv::Rect,std::pair<size_t, std::vector<float>>>> toSkip;
    for(const std::pair<cv::Rect, std::pair<size_t, std::vector<float>>> &r : allCandidates) {
        toSkip.push_back(std::make_pair(
                             cv::Rect(r.first.x/factor, r.first.y/factor, r.first.width/factor, r.first.height/factor),
                             r.second));
    }

    auto intersect = [&](const cv::Rect &roi_rect, const std::pair<cv::Rect, std::pair<size_t, std::vector<float>>> &skip) {
        cv::Rect intersection(skip.first & roi_rect);
        return intersection.area() > 0;
    };

    for(auto y=0; y<level.rows - _windowSize.height; y+=_step) {
        for(auto x=0; x<level.cols - _windowSize.width; x+=_step) {
            cv::Rect roi_rect(x, y, _windowSize.width, _windowSize.height);

            // if roi_rect intersect a toSkip element, lets continue
            auto exists = std::find_if(toSkip.begin(), toSkip.end(), [&](const std::pair<cv::Rect,std::pair<size_t, std::vector<float>>> &skip) {
                return intersect(roi_rect, skip);
            });

            if(exists != toSkip.end() && exists->second.first > 0) { // intersection exists and ROI hitted more than once
                x+= winSize.width + _step;
                continue;
            }

            cv::Mat1b roi = level(roi_rect);
            if(_vc->classify(roi)) { // variance
                //std::cout << "V";
                float c1,c2,c3,c4;
                if(_fc->classify(roi,&c1,&c2,&c3,&c4)) { // features (shape). Distance from mined pattern
                    //std::cout << "F";
                    if(_sc->classify(roi)) { // svm to refine
                        //std::cout << "S";
                        std::cout << "in" << std::endl;
                        cv::Rect destPos(std::floor(x*factor), std::floor(y*factor), winSize.width, winSize.height);

                        // if roi_rect intersect a toSkip element, increment the hit counter and increase size of rect
                        auto exists = std::find_if(allCandidates.begin(), allCandidates.end(), [&](const std::pair<cv::Rect, std::pair<size_t, std::vector<float>>> &skip) {
                            return intersect(destPos, skip);
                        });

                        if(exists != allCandidates.end()) { // intersection exists
                            auto newX = std::min(exists->first.x, destPos.x);
                            auto newY = std::min(exists->first.y, destPos.y);
                            auto newWidth = 0, newHeight = 0;
                            if(exists->first.x + exists->first.width > destPos.x + destPos.width) {
                                newWidth = exists->first.x + exists->first.width - newX;
                            } else {
                                newWidth = destPos.x + destPos.width - newX;
                            }

                            if(exists->first.y + exists->first.height > destPos.y + destPos.height) {
                                newHeight = exists->first.y + exists->first.height - newY;
                            } else {
                                newHeight = destPos.y + destPos.height - newY;
                            }

                            exists->first = cv::Rect(newX, newY, newWidth, newHeight);
                            exists->second.first++;

                            toSkip.push_back(*exists);
                            std::cout << "Hitted " << exists->second.first << " time(s)" << std::endl;
                        } else {
                            std::vector<float> costs;
                            costs.reserve(4);
                            costs.push_back(c1);
                            costs.push_back(c2);
                            costs.push_back(c3);
                            costs.push_back(c4);
                            auto pair = std::make_pair(destPos,std::make_pair(0,costs));
                            allCandidates.push_back(pair);
                            // add current roi to toSkip vector
                            toSkip.push_back(pair);
                        }
                        x+=_windowSize.height + _step;
                    }
                }
                //std::cout << std::endl;
            }

        }
    }
}
