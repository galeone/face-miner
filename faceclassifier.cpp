#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _windowSize = size;
    _step = 2;
}

//Returns true if contains some faces. Hilight with a rectangle the face on the image.
bool FaceClassifier::classify(cv::Mat &image) {
    std::vector<std::pair<cv::Rect, size_t>> allCandidates;
    size_t iter_count = 0;
    cv::Mat1b gray = Preprocessor::gray(image);

    // the scale should change in function of the image dimensions.
    // a bigger image is tested a lots of times with respect to a little one.
    // lets find out a scale factor that will make the number of levels the same for every image.

    std::vector<cv::Size> levelSearched;
    auto imgArea = cv::Rect(0,0, image.cols, image.rows).area();
    std::cout << "image area: " << imgArea << "\n";
    auto windowArea = cv::Rect(0,0,19,19).area();
    std::cout << "sample area: " << windowArea << "\n";
    // we can define, with a good approximation the scale factor to obtaion the desired number of layer
    size_t desiredLayers = 14;
    std::cout << "[!] Desider layers: " << desiredLayers << std::endl;
    std::vector<float> factors = {1.4f, 1 + (float)std::log10(imgArea/windowArea)/desiredLayers};
    for(float _scaleFactor : factors) {
        std::cout << "[!] Scale factor: " << _scaleFactor << std::endl;
        // pyramid downsampling
        // from smaller to bigger.
        // avoid to search where a face in a lower scale is found < image, scale factor >
        std::vector<std::pair<cv::Mat1b, float>> pyramid;
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
        for(auto rit = pyramid.rbegin(); rit != pyramid.rend(); rit++) {
            cv::Mat1b level = (*rit).first;
            float factor = (*rit).second;
            if(std::find(levelSearched.begin(), levelSearched.end(), level.size()) != levelSearched.end()) {
                std::cout << "Skipping level: " << level.size() << ": " << factor << std::endl;
                continue;
            }
            std::cout << "Searching on: " << level.size() << ": " << factor << std::endl;
            _slidingSearch(level,factor,allCandidates);
            levelSearched.push_back(level.size());
        }
    }
    
    //cv::groupRectangles(allCandidates, 1, 0.6);
    bool found = false;
    for(const std::pair<cv::Rect, size_t> &hits: allCandidates) {
        if(hits.second > 0) {
            found = true;
            cv::rectangle(image,hits.first, cv::Scalar(255,255,0));
        }
    }
    
    return found;
}

// allCandidates contains the previous positions, scaled to the original dimension of image of the found face
// thus we can exploit this thing to skip a big section of the level = the intersection of the candidates
// scaled by factor (thus is compatible with level) with level
void FaceClassifier::_slidingSearch(cv::Mat1b &level, float factor, std::vector<std::pair<cv::Rect, size_t>> &allCandidates) {
    cv::Size winSize(std::ceil(_windowSize.width*factor)+3, std::ceil(_windowSize.height*factor)+3);
    
    std::vector<std::pair<cv::Rect, size_t>> toSkip;
    for(const std::pair<cv::Rect, size_t> &r : allCandidates) {
        toSkip.push_back(std::make_pair(
                             cv::Rect(r.first.x/factor, r.first.y/factor, r.first.width/factor, r.first.height/factor),
                             r.second));
    }

    auto intersect = [&](const cv::Rect &roi_rect, const std::pair<cv::Rect, size_t> &skip) {
        cv::Rect intersection(skip.first & roi_rect);
        return intersection.area() > 0;
        /*cv::Point skipCenter((skip.first.x + skip.first.width)/2, (skip.first.y + skip.first.height)/2),
                roiCenter((roi_rect.x + roi_rect.width)/2, (roi_rect.y + roi_rect.height)/2);
        return cv::norm(skipCenter - roiCenter) < winSize.width/2;*/
    };
    
    for(auto y=0; y<level.rows - _windowSize.height; y+=_step) {
        for(auto x=0; x<level.cols - _windowSize.width; x+=_step) {
            cv::Rect roi_rect(x, y, _windowSize.width, _windowSize.height);
            
            // if roi_rect intersect a toSkip element, lets continue
            auto exists = std::find_if(toSkip.begin(), toSkip.end(), [&](const std::pair<cv::Rect, size_t> &skip) {
                return intersect(roi_rect, skip);
            });
            
            if(exists != toSkip.end() && exists->second > 0) { // intersection exists and ROI hitted more than once
                x+= winSize.width + _step;
                continue;
            }
            
            cv::Mat1b roi = level(roi_rect);
            if(_vc->classify(roi)) { // variance
                //std::cout << "V";
                if(_fc->classify(roi)) { // features (shape). Distance from mined pattern
                    //std::cout << "F";
                    if(_sc->classify(roi)) { // svm to refine
                        //std::cout << "S";
                        std::cout << "in" << std::endl;
                        cv::Rect destPos(std::floor(x*factor), std::floor(y*factor), winSize.width, winSize.height);

                        // if roi_rect intersect a toSkip element, increment the hit counter and increase size of rect
                        auto exists = std::find_if(allCandidates.begin(), allCandidates.end(), [&](const std::pair<cv::Rect, size_t> &skip) {
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
                            exists->second++;

                            toSkip.push_back(*exists);
                            std::cout << "Hitted " << exists->second << " times" << std::endl;
                        } else {
                            auto pair = std::make_pair(destPos,0);
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
