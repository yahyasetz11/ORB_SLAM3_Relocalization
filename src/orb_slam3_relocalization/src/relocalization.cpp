#include "relocalization.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <iomanip>
#include <limits>
#include <random>
#include <sophus/se3.hpp>

#define M_PI 3.14159265358979323846

namespace Relocalization
{

    RelocalizationModule::RelocalizationModule(const std::string &vocabPath,
                                               const std::string &configPath)
        : mVocabPath(vocabPath), mConfigPath(configPath), mpVocabulary(nullptr),
          mpAtlas(nullptr), mpSLAM(nullptr), mpKeyFrameDB(nullptr),
          mMinX(1e10), mMaxX(-1e10), mMinZ(1e10), mMaxZ(-1e10), mScale(1.0f),
          mMapZoomScale(1.25f), mMapOffsetX(0), mMapOffsetY(0)
    {
        if (!loadConfig())
        {
            std::cerr << "Failed to load configuration from " << configPath << std::endl;
            return;
        }

        std::cout << "[INFO] Loading vocabulary from: " << mVocabPath << std::endl;
        mpVocabulary = new ORB_SLAM3::ORBVocabulary();

        bool bVocLoad = false;
        try
        {
            bVocLoad = mpVocabulary->loadFromTextFile(mVocabPath);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Exception loading vocabulary: " << e.what() << std::endl;
        }

        if (!bVocLoad)
        {
            std::cerr << "[ERROR] Failed to load vocabulary!" << std::endl;
        }
        else
        {
            std::cout << "[INFO] Vocabulary loaded successfully" << std::endl;
            mpKeyFrameDB = new ORB_SLAM3::KeyFrameDatabase(*mpVocabulary);
            std::cout << "[INFO] KeyFrame database created" << std::endl;
        }

        mpORBextractor = std::make_unique<ORB_SLAM3::ORBextractor>(1000, 1.2, 8, 20, 7);

        std::cout << "Relocalization module initialized" << std::endl;
    }

    RelocalizationModule::~RelocalizationModule()
    {
        if (mpKeyFrameDB)
        {
            delete mpKeyFrameDB;
            mpKeyFrameDB = nullptr;
        }

        if (mpVocabulary)
        {
            delete mpVocabulary;
            mpVocabulary = nullptr;
        }

        if (mpSLAM)
        {
            mpSLAM->Shutdown();
            delete mpSLAM;
        }
    }

    bool RelocalizationModule::loadMap()
    {
        std::cout << "Loading map using System class..." << std::endl;

        try
        {
            mpSLAM = new ORB_SLAM3::System(mVocabPath, mConfigPath,
                                           ORB_SLAM3::System::MONOCULAR, false);

            std::this_thread::sleep_for(std::chrono::seconds(2));

            mpAtlas = mpSLAM->mpAtlas;
            if (!mpAtlas)
            {
                std::cerr << "✗ Failed to access atlas" << std::endl;
                return false;
            }

            mpVocabulary = mpSLAM->mpVocabulary;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }

        std::vector<ORB_SLAM3::Map *> vpMaps = mpAtlas->GetAllMaps();
        ORB_SLAM3::Map *pMap = nullptr;

        for (auto pMapCandidate : vpMaps)
        {
            auto kfs = pMapCandidate->GetAllKeyFrames();
            if (!kfs.empty())
            {
                pMap = pMapCandidate;
                break;
            }
        }

        if (!pMap)
        {
            std::cerr << "No map with data found!" << std::endl;
            return false;
        }

        mvKeyFrames = pMap->GetAllKeyFrames();
        mvMapPoints = pMap->GetAllMapPoints();

        std::cout << "KeyFrames: " << mvKeyFrames.size() << std::endl;
        std::cout << "MapPoints: " << mvMapPoints.size() << std::endl;

        // Compute BoW vectors
        std::cout << "[INFO] Computing BoW vectors..." << std::endl;
        int processedCount = 0;
        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->isBad())
            {
                pKF->SetORBVocabulary(mpVocabulary);
                pKF->ComputeBoW();
                mpKeyFrameDB->add(pKF);
                processedCount++;
            }
        }
        std::cout << "[INFO] BoW computed for " << processedCount << " keyframes" << std::endl;

        // Extract map points and calculate bounds
        mMapPointsViz.clear();
        for (auto pMP : mvMapPoints)
        {
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f pos = pMP->GetWorldPos();
                cv::Point3f pt(pos(0), pos(1), pos(2));
                mMapPointsViz.push_back(pt);

                // Update bounds
                mMinX = std::min(mMinX, pt.x);
                mMaxX = std::max(mMaxX, pt.x);
                mMinZ = std::min(mMinZ, pt.z);
                mMaxZ = std::max(mMaxZ, pt.z);
            }
        }

        std::cout << "Valid map points: " << mMapPointsViz.size() << std::endl;

        if (mExportPCD)
        {
            exportMapToPCD(mPCDPath);
        }

        mpSLAM->Shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        delete mpSLAM;
        mpSLAM = nullptr;

        return true;
    }

    bool RelocalizationModule::loadConfig()
    {
        cv::FileStorage fs(mConfigPath, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "Failed to open config file: " << mConfigPath << std::endl;
            return false;
        }

        fs["System.LoadAtlasFromFile"] >> mMapPath;
        if (mMapPath.empty())
        {
            std::cerr << "System.LoadAtlasFromFile not found in config" << std::endl;
            return false;
        }

        float fx = fs["Camera1.fx"];
        float fy = fs["Camera1.fy"];
        float cx = fs["Camera1.cx"];
        float cy = fs["Camera1.cy"];

        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;

        mDistCoef = cv::Mat::zeros(5, 1, CV_32F);
        if (!fs["Camera1.k1"].empty())
            mDistCoef.at<float>(0) = fs["Camera1.k1"];
        if (!fs["Camera1.k2"].empty())
            mDistCoef.at<float>(1) = fs["Camera1.k2"];
        if (!fs["Camera1.p1"].empty())
            mDistCoef.at<float>(2) = fs["Camera1.p1"];
        if (!fs["Camera1.p2"].empty())
            mDistCoef.at<float>(3) = fs["Camera1.p2"];
        if (!fs["Camera1.k3"].empty())
            mDistCoef.at<float>(4) = fs["Camera1.k3"];

        mFrameSkip = fs["Relocalization.FrameSkip"].empty() ? 5 : (int)fs["Relocalization.FrameSkip"];
        mMinInliers = fs["Relocalization.MinInliers"].empty() ? 1 : (int)fs["Relocalization.MinInliers"];
        mMinMatches = fs["Relocalization.MinMatches"].empty() ? 10 : (int)fs["Relocalization.MinMatches"];
        mHammingThreshold = fs["Relocalization.HammingThreshold"].empty() ? 80 : (int)fs["Relocalization.HammingThreshold"];
        mHammingPreFilterThreshold = fs["Relocalization.HammingPreFilter"].empty() ? 100 : (int)fs["Relocalization.HammingPreFilter"];
        mRansacInlierThreshold3D = fs["Relocalization.RansacInlierThreshold3D"].empty() ? 0.10f : (float)fs["Relocalization.RansacInlierThreshold3D"];
        mRansacIterations = fs["Relocalization.RansacIterations"].empty() ? 50 : (int)fs["Relocalization.RansacIterations"];
        mBowThreshold = fs["Relocalization.BowSimilarityThreshold"].empty() ? 0.05f : (float)fs["Relocalization.BowSimilarityThreshold"];
        mMaxCandidates = fs["Relocalization.MaxCandidates"].empty() ? 5 : (int)fs["Relocalization.MaxCandidates"];

        int procW = fs["Visualization.ProcessWidth"].empty() ? 640 : (int)fs["Visualization.ProcessWidth"];
        int procH = fs["Visualization.ProcessHeight"].empty() ? 480 : (int)fs["Visualization.ProcessHeight"];
        mProcessSize = cv::Size(procW, procH);

        int dispW = fs["Visualization.DisplayWidth"].empty() ? 1280 : (int)fs["Visualization.DisplayWidth"];
        int dispH = fs["Visualization.DisplayHeight"].empty() ? 960 : (int)fs["Visualization.DisplayHeight"];
        mDisplaySize = cv::Size(dispW, dispH);

        mVisualizationEnabled = fs["Visualization.Enabled"].empty() ? true : (int)fs["Visualization.Enabled"] != 0;
        mExportPCD = fs["Visualization.ExportPCD"].empty() ? true : (int)fs["Visualization.ExportPCD"] != 0;
        fs["Visualization.PCDPath"] >> mPCDPath;
        if (mPCDPath.empty())
            mPCDPath = "map_points.pcd";

        // Map visualization parameters - DEBUG
        std::cout << "\n[DEBUG] Loading map visualization parameters..." << std::endl;
        std::cout << "  Checking Map.ZoomScale: " << (fs["Map.ZoomScale"].empty() ? "EMPTY" : "FOUND") << std::endl;
        std::cout << "  Checking Map.OffsetX: " << (fs["Map.OffsetX"].empty() ? "EMPTY" : "FOUND") << std::endl;
        std::cout << "  Checking Map.OffsetY: " << (fs["Map.OffsetY"].empty() ? "EMPTY" : "FOUND") << std::endl;

        mMapZoomScale = fs["Map.ZoomScale"].empty() ? 1.25f : (float)fs["Map.ZoomScale"];
        mMapOffsetX = fs["Map.OffsetX"].empty() ? 0 : (int)fs["Map.OffsetX"];
        mMapOffsetY = fs["Map.OffsetY"].empty() ? 0 : (int)fs["Map.OffsetY"];

        std::cout << "  Loaded Map.ZoomScale: " << mMapZoomScale << std::endl;
        std::cout << "  Loaded Map.OffsetX: " << mMapOffsetX << std::endl;
        std::cout << "  Loaded Map.OffsetY: " << mMapOffsetY << std::endl;

        fs.release();
        return true;
    }

    void RelocalizationModule::extractFeatures(const cv::Mat &frame,
                                               std::vector<cv::KeyPoint> &keypoints,
                                               cv::Mat &descriptors)
    {
        if (frame.empty())
            return;

        cv::Mat gray;
        if (frame.channels() == 3)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = frame.clone();
        }

        std::vector<int> vLappingArea = {0, 0};
        (*mpORBextractor)(gray, cv::Mat(), keypoints, descriptors, vLappingArea);
    }

    std::vector<ORB_SLAM3::KeyFrame *> RelocalizationModule::detectRelocalizationCandidates(
        const cv::Mat &descriptors)
    {
        std::vector<ORB_SLAM3::KeyFrame *> vpCandidates;

        if (!mpKeyFrameDB || !mpVocabulary || descriptors.empty())
        {
            return vpCandidates;
        }

        DBoW2::BowVector currentBowVec;
        DBoW2::FeatureVector currentFeatVec;

        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(descriptors.rows);
        for (int i = 0; i < descriptors.rows; ++i)
        {
            vCurrentDesc.push_back(descriptors.row(i).clone());
        }

        mpVocabulary->transform(vCurrentDesc, currentBowVec, currentFeatVec, 4);

        std::vector<std::pair<float, ORB_SLAM3::KeyFrame *>> vScoreAndMatch;

        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->isBad() && !pKF->mBowVec.empty())
            {
                float score = mpVocabulary->score(currentBowVec, pKF->mBowVec);
                if (score > mBowThreshold)
                {
                    vScoreAndMatch.push_back(std::make_pair(score, pKF));
                }
            }
        }

        std::sort(vScoreAndMatch.begin(), vScoreAndMatch.end(),
                  [](const auto &a, const auto &b)
                  { return a.first > b.first; });

        int nCandidates = std::min(mMaxCandidates, (int)vScoreAndMatch.size());
        for (int i = 0; i < nCandidates; i++)
        {
            vpCandidates.push_back(vScoreAndMatch[i].second);
        }

        return vpCandidates;
    }

    void RelocalizationModule::debugStatus()
    {
        std::cout << "\n=== RELOCALIZATION MODULE STATUS ===" << std::endl;
        std::cout << "Vocabulary: " << (mpVocabulary ? "YES" : "NO") << std::endl;
        std::cout << "KeyFrames: " << mvKeyFrames.size() << std::endl;
        std::cout << "Map points: " << mvMapPoints.size() << std::endl;
        std::cout << "====================================" << std::endl;
    }

    bool RelocalizationModule::matchWithKeyFrame(
        const std::vector<cv::KeyPoint> &keypoints,
        const cv::Mat &descriptors,
        ORB_SLAM3::KeyFrame *pKF,
        std::vector<cv::Point3f> &points3D,
        std::vector<cv::Point2f> &points2D,
        std::vector<float> &hammingDists)
    {
        points3D.clear();
        points2D.clear();
        hammingDists.clear();

        const std::vector<cv::KeyPoint> &vKFKeyPoints = pKF->mvKeysUn;
        const cv::Mat &KFDescriptors = pKF->mDescriptors;

        if (KFDescriptors.empty())
            return false;

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(descriptors, KFDescriptors, knnMatches, 2);

        const float ratioThresh = 0.75f;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i].size() < 2)
                continue;

            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            {
                int trainIdx = knnMatches[i][0].trainIdx;
                ORB_SLAM3::MapPoint *pMP = pKF->GetMapPoint(trainIdx);

                if (pMP && !pMP->isBad())
                {
                    Eigen::Vector3f pos = pMP->GetWorldPos();
                    points3D.push_back(cv::Point3f(pos(0), pos(1), pos(2)));

                    int queryIdx = knnMatches[i][0].queryIdx;
                    points2D.push_back(keypoints[queryIdx].pt);
                    hammingDists.push_back((float)knnMatches[i][0].distance);
                }
            }
        }

        return points3D.size() >= (size_t)mMinMatches;
    }

    cv::Point3f RelocalizationModule::computePosition(const cv::Mat &rvec,
                                                      const cv::Mat &tvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat pos = -R.t() * tvec;
        return cv::Point3f(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
    }

    cv::Point2f RelocalizationModule::project3DTo2D(const cv::Point3f &pt3D, int mapHeight)
    {
        float rangeX = mMaxX - mMinX;
        float rangeZ = mMaxZ - mMinZ;

        // Use more of the available space
        float marginX = 100.0f;
        float marginY = 100.0f;
        float availableWidth = mDisplaySize.width - 2 * marginX;
        float availableHeight = mapHeight - 2 * marginY;

        // Apply configurable zoom factor
        float baseScale = std::min(availableWidth / rangeX, availableHeight / rangeZ);
        float scale = baseScale * mMapZoomScale;

        // Debug output (only print once)
        static bool debugPrinted = false;
        if (!debugPrinted)
        {
            std::cout << "[DEBUG] Map projection:" << std::endl;
            std::cout << "  Base scale: " << baseScale << std::endl;
            std::cout << "  Zoom scale: " << mMapZoomScale << std::endl;
            std::cout << "  Final scale: " << scale << std::endl;
            std::cout << "  Map range: X=" << rangeX << ", Z=" << rangeZ << std::endl;
            debugPrinted = true;
        }

        // Center the map in the available space with configurable offset
        float scaledWidth = rangeX * scale;
        float scaledHeight = rangeZ * scale;
        float offsetX = marginX + (availableWidth - scaledWidth) / 2 + mMapOffsetX;
        float offsetY = marginY + (availableHeight - scaledHeight) / 2 + mMapOffsetY;

        int x = (int)(offsetX + (pt3D.x - mMinX) * scale);
        int y = (int)(mapHeight - offsetY - (pt3D.z - mMinZ) * scale);

        return cv::Point2f(x, y);
    }

    LocationResult RelocalizationModule::processFrame(const cv::Mat &frame, double timestamp)
    {
        LocationResult result;
        result.success = false;
        result.matchedKeyFrameId = -1;
        result.numInliers = 0;
        result.totalMatches = 0;
        result.confidence = 0.0f;
        result.bowScore = 0.0f;

        cv::Mat processFrame;
        if (frame.size() != mProcessSize)
        {
            cv::resize(frame, processFrame, mProcessSize);
        }
        else
        {
            processFrame = frame.clone();
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractFeatures(processFrame, keypoints, descriptors);

        if (keypoints.empty())
            return result;

        // Store all keypoints for visualization
        result.queryKeypoints = keypoints;

        auto candidates = detectRelocalizationCandidates(descriptors);

        float bestRankScore = -1.0f;
        for (auto pKF : candidates)
        {
            std::vector<cv::Point3f> points3D;
            std::vector<cv::Point2f> points2D;
            std::vector<float> hammingDists;

            if (matchWithKeyFrame(keypoints, descriptors, pKF, points3D, points2D, hammingDists))
            {
                DBoW2::BowVector currentBowVec;
                DBoW2::FeatureVector currentFeatVec;
                std::vector<cv::Mat> vCurrentDesc;
                vCurrentDesc.reserve(descriptors.rows);
                for (int i = 0; i < descriptors.rows; ++i)
                    vCurrentDesc.push_back(descriptors.row(i).clone());
                mpVocabulary->transform(vCurrentDesc, currentBowVec, currentFeatVec, 4);
                float bowScore = mpVocabulary->score(currentBowVec, pKF->mBowVec);

                // Stage 1: loose Hamming pre-filter — remove obviously bad matches
                std::vector<cv::Point3f> preFiltered3D;
                std::vector<cv::Point2f> preFiltered2D;
                std::vector<int> preFilteredIndices;
                for (int i = 0; i < (int)points3D.size(); ++i)
                {
                    if (hammingDists[i] < mHammingPreFilterThreshold)
                    {
                        preFiltered3D.push_back(points3D[i]);
                        preFiltered2D.push_back(points2D[i]);
                        preFilteredIndices.push_back(i);
                    }
                }
                if ((int)preFiltered3D.size() < 4)
                    continue;

                // Stage 2: RANSAC — sample 4 points, solve PnP, score via 3D distance
                const int N = (int)preFiltered3D.size();
                double fx = (double)mK.at<float>(0, 0);
                double fy = (double)mK.at<float>(1, 1);
                double cx_k = (double)mK.at<float>(0, 2);
                double cy_k = (double)mK.at<float>(1, 2);

                std::vector<int> bestInlierLocalIndices;
                cv::Mat bestRvec, bestTvec;

                std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> randIdx(0, N - 1);

                for (int iter = 0; iter < mRansacIterations; ++iter)
                {
                    // Sample 4 unique correspondences
                    std::vector<int> sample;
                    sample.reserve(4);
                    while ((int)sample.size() < 4)
                    {
                        int idx = randIdx(rng);
                        if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                            sample.push_back(idx);
                    }

                    std::vector<cv::Point3f> s3D;
                    std::vector<cv::Point2f> s2D;
                    for (int idx : sample)
                    {
                        s3D.push_back(preFiltered3D[idx]);
                        s2D.push_back(preFiltered2D[idx]);
                    }

                    cv::Mat rvec_s, tvec_s;
                    if (!cv::solvePnP(s3D, s2D, mK, mDistCoef, rvec_s, tvec_s, false, cv::SOLVEPNP_P3P))
                        continue;

                    cv::Mat R_s;
                    cv::Rodrigues(rvec_s, R_s);
                    R_s.convertTo(R_s, CV_64F);
                    cv::Mat t_s;
                    tvec_s.convertTo(t_s, CV_64F);

                    // Score all pre-filtered matches using 3D distance in camera space:
                    // X_c = R*X_w + t; X_c' = Zc*[(u-cx)/fx, (v-cy)/fy, 1]; dist = ||X_c - X_c'||
                    std::vector<int> iterInliers;
                    for (int i = 0; i < N; ++i)
                    {
                        double Xw = preFiltered3D[i].x, Yw = preFiltered3D[i].y, Zw = preFiltered3D[i].z;
                        double Xc = R_s.at<double>(0,0)*Xw + R_s.at<double>(0,1)*Yw + R_s.at<double>(0,2)*Zw + t_s.at<double>(0);
                        double Yc = R_s.at<double>(1,0)*Xw + R_s.at<double>(1,1)*Yw + R_s.at<double>(1,2)*Zw + t_s.at<double>(1);
                        double Zc = R_s.at<double>(2,0)*Xw + R_s.at<double>(2,1)*Yw + R_s.at<double>(2,2)*Zw + t_s.at<double>(2);
                        if (Zc <= 0) continue;

                        double Xc_obs = Zc * (preFiltered2D[i].x - cx_k) / fx;
                        double Yc_obs = Zc * (preFiltered2D[i].y - cy_k) / fy;
                        double dist3d = std::sqrt((Xc - Xc_obs)*(Xc - Xc_obs) + (Yc - Yc_obs)*(Yc - Yc_obs));

                        if (dist3d < (double)mRansacInlierThreshold3D)
                            iterInliers.push_back(i);
                    }

                    if ((int)iterInliers.size() > (int)bestInlierLocalIndices.size())
                    {
                        bestInlierLocalIndices = iterInliers;
                        bestRvec = rvec_s.clone();
                        bestTvec = tvec_s.clone();
                    }
                }

                if ((int)bestInlierLocalIndices.size() < 4)
                    continue;

                // Stage 3: LM refinement on all RANSAC inliers (uniform weights)
                std::vector<cv::Point3f> inlier3D;
                std::vector<cv::Point2f> inlier2D;
                std::vector<int> inlierOriginalIndices;
                for (int localIdx : bestInlierLocalIndices)
                {
                    inlier3D.push_back(preFiltered3D[localIdx]);
                    inlier2D.push_back(preFiltered2D[localIdx]);
                    inlierOriginalIndices.push_back(preFilteredIndices[localIdx]);
                }

                // LM pass 1: refine on all RANSAC inliers
                std::vector<float> uniform_weights(inlier3D.size(), 1.0f);
                WeightedPnPResult lm1 = solvePnPWeighted(
                    inlier3D, inlier2D, uniform_weights,
                    bestRvec, bestTvec, 50, 1e-6, 8.0f);

                if (!lm1.success)
                    continue;

                // LM pass 2: re-run on the strict 8px inlier subset from pass 1
                std::vector<cv::Point3f> strict3D;
                std::vector<cv::Point2f> strict2D;
                std::vector<int> strictOriginalIndices;
                for (int lmIdx : lm1.inlierIndices)
                {
                    strict3D.push_back(inlier3D[lmIdx]);
                    strict2D.push_back(inlier2D[lmIdx]);
                    strictOriginalIndices.push_back(inlierOriginalIndices[lmIdx]);
                }

                if ((int)strict3D.size() < 4)
                    continue;

                std::vector<float> strict_weights(strict3D.size(), 1.0f);
                WeightedPnPResult lm2 = solvePnPWeighted(
                    strict3D, strict2D, strict_weights,
                    lm1.rvec, lm1.tvec, 50, 1e-6, 8.0f);

                if (!lm2.success)
                    continue;

                {
                    float reproj = std::max(lm2.meanReprojectionError, 0.01f);
                    float rankScore = (float)lm2.numInliers / reproj;

                    float inlierRatio = (float)lm2.numInliers / points3D.size();

                    std::cout << "[ORB] descriptor size: " << descriptors.rows << " x " << descriptors.cols << std::endl;
                    std::cout << "inlierRatio: " << inlierRatio << std::endl;
                    std::cout << "BoW score: " << bowScore << std::endl;
                    std::cout << "inlierSize: " << lm2.numInliers << std::endl;
                    std::cout << "reproj_error: " << lm2.meanReprojectionError << std::endl;
                    std::cout << "rankScore: " << rankScore << std::endl;

                    if (rankScore > bestRankScore)
                    {
                        bestRankScore = rankScore;

                        // Map lm2 inlier indices (local to strict3D) back to original points3D indices
                        std::vector<int> finalInlierOriginalIndices;
                        finalInlierOriginalIndices.reserve(lm2.inlierIndices.size());
                        for (int idx : lm2.inlierIndices)
                            finalInlierOriginalIndices.push_back(strictOriginalIndices[idx]);

                        result.success = true;
                        result.position = computePosition(lm2.rvec, lm2.tvec);
                        result.matchedKeyFrameId = pKF->mnId;
                        result.numInliers = lm2.numInliers;
                        result.totalMatches = points3D.size();
                        result.confidence = inlierRatio * 100.0f;
                        result.bowScore = bowScore;

                        result.matched2DPoints = points2D;
                        result.matched3DPoints = points3D;
                        result.inlierIndices = finalInlierOriginalIndices;

                        result.rvec = lm2.rvec.clone();
                        result.tvec = lm2.tvec.clone();

                        mCurrentPosition = result.position;
                    }
                }
            }
        }

        return result;
    }

    cv::Mat RelocalizationModule::createMapVisualization(const LocationResult &result, cv::Size targetSize)
    {
        cv::Mat mapViz(targetSize, CV_8UC3, cv::Scalar(255, 255, 255));

        // STEP 1: Draw grid and axes FIRST (bottom layer)
        drawGrid(mapViz, targetSize.height);

        // STEP 2: Draw all map points (small gray dots)
        for (const auto &pt : mMapPointsViz)
        {
            cv::Point2f pt2d = project3DTo2D(pt, targetSize.height);
            cv::circle(mapViz, pt2d, 1, cv::Scalar(180, 180, 180), -1);
        }

        // STEP 3: Highlight matched 3D points (yellow)
        if (result.success)
        {
            for (size_t i = 0; i < result.matched3DPoints.size(); i++)
            {
                // Check if this match is an inlier
                bool isInlier = false;
                for (int inlierIdx : result.inlierIndices)
                {
                    if (inlierIdx == (int)i)
                    {
                        isInlier = true;
                        break;
                    }
                }

                if (isInlier)
                {
                    cv::Point2f pt2d = project3DTo2D(result.matched3DPoints[i], targetSize.height);
                    cv::circle(mapViz, pt2d, 4, cv::Scalar(0, 255, 255), -1); // Cyan for inliers
                    cv::circle(mapViz, pt2d, 6, cv::Scalar(0, 200, 0), 2);    // Green border
                }
            }

            // STEP 4: Draw current position with ORIENTED TRIANGLE
            cv::Point2f posPoint = project3DTo2D(mCurrentPosition, targetSize.height);

            // Calculate orientation angle from rvec
            float yaw = 0.0f;
            if (!result.rvec.empty())
            {
                cv::Mat R;
                cv::Rodrigues(result.rvec, R);

                // Extract yaw from rotation matrix (rotation around Y axis)
                // For top-down view, we want rotation in XZ plane
                yaw = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2));

                // Adjust for map orientation (Y-axis points up on screen, but we use Z)
                yaw = -yaw - M_PI / 2;
            }

            // Draw oriented triangle (RED with yellow border)
            drawOrientedTriangle(mapViz, posPoint, yaw, 15, cv::Scalar(0, 0, 255), -1);
            drawOrientedTriangle(mapViz, posPoint, yaw, 15, cv::Scalar(0, 255, 255), 2);

            // Label
            cv::putText(mapViz, "YOU",
                        cv::Point(posPoint.x + 22, posPoint.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

            // STEP 5: Display coordinates in top-right corner
            std::ostringstream coordText;
            coordText << "Position:";
            cv::putText(mapViz, coordText.str(),
                        cv::Point(targetSize.width - 180, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            coordText.str("");
            coordText << "X: " << std::fixed << std::setprecision(2) << mCurrentPosition.x;
            cv::putText(mapViz, coordText.str(),
                        cv::Point(targetSize.width - 180, 55),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            coordText.str("");
            coordText << "Y: " << std::fixed << std::setprecision(2) << mCurrentPosition.z;
            cv::putText(mapViz, coordText.str(),
                        cv::Point(targetSize.width - 180, 78),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 150, 0), 2);

            // coordText.str("");
            // coordText << "Z: " << std::fixed << std::setprecision(2) << mCurrentPosition.y;
            // cv::putText(mapViz, coordText.str(),
            //             cv::Point(targetSize.width - 180, 101),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        // Title
        cv::putText(mapViz, "Map (Top-Down View - X/Y Plane)",
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        return mapViz;
    }

    void RelocalizationModule::processVideo(const std::string &videoPath, bool visualize)
    {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened())
        {
            std::cerr << "Cannot open video: " << videoPath << std::endl;
            return;
        }

        visualize = mVisualizationEnabled;

        if (!visualize)
        {
            std::cout << "[INFO] Visualization disabled - running in headless mode" << std::endl;
        }

        cv::Mat frame;
        int frameCount = 0;
        int successCount = 0;

        std::cout << "\n=== Processing video ===" << std::endl;

        while (cap.read(frame))
        {
            frameCount++;
            if (frameCount % mFrameSkip != 0)
                continue;

            std::cout << "\n--- Frame " << frameCount << " ---" << std::endl;

            double video_ts = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
            auto result = processFrame(frame, video_ts);

            if (result.success)
            {
                successCount++;

                if (visualize)
                {
                    try
                    {
                        // Create display frame
                        cv::Mat displayFrame;
                        cv::resize(frame, displayFrame, mDisplaySize);

                        // Draw ALL ORB keypoints (small gray circles)
                        for (const auto &kp : result.queryKeypoints)
                        {
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f scaledPt(kp.pt.x * scale, kp.pt.y * scale);
                            cv::circle(displayFrame, scaledPt, 2, cv::Scalar(150, 150, 150), -1);
                        }

                        // Highlight inlier matches (bright green)
                        for (int inlierIdx : result.inlierIndices)
                        {
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f scaledPt(result.matched2DPoints[inlierIdx].x * scale,
                                                 result.matched2DPoints[inlierIdx].y * scale);
                            cv::circle(displayFrame, scaledPt, 5, cv::Scalar(0, 255, 0), 2);
                        }

                        // Create map visualization
                        cv::Mat mapViz = createMapVisualization(result, mDisplaySize);

                        // Create combined view (side by side)
                        cv::Mat combined(mDisplaySize.height, mDisplaySize.width * 2, CV_8UC3);
                        displayFrame.copyTo(combined(cv::Rect(0, 0, mDisplaySize.width, mDisplaySize.height)));
                        mapViz.copyTo(combined(cv::Rect(mDisplaySize.width, 0, mDisplaySize.width, mDisplaySize.height)));

                        // Draw connection lines for inlier matches
                        for (int inlierIdx : result.inlierIndices)
                        {
                            // 2D point in video (left side)
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f pt2D(result.matched2DPoints[inlierIdx].x * scale,
                                             result.matched2DPoints[inlierIdx].y * scale);

                            // 3D point projected to map (right side)
                            cv::Point2f pt3DProj = project3DTo2D(result.matched3DPoints[inlierIdx],
                                                                 mDisplaySize.height);
                            pt3DProj.x += mDisplaySize.width; // Offset for right panel

                            // Draw line (green for inliers)
                            cv::line(combined, pt2D, pt3DProj, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        }

                        // Status overlay on video side
                        cv::Scalar statusColor;
                        std::string status;
                        if (result.confidence >= 70)
                        {
                            statusColor = cv::Scalar(0, 255, 0);
                            status = "EXCELLENT";
                        }
                        else if (result.confidence >= 50)
                        {
                            statusColor = cv::Scalar(0, 200, 255);
                            status = "GOOD";
                        }
                        else
                        {
                            statusColor = cv::Scalar(0, 165, 255);
                            status = "WEAK";
                        }

                        cv::putText(combined, "LOCALIZED - " + status,
                                    cv::Point(30, 40),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 2);

                        std::ostringstream info;
                        info << "Inliers: " << result.numInliers << "/" << result.totalMatches
                             << " | Conf: " << std::fixed << std::setprecision(1)
                             << result.confidence << "%";
                        cv::putText(combined, info.str(),
                                    cv::Point(30, 70),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

                        if (video_ts > 0.0)
                        {
                            std::ostringstream ts;
                            ts << "t=" << std::fixed << std::setprecision(3) << video_ts << "s";
                            cv::putText(combined, ts.str(),
                                        cv::Point(30, 95),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
                        }

                        cv::imshow("Relocalization: Video + Map", combined);

                        int key = cv::waitKey(1);
                        if (key == 27)
                        {
                            std::cout << "\nESC pressed - stopping" << std::endl;
                            break;
                        }
                    }
                    catch (const cv::Exception &e)
                    {
                        std::cerr << "\n[ERROR] Display error: " << e.what() << std::endl;
                        mVisualizationEnabled = false;
                        visualize = false;
                    }
                }
            }
        }

        std::cout << "\n=== Video processing complete ===" << std::endl;
        std::cout << "Success rate: " << (100.0 * successCount / (frameCount / mFrameSkip))
                  << "%" << std::endl;
    }

    void RelocalizationModule::processWebcam(int cameraId)
    {
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened())
        {
            std::cerr << "Cannot open webcam device " << cameraId << std::endl;
            return;
        }

        bool visualize = mVisualizationEnabled;
        if (!visualize)
        {
            std::cout << "[INFO] Visualization disabled - running in headless mode" << std::endl;
        }

        cv::Mat frame;
        int frameCount = 0;
        int successCount = 0;

        std::cout << "\n=== Processing webcam (device " << cameraId << ") ===" << std::endl;
        std::cout << "Press ESC to stop" << std::endl;

        auto webcam_start = std::chrono::steady_clock::now();

        while (true)
        {
            if (!cap.read(frame) || frame.empty())
            {
                std::cerr << "[WARNING] Failed to grab frame from webcam" << std::endl;
                continue;
            }

            frameCount++;
            if (frameCount % mFrameSkip != 0)
                continue;

            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - webcam_start)
                                 .count();
            auto result = processFrame(frame, elapsed);

            if (result.success)
                successCount++;

            if (visualize)
            {
                try
                {
                    cv::Mat displayFrame;
                    cv::resize(frame, displayFrame, mDisplaySize);

                    if (result.success)
                    {
                        for (const auto &kp : result.queryKeypoints)
                        {
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f scaledPt(kp.pt.x * scale, kp.pt.y * scale);
                            cv::circle(displayFrame, scaledPt, 2, cv::Scalar(150, 150, 150), -1);
                        }

                        for (int inlierIdx : result.inlierIndices)
                        {
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f scaledPt(result.matched2DPoints[inlierIdx].x * scale,
                                                 result.matched2DPoints[inlierIdx].y * scale);
                            cv::circle(displayFrame, scaledPt, 5, cv::Scalar(0, 255, 0), 2);
                        }

                        cv::Mat mapViz = createMapVisualization(result, mDisplaySize);
                        cv::Mat combined(mDisplaySize.height, mDisplaySize.width * 2, CV_8UC3);
                        displayFrame.copyTo(combined(cv::Rect(0, 0, mDisplaySize.width, mDisplaySize.height)));
                        mapViz.copyTo(combined(cv::Rect(mDisplaySize.width, 0, mDisplaySize.width, mDisplaySize.height)));

                        for (int inlierIdx : result.inlierIndices)
                        {
                            float scale = (float)mDisplaySize.width / mProcessSize.width;
                            cv::Point2f pt2D(result.matched2DPoints[inlierIdx].x * scale,
                                             result.matched2DPoints[inlierIdx].y * scale);
                            cv::Point2f pt3DProj = project3DTo2D(result.matched3DPoints[inlierIdx], mDisplaySize.height);
                            pt3DProj.x += mDisplaySize.width;
                            cv::line(combined, pt2D, pt3DProj, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        }

                        cv::Scalar statusColor;
                        std::string status;
                        if (result.confidence >= 70)
                        {
                            statusColor = cv::Scalar(0, 255, 0);
                            status = "EXCELLENT";
                        }
                        else if (result.confidence >= 50)
                        {
                            statusColor = cv::Scalar(0, 200, 255);
                            status = "GOOD";
                        }
                        else
                        {
                            statusColor = cv::Scalar(0, 165, 255);
                            status = "WEAK";
                        }

                        cv::putText(combined, "LOCALIZED - " + status,
                                    cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 2);

                        std::ostringstream info;
                        info << "Inliers: " << result.numInliers << "/" << result.totalMatches
                             << " | Conf: " << std::fixed << std::setprecision(1)
                             << result.confidence << "%";
                        cv::putText(combined, info.str(),
                                    cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

                        if (elapsed > 0.0)
                        {
                            std::ostringstream ts;
                            ts << "t=" << std::fixed << std::setprecision(3) << elapsed << "s";
                            cv::putText(combined, ts.str(),
                                        cv::Point(30, 95), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
                        }

                        cv::imshow("Relocalization: Webcam + Map", combined);
                    }
                    else
                    {
                        cv::Mat mapViz = createMapVisualization(result, mDisplaySize);
                        cv::Mat combined(mDisplaySize.height, mDisplaySize.width * 2, CV_8UC3);
                        displayFrame.copyTo(combined(cv::Rect(0, 0, mDisplaySize.width, mDisplaySize.height)));
                        mapViz.copyTo(combined(cv::Rect(mDisplaySize.width, 0, mDisplaySize.width, mDisplaySize.height)));

                        cv::putText(combined, "SEARCHING...",
                                    cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 100, 255), 2);

                        if (elapsed > 0.0)
                        {
                            std::ostringstream ts;
                            ts << "t=" << std::fixed << std::setprecision(3) << elapsed << "s";
                            cv::putText(combined, ts.str(),
                                        cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
                        }

                        cv::imshow("Relocalization: Webcam + Map", combined);
                    }

                    int key = cv::waitKey(1);
                    if (key == 27)
                    {
                        std::cout << "\nESC pressed - stopping" << std::endl;
                        break;
                    }
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "\n[ERROR] Display error: " << e.what() << std::endl;
                    mVisualizationEnabled = false;
                    visualize = false;
                }
            }
        }

        int processed = frameCount / mFrameSkip;
        std::cout << "\n=== Webcam processing stopped ===" << std::endl;
        if (processed > 0)
            std::cout << "Success rate: " << (100.0 * successCount / processed) << "%" << std::endl;
    }

    void RelocalizationModule::exportMapToPCD(const std::string &outputPath)
    {
        std::ofstream file(outputPath);
        if (!file.is_open())
        {
            std::cerr << "Cannot open file for writing: " << outputPath << std::endl;
            return;
        }

        file << "# .PCD v0.7 - Point Cloud Data file format\n";
        file << "VERSION 0.7\n";
        file << "FIELDS x y z\n";
        file << "SIZE 4 4 4\n";
        file << "TYPE F F F\n";
        file << "COUNT 1 1 1\n";
        file << "WIDTH " << mMapPointsViz.size() << "\n";
        file << "HEIGHT 1\n";
        file << "VIEWPOINT 0 0 0 1 0 0 0\n";
        file << "POINTS " << mMapPointsViz.size() << "\n";
        file << "DATA ascii\n";

        for (const auto &pt : mMapPointsViz)
        {
            file << pt.x << " " << pt.y << " " << pt.z << "\n";
        }

        file.close();
        std::cout << "Map exported to " << outputPath << std::endl;
    }
    void RelocalizationModule::drawOrientedTriangle(cv::Mat &img, const cv::Point2f &center,
                                                    float angle, float size,
                                                    const cv::Scalar &color, int thickness)
    {
        // Create triangle points (pointing upward initially)
        std::vector<cv::Point2f> trianglePoints;                           // FIX: Added <cv::Point2f>
        trianglePoints.push_back(cv::Point2f(size, 0));                    // Top point
        trianglePoints.push_back(cv::Point2f(-size * 0.6f, size * 0.5f));  // Bottom left
        trianglePoints.push_back(cv::Point2f(-size * 0.6f, -size * 0.5f)); // Bottom right

        // Rotate and translate
        float cosA = cos(angle);
        float sinA = sin(angle);
        std::vector<cv::Point> rotatedPoints; // FIX: Added <cv::Point>

        for (const auto &pt : trianglePoints)
        {
            float x = pt.x * cosA - pt.y * sinA + center.x;
            float y = pt.x * sinA + pt.y * cosA + center.y;
            rotatedPoints.push_back(cv::Point((int)x, (int)y));
        }

        // Draw filled triangle
        if (thickness == -1)
        {
            cv::fillConvexPoly(img, rotatedPoints, color);
        }
        else
        {
            for (size_t i = 0; i < rotatedPoints.size(); i++)
            {
                cv::line(img, rotatedPoints[i], rotatedPoints[(i + 1) % rotatedPoints.size()],
                         color, thickness);
            }
        }
    }

    void RelocalizationModule::drawGrid(cv::Mat &img, int mapHeight)
    {
        float rangeX = mMaxX - mMinX;
        float rangeZ = mMaxZ - mMinZ;

        float marginX = 100.0f;
        float marginY = 100.0f;
        float availableWidth = mDisplaySize.width - 2 * marginX;
        float availableHeight = mapHeight - 2 * marginY;

        float baseScale = std::min(availableWidth / rangeX, availableHeight / rangeZ);
        float scale = baseScale * mMapZoomScale;

        float scaledWidth = rangeX * scale;
        float scaledHeight = rangeZ * scale;
        float offsetX = marginX + (availableWidth - scaledWidth) / 2 + mMapOffsetX;
        float offsetY = marginY + (availableHeight - scaledHeight) / 2 + mMapOffsetY;

        // Determine grid spacing (aim for ~10 grid lines)
        float gridSpacingWorld = std::max(0.5f, std::ceil(std::max(rangeX, rangeZ) / 10.0f));

        // Draw vertical grid lines (X axis)
        float startX = std::floor(mMinX / gridSpacingWorld) * gridSpacingWorld;
        for (float x = startX; x <= mMaxX; x += gridSpacingWorld)
        {
            int screenX = (int)(offsetX + (x - mMinX) * scale);

            cv::Scalar lineColor;
            int lineThickness;

            if (std::abs(x) < 0.01f) // X = 0 axis
            {
                lineColor = cv::Scalar(0, 0, 255); // Red for X axis
                lineThickness = 2;
            }
            else
            {
                lineColor = cv::Scalar(220, 220, 220); // Light gray for grid
                lineThickness = 1;
            }

            cv::line(img,
                     cv::Point(screenX, 0),
                     cv::Point(screenX, mapHeight),
                     lineColor, lineThickness);

            // Label
            if (std::abs(x) > 0.01f || x == startX)
            {
                std::ostringstream label;
                label << std::fixed << std::setprecision(1) << x;
                cv::putText(img, label.str(),
                            cv::Point(screenX + 3, mapHeight - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3,
                            cv::Scalar(100, 100, 100), 1);
            }
        }

        // Draw horizontal grid lines (Z axis)
        float startZ = std::floor(mMinZ / gridSpacingWorld) * gridSpacingWorld;
        for (float z = startZ; z <= mMaxZ; z += gridSpacingWorld)
        {
            int screenY = (int)(mapHeight - offsetY - (z - mMinZ) * scale);

            cv::Scalar lineColor;
            int lineThickness;

            if (std::abs(z) < 0.01f) // Z = 0 axis
            {
                lineColor = cv::Scalar(0, 255, 0); // Green for Z axis
                lineThickness = 2;
            }
            else
            {
                lineColor = cv::Scalar(220, 220, 220); // Light gray for grid
                lineThickness = 1;
            }

            cv::line(img,
                     cv::Point(0, screenY),
                     cv::Point(mDisplaySize.width, screenY),
                     lineColor, lineThickness);

            // Label
            if (std::abs(z) > 0.01f || z == startZ)
            {
                std::ostringstream label;
                label << std::fixed << std::setprecision(1) << z;
                cv::putText(img, label.str(),
                            cv::Point(5, screenY - 3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3,
                            cv::Scalar(100, 100, 100), 1);
            }
        }

        // Draw axis labels
        cv::putText(img, "X",
                    cv::Point(mDisplaySize.width - 25, mapHeight - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        cv::putText(img, "Y",
                    cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // Draw origin marker
        cv::Point2f origin = project3DTo2D(cv::Point3f(0, 0, 0), mapHeight);
        cv::circle(img, origin, 8, cv::Scalar(255, 0, 0), 2); // Blue circle
        cv::putText(img, "Origin",
                    cv::Point(origin.x + 12, origin.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }

    // ── Semantic weight per YOLO class ID ────────────────────────────────────────
    // All YOLO-detected objects get the same landmark weight (1.0).
    // To give specific classes a different weight, add cases here.
    static float getSemanticWeight(int /*cls_id*/)
    {
        return 1.0f;
        // switch (cls_id) {
        //   case 39: return 0.8f;  // bottle — less permanent
        //   case 56: return 1.0f;  // chair
        //   default: return 1.0f;
        // }
    }

    // ── 2×6 Jacobian of projection w.r.t. SE3 left perturbation ─────────────────
    // J = ∂proj/∂δξ  (positive convention; matches solve(b) in solvePnPWeighted)
    Eigen::Matrix<double, 2, 6> RelocalizationModule::computeProjectionJacobian(
        const Eigen::Vector3d &Pc, double fx, double fy)
    {
        double Xc = Pc(0), Yc = Pc(1), Zc = Pc(2);
        double Zc_inv = 1.0 / Zc;
        double Zc_inv2 = Zc_inv * Zc_inv;

        Eigen::Matrix<double, 2, 6> J;
        // Row 0: ∂u/∂δξ = [∂u/∂t | ∂u/∂φ]
        J(0, 0) = fx * Zc_inv;
        J(0, 1) = 0.0;
        J(0, 2) = -fx * Xc * Zc_inv2;
        J(0, 3) = -fx * Xc * Yc * Zc_inv2;
        J(0, 4) = fx + fx * Xc * Xc * Zc_inv2;
        J(0, 5) = -fx * Yc * Zc_inv;
        // Row 1: ∂v/∂δξ
        J(1, 0) = 0.0;
        J(1, 1) = fy * Zc_inv;
        J(1, 2) = -fy * Yc * Zc_inv2;
        J(1, 3) = -(fy + fy * Yc * Yc * Zc_inv2);
        J(1, 4) = fy * Xc * Yc * Zc_inv2;
        J(1, 5) = fy * Xc * Zc_inv;
        return J;
    }

    // ── Mean unweighted reprojection error (public, for comparison logging) ───────
    float RelocalizationModule::computeMeanReprojError(
        const std::vector<cv::Point3f> &points3D,
        const std::vector<cv::Point2f> &points2D,
        const cv::Mat &rvec, const cv::Mat &tvec)
    {
        if (points3D.empty())
            return 0.0f;
        std::vector<cv::Point2f> projected;
        cv::projectPoints(points3D, rvec, tvec, mK, mDistCoef, projected);
        float total = 0.0f;
        for (size_t i = 0; i < projected.size(); ++i)
        {
            float du = points2D[i].x - projected[i].x;
            float dv = points2D[i].y - projected[i].y;
            total += std::sqrt(du * du + dv * dv);
        }
        return total / (float)projected.size();
    }

    // ── Weight assignment from pre-segmented landmark regions ────────────────────
    // Uses region.keypoints (already separated in relocalization_node) — no bbox check here.
    std::vector<float> RelocalizationModule::assignWeightsFromLandmarks(
        const std::vector<cv::Point2f> &matched2DPoints,
        const std::vector<LandmarkRegion> &landmarkRegions,
        float permanentWeight,
        float backgroundWeight)
    {
        std::vector<float> weights(matched2DPoints.size(), backgroundWeight);

        for (size_t i = 0; i < matched2DPoints.size(); ++i)
        {
            for (const auto &region : landmarkRegions)
            {
                for (const auto &kp : region.keypoints)
                {
                    if (std::abs(kp.pt.x - matched2DPoints[i].x) < 0.5f &&
                        std::abs(kp.pt.y - matched2DPoints[i].y) < 0.5f)
                    {
                        float w = getSemanticWeight(region.cls_id) * permanentWeight;
                        weights[i] = std::max(weights[i], w);
                        break;
                    }
                }
            }
        }
        return weights;
    }

    // ── Hamming-distance filter via BFMatcher ────────────────────────────────────
    void RelocalizationModule::filterByHammingDistance(
        const std::vector<cv::Point3f> &points3D,
        const std::vector<cv::Point2f> &points2D,
        const cv::Mat &queryDescriptors,
        const cv::Mat &trainDescriptors,
        std::vector<cv::Point3f> &filtered3D,
        std::vector<cv::Point2f> &filtered2D,
        std::vector<int> &filteredIndices,
        int threshold)
    {
        int thr = (threshold < 0) ? mHammingThreshold : threshold;
        filtered3D.clear();
        filtered2D.clear();
        filteredIndices.clear();

        // crossCheck=true: only keep mutually best matches between the paired descriptors
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(queryDescriptors, trainDescriptors, matches);

        for (const auto &m : matches)
        {
            if (m.distance < thr)
            {
                int i = m.queryIdx;
                filtered3D.push_back(points3D[i]);
                filtered2D.push_back(points2D[i]);
                filteredIndices.push_back(i);
            }
        }
    }

    // ── Standard PnP wrapper ─────────────────────────────────────────────────────
    bool RelocalizationModule::solvePnP(
        const std::vector<cv::Point3f> &points3D,
        const std::vector<cv::Point2f> &points2D,
        const cv::Mat &queryDescriptors,
        const cv::Mat &trainDescriptors,
        cv::Mat &rvec, cv::Mat &tvec,
        int method)
    {
        std::vector<cv::Point3f> filtered3D;
        std::vector<cv::Point2f> filtered2D;
        std::vector<int> filteredIndices;
        filterByHammingDistance(points3D, points2D, queryDescriptors, trainDescriptors,
                                filtered3D, filtered2D, filteredIndices);
        if (filtered3D.size() < 4)
            return false;
        return cv::solvePnP(filtered3D, filtered2D, mK, mDistCoef,
                            rvec, tvec, false, method);
    }

    // ── Weighted PnP via Gauss-Newton / Levenberg-Marquardt ──────────────────────
    WeightedPnPResult RelocalizationModule::solvePnPWeighted(
        const std::vector<cv::Point3f> &points3D,
        const std::vector<cv::Point2f> &points2D,
        const std::vector<float> &weights,
        const cv::Mat &rvec_hint,
        const cv::Mat &tvec_hint,
        int maxIterations,
        double convergenceThreshold,
        float inlierThresholdPx)
    {
        WeightedPnPResult result;
        result.success = false;
        result.iterations = 0;

        const int n = (int)points3D.size();
        if (n < 6 || (int)points2D.size() != n || (int)weights.size() != n)
            return result;

        // Camera intrinsics (mK is CV_32F)
        double fx = (double)mK.at<float>(0, 0);
        double fy = (double)mK.at<float>(1, 1);
        double cx = (double)mK.at<float>(0, 2);
        double cy = (double)mK.at<float>(1, 2);

        // Normalize weights so mean = 1  (preserves relative weighting, stable Hessian scale)
        float w_sum = 0.0f;
        for (float w : weights)
            w_sum += w;
        float w_mean = w_sum / n;
        std::vector<double> w_norm(n);
        for (int i = 0; i < n; ++i)
            w_norm[i] = (double)(weights[i] / w_mean);

        // Warm-start: prefer caller-supplied RANSAC pose; fall back to fresh EPnP
        cv::Mat rvec_init, tvec_init;
        if (!rvec_hint.empty() && !tvec_hint.empty())
        {
            rvec_hint.convertTo(rvec_init, CV_64F);
            tvec_hint.convertTo(tvec_init, CV_64F);
        }
        else
        {
            bool init_ok = cv::solvePnP(points3D, points2D, mK, mDistCoef,
                                        rvec_init, tvec_init, false,
                                        cv::SOLVEPNP_EPNP);
            if (!init_ok)
            {
                rvec_init = cv::Mat::zeros(3, 1, CV_64F);
                tvec_init = cv::Mat::zeros(3, 1, CV_64F);
            }
        }

        // Convert rvec/tvec → Sophus SE3
        cv::Mat R_init;
        cv::Rodrigues(rvec_init, R_init);
        // Ensure R_init is CV_64F
        if (R_init.type() != CV_64F)
            R_init.convertTo(R_init, CV_64F);
        if (tvec_init.type() != CV_64F)
            tvec_init.convertTo(tvec_init, CV_64F);

        Eigen::Matrix3d R_e;
        Eigen::Vector3d t_e;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R_e(i, j) = R_init.at<double>(i, j);
        for (int i = 0; i < 3; ++i)
            t_e(i) = tvec_init.at<double>(i);

        Sophus::SE3d T_cw(R_e, t_e);

        // Levenberg-Marquardt
        double lambda = 1e-3;
        double cost = std::numeric_limits<double>::max();
        const Sophus::SE3d T_init = T_cw;
        Sophus::SE3d T_last_good = T_cw;
        Sophus::SE3d T_prev = T_cw;

        for (int iter = 0; iter < maxIterations; ++iter)
        {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            double new_cost = 0.0;

            int valid_pts = 0;
            for (int i = 0; i < n; ++i)
            {
                Eigen::Vector3d Xw(points3D[i].x, points3D[i].y, points3D[i].z);
                Eigen::Vector3d Pc = T_cw * Xw;
                if (Pc(2) <= 0)
                    continue;

                double u_proj = fx * Pc(0) / Pc(2) + cx;
                double v_proj = fy * Pc(1) / Pc(2) + cy;

                // r = obs - proj
                Eigen::Vector2d r(points2D[i].x - u_proj,
                                  points2D[i].y - v_proj);

                Eigen::Matrix<double, 2, 6> J = computeProjectionJacobian(Pc, fx, fy);

                double w = w_norm[i];
                H += w * J.transpose() * J;
                // GN: H δξ = Σ w Jᵀ r  (valid when J = +∂proj/∂ξ, r = obs-proj)
                b += w * J.transpose() * r;
                new_cost += w * r.squaredNorm();
                ++valid_pts;
            }

            if (valid_pts < 4)
            {
                T_cw = T_last_good;
                break;
            }
            T_last_good = T_cw;

            // LM damping — additive identity term prevents singular H when a DOF is unobservable
            Eigen::Matrix<double, 6, 6> H_damped = H;
            for (int k = 0; k < 6; ++k)
                H_damped(k, k) += lambda * std::max(H(k, k), 1e-8);

            Eigen::Matrix<double, 6, 1> delta = H_damped.ldlt().solve(b);

            if (!delta.allFinite())
            {
                lambda *= 10.0;
                result.iterations = iter + 1;
                continue;
            }

            Sophus::SE3d T_new = Sophus::SE3d::exp(delta) * T_cw;

            if (new_cost < cost)
            {
                T_prev = T_cw;
                T_cw = T_new;
                cost = new_cost;
                lambda /= 10.0;
            }
            else
            {
                T_cw = T_prev;
                lambda *= 10.0;
            }

            result.iterations = iter + 1;
            if (delta.norm() < convergenceThreshold)
                break;
        }

        // Extract final rvec/tvec from converged Sophus pose
        Eigen::Matrix3d R_final = T_cw.rotationMatrix();
        Eigen::Vector3d t_final = T_cw.translation();

        cv::Mat R_cv(3, 3, CV_64F), t_cv(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
                R_cv.at<double>(i, j) = R_final(i, j);
            t_cv.at<double>(i) = t_final(i);
        }
        cv::Mat rvec_final;
        cv::Rodrigues(R_cv, rvec_final);

        // Count inliers using unweighted pixel threshold (fair comparison with solvePnPRansac)
        std::vector<int> inliers;
        for (int i = 0; i < n; ++i)
        {
            Eigen::Vector3d Xw(points3D[i].x, points3D[i].y, points3D[i].z);
            Eigen::Vector3d Pc = T_cw * Xw;
            if (Pc(2) <= 0)
                continue;
            float u_proj = (float)(fx * Pc(0) / Pc(2) + cx);
            float v_proj = (float)(fy * Pc(1) / Pc(2) + cy);
            float du = points2D[i].x - u_proj;
            float dv = points2D[i].y - v_proj;
            if (std::sqrt(du * du + dv * dv) < inlierThresholdPx)
                inliers.push_back(i);
        }

        if ((int)inliers.size() < 4)
        {
            // LM converged to a bad local minimum — fall back to the initial
            // (RANSAC) pose, which is guaranteed to have all input points as inliers.
            inliers.clear();
            for (int i = 0; i < n; ++i)
            {
                Eigen::Vector3d Xw(points3D[i].x, points3D[i].y, points3D[i].z);
                Eigen::Vector3d Pc = T_init * Xw;
                if (Pc(2) <= 0)
                    continue;
                float u_proj = (float)(fx * Pc(0) / Pc(2) + cx);
                float v_proj = (float)(fy * Pc(1) / Pc(2) + cy);
                float du = points2D[i].x - u_proj;
                float dv = points2D[i].y - v_proj;
                if (std::sqrt(du * du + dv * dv) < inlierThresholdPx)
                    inliers.push_back(i);
            }
            if ((int)inliers.size() < 4)
                return result;
            // Use the initial pose for rvec/tvec extraction below
            Eigen::Matrix3d R_fb = T_init.rotationMatrix();
            Eigen::Vector3d t_fb = T_init.translation();
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    R_cv.at<double>(i, j) = R_fb(i, j);
                t_cv.at<double>(i) = t_fb(i);
            }
            cv::Rodrigues(R_cv, rvec_final);
        }

        result.success = true;
        result.rvec = rvec_final.clone();
        result.tvec = t_cv.clone();
        result.numInliers = (int)inliers.size();
        result.totalCorrespondences = n;
        result.inlierIndices = std::move(inliers);
        result.weightedReprojectionError = (float)(cost / n);
        result.meanReprojectionError = computeMeanReprojError(
            points3D, points2D, rvec_final, t_cv);
        result.position = computePosition(rvec_final, t_cv);

        return result;
    }

} // namespace Relocalization