#include "relocalization.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <iomanip>

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
        std::vector<cv::Point2f> &points2D)
    {
        points3D.clear();
        points2D.clear();

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
                }
            }
        }

        return points3D.size() >= (size_t)mMinMatches;
    }

    bool RelocalizationModule::solvePnP(const std::vector<cv::Point3f> &points3D,
                                        const std::vector<cv::Point2f> &points2D,
                                        cv::Mat &rvec, cv::Mat &tvec,
                                        std::vector<int> &inliers)
    {
        if (points3D.size() < (size_t)mMinMatches)
            return false;

        cv::Mat inliersMask;
        bool success = cv::solvePnPRansac(points3D, points2D, mK, mDistCoef,
                                          rvec, tvec, false, 300, 8.0, 0.9,
                                          inliersMask, cv::SOLVEPNP_EPNP);

        if (!success)
            return false;

        inliers.clear();
        int numInliers = cv::countNonZero(inliersMask);
        for (int i = 0; i < inliersMask.rows; i++)
        {
            if (inliersMask.at<uchar>(i))
            {
                inliers.push_back(i);
            }
        }

        return numInliers >= mMinInliers;
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

    LocationResult RelocalizationModule::processFrame(const cv::Mat &frame)
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

        float bestScore = 0.0f;
        for (auto pKF : candidates)
        {
            std::vector<cv::Point3f> points3D;
            std::vector<cv::Point2f> points2D;

            if (matchWithKeyFrame(keypoints, descriptors, pKF, points3D, points2D))
            {
                DBoW2::BowVector currentBowVec;
                DBoW2::FeatureVector currentFeatVec;
                std::vector<cv::Mat> vCurrentDesc;
                vCurrentDesc.reserve(descriptors.rows);
                for (int i = 0; i < descriptors.rows; ++i)
                {
                    vCurrentDesc.push_back(descriptors.row(i).clone());
                }
                mpVocabulary->transform(vCurrentDesc, currentBowVec, currentFeatVec, 4);
                float bowScore = mpVocabulary->score(currentBowVec, pKF->mBowVec);

                cv::Mat rvec, tvec;
                std::vector<int> inliers;

                if (solvePnP(points3D, points2D, rvec, tvec, inliers))
                {
                    float inlierRatio = (float)inliers.size() / points3D.size();
                    float confidence = std::min(100.0f,
                                                (inlierRatio * 60.0f) +
                                                    (bowScore * 40.0f * 100.0f) +
                                                    (std::min(50, (int)inliers.size()) * 0.8f));

                    if (confidence > bestScore)
                    {
                        bestScore = confidence;

                        result.success = true;
                        result.position = computePosition(rvec, tvec);
                        result.matchedKeyFrameId = pKF->mnId;
                        result.numInliers = inliers.size();
                        result.totalMatches = points3D.size();
                        result.confidence = confidence;
                        result.bowScore = bowScore;

                        // Store match data for visualization
                        result.matched2DPoints = points2D;
                        result.matched3DPoints = points3D;
                        result.inlierIndices = inliers;

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

        // Draw all map points (small gray dots)
        for (const auto &pt : mMapPointsViz)
        {
            cv::Point2f pt2d = project3DTo2D(pt, targetSize.height);
            cv::circle(mapViz, pt2d, 1, cv::Scalar(180, 180, 180), -1);
        }

        // Highlight matched 3D points (yellow)
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

            // Draw current position (RED - on top layer)
            cv::Point2f posPoint = project3DTo2D(mCurrentPosition, targetSize.height);
            cv::circle(mapViz, posPoint, 10, cv::Scalar(0, 0, 255), -1);  // Red dot
            cv::circle(mapViz, posPoint, 18, cv::Scalar(0, 255, 255), 3); // Yellow ring

            // Text on top layer
            cv::putText(mapViz, "YOU ARE HERE",
                        cv::Point(posPoint.x + 25, posPoint.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        // Title
        cv::putText(mapViz, "Map (Top-Down View)",
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

            auto result = processFrame(frame);

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

} // namespace Relocalization