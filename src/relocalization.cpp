#include "relocalization.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unistd.h> // For getcwd

namespace Relocalization
{

    RelocalizationModule::RelocalizationModule(const std::string &vocabPath,
                                               const std::string &configPath)
        : mVocabPath(vocabPath), mConfigPath(configPath), mpVocabulary(nullptr), mpAtlas(nullptr), mpSLAM(nullptr), mpKeyFrameDB(nullptr)
    {
        // Load configuration from YAML
        if (!loadConfig())
        {
            std::cerr << "Failed to load configuration from " << configPath << std::endl;
            return;
        }
        // Initialize vocabulary
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
            // You might want to throw an exception or handle this error
        }
        else
        {
            std::cout << "[INFO] Vocabulary loaded successfully" << std::endl;

            // Create KeyFrame Database
            mpKeyFrameDB = new ORB_SLAM3::KeyFrameDatabase(*mpVocabulary);
            std::cout << "[INFO] KeyFrame database created" << std::endl;
        }

        if (bVocLoad)
        {
            std::cout << "[INFO] Vocabulary loaded successfully." << std::endl;
            std::cout << "[INFO] Vocabulary empty()? " << (mpVocabulary->empty() ? "YES" : "NO") << std::endl;
            std::cout << "[INFO] Vocabulary size() (words): " << mpVocabulary->size() << std::endl;
        }
        else
        {
            std::cerr << "[ERROR] Failed to load vocabulary from: " << mVocabPath << std::endl;
        }

        // Initialize ORB extractor with settings from config
        mpORBextractor = std::make_unique<ORB_SLAM3::ORBextractor>(1000, 1.2, 8, 20, 7);

        std::cout << "Relocalization module initialized" << std::endl;
        std::cout << "  Map: " << mMapPath << std::endl;
        std::cout << "  Camera: fx=" << mK.at<float>(0, 0) << ", fy=" << mK.at<float>(1, 1)
                  << ", cx=" << mK.at<float>(0, 2) << ", cy=" << mK.at<float>(1, 2) << std::endl;
    }

    RelocalizationModule::~RelocalizationModule()
    {
        // Clean up database BEFORE vocabulary
        if (mpKeyFrameDB)
        {
            delete mpKeyFrameDB;
            mpKeyFrameDB = nullptr;
        }

        // Your existing cleanup for vocabulary and other members
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
        std::cout << "Preparing to load map..." << std::endl;

        // Debug: Show current working directory
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr)
        {
            std::cout << "\nDebug Info:" << std::endl;
            std::cout << "  Current working directory: " << cwd << std::endl;
            std::cout << "  Map path from config: " << mMapPath << std::endl;
        }

        std::cout << "\nLoading map using System class..." << std::endl;

        try
        {
            mpSLAM = new ORB_SLAM3::System(mVocabPath, mConfigPath,
                                           ORB_SLAM3::System::MONOCULAR, false);

            std::cout << "✓ System created, atlas loading..." << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(2));

            mpAtlas = mpSLAM->mpAtlas;
            if (!mpAtlas)
            {
                std::cerr << "✗ Failed to access atlas" << std::endl;
                return false;
            }

            // Use System's vocabulary (same one used to create the map)
            mpVocabulary = mpSLAM->mpVocabulary;

            std::cout << "✓ Atlas and vocabulary accessed successfully!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }

        // Get ALL maps and find the one with data
        std::vector<ORB_SLAM3::Map *> vpMaps = mpAtlas->GetAllMaps();

        ORB_SLAM3::Map *pMap = nullptr;
        std::cout << "\nSearching for map with data..." << std::endl;
        std::cout << "Total maps in atlas: " << vpMaps.size() << std::endl;

        for (auto pMapCandidate : vpMaps)
        {
            auto kfs = pMapCandidate->GetAllKeyFrames();
            if (!kfs.empty())
            {
                pMap = pMapCandidate;
                std::cout << "Found map with " << kfs.size() << " keyframes (Map ID: "
                          << pMap->GetId() << ")" << std::endl;
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

        std::cout << "\nMap data extracted:" << std::endl;
        std::cout << "  KeyFrames: " << mvKeyFrames.size() << std::endl;
        std::cout << "  MapPoints: " << mvMapPoints.size() << std::endl;

        if (mvKeyFrames.empty() || mvMapPoints.empty())
        {
            std::cerr << "Warning: Map is empty!" << std::endl;
            return false;
        }

        // ============ ADD THIS SECTION BEFORE SHUTDOWN ============
        // CRITICAL: Compute BoW while the system is still active
        if (!mpVocabulary || !mpKeyFrameDB)
        {
            std::cerr << "[ERROR] Vocabulary or Database not initialized!" << std::endl;
            return false;
        }

        std::cout << "[INFO] Computing BoW vectors for " << mvKeyFrames.size() << " keyframes..." << std::endl;

        int processedCount = 0;
        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->isBad())
            {
                // Set vocabulary reference for keyframe
                pKF->SetORBVocabulary(mpVocabulary);

                // Compute BoW vectors - THIS MUST BE DONE BEFORE SHUTDOWN!
                pKF->ComputeBoW();

                // Add to database for fast searching
                mpKeyFrameDB->add(pKF);

                processedCount++;
                if (processedCount % 100 == 0)
                {
                    std::cout << "[DEBUG] Processed " << processedCount << " keyframes..." << std::endl;
                }
            }
        }

        // setelah ComputeBoW() + add ke DB
        int cntKFwithBoW = 0;
        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->mBowVec.empty())
                cntKFwithBoW++;
        }
        std::cout << "[DEBUG] KeyFrames total: " << mvKeyFrames.size()
                  << ", KeyFrames with BoW: " << cntKFwithBoW << std::endl;

        std::cout << "[DEBUG] KeyFrameDB size (if method exists): ";
#ifdef HAVE_KFDB_SIZE_METHOD
        std::cout << mpKeyFrameDB->size() << std::endl;
#else
        std::cout << "(no size() method available in this DB impl)" << std::endl;
#endif

        std::cout << "[INFO] BoW computed for " << processedCount << " keyframes" << std::endl;
        // ============ END OF NEW SECTION ============

        mMapPointsViz.clear();
        for (auto pMP : mvMapPoints)
        {
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f pos = pMP->GetWorldPos();
                mMapPointsViz.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
            }
        }

        std::cout << "  Valid map points: " << mMapPointsViz.size() << std::endl;

        if (mExportPCD)
        {
            exportMapToPCD(mPCDPath);
        }

        std::cout << "\nShutting down System to release resources..." << std::endl;
        mpSLAM->Shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        delete mpSLAM;
        mpSLAM = nullptr;
        std::cout << "✓ System shut down successfully" << std::endl;

        return true;
    }

    bool RelocalizationModule::loadConfig()
    {
        std::cout << "Loading configuration from " << mConfigPath << std::endl;

        cv::FileStorage fs(mConfigPath, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "Failed to open config file: " << mConfigPath << std::endl;
            return false;
        }

        // Map path
        fs["System.LoadAtlasFromFile"] >> mMapPath;
        if (mMapPath.empty())
        {
            std::cerr << "System.LoadAtlasFromFile not found in config" << std::endl;
            return false;
        }

        // Camera parameters
        float fx = fs["Camera1.fx"];
        float fy = fs["Camera1.fy"];
        float cx = fs["Camera1.cx"];
        float cy = fs["Camera1.cy"];

        if (fx == 0.0f || fy == 0.0f)
        {
            std::cerr << "Invalid camera parameters in config" << std::endl;
            return false;
        }

        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;

        // Distortion coefficients
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

        // Relocalization settings
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

        // Visualization settings
        mVisualizationEnabled = fs["Visualization.Enabled"].empty() ? true : (int)fs["Visualization.Enabled"] != 0;
        mExportPCD = fs["Visualization.ExportPCD"].empty() ? true : (int)fs["Visualization.ExportPCD"] != 0;
        fs["Visualization.PCDPath"] >> mPCDPath;
        if (mPCDPath.empty())
            mPCDPath = "map_points.pcd";

        fs.release();

        std::cout << "Configuration loaded successfully" << std::endl;
        std::cout << "  Map: " << mMapPath << std::endl;
        std::cout << "  Camera: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
        std::cout << "  Distortion: k1=" << mDistCoef.at<float>(0) << ", k2=" << mDistCoef.at<float>(1) << std::endl;
        std::cout << "  Process size: " << mProcessSize << std::endl; // ⭐ NEW
        std::cout << "  Display size: " << mDisplaySize << std::endl; // ⭐ NEW

        return true;
    }

    void RelocalizationModule::extractFeatures(const cv::Mat &frame,
                                               std::vector<cv::KeyPoint> &keypoints,
                                               cv::Mat &descriptors)
    {
        std::cout << "[DEBUG] extractFeatures: Frame size=" << frame.cols << "x" << frame.rows
                  << ", channels=" << frame.channels() << std::endl;

        if (frame.empty())
        {
            std::cerr << "[ERROR] Frame is empty!" << std::endl;
            return;
        }

        cv::Mat gray;
        if (frame.channels() == 3)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = frame.clone();
        }

        if (gray.empty() || !gray.isContinuous())
        {
            cerr << "Invalid gray image!" << endl;
            return;
        }

        std::cout << "[DEBUG] Gray image size=" << gray.cols << "x" << gray.rows << std::endl;
        std::cout << "[DEBUG] mpORBextractor pointer=" << mpORBextractor.get() << std::endl;

        if (!mpORBextractor)
        {
            std::cerr << "[ERROR] ORB extractor is null!" << std::endl;
            return;
        }

        std::cout << "[DEBUG] Calling ORB extractor..." << std::endl;

        // Extract ORB features
        // Note: ORBextractor requires an additional vector<int> parameter
        std::vector<int> vLappingArea = {0, 0};

        cout << "vLappingArea size: " << vLappingArea.size() << endl;

        (*mpORBextractor)(gray, cv::Mat(), keypoints, descriptors, vLappingArea);
        std::cout << "[DEBUG] Extraction complete, found " << keypoints.size() << " keypoints" << std::endl;
    }

    std::vector<ORB_SLAM3::KeyFrame *> RelocalizationModule::detectRelocalizationCandidates(
        const cv::Mat &descriptors)
    {

        std::vector<ORB_SLAM3::KeyFrame *> vpCandidates;

        if (!mpKeyFrameDB || !mpVocabulary)
        {
            std::cerr << "[ERROR] Database or Vocabulary not initialized!" << std::endl;
            return vpCandidates;
        }

        std::cout << "[DEBUG] Manual BoW-based matching..." << std::endl;

        // manual BoW-based matching computation
        DBoW2::BowVector currentBowVec;
        DBoW2::FeatureVector currentFeatVec;

        std::cout << "[DEBUG] Converting descriptors to BoW representation..." << std::endl;
        std::cout << "Descriptor size: " << descriptors.rows << "x" << descriptors.cols
                  << " type=" << descriptors.type() << std::endl;

        if (descriptors.empty())
        {
            std::cerr << "[ERROR] descriptors.empty() == true -> skipping BoW transform\n";
            return vpCandidates;
        }
        std::cout << "[DEBUG] Descriptor size: " << descriptors.rows << "x" << descriptors.cols
                  << " type=" << descriptors.type() << std::endl;

        if (mpVocabulary->empty())
        {
            std::cerr << "[ERROR] Vocabulary is EMPTY before transform. Check load path and format.\n";
            return vpCandidates;
        }
        std::cout << "[DEBUG] Vocabulary size (words): " << mpVocabulary->size() << std::endl;

        // prepare vector<Mat> version (recommended)
        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(descriptors.rows);
        for (int i = 0; i < descriptors.rows; ++i)
        {
            vCurrentDesc.push_back(descriptors.row(i).clone()); // clone untuk safety
        }

        // Convert descriptors to BoW representation
        mpVocabulary->transform(vCurrentDesc, currentBowVec, currentFeatVec, 4);

        std::cout << "[DEBUG] BoW vector size: " << currentBowVec.size()
                  << ", Feature vector size: " << currentFeatVec.size() << std::endl;
        std::vector<std::pair<float, ORB_SLAM3::KeyFrame *>> vScoreAndMatch;

        std::cout << "[DEBUG] Compute similarity score..." << std::endl;
        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->isBad() && !pKF->mBowVec.empty())
            {
                // Compute similarity score
                float score = mpVocabulary->score(currentBowVec, pKF->mBowVec);

                if (score > mBowThreshold)
                {
                    vScoreAndMatch.push_back(std::make_pair(score, pKF));
                }
            }
        }

        std::cout << "[DEBUG] Sort by score..." << std::endl;

        // Sort by score (highest first)
        std::sort(vScoreAndMatch.begin(), vScoreAndMatch.end(),
                  [](const auto &a, const auto &b)
                  { return a.first > b.first; });

        // Take top candidates
        int nCandidates = std::min(mMaxCandidates, (int)vScoreAndMatch.size());
        for (int i = 0; i < nCandidates; i++)
        {
            vpCandidates.push_back(vScoreAndMatch[i].second);
            std::cout << "[DEBUG] Candidate " << i << ": KF "
                      << vScoreAndMatch[i].second->mnId
                      << " (score: " << vScoreAndMatch[i].first << ")" << std::endl;
        }

        std::cout << "[DEBUG] Found " << vpCandidates.size() << " candidate keyframes" << std::endl;
        return vpCandidates;
    }

    void RelocalizationModule::debugStatus()
    {
        std::cout << "\n=== RELOCALIZATION MODULE STATUS ===" << std::endl;
        std::cout << "Vocabulary loaded: " << (mpVocabulary ? "YES" : "NO") << std::endl;
        if (mpVocabulary)
        {
            std::cout << "  Vocabulary size: " << mpVocabulary->size() << std::endl;
        }
        std::cout << "Database created: " << (mpKeyFrameDB ? "YES" : "NO") << std::endl;
        std::cout << "KeyFrames loaded: " << mvKeyFrames.size() << std::endl;

        int bowCount = 0;
        for (auto pKF : mvKeyFrames)
        {
            if (pKF && !pKF->mBowVec.empty())
                bowCount++;
        }
        std::cout << "  KeyFrames with BoW: " << bowCount << std::endl;
        std::cout << "Map points: " << mvMapPoints.size() << std::endl;
        std::cout << "  Visualization points: " << mMapPointsViz.size() << std::endl;
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

        // Get keyframe features
        const std::vector<cv::KeyPoint> &vKFKeyPoints = pKF->mvKeysUn;
        const cv::Mat &KFDescriptors = pKF->mDescriptors;

        if (KFDescriptors.empty())
        {
            return false;
        }

        // Match descriptors using BFMatcher
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(descriptors, KFDescriptors, knnMatches, 2);

        // Apply ratio test (Lowe's ratio test)
        const float ratioThresh = 0.75f;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i].size() < 2)
                continue;

            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            {
                int trainIdx = knnMatches[i][0].trainIdx;

                // Get corresponding 3D point
                ORB_SLAM3::MapPoint *pMP = pKF->GetMapPoint(trainIdx);
                if (pMP && !pMP->isBad())
                {
                    // GetWorldPos() returns Eigen::Vector3f
                    Eigen::Vector3f pos = pMP->GetWorldPos();
                    points3D.push_back(cv::Point3f(pos(0), pos(1), pos(2)));

                    int queryIdx = knnMatches[i][0].queryIdx;
                    points2D.push_back(keypoints[queryIdx].pt);
                }
            }
        }

        return points3D.size() >= (size_t)mMinMatches; // Use config value
    }

    bool RelocalizationModule::solvePnP(const std::vector<cv::Point3f> &points3D,
                                        const std::vector<cv::Point2f> &points2D,
                                        cv::Mat &rvec, cv::Mat &tvec,
                                        std::vector<int> &inliers)
    {

        std::cout << "[DEBUG] Check points3D size, with size mMinMatched: " << (size_t)mMinMatches << std::endl;

        if (points3D.size() < (size_t)mMinMatches)
        {
            return false;
        }

        // for (int i = 0; i <= points3D.size(); i++)
        // {
        //     cv::Point3f pg = points3D[i];
        //     if (!cv::checkRange(pg.x) || !cv::checkRange(pg.y) || !cv::checkRange(pg.z))
        //     {
        //         std::cout << "[WARN] Invalid 3D point at index " << i << ": " << pg << std::endl;
        //     }

        //     std::cout << "[DEBUG] PnP Input sizes: 3D=" << points3D.size()
        //               << ", 2D=" << points2D.size() << std::endl;
        // }

        if (points3D.size() > 0)
        {
            std::cout << "[DEBUG] Sample 3D point: " << points3D[0] << std::endl;
            std::cout << "[DEBUG] Sample 2D point: " << points2D[0] << std::endl;
        }

        std::cout << "[DEBUG] Camera matrix: " << mK << std::endl;

        std::cout << "Solve PnP with RANSAC" << std::endl;
        // Solve PnP with RANSAC
        cv::Mat inliersMask;
        bool success = cv::solvePnPRansac(points3D, points2D, mK, mDistCoef,
                                          rvec, tvec, false, 300, 8.0, 0.9,
                                          inliersMask, cv::SOLVEPNP_EPNP);

        if (!success)
        {
            std::cout << "Failed solve PnP with RANSAC" << std::endl;
            return false;
        }

        // Count inliers
        inliers.clear();
        int numInliers = cv::countNonZero(inliersMask);
        std::cout << "[DEBUG] Total Inliers size: " << numInliers << std::endl;
        for (int i = 0; i < inliersMask.rows; i++)
        {
            if (inliersMask.at<uchar>(i))
            {
                inliers.push_back(i);
            }
        }

        std::cout << "[DEBUG] Check Inliers size: " << inliers.size() << std::endl;

        return numInliers >= (size_t)mMinInliers; // Use config value
    }

    cv::Point3f RelocalizationModule::computePosition(const cv::Mat &rvec,
                                                      const cv::Mat &tvec)
    {
        // Convert rotation vector to rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Camera position in world coordinates: -R^T * t
        cv::Mat pos = -R.t() * tvec;

        return cv::Point3f(pos.at<double>(0),
                           pos.at<double>(1),
                           pos.at<double>(2));
    }

    LocationResult RelocalizationModule::processFrame(const cv::Mat &frame)
    {
        std::cout << "[DEBUG] Starting processFrame..." << std::endl;
        std::cout << "[DEBUG] Input frame size: " << frame.cols << "×" << frame.rows << std::endl;

        LocationResult result;
        result.success = false;
        result.matchedKeyFrameId = -1;
        result.numInliers = 0;
        result.totalMatches = 0;
        result.confidence = 0.0f;
        result.bowScore = 0.0f;

        // ⭐ CRITICAL: Resize frame to match map resolution
        cv::Mat processFrame;
        if (frame.size() != mProcessSize)
        {
            cv::resize(frame, processFrame, mProcessSize);
            std::cout << "[DEBUG] Resized to process size: " << mProcessSize << std::endl;
        }
        else
        {
            processFrame = frame.clone();
            std::cout << "[DEBUG] Frame already at process size" << std::endl;
        }

        // Extract features from the RESIZED frame
        std::cout << "[DEBUG] Extracting features..." << std::endl;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractFeatures(processFrame, keypoints, descriptors);

        if (keypoints.empty())
        {
            std::cout << "No features extracted from frame" << std::endl;
            return result;
        }

        std::cout << "Extracted " << keypoints.size() << " features" << std::endl;

        // Find candidate keyframes
        auto candidates = detectRelocalizationCandidates(descriptors);
        std::cout << "Found " << candidates.size() << " candidate keyframes" << std::endl;

        // Try to match with each candidate
        float bestScore = 0.0f;
        for (auto pKF : candidates)
        {
            std::cout << "[DEBUG] Checking keyframe " << pKF->mnId << std::endl;
            std::vector<cv::Point3f> points3D;
            std::vector<cv::Point2f> points2D;

            if (matchWithKeyFrame(keypoints, descriptors, pKF, points3D, points2D))
            {
                std::cout << "Matched " << points3D.size() << " points with KF "
                          << pKF->mnId << std::endl;

                // Calculate BoW score
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

                // Solve PnP
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

                        mCurrentPosition = result.position;

                        std::cout << "✓ Relocalization successful!" << std::endl;
                        std::cout << "  Position: [" << result.position.x << ", "
                                  << result.position.y << ", " << result.position.z << "]" << std::endl;
                        std::cout << "  Inliers: " << result.numInliers
                                  << " / " << result.totalMatches
                                  << " (" << (inlierRatio * 100) << "%)" << std::endl;
                        std::cout << "  BoW Score: " << bowScore << std::endl;
                        std::cout << "  Confidence: " << result.confidence << "%" << std::endl;
                    }
                }
            }
        }

        if (!result.success)
        {
            std::cout << "✗ Relocalization failed" << std::endl;
        }

        return result;
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

        // ⭐ Calculate text scaling based on display size
        float textScale = mDisplaySize.width / 640.0f;

        std::cout << "\n=== Processing video ===" << std::endl;
        std::cout << "Process resolution: " << mProcessSize << std::endl;
        std::cout << "Display resolution: " << mDisplaySize << std::endl;

        while (cap.read(frame))
        {
            frameCount++;
            if (frameCount % mFrameSkip != 0)
                continue;

            std::cout << "\n--- Frame " << frameCount << " ---" << std::endl;

            // ⭐ processFrame() handles resizing internally
            auto result = processFrame(frame);

            if (result.success)
            {
                successCount++;

                if (visualize)
                {
                    try
                    {
                        visualizeLocation(result);

                        // ⭐ Create display frame at display resolution
                        cv::Mat displayFrame;
                        cv::resize(frame, displayFrame, mDisplaySize);

                        // Status text - color coded by confidence
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

                        // ⭐ HALF THE SIZE - Scaled UI elements
                        int baseX = (int)(30 * textScale);
                        int baseY = (int)(40 * textScale);
                        float fontSize = 0.5f * textScale;                 // ⭐ Was 1.0f, now 0.5f
                        int thickness = std::max(1, (int)(1 * textScale)); // ⭐ Was 2, now 1

                        cv::putText(displayFrame, "LOCALIZED - " + status,
                                    cv::Point(baseX, baseY),
                                    cv::FONT_HERSHEY_SIMPLEX, fontSize, statusColor, thickness);

                        // Position
                        std::ostringstream posStream;
                        posStream << std::fixed << std::setprecision(2);
                        posStream << "Pos: [" << result.position.x << ", "
                                  << result.position.y << ", " << result.position.z << "]";
                        cv::putText(displayFrame, posStream.str(),
                                    cv::Point(baseX, (int)(80 * textScale)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.3f * textScale, // ⭐ Was 0.6f, now 0.3f
                                    cv::Scalar(255, 255, 255), thickness);

                        // Confidence bar
                        int barY = (int)(100 * textScale);
                        int barHeight = (int)(30 * textScale);
                        int barWidth = (int)(300 * textScale);

                        cv::rectangle(displayFrame,
                                      cv::Point(baseX, barY),
                                      cv::Point(baseX + (int)(result.confidence * barWidth / 100),
                                                barY + barHeight),
                                      statusColor, -1);
                        cv::rectangle(displayFrame,
                                      cv::Point(baseX, barY),
                                      cv::Point(baseX + barWidth, barY + barHeight),
                                      cv::Scalar(255, 255, 255), 2);

                        std::ostringstream confStream;
                        confStream << std::fixed << std::setprecision(1);
                        confStream << "Confidence: " << result.confidence << "%";
                        cv::putText(displayFrame, confStream.str(),
                                    cv::Point(baseX, barY + barHeight + (int)(25 * textScale)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.3f * textScale, // ⭐ Was 0.6f, now 0.3f
                                    statusColor, thickness);

                        // Inliers info
                        std::ostringstream inlierStream;
                        inlierStream << "Inliers: " << result.numInliers
                                     << "/" << result.totalMatches
                                     << " (" << (result.numInliers * 100 / result.totalMatches) << "%)";
                        cv::putText(displayFrame, inlierStream.str(),
                                    cv::Point(baseX, barY + barHeight + (int)(55 * textScale)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.3f * textScale, // ⭐ Was 0.6f, now 0.3f
                                    cv::Scalar(255, 255, 255), thickness);

                        // BoW score
                        std::ostringstream bowStream;
                        bowStream << std::fixed << std::setprecision(3);
                        bowStream << "BoW Score: " << result.bowScore;
                        cv::putText(displayFrame, bowStream.str(),
                                    cv::Point(baseX, barY + barHeight + (int)(85 * textScale)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.3f * textScale, // ⭐ Was 0.6f, now 0.3f
                                    cv::Scalar(255, 255, 255), thickness);

                        cv::imshow("Current Frame", displayFrame);

                        int key = cv::waitKey(1);
                        if (key == 27)
                        {
                            std::cout << "\nESC pressed - stopping visualization" << std::endl;
                            break;
                        }
                    }
                    catch (const cv::Exception &e)
                    {
                        std::cerr << "\n[ERROR] Display error: " << e.what() << std::endl;
                        std::cerr << "Disabling visualization and continuing..." << std::endl;
                        mVisualizationEnabled = false;
                        visualize = false;
                    }
                }
            }
        }

        std::cout << "\n=== Video processing complete ===" << std::endl;
        std::cout << "Success rate: " << (100.0 * successCount / (frameCount / mFrameSkip)) << "%" << std::endl;
    }

    void RelocalizationModule::visualizeLocation(const LocationResult &result)
    {
        try
        {
            // Create a 2D top-down view of the map
            cv::Mat mapViz(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));

            // Find map bounds for scaling
            float minX = 1e10, maxX = -1e10, minZ = 1e10, maxZ = -1e10;
            for (const auto &pt : mMapPointsViz)
            {
                minX = std::min(minX, pt.x);
                maxX = std::max(maxX, pt.x);
                minZ = std::min(minZ, pt.z);
                maxZ = std::max(maxZ, pt.z);
            }

            float rangeX = maxX - minX;
            float rangeZ = maxZ - minZ;
            float scale = std::min(700.0f / rangeX, 700.0f / rangeZ);

            // Draw map points
            for (const auto &pt : mMapPointsViz)
            {
                int x = 50 + (int)((pt.x - minX) * scale);
                int y = 750 - (int)((pt.z - minZ) * scale);
                cv::circle(mapViz, cv::Point(x, y), 1, cv::Scalar(100, 100, 100), -1);
            }

            // Draw current position
            int posX = 50 + (int)((mCurrentPosition.x - minX) * scale);
            int posY = 750 - (int)((mCurrentPosition.z - minZ) * scale);
            cv::circle(mapViz, cv::Point(posX, posY), 8, cv::Scalar(0, 0, 255), -1);
            cv::circle(mapViz, cv::Point(posX, posY), 15, cv::Scalar(0, 255, 255), 2);

            // Add text
            cv::putText(mapViz, "YOU ARE HERE", cv::Point(posX + 20, posY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            cv::putText(mapViz, "Map (Top-Down View)", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            cv::imshow("Relocalization - Map Visualization", mapViz);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "[WARNING] Could not display map visualization: " << e.what() << std::endl;
            mVisualizationEnabled = false;
        }
    }

    void RelocalizationModule::exportMapToPCD(const std::string &outputPath)
    {
        std::ofstream file(outputPath);
        if (!file.is_open())
        {
            std::cerr << "Cannot open file for writing: " << outputPath << std::endl;
            return;
        }

        // Write PCD header
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

        // Write points
        for (const auto &pt : mMapPointsViz)
        {
            file << pt.x << " " << pt.y << " " << pt.z << "\n";
        }

        file.close();
        std::cout << "Map exported to " << outputPath << std::endl;
    }

} // namespace Relocalization