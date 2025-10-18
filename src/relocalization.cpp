#include "relocalization.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unistd.h> // For getcwd
#include <random>

namespace Relocalization
{

    LSHMatcher::LSHMatcher(int numHashTables, int numHashBits)
        : mNumHashTables(numHashTables), mNumHashBits(numHashBits)
    {
        InitializeHashFunctions();
        mvHashTables.resize(mNumHashTables);
        std::cout << "[LSH] Initialized with " << numHashTables << " tables, "
                  << numHashBits << " bits per hash" << std::endl;
    }

    void LSHMatcher::InitializeHashFunctions()
    {
        mvHashFunctions.resize(mNumHashTables);
        std::mt19937 gen(12345);

        for (int i = 0; i < mNumHashTables; i++)
        {
            std::vector<int> bitIndices(256);
            for (int j = 0; j < 256; j++)
                bitIndices[j] = j;

            std::shuffle(bitIndices.begin(), bitIndices.end(), gen);
            bitIndices.resize(mNumHashBits);
            std::sort(bitIndices.begin(), bitIndices.end());

            mvHashFunctions[i] = bitIndices;
        }
    }

    size_t LSHMatcher::ComputeHash(const cv::Mat &descriptor, int tableIdx)
    {
        return ExtractBits(descriptor, mvHashFunctions[tableIdx]);
    }

    size_t LSHMatcher::ExtractBits(const cv::Mat &descriptor,
                                   const std::vector<int> &bitIndices)
    {
        size_t hash = 0;
        const unsigned char *desc = descriptor.ptr<unsigned char>(0);

        for (size_t i = 0; i < bitIndices.size(); i++)
        {
            int bitIdx = bitIndices[i];
            int byteIdx = bitIdx / 8;
            int bitOffset = bitIdx % 8;

            bool bit = (desc[byteIdx] >> bitOffset) & 1;
            if (bit)
                hash |= (1ULL << i);
        }

        return hash;
    }

    void LSHMatcher::BuildHashTables(const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints)
    {
        Clear();

        int validCount = 0;
        for (size_t i = 0; i < vpMapPoints.size(); i++)
        {
            ORB_SLAM3::MapPoint *pMP = vpMapPoints[i];
            if (!pMP || pMP->isBad())
                continue;

            cv::Mat descriptor = pMP->GetDescriptor();
            if (descriptor.empty())
                continue;

            for (int t = 0; t < mNumHashTables; t++)
            {
                size_t hash = ComputeHash(descriptor, t);
                mvHashTables[t][hash].push_back(i);
            }
            validCount++;
        }
    }

    int LSHMatcher::SearchByLSH(
        const cv::Mat &descriptors,
        const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints,
        std::vector<ORB_SLAM3::MapPoint *> &vpMatched,
        int th)
    {
        int nmatches = 0;
        vpMatched = std::vector<ORB_SLAM3::MapPoint *>(descriptors.rows, nullptr);

        for (int i = 0; i < descriptors.rows; i++)
        {
            cv::Mat desc = descriptors.row(i);

            std::unordered_map<int, int> candidateCounts;

            for (int t = 0; t < mNumHashTables; t++)
            {
                size_t hash = ComputeHash(desc, t);

                auto it = mvHashTables[t].find(hash);
                if (it != mvHashTables[t].end())
                {
                    for (int idx : it->second)
                    {
                        candidateCounts[idx]++;
                    }
                }
            }

            int bestDist = 256;
            int bestIdx = -1;

            for (auto &p : candidateCounts)
            {
                int idx = p.first;
                if (idx >= (int)vpMapPoints.size())
                    continue;

                ORB_SLAM3::MapPoint *pMP = vpMapPoints[idx];

                if (!pMP || pMP->isBad())
                    continue;

                cv::Mat mpDesc = pMP->GetDescriptor();
                int dist = HammingDistance(desc, mpDesc);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist < th && bestIdx >= 0)
            {
                vpMatched[i] = vpMapPoints[bestIdx];
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSHMatcher::HammingDistance(const cv::Mat &a, const cv::Mat &b)
    {
        int dist = 0;
        const unsigned char *pa = a.ptr<unsigned char>(0);
        const unsigned char *pb = b.ptr<unsigned char>(0);

        for (int i = 0; i < a.cols; i++)
        {
            unsigned char v = pa[i] ^ pb[i];
            v = v - ((v >> 1) & 0x55);
            v = (v & 0x33) + ((v >> 2) & 0x33);
            dist += ((v + (v >> 4) & 0xF) * 0x1);
        }

        return dist;
    }

    void LSHMatcher::Clear()
    {
        for (auto &table : mvHashTables)
            table.clear();
    }

    RelocalizationModule::RelocalizationModule(const std::string &vocabPath,
                                               const std::string &configPath)
        : mVocabPath(vocabPath), mConfigPath(configPath),
          mpVocabulary(nullptr), mpAtlas(nullptr), mpSLAM(nullptr),
          mpKeyFrameDB(nullptr),
          // State machine initialization
          mCurrentState(COMPLETELY_LOST),
          mStateAtomic(COMPLETELY_LOST),
          mHaveLastKnownPosition(false),
          mLostDuration(0.0),
          mLostTimeout(5.0),
          mStopThreads(false),
          mLSHFoundPosition(false),
          mVisOdomRunning(false),
          mSmoothingFactor(0.3f), // Moderate smoothing
          mSmoothingFrames(10),
          mFrameCount(0),
          // LSH defaults
          mVisibilityRadius(5.0f),
          mLSHNumTables(11),
          mLSHNumBits(14),
          mLSHHammingThreshold(50),
          mPROSACMaxIterations(100),
          mTimeoutReached(false)
    {
        // Load configuration
        if (!loadConfig())
        {
            std::cerr << "[ERROR] Failed to load configuration" << std::endl;
            return;
        }

        // Load vocabulary
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
        }

        // Initialize ORB extractor
        mpORBextractor = std::make_unique<ORB_SLAM3::ORBextractor>(1000, 1.2, 8, 20, 7);

        // Initialize LSH Matcher
        mpLSHMatcher = std::make_unique<LSHMatcher>(mLSHNumTables, mLSHNumBits);
        std::cout << "[INFO] LSH matcher initialized for local relocalization" << std::endl;

        std::cout << "\n========================================" << std::endl;
        std::cout << " Hybrid Relocalization System" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "State: COMPLETELY_LOST (will use DBoW2 first)" << std::endl;
        std::cout << "LSH parameters:" << std::endl;
        std::cout << "  - Hash tables: " << mLSHNumTables << std::endl;
        std::cout << "  - Bits per hash: " << mLSHNumBits << std::endl;
        std::cout << "  - Visibility radius: " << mVisibilityRadius << "m" << std::endl;
        std::cout << "  - Lost timeout: " << mLostTimeout << "s" << std::endl;
        std::cout << "========================================\n"
                  << std::endl;
    }

    RelocalizationModule::~RelocalizationModule()
    {
        // Stop parallel threads if running
        stopParallelTracking();

        // Clean up database before vocabulary
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

        // Map path (ORB-SLAM3 format)
        fs["System.LoadAtlasFromFile"] >> mMapPath;
        if (mMapPath.empty())
        {
            std::cerr << "System.LoadAtlasFromFile not found in config" << std::endl;
            return false;
        }

        // Camera parameters (ORB-SLAM3 format uses Camera1.*)
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

        // Distortion coefficients (OpenCV format: k1, k2, p1, p2, k3)
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

        // Relocalization settings (with defaults)
        mFrameSkip = fs["Relocalization.FrameSkip"].empty() ? 5 : (int)fs["Relocalization.FrameSkip"];
        mMinInliers = fs["Relocalization.MinInliers"].empty() ? 1 : (int)fs["Relocalization.MinInliers"];
        mMinMatches = fs["Relocalization.MinMatches"].empty() ? 10 : (int)fs["Relocalization.MinMatches"];
        mBowThreshold = fs["Relocalization.BowSimilarityThreshold"].empty() ? 0.05f : (float)fs["Relocalization.BowSimilarityThreshold"];
        mMaxCandidates = fs["Relocalization.MaxCandidates"].empty() ? 5 : (int)fs["Relocalization.MaxCandidates"];
        mLostTimeout = fs["Relocalization.LostTimeout"].empty() ? 5 : (int)fs["Relocalization.LostTimeout"];
        // Visualization settings (with defaults)
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

        if (!success || inliersMask.empty())
        {
            std::cout << "Failed solve PnP with RANSAC" << std::endl;
            return false;
        }

        // Count inliers
        inliers.clear();
        if (inliersMask.rows == (int)points3D.size())
        {
            // Dense mask: each row is 0 or 1
            for (int i = 0; i < inliersMask.rows; i++)
            {
                if (inliersMask.at<uchar>(i, 0) != 0)
                {
                    inliers.push_back(i);
                }
            }
        }
        else
        {
            // Sparse mask: contains only inlier indices
            for (int i = 0; i < inliersMask.rows; i++)
            {
                inliers.push_back(inliersMask.at<int>(i, 0));
            }
        }

        int numInliers = inliers.size();

        std::cout << "[DEBUG] Check Inliers size: " << inliers.size() << std::endl;

        return numInliers >= (size_t)mMinInliers; // Use config value
    }

    bool RelocalizationModule::solvePnPWithPROSAC(
        const std::vector<cv::Point3f> &points3D,
        const std::vector<cv::Point2f> &points2D,
        cv::Mat &rvec, cv::Mat &tvec,
        std::vector<int> &inliers)
    {
        if (points3D.size() < 4)
        {
            std::cout << "[PROSAC] Not enough points for PnP" << std::endl;
            return false;
        }

        auto t1 = std::chrono::steady_clock::now();

        cv::Mat inliersMat;

        // Use P3P as mentioned in paper, with RANSAC
        // OpenCV doesn't have PROSAC, but RANSAC with quality-sorted data is similar
        bool success = cv::solvePnPRansac(
            points3D,
            points2D,
            mK,
            mDistCoef,
            rvec,
            tvec,
            false,                // useExtrinsicGuess
            mPROSACMaxIterations, // Paper uses 100 iterations
            8.0f,                 // reprojectionError
            0.99,                 // confidence
            inliersMat,
            cv::SOLVEPNP_P3P // Use P3P as mentioned in paper
        );

        auto t2 = std::chrono::steady_clock::now();
        double tPROSAC = std::chrono::duration_cast<std::chrono::duration<double,
                                                                          std::milli>>(t2 - t1)
                             .count();

        if (success && !inliersMat.empty())
        {
            inliers.clear();
            inliers.reserve(inliersMat.rows);

            for (int i = 0; i < inliersMat.rows; i++)
            {
                inliers.push_back(inliersMat.at<int>(i, 0));
            }

            std::cout << "[PROSAC] Found " << inliers.size() << " inliers in "
                      << tPROSAC << " ms" << std::endl;

            return inliers.size() >= (size_t)mMinInliers;
        }

        std::cout << "[PROSAC] Failed to find solution" << std::endl;
        return false;
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

    void RelocalizationModule::processVideo(const std::string &videoPath, bool visualize)
    {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened())
        {
            std::cerr << "Cannot open video: " << videoPath << std::endl;
            return;
        }

        // Override visualize parameter with config setting
        visualize = mVisualizationEnabled;

        if (visualize)
        {
            cv::namedWindow("Current Frame", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Relocalization - Map Visualization", cv::WINDOW_AUTOSIZE);
        }

        cv::Mat frame;
        int frameCount = 0;
        int successCount = 0;

        std::cout << "\n=== Processing video ===" << std::endl;
        std::cout << "Frame skip: every " << mFrameSkip << " frames" << std::endl;

        while (cap.read(frame))
        {
            frameCount++;

            if (frameCount % mFrameSkip != 0)
            {
                // Still need to handle window events even when skipping
                if (visualize)
                {
                    char key = cv::waitKey(1);
                    if (key == 27) // ESC to quit
                        break;
                }
                continue;
            }

            std::cout << "\n--- Frame " << frameCount << " ---" << std::endl;

            auto result = processFrame(frame);

            if (visualize)
            {
                // Check if windows are still valid
                try
                {
                    visualizeLocation(result);

                    cv::Mat display = frame.clone();

                    if (result.success)
                    {
                        cv::putText(display, "LOCALIZED", cv::Point(30, 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                        successCount++;
                    }
                    else
                    {
                        cv::putText(display, "TRACKING LOST", cv::Point(30, 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    }

                    std::string posText = "Pos: [" +
                                          std::to_string(result.position.x) + ", " +
                                          std::to_string(result.position.y) + ", " +
                                          std::to_string(result.position.z) + "]";
                    cv::putText(display, posText, cv::Point(30, 70),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                    cv::imshow("Current Frame", display);

                    char key = cv::waitKey(10);
                    if (key == 27) // ESC
                    {
                        std::cout << "\nESC pressed, stopping..." << std::endl;
                        break;
                    }
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "\n[WARNING] OpenCV GUI error: " << e.what() << std::endl;
                    std::cerr << "Continuing without visualization..." << std::endl;
                    visualize = false; // Disable visualization to prevent further errors
                }
            }
        }

        if (visualize)
        {
            cv::destroyAllWindows();
            cv::waitKey(1); // Let events process
        }

        std::cout << "\n=== Video processing complete ===" << std::endl;
        std::cout << "Total frames processed: " << frameCount / mFrameSkip << std::endl;
        std::cout << "Successful localizations: " << successCount << std::endl;
    }

    void RelocalizationModule::visualizeLocation(const LocationResult &result)
    {
        if (!result.success || mMapPointsViz.empty())
        {
            return;
        }

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
        cv::waitKey(1);
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

    // ========== State Management ==========

    void RelocalizationModule::setState(TrackingState newState)
    {
        std::lock_guard<std::mutex> lock(mStateMutex);

        TrackingState oldState = mCurrentState;

        if (oldState != newState)
        {
            std::cout << "\n[STATE] Transition: ";

            switch (oldState)
            {
            case COMPLETELY_LOST:
                std::cout << "COMPLETELY_LOST";
                break;
            case FOUND_POSITION:
                std::cout << "FOUND_POSITION";
                break;
            case CURRENTLY_LOST:
                std::cout << "CURRENTLY_LOST";
                break;
            }

            std::cout << " → ";

            switch (newState)
            {
            case COMPLETELY_LOST:
                std::cout << "COMPLETELY_LOST";
                break;
            case FOUND_POSITION:
                std::cout << "FOUND_POSITION";
                break;
            case CURRENTLY_LOST:
                std::cout << "CURRENTLY_LOST";
                break;
            }

            std::cout << std::endl;

            mCurrentState = newState;
            mStateAtomic.store(newState);

            handleStateTransition(oldState, newState);
        }
    }

    TrackingState RelocalizationModule::getState()
    {
        return mStateAtomic.load();
    }

    void RelocalizationModule::handleStateTransition(TrackingState oldState, TrackingState newState)
    {
        // Entering CURRENTLY_LOST - start parallel threads
        if (newState == CURRENTLY_LOST && oldState != CURRENTLY_LOST)
        {
            std::cout << "[STATE] Starting parallel VisOdom + LSH threads" << std::endl;
            mTimeWhenLost = std::chrono::steady_clock::now();
            mLostDuration = 0.0;
            startParallelTracking();
        }

        // Leaving CURRENTLY_LOST - stop parallel threads
        if (oldState == CURRENTLY_LOST && newState != CURRENTLY_LOST)
        {
            std::cout << "[STATE] Stopping parallel threads" << std::endl;
            stopParallelTracking();
        }

        // Entering FOUND_POSITION - update last known position
        if (newState == FOUND_POSITION)
        {
            mHaveLastKnownPosition = true;
            mLastKnownPosition = mCurrentPosition;
            std::cout << "[STATE] Position found: [" << mCurrentPosition.x << ", "
                      << mCurrentPosition.y << ", " << mCurrentPosition.z << "]" << std::endl;
        }
    }

    // ========== Parallel Thread Management ==========

    void RelocalizationModule::startParallelTracking()
    {
        // Stop any existing threads first
        stopParallelTracking();

        // Reset flags
        mStopThreads.store(false);
        mLSHFoundPosition.store(false);
        mVisOdomRunning.store(true);

        // Start Visual Odometry thread
        mpVisOdomThread = std::make_unique<std::thread>(
            &RelocalizationModule::runVisualOdometryThread, this);

        // Start LSH thread
        mpLSHThread = std::make_unique<std::thread>(
            &RelocalizationModule::runLSHThread, this);

        std::cout << "[PARALLEL] Both threads started" << std::endl;
    }

    void RelocalizationModule::stopParallelTracking()
    {
        mStopThreads.store(true);

        // Wait for threads to finish
        if (mpVisOdomThread && mpVisOdomThread->joinable())
        {
            mpVisOdomThread->join();
            mpVisOdomThread.reset();
        }

        if (mpLSHThread && mpLSHThread->joinable())
        {
            mpLSHThread->join();
            mpLSHThread.reset();
        }

        mVisOdomRunning.store(false);
    }

    // ========== Visual Odometry Thread ==========

    void RelocalizationModule::runVisualOdometryThread()
    {
        std::cout << "[VISODOM] Thread started" << std::endl;

        while (mCurrentFrame.empty() && !mStopThreads.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (mStopThreads.load())
        {
            std::cout << "[VISODOM] Stopped before initialization" << std::endl;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mPositionMutex);
            std::cout << "[VISODOM TEST] First frame: "
                      << mCurrentFrame.cols << "x" << mCurrentFrame.rows
                      << ", channels=" << mCurrentFrame.channels()
                      << ", type=" << mCurrentFrame.type()
                      << " (CV_8UC1=" << CV_8UC1 << ", CV_8UC3=" << CV_8UC3 << ")" << std::endl;
        }

        cv::Mat prevFrameGray;
        std::vector<cv::Point2f> prevPoints;
        int frameCounter = 0;

        while (!mStopThreads.load() && getState() == CURRENTLY_LOST)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS

            cv::Mat currentFrame;
            {
                std::lock_guard<std::mutex> lock(mPositionMutex);
                if (mCurrentFrame.empty())
                    continue;
                currentFrame = mCurrentFrame.clone();
            }

            // ✅ CRITICAL: Convert to grayscale
            cv::Mat currentFrameGray;
            try
            {
                if (currentFrame.channels() == 3)
                {
                    cv::cvtColor(currentFrame, currentFrameGray, cv::COLOR_BGR2GRAY);
                }
                else if (currentFrame.channels() == 1)
                {
                    currentFrameGray = currentFrame.clone();
                }
                else
                {
                    std::cerr << "[VISODOM ERROR] Unexpected frame channels: "
                              << currentFrame.channels() << std::endl;
                    continue;
                }

                // Verify the grayscale conversion worked
                if (currentFrameGray.type() != CV_8UC1)
                {
                    std::cerr << "[VISODOM ERROR] Frame is not CV_8UC1, type="
                              << currentFrameGray.type() << std::endl;
                    continue;
                }
            }
            catch (const cv::Exception &e)
            {
                std::cerr << "[VISODOM ERROR] cvtColor failed: " << e.what() << std::endl;
                continue;
            }

            if (prevFrameGray.empty())
            {
                // First frame - initialize
                prevFrameGray = currentFrameGray.clone();

                try
                {
                    // Detect features for tracking (now using grayscale!)
                    cv::goodFeaturesToTrack(currentFrameGray, prevPoints, 200, 0.01, 10);

                    std::cout << "[VISODOM] Initialized with " << prevPoints.size()
                              << " points (frame type: " << currentFrameGray.type() << ")" << std::endl;
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "[VISODOM ERROR] goodFeaturesToTrack failed: "
                              << e.what() << std::endl;
                    prevFrameGray.release();
                    continue;
                }

                if (prevPoints.empty())
                {
                    std::cerr << "[VISODOM ERROR] No features detected!" << std::endl;
                    prevFrameGray.release();
                }
                continue;
            }

            // Track features from previous frame to current
            std::vector<cv::Point2f> currPoints;
            std::vector<uchar> status;
            std::vector<float> err;

            try
            {
                cv::calcOpticalFlowPyrLK(
                    prevFrameGray, currentFrameGray, // Both grayscale!
                    prevPoints, currPoints,
                    status, err);
            }
            catch (const cv::Exception &e)
            {
                std::cerr << "[VISODOM ERROR] Optical flow failed: " << e.what() << std::endl;
                prevPoints.clear();
                prevFrameGray.release();
                continue;
            }

            // Filter good matches
            std::vector<cv::Point2f> goodPrev, goodCurr;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i] && i < prevPoints.size() && i < currPoints.size())
                {
                    goodPrev.push_back(prevPoints[i]);
                    goodCurr.push_back(currPoints[i]);
                }
            }

            if (goodPrev.size() < 10)
            {
                std::cout << "[VISODOM] Too few tracked points (" << goodPrev.size()
                          << "), re-detecting..." << std::endl;

                try
                {
                    prevPoints.clear();
                    cv::goodFeaturesToTrack(currentFrameGray, prevPoints, 200, 0.01, 10);
                    prevFrameGray = currentFrameGray.clone();
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "[VISODOM ERROR] Feature re-detection failed: "
                              << e.what() << std::endl;
                    prevFrameGray.release();
                }
                continue;
            }

            // Estimate motion (simple 2D translation)
            cv::Point2f motion(0, 0);
            for (size_t i = 0; i < goodPrev.size(); i++)
            {
                motion.x += (goodCurr[i].x - goodPrev[i].x);
                motion.y += (goodCurr[i].y - goodPrev[i].y);
            }
            motion.x /= goodPrev.size();
            motion.y /= goodPrev.size();

            // Convert pixel motion to world motion (rough estimate)
            float scale = 0.001f;
            float dx = motion.x * scale;
            float dz = motion.y * scale;

            // Update estimated position
            cv::Point3f estimatedPos;
            {
                std::lock_guard<std::mutex> lock(mPositionMutex);
                estimatedPos = mVisOdomPosition;
                estimatedPos.x += dx;
                estimatedPos.z += dz;
                mVisOdomPosition = estimatedPos;
            }

            if (frameCounter % 10 == 0)
            {
                std::cout << "[VISODOM] Estimate: [" << estimatedPos.x << ", "
                          << estimatedPos.y << ", " << estimatedPos.z << "]"
                          << " (tracking " << goodPrev.size() << " points)" << std::endl;
            }

            // Prepare for next iteration
            prevPoints = currPoints;
            prevFrameGray = currentFrameGray.clone();
            frameCounter++;

            // Check timeout
            if (isTimeout())
            {
                std::cout << "[VISODOM] Timeout reached!" << std::endl;
                mTimeoutReached.store(true); // ← Set flag instead of setState()
                break;                       // Exit thread
            }
        }

        std::cout << "[VISODOM] Thread stopped" << std::endl;
    }

    cv::Point3f RelocalizationModule::estimatePositionFromTracking(const cv::Mat &frame)
    {
        // Use ORB-SLAM3's tracking to estimate position
        // This is a simplified version - in reality, you'd use SLAM's motion model

        // For now, apply simple dead reckoning from last known position
        // In real implementation, this would use ORB-SLAM3's TrackWithMotionModel

        cv::Point3f estimatedPos = mLastKnownPosition;

        // TODO: Integrate with ORB-SLAM3's actual tracking
        // For now, just use last known position with small drift simulation
        double timeLost = getTimeSinceLost();

        // Assume slow drift (0.1 m/s uncertainty)
        float driftX = (rand() % 100 - 50) / 1000.0f * timeLost;
        float driftZ = (rand() % 100 - 50) / 1000.0f * timeLost;

        estimatedPos.x += driftX;
        estimatedPos.z += driftZ;

        return estimatedPos;
    }

    // ========== LSH Thread ==========

    void RelocalizationModule::runLSHThread()
    {
        std::cout << "[LSH] Thread started, searching every 5 frames" << std::endl;

        int attemptCount = 0;
        int frameCounter = 0;

        while (!mStopThreads.load() && getState() == CURRENTLY_LOST)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            frameCounter++;

            if (frameCounter % 5 != 0)
                continue;

            cv::Mat currentFrame;
            {
                std::lock_guard<std::mutex> lock(mPositionMutex);
                if (mCurrentFrame.empty())
                    continue;
                currentFrame = mCurrentFrame.clone();
            }

            if (!currentFrame.empty())
            {
                attemptCount++;
                std::cout << "\n[LSH] Attempt #" << attemptCount << std::endl;

                try
                {
                    LocationResult result = localRelocalizationLSH(currentFrame);

                    if (result.success)
                    {
                        std::cout << "[LSH] ✓ Position found!" << std::endl;

                        {
                            std::lock_guard<std::mutex> lock(mPositionMutex);
                            mLSHPosition = result.position;
                        }

                        mLSHFoundPosition.store(true);

                        cv::Point3f smoothedPos = smoothPositionTransition(
                            mVisOdomPosition,
                            result.position,
                            mSmoothingFactor);

                        mCurrentPosition = smoothedPos;

                        setState(FOUND_POSITION);
                        break;
                    }
                    else
                    {
                        std::cout << "[LSH] ✗ No match found" << std::endl;
                    }
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "[LSH ERROR] Exception during relocalization: "
                              << e.what() << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[LSH ERROR] Standard exception: " << e.what() << std::endl;
                }
            }

            if (isTimeout())
            {
                std::cout << "[LSH] Timeout reached! Giving up..." << std::endl;
                mTimeoutReached.store(true); // ← Set flag instead of setState()
                break;                       // Exit thread
            }
        }

        std::cout << "[LSH] Thread stopped after " << attemptCount << " attempts" << std::endl;
    }

    // ========== Position Smoothing ==========

    cv::Point3f RelocalizationModule::smoothPositionTransition(
        const cv::Point3f &from,
        const cv::Point3f &to,
        float alpha)
    {
        // Linear interpolation (LERP)
        // alpha = 0.0: instant jump to 'to'
        // alpha = 1.0: stay at 'from'
        // alpha = 0.3: smooth blend (30% old, 70% new)

        cv::Point3f smoothed;
        smoothed.x = from.x * alpha + to.x * (1.0f - alpha);
        smoothed.y = from.y * alpha + to.y * (1.0f - alpha);
        smoothed.z = from.z * alpha + to.z * (1.0f - alpha);

        float distance = cv::norm(to - from);

        std::cout << "[SMOOTH] Correcting position:" << std::endl;
        std::cout << "  VisOdom: [" << from.x << ", " << from.y << ", " << from.z << "]" << std::endl;
        std::cout << "  LSH:     [" << to.x << ", " << to.y << ", " << to.z << "]" << std::endl;
        std::cout << "  Smooth:  [" << smoothed.x << ", " << smoothed.y << ", " << smoothed.z << "]" << std::endl;
        std::cout << "  Distance corrected: " << distance << " m" << std::endl;

        return smoothed;
    }

    // ========== Utility Methods ==========

    double RelocalizationModule::getTimeSinceLost()
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(
                   now - mTimeWhenLost)
            .count();
    }

    bool RelocalizationModule::isTimeout()
    {
        mLostDuration = getTimeSinceLost();
        return mLostDuration >= mLostTimeout;
    }

    LocationResult RelocalizationModule::processFrame(const cv::Mat &frame)
    {
        mFrameCount++;

        // Store current frame for threads to access
        {
            std::lock_guard<std::mutex> lock(mPositionMutex);
            mCurrentFrame = frame.clone();

            // Debug: verify frame type
            std::cout << "[DEBUG] Stored frame: " << frame.cols << "x" << frame.rows
                      << ", channels=" << frame.channels()
                      << ", type=" << frame.type() << std::endl;
        }

        LocationResult result;
        result.success = false;

        TrackingState currentState = getState();

        std::cout << "\n========== Frame " << mFrameCount << " ==========" << std::endl;
        std::cout << "[STATE] Current: ";
        switch (currentState)
        {
        case COMPLETELY_LOST:
            std::cout << "COMPLETELY_LOST";
            break;
        case FOUND_POSITION:
            std::cout << "FOUND_POSITION";
            break;
        case CURRENTLY_LOST:
            std::cout << "CURRENTLY_LOST";
            break;
        }
        std::cout << std::endl;

        // ========== STATE MACHINE ==========

        switch (currentState)
        {
        case COMPLETELY_LOST:
        {
            // ===== Use DBoW2 for global relocalization =====
            std::cout << "[METHOD] DBoW2 global relocalization" << std::endl;
            result = globalRelocalizationDBoW2(frame);

            if (result.success)
            {
                // Validate DBoW2 result (same validation as LSH!)
                if (result.numInliers < mMinInliers)
                {
                    std::cout << "[REJECT] DBoW2: Too few inliers: "
                              << result.numInliers << std::endl;
                    break;
                }

                // Accept position
                mCurrentPosition = result.position;
                mLastKnownPosition = result.position;
                mHaveLastKnownPosition = true;

                // Initialize VisOdom position
                mVisOdomPosition = result.position;

                std::cout << "[SUCCESS] Initial position found!" << std::endl;
                std::cout << "  Position: [" << result.position.x << ", "
                          << result.position.y << ", " << result.position.z << "]" << std::endl;
                std::cout << "  Inliers: " << result.numInliers << std::endl;

                setState(FOUND_POSITION);
            }
            else
            {
                std::cout << "[FAILED] DBoW2 global relocalization failed" << std::endl;
            }
            break;
        }

        case FOUND_POSITION:
        {
            // ===== Continuous tracking with LSH =====
            std::cout << "[METHOD] LSH local relocalization (visibility-constrained)" << std::endl;

            // Only relocalize every 5 frames
            if (mFrameCount % 5 != 0)
            {
                std::cout << "[SKIP] Maintaining current position" << std::endl;
                result.success = true;
                result.position = mCurrentPosition;
                break;
            }

            // Try LSH relocalization with local search
            LocationResult lshResult = localRelocalizationLSH(frame);

            // ===== VALIDATE LSH RESULT =====

            if (!lshResult.success)
            {
                std::cout << "[WARNING] LSH failed - entering CURRENTLY_LOST" << std::endl;
                setState(CURRENTLY_LOST);
                result.success = false;
                result.position = mCurrentPosition;
                break;
            }

            // Check minimum inliers
            if (lshResult.numInliers < 15)
            {
                std::cout << "[REJECT] Too few inliers: " << lshResult.numInliers << std::endl;
                setState(CURRENTLY_LOST);
                result.success = false;
                result.position = mCurrentPosition;
                break;
            }

            // Check maximum movement
            float dx = lshResult.position.x - mCurrentPosition.x;
            float dy = lshResult.position.y - mCurrentPosition.y;
            float dz = lshResult.position.z - mCurrentPosition.z;
            float distance = sqrt(dx * dx + dy * dy + dz * dz);

            float maxMovement = 0.5f; // 0.5m max between frames

            if (distance > maxMovement)
            {
                std::cout << "[REJECT] Position jump too large: " << distance << " m" << std::endl;
                std::cout << "  (Max allowed: " << maxMovement << " m)" << std::endl;
                setState(CURRENTLY_LOST);
                result.success = false;
                result.position = mCurrentPosition;
                break;
            }

            // ===== ACCEPT: Smooth Update =====
            cv::Point3f smoothPos = smoothPositionTransition(
                mCurrentPosition,
                lshResult.position,
                0.3f // 30% old, 70% new
            );

            mCurrentPosition = smoothPos;
            mLastKnownPosition = smoothPos;
            mVisOdomPosition = smoothPos; // Keep VisOdom synced

            result.success = true;
            result.position = smoothPos;
            result.numInliers = lshResult.numInliers;

            std::cout << "[ACCEPT] ✓ Position updated" << std::endl;
            std::cout << "  Movement: " << distance << " m" << std::endl;
            std::cout << "  Inliers: " << lshResult.numInliers << std::endl;
            std::cout << "  New pos: [" << smoothPos.x << ", "
                      << smoothPos.y << ", " << smoothPos.z << "]" << std::endl;

            break;
        }

        case CURRENTLY_LOST:
        {
            // ===== Parallel threads are running =====
            std::cout << "[METHOD] Parallel VisOdom + LSH recovery" << std::endl;
            std::cout << "[INFO] Time lost: " << getTimeSinceLost() << "s / "
                      << mLostTimeout << "s" << std::endl;

            if (mTimeoutReached.load())
            {
                std::cout << "[TIMEOUT] Parallel threads timed out, returning to COMPLETELY_LOST" << std::endl;
                mTimeoutReached.store(false); // Reset flag
                setState(COMPLETELY_LOST);    // Now safe - called from main thread!
                result.success = false;
                result.position = mCurrentPosition;
                break;
            }

            // Check if LSH thread found position
            if (mLSHFoundPosition.load())
            {
                std::cout << "[CURRENTLY_LOST] ✓ LSH found position!" << std::endl;

                // Validate before accepting (same checks as FOUND_POSITION)
                float dx = mLSHPosition.x - mVisOdomPosition.x;
                float dy = mLSHPosition.y - mVisOdomPosition.y;
                float dz = mLSHPosition.z - mVisOdomPosition.z;
                float distance = sqrt(dx * dx + dy * dy + dz * dz);

                if (distance > 2.0f) // More lenient for recovery
                {
                    std::cout << "[WARNING] LSH position differs from VisOdom by "
                              << distance << " m" << std::endl;
                }

                result.success = true;
                result.position = mCurrentPosition; // Already updated by thread
                // State transition already handled by LSH thread
            }
            else
            {
                std::cout << "[CURRENTLY_LOST] Still searching..." << std::endl;

                // Return VisOdom estimate
                {
                    std::lock_guard<std::mutex> lock(mPositionMutex);
                    result.position = mVisOdomPosition;
                }
                result.success = false;
            }

            break;
        }
        }

        return result;
    }

    // ========== DBoW2 Global Relocalization ==========

    LocationResult RelocalizationModule::globalRelocalizationDBoW2(const cv::Mat &frame)
    {
        auto tStart = std::chrono::steady_clock::now();

        LocationResult result;
        result.success = false;

        // Extract features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractFeatures(frame, keypoints, descriptors);

        if (keypoints.empty())
        {
            std::cout << "[DBOW2] No features extracted" << std::endl;
            return result;
        }

        std::cout << "[DBOW2] Extracted " << keypoints.size() << " features" << std::endl;

        // Find candidate keyframes using DBoW2
        auto candidates = detectRelocalizationCandidates(descriptors);

        if (candidates.empty())
        {
            std::cout << "[DBOW2] No candidates found" << std::endl;
            return result;
        }

        std::cout << "[DBOW2] Found " << candidates.size() << " candidate keyframes" << std::endl;

        // Try to match with each candidate
        for (auto pKF : candidates)
        {
            std::vector<cv::Point3f> points3D;
            std::vector<cv::Point2f> points2D;

            if (matchWithKeyFrame(keypoints, descriptors, pKF, points3D, points2D))
            {
                // Solve PnP
                cv::Mat rvec, tvec;
                std::vector<int> inliers;

                if (solvePnP(points3D, points2D, rvec, tvec, inliers))
                {
                    result.success = true;
                    result.position = computePosition(rvec, tvec);
                    result.matchedKeyFrameId = pKF->mnId;
                    result.numInliers = inliers.size();

                    auto tEnd = std::chrono::steady_clock::now();
                    double tTotal = std::chrono::duration_cast<std::chrono::duration<double,
                                                                                     std::milli>>(tEnd - tStart)
                                        .count();

                    std::cout << "[DBOW2] ✓✓✓ SUCCESS!" << std::endl;
                    std::cout << "  Position: [" << result.position.x << ", "
                              << result.position.y << ", " << result.position.z << "]" << std::endl;
                    std::cout << "  Matched KF: " << result.matchedKeyFrameId << std::endl;
                    std::cout << "  Inliers: " << result.numInliers << std::endl;
                    std::cout << "  Time: " << tTotal << " ms" << std::endl;

                    return result;
                }
            }
        }

        auto tEnd = std::chrono::steady_clock::now();
        double tTotal = std::chrono::duration_cast<std::chrono::duration<double,
                                                                         std::milli>>(tEnd - tStart)
                            .count();

        std::cout << "[DBOW2] Failed after " << tTotal << " ms" << std::endl;
        return result;
    }

    // ========== LSH Local Relocalization ==========

    LocationResult RelocalizationModule::localRelocalizationLSH(const cv::Mat &frame)
    {
        auto tStart = std::chrono::steady_clock::now();

        LocationResult result;
        result.success = false;

        // Extract features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractFeatures(frame, keypoints, descriptors);

        if (keypoints.empty())
        {
            return result;
        }

        // Get candidate map points within visibility radius
        auto vpCandidates = getCandidateMapPointsLSH();

        if (vpCandidates.empty())
        {
            std::cout << "[LSH] No candidates in radius" << std::endl;
            return result;
        }

        // Match using LSH
        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;

        if (matchWithLSH(keypoints, descriptors, vpCandidates, points3D, points2D))
        {
            // Solve PnP with PROSAC
            cv::Mat rvec, tvec;
            std::vector<int> inliers;

            if (solvePnPWithPROSAC(points3D, points2D, rvec, tvec, inliers))
            {
                result.success = true;
                result.position = computePosition(rvec, tvec);
                result.numInliers = inliers.size();

                auto tEnd = std::chrono::steady_clock::now();
                double tTotal = std::chrono::duration_cast<std::chrono::duration<double,
                                                                                 std::milli>>(tEnd - tStart)
                                    .count();

                std::cout << "[LSH] ✓ Success in " << tTotal << " ms" << std::endl;
                std::cout << "  Inliers: " << result.numInliers << std::endl;

                return result;
            }
        }

        return result;
    }

    // ========== Get Candidate Map Points for LSH ==========

    std::vector<ORB_SLAM3::MapPoint *> RelocalizationModule::getCandidateMapPointsLSH()
    {
        std::vector<ORB_SLAM3::MapPoint *> vpCandidates;

        if (!mHaveLastKnownPosition)
        {
            std::cout << "[LSH] No last known position, using full map" << std::endl;
            return mvMapPoints;
        }

        // Use visual odometry position as search center
        cv::Point3f searchCenter;
        {
            std::lock_guard<std::mutex> lock(mPositionMutex);
            searchCenter = mVisOdomPosition;
        }

        // Get keyframes within radius
        for (auto pKF : mvKeyFrames)
        {
            if (pKF->isBad())
                continue;

            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Vector3f kfPos = Twc.translation();

            float dist = sqrt(
                pow(kfPos(0) - searchCenter.x, 2) +
                pow(kfPos(1) - searchCenter.y, 2) +
                pow(kfPos(2) - searchCenter.z, 2));

            if (dist < mVisibilityRadius)
            {
                const std::vector<ORB_SLAM3::MapPoint *> vpMPs = pKF->GetMapPointMatches();
                for (auto pMP : vpMPs)
                {
                    if (pMP && !pMP->isBad())
                        vpCandidates.push_back(pMP);
                }
            }
        }

        // Remove duplicates
        std::sort(vpCandidates.begin(), vpCandidates.end());
        vpCandidates.erase(std::unique(vpCandidates.begin(), vpCandidates.end()),
                           vpCandidates.end());

        std::cout << "[LSH] Selected " << vpCandidates.size()
                  << " candidates within " << mVisibilityRadius << "m" << std::endl;

        return vpCandidates;
    }

    // ========== Match with LSH ==========

    bool RelocalizationModule::matchWithLSH(
        const std::vector<cv::KeyPoint> &keypoints,
        const cv::Mat &descriptors,
        const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints,
        std::vector<cv::Point3f> &points3D,
        std::vector<cv::Point2f> &points2D)
    {
        points3D.clear();
        points2D.clear();

        if (vpMapPoints.empty())
            return false;

        // Build hash tables
        mpLSHMatcher->BuildHashTables(vpMapPoints);

        // Perform LSH matching
        std::vector<ORB_SLAM3::MapPoint *> vpMatched;
        int nmatches = mpLSHMatcher->SearchByLSH(
            descriptors,
            vpMapPoints,
            vpMatched,
            mLSHHammingThreshold);

        if (nmatches < mMinMatches)
            return false;

        // Extract 2D-3D correspondences
        for (size_t i = 0; i < vpMatched.size(); i++)
        {
            if (vpMatched[i])
            {
                ORB_SLAM3::MapPoint *pMP = vpMatched[i];
                if (!pMP->isBad())
                {
                    Eigen::Vector3f pos = pMP->GetWorldPos();
                    points3D.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
                    points2D.push_back(keypoints[i].pt);
                }
            }
        }

        return points3D.size() >= (size_t)mMinMatches;
    }

} // namespace Relocalization