#ifndef RELOCALIZATION_H
#define RELOCALIZATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include "System.h"
#include "Atlas.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"

namespace Relocalization
{

    enum TrackingState
    {
        COMPLETELY_LOST = 0, // No position known - use DBoW2
        FOUND_POSITION = 1,  // Tracking OK - normal operation
        CURRENTLY_LOST = 2   // Recently lost - use parallel VisOdom + LSH
    };

    struct LocationResult
    {
        bool success;
        cv::Point3f position;
        int matchedKeyFrameId;
        int numInliers;
        std::vector<cv::DMatch> matches;
    };

    class LSHMatcher
    {
    public:
        LSHMatcher(int numHashTables = 11, int numHashBits = 14);

        void BuildHashTables(const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints);

        int SearchByLSH(
            const cv::Mat &descriptors,
            const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints,
            std::vector<ORB_SLAM3::MapPoint *> &vpMatched,
            int th = 50);

        void Clear();

    private:
        int mNumHashTables;
        int mNumHashBits;
        std::vector<std::unordered_map<size_t, std::vector<int>>> mvHashTables;
        std::vector<std::vector<int>> mvHashFunctions;

        void InitializeHashFunctions();
        size_t ComputeHash(const cv::Mat &descriptor, int tableIdx);
        size_t ExtractBits(const cv::Mat &descriptor, const std::vector<int> &bitIndices);
        static int HammingDistance(const cv::Mat &a, const cv::Mat &b);
    };

    class RelocalizationModule
    {
    public:
        RelocalizationModule(const std::string &vocabPath,
                             const std::string &configPath);
        ~RelocalizationModule();

        bool loadMap();
        LocationResult processFrame(const cv::Mat &frame);
        void processVideo(const std::string &videoPath, bool visualize = true);
        void visualizeLocation(const LocationResult &result);
        void exportMapToPCD(const std::string &outputPath);
        void debugStatus();

    private:
        // Map data
        ORB_SLAM3::System *mpSLAM;
        ORB_SLAM3::Atlas *mpAtlas;
        ORB_SLAM3::ORBVocabulary *mpVocabulary;
        ORB_SLAM3::KeyFrameDatabase *mpKeyFrameDB;
        std::vector<ORB_SLAM3::KeyFrame *> mvKeyFrames;
        std::vector<ORB_SLAM3::MapPoint *> mvMapPoints;
        std::unique_ptr<ORB_SLAM3::ORBextractor> mpORBextractor;

        std::unique_ptr<LSHMatcher> mpLSHMatcher;

        cv::Mat mK;
        cv::Mat mDistCoef;

        std::string mMapPath;
        std::string mVocabPath;
        std::string mConfigPath;

        int mFrameSkip;
        int mMinInliers;
        int mMinMatches;
        float mBowThreshold;
        int mMaxCandidates;
        bool mVisualizationEnabled;
        bool mExportPCD;
        std::string mPCDPath;

        float mVisibilityRadius;
        int mLSHNumTables;
        int mLSHNumBits;
        int mLSHHammingThreshold;
        int mPROSACMaxIterations;

        TrackingState mCurrentState;
        std::atomic<TrackingState> mStateAtomic; // Thread-safe state

        cv::Point3f mLastKnownPosition;
        cv::Point3f mCurrentPosition;
        cv::Point3f mVisOdomPosition; // Position from visual odometry
        cv::Point3f mLSHPosition;     // Position from LSH

        bool mHaveLastKnownPosition;
        std::chrono::steady_clock::time_point mTimeWhenLost;
        double mLostDuration; // How long we've been lost
        float mLostTimeout;   // 30 seconds default
        std::atomic<bool> mTimeoutReached;

        // ========== NEW: Parallel Thread Management ==========
        std::unique_ptr<std::thread> mpVisOdomThread;
        std::unique_ptr<std::thread> mpLSHThread;
        std::atomic<bool> mStopThreads;
        std::atomic<bool> mLSHFoundPosition;
        std::atomic<bool> mVisOdomRunning;

        std::mutex mPositionMutex;
        std::mutex mStateMutex;

        // ========== Position Smoothing ==========
        float mSmoothingFactor; // 0.0 = instant, 1.0 = very smooth
        int mSmoothingFrames;

        std::vector<cv::Point3f> mMapPointsViz;

        int mFrameCount;
        cv::Mat mCurrentFrame;
        std::vector<cv::KeyPoint> mCurrentKeypoints;
        cv::Mat mCurrentDescriptors;

        bool loadConfig();
        void extractFeatures(const cv::Mat &frame,
                             std::vector<cv::KeyPoint> &keypoints,
                             cv::Mat &descriptors);
        // ========== DBoW2 Methods (Global Relocalization) ==========
        std::vector<ORB_SLAM3::KeyFrame *> detectRelocalizationCandidates(
            const cv::Mat &descriptors);

        bool matchWithKeyFrame(const std::vector<cv::KeyPoint> &keypoints,
                               const cv::Mat &descriptors,
                               ORB_SLAM3::KeyFrame *pKF,
                               std::vector<cv::Point3f> &points3D,
                               std::vector<cv::Point2f> &points2D);

        LocationResult globalRelocalizationDBoW2(const cv::Mat &frame);

        // ========== LSH Methods (Local Relocalization) ==========
        std::vector<ORB_SLAM3::MapPoint *> getCandidateMapPointsLSH();

        bool matchWithLSH(const std::vector<cv::KeyPoint> &keypoints,
                          const cv::Mat &descriptors,
                          const std::vector<ORB_SLAM3::MapPoint *> &vpMapPoints,
                          std::vector<cv::Point3f> &points3D,
                          std::vector<cv::Point2f> &points2D);

        LocationResult localRelocalizationLSH(const cv::Mat &frame);

        // ========== Visual Odometry Methods ==========
        void runVisualOdometryThread();
        cv::Point3f estimatePositionFromTracking(const cv::Mat &frame);

        // ========== LSH Thread Methods ==========
        void runLSHThread();

        // ========== Pose Estimation ==========
        bool solvePnP(const std::vector<cv::Point3f> &points3D,
                      const std::vector<cv::Point2f> &points2D,
                      cv::Mat &rvec, cv::Mat &tvec,
                      std::vector<int> &inliers);

        bool solvePnPWithPROSAC(const std::vector<cv::Point3f> &points3D,
                                const std::vector<cv::Point2f> &points2D,
                                cv::Mat &rvec, cv::Mat &tvec,
                                std::vector<int> &inliers);

        cv::Point3f computePosition(const cv::Mat &rvec, const cv::Mat &tvec);

        // ========== State Management ==========
        void setState(TrackingState newState);
        TrackingState getState();

        void handleStateTransition(TrackingState oldState, TrackingState newState);

        // ========== Position Correction ==========
        cv::Point3f smoothPositionTransition(const cv::Point3f &from,
                                             const cv::Point3f &to,
                                             float alpha);

        // ========== Utility ==========
        double getTimeSinceLost();
        bool isTimeout();
        void startParallelTracking();
        void stopParallelTracking();
    };

} // namespace Relocalization

#endif