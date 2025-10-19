#ifndef RELOCALIZATION_H
#define RELOCALIZATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include <memory>

#include "System.h"
#include "Atlas.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"

namespace Relocalization
{

    struct LocationResult
    {
        bool success;
        cv::Point3f position;
        int matchedKeyFrameId;
        int numInliers;
        std::vector<cv::DMatch> matches;
        int totalMatches;
        float confidence;
        float bowScore;
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
        void setVisualizationEnabled(bool enabled)
        {
            mVisualizationEnabled = enabled;
        }

    private:
        // Map data
        ORB_SLAM3::System *mpSLAM;
        ORB_SLAM3::Atlas *mpAtlas;
        ORB_SLAM3::ORBVocabulary *mpVocabulary;
        ORB_SLAM3::KeyFrameDatabase *mpKeyFrameDB;
        std::vector<ORB_SLAM3::KeyFrame *> mvKeyFrames;
        std::vector<ORB_SLAM3::MapPoint *> mvMapPoints;

        std::unique_ptr<ORB_SLAM3::ORBextractor> mpORBextractor;

        cv::Mat mK;
        cv::Mat mDistCoef;

        cv::Size mProcessSize;
        cv::Size mDisplaySize;

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

        std::vector<cv::Point3f> mMapPointsViz;
        cv::Point3f mCurrentPosition;

        bool loadConfig();
        void extractFeatures(const cv::Mat &frame,
                             std::vector<cv::KeyPoint> &keypoints,
                             cv::Mat &descriptors);
        std::vector<ORB_SLAM3::KeyFrame *> detectRelocalizationCandidates(
            const cv::Mat &descriptors);
        bool matchWithKeyFrame(const std::vector<cv::KeyPoint> &keypoints,
                               const cv::Mat &descriptors,
                               ORB_SLAM3::KeyFrame *pKF,
                               std::vector<cv::Point3f> &points3D,
                               std::vector<cv::Point2f> &points2D);
        bool solvePnP(const std::vector<cv::Point3f> &points3D,
                      const std::vector<cv::Point2f> &points2D,
                      cv::Mat &rvec, cv::Mat &tvec,
                      std::vector<int> &inliers);
        cv::Point3f computePosition(const cv::Mat &rvec, const cv::Mat &tvec);
    };

} // namespace Relocalization

#endif