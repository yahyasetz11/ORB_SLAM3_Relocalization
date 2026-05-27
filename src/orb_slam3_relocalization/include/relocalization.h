#ifndef RELOCALIZATION_H
#define RELOCALIZATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
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

    // Keypoints belonging to a single detected landmark region (e.g. door, pillar)
    struct LandmarkRegion
    {
        int cls_id;                              // YOLO class id (e.g. 164 = door)
        cv::Rect bbox;                           // bounding box in process resolution
        std::vector<cv::KeyPoint> keypoints;     // all ORB keypoints inside this bbox
    };

    struct WeightedPnPResult
    {
        bool success;
        cv::Point3f position;
        int numInliers;
        int totalCorrespondences;
        float meanReprojectionError;
        float weightedReprojectionError;
        int iterations;
        std::vector<int> inlierIndices;
        cv::Mat rvec;
        cv::Mat tvec;
    };

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

        // For visualization
        std::vector<cv::KeyPoint> queryKeypoints;
        std::vector<cv::Point2f> matched2DPoints;
        std::vector<cv::Point3f> matched3DPoints;
        std::vector<int> inlierIndices;

        cv::Mat rvec;
        cv::Mat tvec;

        // Keypoints segmented by landmark bounding boxes (populated by relocalization_node)
        std::vector<LandmarkRegion> landmarkRegions;
    };

    class RelocalizationModule
    {
    public:
        RelocalizationModule(const std::string &vocabPath,
                             const std::string &configPath);
        ~RelocalizationModule();

        bool loadMap();
        LocationResult processFrame(const cv::Mat &frame, double timestamp = 0.0);
        void processVideo(const std::string &videoPath, bool visualize = true);
        void processWebcam(int cameraId = 0);
        void debugStatus();
        void setVisualizationEnabled(bool enabled) { mVisualizationEnabled = enabled; }
        cv::Size getDisplaySize() const { return mDisplaySize; }
        cv::Size getProcessSize() const { return mProcessSize; }

        cv::Mat createMapVisualization(const LocationResult &result, cv::Size targetSize);
        cv::Point2f project3DTo2D(const cv::Point3f &pt3D, int mapHeight);

        // Weighted PnP (Gauss-Newton / Levenberg-Marquardt)
        WeightedPnPResult solvePnPWeighted(
            const std::vector<cv::Point3f> &points3D,
            const std::vector<cv::Point2f> &points2D,
            const std::vector<float> &weights,
            const cv::Mat &rvec_hint = cv::Mat(),
            const cv::Mat &tvec_hint = cv::Mat(),
            int maxIterations = 100,
            double convergenceThreshold = 1e-6,
            float inlierThresholdPx = 2.0f);

        std::vector<float> assignWeightsFromLandmarks(
            const std::vector<cv::Point2f> &matched2DPoints,
            const std::vector<LandmarkRegion> &landmarkRegions,
            float permanentWeight = 1.0f,
            float backgroundWeight = 0.3f);

        float computeMeanReprojError(
            const std::vector<cv::Point3f> &points3D,
            const std::vector<cv::Point2f> &points2D,
            const cv::Mat &rvec, const cv::Mat &tvec);

        Eigen::Matrix<double, 2, 6> computeProjectionJacobian(
            const Eigen::Vector3d &Pc, double fx, double fy);

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
        int mHammingThreshold;
        float mBowThreshold;
        int mMaxCandidates;
        bool mVisualizationEnabled;
        bool mExportPCD;
        std::string mPCDPath;

        std::vector<cv::Point3f> mMapPointsViz;
        cv::Point3f mCurrentPosition;

        // For map scaling
        float mMinX, mMaxX, mMinZ, mMaxZ;
        float mScale;

        // Map visualization parameters
        float mMapZoomScale;
        int mMapOffsetX;
        int mMapOffsetY;

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
                               std::vector<cv::Point2f> &points2D,
                               std::vector<float> &hammingDists);
        cv::Point3f computePosition(const cv::Mat &rvec, const cv::Mat &tvec);
        void exportMapToPCD(const std::string &outputPath);

        void drawOrientedTriangle(cv::Mat &img, const cv::Point2f &center,
                                  float angle, float size,
                                  const cv::Scalar &color, int thickness = -1);
        void drawGrid(cv::Mat &img, int mapHeight);
    };

} // namespace Relocalization

#endif