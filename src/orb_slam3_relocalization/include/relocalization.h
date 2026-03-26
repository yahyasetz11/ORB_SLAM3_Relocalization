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

    // Keypoints belonging to a single detected landmark region (e.g. door, pillar)
    struct LandmarkRegion
    {
        int cls_id;                              // YOLO class id (e.g. 164 = door)
        cv::Rect bbox;                           // bounding box in process resolution
        std::vector<cv::KeyPoint> keypoints;     // all ORB keypoints inside this bbox
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
        LocationResult processFrame(const cv::Mat &frame);
        void processVideo(const std::string &videoPath, bool visualize = true);
        void processWebcam(int cameraId = 0);
        void debugStatus();
        void setVisualizationEnabled(bool enabled) { mVisualizationEnabled = enabled; }
        cv::Size getDisplaySize() const { return mDisplaySize; }
        cv::Size getProcessSize() const { return mProcessSize; }

        cv::Mat createMapVisualization(const LocationResult &result, cv::Size targetSize);
        cv::Point2f project3DTo2D(const cv::Point3f &pt3D, int mapHeight);

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
                               std::vector<cv::Point2f> &points2D);
        bool solvePnP(const std::vector<cv::Point3f> &points3D,
                      const std::vector<cv::Point2f> &points2D,
                      cv::Mat &rvec, cv::Mat &tvec,
                      std::vector<int> &inliers);
        cv::Point3f computePosition(const cv::Mat &rvec, const cv::Mat &tvec);
        void exportMapToPCD(const std::string &outputPath);

        void drawOrientedTriangle(cv::Mat &img, const cv::Point2f &center,
                                  float angle, float size,
                                  const cv::Scalar &color, int thickness = -1);
        void drawGrid(cv::Mat &img, int mapHeight);
    };

} // namespace Relocalization

#endif