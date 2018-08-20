/*
 * Ubitrack - Library for Ubiquitous Tracking
 * Copyright 2006, Technische Universitaet Muenchen, and individual
 3* contributors as indicated by the @authors tag. See the
 * copyright.txt in the distribution for a full listing of individual
 * contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

/**
 * @ingroup vision_components
 * @file
 * 
 *
 * @author Joe Bedard <bedard@in.tum.de>
 */


#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <strstream>
#include <log4cpp/Category.hh>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/numeric/ublas/blas.hpp>

#include <utDataflow/PushSupplier.h>
#include <utDataflow/PullSupplier.h>
#include <utDataflow/Component.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>
#include <utMeasurement/TimestampSync.h>
#include <utUtil/OS.h>
#include <utUtil/TracingProvider.h>
#include <utUtil/BlockTimer.h>
#include <opencv/cv.h>
#include <utVision/Image.h>
#include <utVision/Undistortion.h>
#include <utDataflow/TriggerComponent.h>
#include <utDataflow/TriggerInPort.h>
#include <utDataflow/TriggerOutPort.h>
#include <utAlgorithm/Homography.h>
#include <utAlgorithm/PoseEstimation2D3D/PlanarPoseEstimation.h>
#include <utAlgorithm/Projection.h>
#include <boost/numeric/ublas/blas.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <LandmarkCoreIncludes.h>
#include <RotationHelpers.h>

using namespace Ubitrack;
using namespace Ubitrack::Vision;

namespace Ubitrack { namespace Drivers {

	// get a logger
	static log4cpp::Category& logger(log4cpp::Category::getInstance("Ubitrack.Vision.OpenFaceTracking"));

/**
 * @ingroup vision_components
 *
 * @par Input Ports
 * None.
 *
 * @par Output Ports
 * \c Output push port of type Ubitrack::Measurement::ImageMeasurement.
 *
 * @par Configuration
 * The configuration tag contains a \c <dsvl_input> configuration.
 * For details, see the DirectShow documentation...
 *
 */
class OpenFaceTracking
	: public Dataflow::TriggerComponent
{
public:

	/** constructor */
   OpenFaceTracking( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph >  );

	/** destructor, waits until thread stops */
	~OpenFaceTracking();

	/** starts the camera */
	void start();

	/** stops the camera */
	void stop();

	void compute(Measurement::Timestamp t);

protected:
	
	// the ports
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_inPortColor;
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_inPortGray;
	Dataflow::PullConsumer< Measurement::Matrix3x3 > m_inIntrinsics;
	Dataflow::PushConsumer< Measurement::Button> m_eventIn;
	Dataflow::PullConsumer<Measurement::Pose> m_referenceHead;

	Dataflow::PushSupplier< Measurement::ImageMeasurement > m_debugPort;
	Dataflow::TriggerOutPort< Measurement::Pose > m_outPort;
	Dataflow::PushSupplier< Measurement::ErrorPose > m_outErrorPort;

	void buttonEvent(Measurement::Button e);

private:

   LandmarkDetector::FaceModelParameters m_det_parameters;
   LandmarkDetector::CLNF * m_face_model;
   Util::BlockTimer m_timerDetectLandmarksInVideo;
   Util::BlockTimer m_timerAll;   
   bool m_LastDetection_success;
   bool m_isTracking;

   Math::Vector3d m_head2Eye;
   int m_initCount = 0;

   double m_minLikelihood;
   int m_maxDelay;
   int m_imageHeight;
   int m_imageWidth;

	// additional covariance
	double m_addErrorX;
	double m_addErrorY;
	double m_addErrorZ;

	// convert from ubitrack enum to visage enum (for image pixel format)
   void printPixelFormat(Measurement::ImageMeasurement image);

   void optimiseForVideo();

   void resetTracker(Measurement::Timestamp t);
};


OpenFaceTracking::OpenFaceTracking( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph )
	: Dataflow::TriggerComponent(sName, subgraph)
	, m_outPort("Output", *this)
	, m_outErrorPort("OutputError", *this)
	, m_inPortColor("ImageInput", *this)
	, m_inPortGray("ImageInputGray", *this)
	, m_inIntrinsics("Intrinsics", *this)
	, m_referenceHead("RefCam2Head", *this)
	, m_eventIn("EventIn", *this, boost::bind(&OpenFaceTracking::buttonEvent, this, _1))
	, m_debugPort("DebugImage", *this)
	, m_timerDetectLandmarksInVideo("OpenFaceTracking.DetectLandmarksInVideo", logger)
	, m_timerAll("OpenFaceTracking.All", logger)	
	, m_LastDetection_success(false)
	, m_isTracking(false)
	, m_minLikelihood(-5)
	, m_maxDelay(30)
	, m_addErrorX(0.0)
	, m_addErrorY(0.0)
	, m_addErrorZ(0.0)
{
	subgraph->m_DataflowAttributes.getAttributeData("minLikelihood", m_minLikelihood);
	subgraph->m_DataflowAttributes.getAttributeData("maxDelay", m_maxDelay);

   if (subgraph->m_DataflowAttributes.hasAttribute("modelFile"))
   {
      std::string modelDir = subgraph->m_DataflowAttributes.getAttributeString("modelFile");
	  LOG4CPP_INFO(logger, "Model File: " << modelDir);

	  m_face_model = new LandmarkDetector::CLNF(modelDir);
	  if (!m_face_model->loaded_successfully)
	  {
		  std::ostringstream os;
		  os << "OpenFace landmark detector failed loading! " << modelDir;
		  UBITRACK_THROW(os.str());
	  }

	  optimiseForVideo();

	  subgraph->m_DataflowAttributes.getAttributeData("addErrorX", m_addErrorX);
	  subgraph->m_DataflowAttributes.getAttributeData("addErrorY", m_addErrorY);
	  subgraph->m_DataflowAttributes.getAttributeData("addErrorZ", m_addErrorZ);
   }
   else 
   {
      std::ostringstream os;
      os << "OpenFace Model File is required, but was not provided!";
      UBITRACK_THROW(os.str());
   }
}

void OpenFaceTracking::printPixelFormat(Measurement::ImageMeasurement image)
{
   using Ubitrack::Vision::Image;
   switch (image->pixelFormat())
   {
   case Image::LUMINANCE:
	   LOG4CPP_INFO(logger, "Image::LUMINANCE");
	   break;
   case Image::RGB:
	   LOG4CPP_INFO(logger, "Image::RGB");
	   break;
   case Image::BGR:
	   LOG4CPP_INFO(logger, "Image::BGR");
	   break;
   case Image::RGBA:
	   LOG4CPP_INFO(logger, "Image::RGBA");
	   break;
   case Image::BGRA:
      LOG4CPP_INFO(logger, "Image::BGRA");
	  break;
   case Image::YUV422:
	   LOG4CPP_INFO(logger, "Image::YUV422");
	   break;
   case Image::YUV411:
	   LOG4CPP_INFO(logger, "Image::YUV411");
	   break;
   case Image::RAW:
	   LOG4CPP_INFO(logger, "Image::RAW");
	   break;
   case Image::DEPTH:
	   LOG4CPP_INFO(logger, "Image::DEPTH");
	   break;
   case Image::UNKNOWN_PIXELFORMAT:
	   LOG4CPP_INFO(logger, "Image::UNKNOWN_PIXELFORMAT");
	   break;
   default:
      break;
   }
}

void OpenFaceTracking::optimiseForVideo()
{
	m_det_parameters.window_sizes_small = vector<int>(4);
	m_det_parameters.window_sizes_init = vector<int>(4);
	
	m_det_parameters.reinit_video_every = -1;
	m_det_parameters.limit_pose = true;
	// default true
	m_det_parameters.refine_hierarchical = false;
	// default true
	m_det_parameters.refine_parameters = false;

	// For fast tracking
	m_det_parameters.window_sizes_small[0] = 0;
	m_det_parameters.window_sizes_small[1] = 9;
	m_det_parameters.window_sizes_small[2] = 7;
	m_det_parameters.window_sizes_small[3] = 0;

	// Just for initialisation
	m_det_parameters.window_sizes_init.at(0) = 11;
	m_det_parameters.window_sizes_init.at(1) = 9;
	m_det_parameters.window_sizes_init.at(2) = 7;
	m_det_parameters.window_sizes_init.at(3) = 5;

	// For first frame use the initialisation
	m_det_parameters.window_sizes_current = m_det_parameters.window_sizes_init;

	m_det_parameters.multi_view = false;
	m_det_parameters.num_optimisation_iteration = 5;

	
	// Off by default (as it might lead to some slight inaccuracies in slowly moving faces)
	//default false
	m_det_parameters.use_face_template = true;
	m_det_parameters.face_template_scale = 0.3f;

	m_det_parameters.sigma = 1.5;
	m_det_parameters.reg_factor = 25;
	m_det_parameters.weight_factor = 0;

	// Parameter optimizations for CE-CLM
	if (m_det_parameters.curr_landmark_detector == ::LandmarkDetector::FaceModelParameters::CECLM_DETECTOR)
	{
		m_det_parameters.sigma = 1.5f * m_det_parameters.sigma;
		m_det_parameters.reg_factor = 0.9f * m_det_parameters.reg_factor;
	}
}

void OpenFaceTracking::compute(Measurement::Timestamp t)
{
	UBITRACK_TIME(m_timerAll);
	//printPixelFormat(imageRGB);
	
	Measurement::ImageMeasurement imageColor = m_inPortColor.get();
	Measurement::ImageMeasurement imageGray = m_inPortGray.get();
	Measurement::Matrix3x3 intrinsics = m_inIntrinsics.get(t);

	m_imageWidth = imageColor->width();
	m_imageHeight = imageColor->height();

	float fx = (*intrinsics)(0, 0);
	float fy = (*intrinsics)(1, 1);
	float cx = -(*intrinsics)(0, 2);
	float cy = -(*intrinsics)(1, 2);

	cv::Mat destColor, destGray;
	if (imageColor->origin() == 0) {
		destColor = imageColor->Mat();
		destGray = imageGray->Mat();
	}
	else {
		// the input image is flipped vertically
		cv::flip(imageColor->Mat(), destColor, 0);
		cv::flip(imageGray->Mat(), destGray, 0);
		cy = imageColor->Mat().rows - 1 + (*intrinsics)(1, 2);
		//LOG4CPP_WARN(logger, "Input image is flipped. Consider flipping in the driver to improve performance.");
	}

	// pass the image to OpenFace
	cv::Vec6d pose;
	std::vector<Math::Vector2d> points2d;
	std::vector<Math::Vector3d> points3d;

	// for debugging
	//cv::imshow("estimateHeadPose imageColor", imageColor);
	//cv::imshow("estimateHeadPose imageGray", imageGray);
	//cv::waitKey(1);

	// perform OpenFace head pose estimation
	// source code ported from OpenFace/exe/FaceLandmarkVid
	Measurement::Timestamp before;
	Measurement::Timestamp after;
	bool detection_success = false;
	{
		UBITRACK_TIME(m_timerDetectLandmarksInVideo);				
		before = Measurement::now();
		detection_success = LandmarkDetector::DetectLandmarksInVideo(destColor, *m_face_model, m_det_parameters, destGray);
		after = Measurement::now();		
	}
	Measurement::Timestamp diff = (after - before) / 1000000l;
	if (!m_isTracking) {
		//LOG4CPP_INFO(logger, "detection: " << detection_success << " in: " << diff << " diffStartToMea " << (before - imageColor.time()) / 1000000l << " diffAfterToMea " << (after - imageColor.time()) / 1000000l);
	}
	


	m_LastDetection_success = detection_success;

	if (!detection_success) {
		m_isTracking = false;
	}
	m_isTracking = detection_success;

	if (m_face_model->failures_in_a_row > 3) {
		resetTracker(imageColor.time());
	}
	
	//LOG4CPP_INFO(logger, "success: " << detection_success <<  " Tracking Confidence: " << m_face_model->detection_certainty << " isTracking: " << m_face_model->tracking_initialised << " scale:" << m_face_model->params_global(0) << " model_likelihood:" << m_face_model->model_likelihood << " failures in row:" << m_face_model->failures_in_a_row);
	
	if (detection_success && diff < m_maxDelay && m_face_model->model_likelihood > m_minLikelihood)
	{
		// Work out the pose of the head from the tracked model
		pose = LandmarkDetector::GetPose(*m_face_model, fx, fy, cx, cy);
		
				

		// convert from millimeters to meters
		float tx = pose[0] / 1000.0f;
		float ty = pose[1] / 1000.0f;
		float tz = pose[2] / 1000.0f;

		// output head pose
		Math::Quaternion headRot = Math::Quaternion(-pose[5], -pose[4], pose[3]);
		Math::Quaternion addRotation = Math::Quaternion(0,1,0,0);
		headRot = headRot * addRotation;
		Math::Vector3d headTrans = Math::Vector3d(tx, -ty, -tz);
		Math::Pose headPose = Math::Pose(headRot, headTrans);

		
		//m_face_model->detection_certainty;
		
		//LOG4CPP_INFO(logger, "Head Translation X Y Z: " << tx << " " << ty << " " << tz);
		//LOG4CPP_INFO(logger, "Head Rotation X Y Z:  " << pose[3] << " " << pose[4] << " " << pose[5]);

		// convert 2D OpenFace landmark points to 2D Ubitrack points
		const int imageHeight = imageColor->height();
		
		int numPoints = m_face_model->detected_landmarks.rows / 2;
		for (int i = 0; i < numPoints; ++i)
		{
			Math::Vector2d point(m_face_model->detected_landmarks.at<float>(i), imageHeight - 1 - m_face_model->detected_landmarks.at<float>(i + numPoints));
			points2d.push_back(point);
		}
		

		// convert 3D OpenFace landmark points to 3D Ubitrack points

	
		
		int n = m_face_model->detected_landmarks.rows / 2;
		cv::Mat_<float> shape3d(n * 3, 1);
		m_face_model->pdm.CalcShape3D(shape3d, m_face_model->params_local);
		shape3d = shape3d.reshape(1, 3);
		
		for (int i = 0; i < shape3d.cols; ++i)
		{
			Math::Vector3d point(-shape3d.at<float>(0, i) / 1000.0f, -shape3d.at<float>(1, i) / 1000.0f, shape3d.at<float>(2, i) / 1000.0f);
			Math::Vector3d relPoint = point;
			points3d.push_back(relPoint);
		}

		if (m_debugPort.isConnected()) {
			// debug drawing of face landmarks
			boost::shared_ptr<Vision::Image> dImage = imageColor->Clone();
			cv::Mat debugImage = dImage->Mat();
			Math::Matrix< double, 3, 4 > poseMat(headPose);
			Math::Matrix3x4d projectionMatrix = boost::numeric::ublas::prod(*intrinsics, poseMat);




			for (int i = 0; i < points3d.size(); i++) {
				Math::Vector4d tmp(points3d[i][0], points3d[i][1], points3d[i][2], 1);
				Math::Vector3d projectedPoint = boost::numeric::ublas::prod(projectionMatrix, tmp);
				Math::Vector2d p2d = points2d[i];
				double wRef = projectedPoint[2];
				projectedPoint = projectedPoint / wRef;
				if (imageColor->origin() == 0) {
					projectedPoint[1] = debugImage.rows - 1 - projectedPoint[1];
					p2d[1] = debugImage.rows - 1 - p2d[1];
				}

				cv::Point2d p1(projectedPoint[0], projectedPoint[1]);
				cv::Point2d p2(p2d[0], p2d[1]);
				cv::circle(debugImage, cv::Point2d(p1), 4, cv::Scalar(255, 0, 0), -1);
				cv::circle(debugImage, cv::Point2d(p2), 3, cv::Scalar(0, 255, 0), -1);
				cv::line(debugImage, p1, p2, cv::Scalar(0, 0, 255), 1);
			}

			m_debugPort.send(Measurement::ImageMeasurement(t, dImage));
		}

		if (points3d.size() > 5) {
			
			double residual = 0.0;

			
			Math::Matrix< double, 3, 4 > poseMat(headPose);

			Math::Matrix3x4d projectionMatrix = boost::numeric::ublas::prod(*intrinsics, poseMat);
	

			// 2D = P * 3D Point
			// Reproject 3D-points to 2D points.
			const std::size_t n_points(points3d.size());
			for (std::size_t i(0); i < n_points; i++)
			{
				Math::Vector< double, 4 > hom((points3d.at(i))[0], (points3d.at(i))[1], (points3d.at(i))[2], 1.0);
				Math::Vector< double, 3 > tmp;

				tmp = boost::numeric::ublas::prod(projectionMatrix, hom);
				double w = tmp[2];
				tmp = tmp / w;
				double d = (tmp[0] - (points2d.at(i))[0]) * (tmp[0] - (points2d.at(i))[0]) + (tmp[1] - (points2d.at(i))[1]) * (tmp[1] - (points2d.at(i))[1]);
				residual += d;
				
			}
			
			/*residual /= points3d.size();
			residual = sqrt( residual );*/
			
			
			Math::Matrix<double, 6, 6> covar = Algorithm::PoseEstimation2D3D::singleCameraPoseError(headPose, points3d, *intrinsics, residual);

			Math::Vector3d headPos = headPose.translation();
			
			cv::Mat_<float> shape = m_face_model->GetShape(fx, fy, cx, cy);
			
			//if (m_initCount < 100 && m_face_model->detection_certainty > 0.9) {
			//	// set head position to average of left+right inner eye landmarks
			//	

			//	double txe = (shape.at<float>(0, 42) + shape.at<float>(0, 39)) / 2.0f / 1000.0f;
			//	double tye = -(shape.at<float>(1, 42) + shape.at<float>(1, 39)) / 2.0f / 1000.0f;
			//	double tze = -(shape.at<float>(2, 42) + shape.at<float>(2, 39)) / 2.0f / 1000.0f;
			//	Math::Vector3d eye(txe, tye, tze);
			//	Math::Pose invHeadPose = ~headPose;
			//	Math::Vector3d head2eye = invHeadPose*eye;

			//	if (m_initCount == 0) {
			//		m_head2Eye = head2eye;
			//	}
			//	else {
			//		const double alpha = 0.05;
			//		//m_head2Eye = head2eye*alpha + m_head2Eye*(1.0 - alpha);
			//		m_head2Eye = head2eye;
			//	}
			//	m_initCount++;				
			//	//LOG4CPP_INFO(logger, "m_head2Eye : " << m_head2Eye);
			//}

			double txe = (shape.at<float>(0, 42) + shape.at<float>(0, 39)) / 2.0f / 1000.0f;
			double tye = -(shape.at<float>(1, 42) + shape.at<float>(1, 39)) / 2.0f / 1000.0f;
			double tze = -(shape.at<float>(2, 42) + shape.at<float>(2, 39)) / 2.0f / 1000.0f;
			headTrans = Math::Vector3d(txe, tye, tze);

			headPose = Math::Pose(headRot, headTrans );

			covar(0, 0) += m_addErrorX * m_addErrorX;
			covar(1, 1) += m_addErrorY * m_addErrorY;
			covar(2, 2) += m_addErrorZ * m_addErrorZ;


			Measurement::Pose meaHeadPose = Measurement::Pose(imageColor.time(), headPose);
			m_outPort.send(meaHeadPose);
			m_outErrorPort.send(Measurement::ErrorPose(t, Math::ErrorPose(headPose, covar)));

		
		}
	}
	else {
		//resetTracker(imageColor.time());
		// must still send debug image if face is not detected
		if (m_debugPort.isConnected())
			m_debugPort.send(Measurement::ImageMeasurement(t, imageColor->Clone()));
	}

}

void OpenFaceTracking::buttonEvent(Measurement::Button e) {
	if (e->m_value == 'r') {
		LOG4CPP_INFO(logger, "reset by user");
		resetTracker(e.time());
	} else if(e->m_value == 'h') {
		LOG4CPP_INFO(logger, "hard reset by user");
		m_face_model->Reset();
	}
}

void OpenFaceTracking::resetTracker(Measurement::Timestamp t) {
	if (m_referenceHead.isConnected()) {
		try {
			Measurement::Pose head = m_referenceHead.get(t);
			Measurement::Matrix3x3 intrinsics = m_inIntrinsics.get(t);

			

			Math::Pose identity;
			Math::Matrix< double, 3, 4 > poseMat(identity);
			
			Math::Matrix3x4d projectionMatrix = boost::numeric::ublas::prod(*intrinsics, poseMat);
			Math::Vector3d pos = head->translation();
			Math::Vector< double, 4 > hom(pos[0], pos[1], pos[2], 1.0);
			Math::Vector< double, 3 > tmp = boost::numeric::ublas::prod(projectionMatrix, hom);
			double w = tmp[2];
			tmp = tmp / w;

			
			tmp[1] = m_imageHeight - 1 - tmp[1];

			double x = tmp[0] / m_imageWidth;
			double y = tmp[1] / m_imageHeight;

			//LOG4CPP_INFO(logger, "search for head at " << tmp[0] << " : " << tmp[1] << "    " << x << " : " << y);

			
			m_face_model->Reset(x,y);

			return;
		}
		catch (...) {

		}
	} 
	// reset with no addition info
	//LOG4CPP_INFO(logger, "hard reset, no info");
	m_face_model->Reset();
}

OpenFaceTracking::~OpenFaceTracking()
{
   delete m_face_model;
}


void OpenFaceTracking::start()
{
	Component::start();
}


void OpenFaceTracking::stop()
{
	Component::stop();
}


} } // namespace Ubitrack::Driver

UBITRACK_REGISTER_COMPONENT( Dataflow::ComponentFactory* const cf ) {
	cf->registerComponent< Ubitrack::Drivers::OpenFaceTracking > ( "OpenFaceTracking" );
}

