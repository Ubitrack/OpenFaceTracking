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

#include <utDataflow/PushSupplier.h>
#include <utDataflow/PullSupplier.h>
#include <utDataflow/Component.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>
#include <utMeasurement/TimestampSync.h>
#include <utUtil/OS.h>
#include <utUtil/TracingProvider.h>
#include <opencv/cv.h>
#include <utVision/Image.h>
#include <utVision/Undistortion.h>

#define _USE_MATH_DEFINES
#include <math.h>

//#include <LandmarkCoreIncludes.h>
#include "LandmarkDetectorInterop.h"
#include "FaceDetectorInterop.h"
#include "FaceAnalyserInterop.h"

using namespace Ubitrack;
using namespace Ubitrack::Vision;
using namespace LandmarkDetectorInterop;
using namespace FaceDetectorInterop;
using namespace FaceAnalyserInterop;

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
	: public Dataflow::Component
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

protected:
	
	// the ports
	Dataflow::PushConsumer< Measurement::ImageMeasurement > m_inPort;

	Dataflow::PushSupplier< Measurement::Pose > m_outPort;
	
	void newImage(Measurement::ImageMeasurement imageRGB);

private:

	int * track_stat;

   bool m_DetectorHaar = false;
   bool m_DetectorHOG = false;
   bool m_DetectorCNN = true;
   bool m_DynamicAUModels = true;
   int m_image_output_size = 112;
   bool m_MaskAligned = true;
   FaceModelParameters * m_face_model_params;
   FaceDetector * m_face_detector;
   CLNF * m_landmark_detector;
   FaceAnalyserManaged * m_face_analyser;

   // perform OpenFace head pose estimation
   // source code ported from OpenFace/gui/OpenFaceOffline/MainWindow.xaml.cs
   double estimateHeadPose(cv::Vec6f & pose, Measurement::ImageMeasurement imageRGB);

   // perform OpenFace head pose estimation
   // source code ported from OpenFace/gui/OpenFaceOffline/MainWindow.xaml.cs ProcessSequence
   double estimateHeadPose2(cv::Vec6f & pose, Measurement::ImageMeasurement imageRGB);

   // convert from ubitrack enum to visage enum (for image pixel format)
   void OpenFaceTracking::printPixelFormat(Measurement::ImageMeasurement imageRGB);
};


OpenFaceTracking::OpenFaceTracking( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph )
	: Dataflow::Component( sName )	
	, m_outPort( "Output", *this )
	, m_inPort( "ImageInput", *this, boost::bind(&OpenFaceTracking::newImage, this, _1))
{
	LOG4CPP_INFO(logger, "enter OpenFaceTracking::OpenFaceTracking");
   if (subgraph->m_DataflowAttributes.hasAttribute("modelDirectory"))
   {
      std::string modelDir = subgraph->m_DataflowAttributes.getAttributeString("modelDirectory");
	  //modelDir = "C:/libraries/JoeSRG/";
	  LOG4CPP_INFO(logger, "Model Directory: " << modelDir);

      //m_face_model_params = new FaceModelParameters();
      m_face_model_params = new FaceModelParameters(modelDir, true, false, false);

      m_face_detector = new FaceDetector(m_face_model_params->GetHaarLocation(), m_face_model_params->GetMTCNNLocation());

      // If MTCNN model not available, use HOG
      if (!m_face_detector->IsMTCNNLoaded())
      {
         m_DetectorCNN = false;
         m_DetectorHOG = true;
      }
      m_face_model_params->SetFaceDetector(m_DetectorHaar, m_DetectorHOG, m_DetectorCNN);
	  m_face_model_params->optimiseForVideo();

      m_landmark_detector = new CLNF(*m_face_model_params);
	  if (!m_landmark_detector->isLoaded())
	  {
		  std::ostringstream os;
		  os << "OpenFace landmark detector failed loading!";
		  UBITRACK_THROW(os.str());
	  }

      //m_face_analyser = new FaceAnalyserManaged(modelDir, m_DynamicAUModels, m_image_output_size, m_MaskAligned);

	  m_landmark_detector->Reset();
   }
   else 
   {
      std::ostringstream os;
      os << "OpenFace Model Directory is required, but was not provided!";
      UBITRACK_THROW(os.str());
   }
   LOG4CPP_INFO(logger, "exit OpenFaceTracking::OpenFaceTracking");
}

void OpenFaceTracking::printPixelFormat(Measurement::ImageMeasurement imageRGB)
{
   using Ubitrack::Vision::Image;
   switch (imageRGB->pixelFormat())
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


void OpenFaceTracking::newImage(Measurement::ImageMeasurement imageRGB)
{
   //printPixelFormat(imageRGB);
   {
      cv::Mat dest;
	  int origin = imageRGB->origin();
	  if (origin == 0) {
		  dest = imageRGB->Mat();
	  }
	  else {
          // the direct show image is flipped 180 degrees
		  cv::rotate(imageRGB->Mat(), dest, cv::RotateFlags::ROTATE_180);
		  //cv::flip(imageRGB->Mat(), dest, 0);
	  }

      // pass the image to OpenFace
      cv::Vec6f pose;
      double confidence = estimateHeadPose2(pose, imageRGB);
	  float tx = pose[0] / 1000.0f;
	  float ty = pose[1] / 1000.0f;
	  float tz = pose[2] / 1000.0f;

      if (confidence > 0.0f)
      {
		  LOG4CPP_INFO(logger, "Tracking Confidence: " << confidence);
		  LOG4CPP_INFO(logger, "Head Translation X Y Z: " << tx << " " << ty << " " << tz);
		  LOG4CPP_INFO(logger, "Head Rotation X Y Z:  " << pose[3] << " " << pose[4] << " " << pose[5]);
      }
     
     // output head pose
      Math::Quaternion headRot = Math::Quaternion(-pose[5], -pose[4], pose[3]);
      Math::Vector3d headTrans = Math::Vector3d(tx, ty, -tz);
      Math::Pose headPose = Math::Pose(headRot, headTrans);
      Measurement::Pose meaHeadPose = Measurement::Pose(imageRGB.time(), headPose);
      m_outPort.send(meaHeadPose);
   }
}

double OpenFaceTracking::estimateHeadPose2(cv::Vec6f & pose, Measurement::ImageMeasurement imageRGB)
{
	float confidence = 0.0f;

	bool detection_succeeding = m_landmark_detector->DetectLandmarksInVideo(imageRGB->Mat(), *m_face_model_params, imageRGB->getGrayscale()->Mat());
	if (detection_succeeding)
	{
		// The face analysis step (for AUs and eye gaze)
		//m_face_analyser->AddNextFrame(imageRGB->Mat(), m_landmark_detector->CalculateAllLandmarks(), detection_succeeding, false);

		// Only the final face will contain the details
		//VisualizeFeatures(frame, visualizer_of, landmarks, landmark_detector.GetVisibilities(), detection_succeeding, i == 0, true, reader.GetFx(), reader.GetFy(), reader.GetCx(), reader.GetCy(), progress);
		//intrisics for LogiC615_5
		float fx = 835.048;
		float fy = 833.865;
		float cx = 413.057;
		float cy = 290.567;
		m_landmark_detector->GetPose(pose, fx, fy, cx, cy);
		confidence = m_landmark_detector->GetConfidence();
	}

	return confidence;
}

double OpenFaceTracking::estimateHeadPose(cv::Vec6f & pose, Measurement::ImageMeasurement imageRGB)
{
   // Detect faces here and return bounding boxes
   vector<cv::Rect_<float>> face_detections;
   vector<float> confidences;
   float confidence = 0.0f;
   if (m_DetectorHOG)
   {
      m_face_detector->DetectFacesHOG(face_detections, imageRGB->getGrayscale()->Mat(), confidences);
   }
   else if (m_DetectorCNN)
   {
      m_face_detector->DetectFacesMTCNN(face_detections, imageRGB->Mat(), confidences);
   }
   else if (m_DetectorHaar)
   {
      m_face_detector->DetectFacesHaar(face_detections, imageRGB->getGrayscale()->Mat(), confidences);
   }

   if (!face_detections.empty())
   {
      bool detection_succeeding = m_landmark_detector->DetectFaceLandmarksInImage(imageRGB->Mat(), face_detections[0], *m_face_model_params, imageRGB->getGrayscale()->Mat());

      auto landmarks = m_landmark_detector->CalculateAllLandmarks();

      // Predict action units
	  //auto au_preds = m_face_analyser->PredictStaticAUsAndComputeFeatures(imageRGB->Mat(), landmarks);

      // Only the final face will contain the details
      //VisualizeFeatures(frame, visualizer_of, landmarks, landmark_detector.GetVisibilities(), detection_succeeding, i == 0, true, reader.GetFx(), reader.GetFy(), reader.GetCx(), reader.GetCy(), progress);
	  //intrisics for LogiC615_5
	  float fx = 835.048;
	  float fy = 833.865;
	  float cx = 413.057;
	  float cy = 290.567;
	  m_landmark_detector->GetPose(pose, fx, fy, cx, cy);
	  confidence = m_landmark_detector->GetConfidence();
   }

   return confidence;
}

OpenFaceTracking::~OpenFaceTracking()
{
   delete m_landmark_detector;
   delete m_face_detector;
   delete m_face_model_params;
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

