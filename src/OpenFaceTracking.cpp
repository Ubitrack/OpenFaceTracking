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
#include <utDataflow/TriggerComponent.h>
#include <utDataflow/TriggerInPort.h>
#include <utDataflow/TriggerOutPort.h>

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
	Dataflow::PushSupplier< Measurement::ImageMeasurement > m_debugPort;
	Dataflow::TriggerOutPort< Measurement::Pose > m_outPort;

private:

   LandmarkDetector::FaceModelParameters m_det_parameters;
   LandmarkDetector::CLNF * m_face_model;

   // perform OpenFace head pose estimation
   // source code ported from OpenFace/exe/FaceLandmarkVid
   double estimateHeadPose(cv::Vec6d & pose, cv::Mat & image, cv::Mat & imageGray, Measurement::Matrix3x3 intrinsics);

   // convert from ubitrack enum to visage enum (for image pixel format)
   void OpenFaceTracking::printPixelFormat(Measurement::ImageMeasurement image);

};


OpenFaceTracking::OpenFaceTracking( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph )
	: Dataflow::TriggerComponent(sName, subgraph)
	, m_outPort("Output", *this)
	, m_inPortColor("ImageInput", *this)
	, m_inPortGray("ImageInputGray", *this)
	, m_inIntrinsics("Intrinsics", *this)
	, m_debugPort("DebugOutput", *this)
{
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

void OpenFaceTracking::compute(Measurement::Timestamp t)
{
	//printPixelFormat(imageRGB);

	Measurement::ImageMeasurement imageRGB = m_inPortColor.get();
	Measurement::ImageMeasurement imageGray = m_inPortGray.get();
	Measurement::Matrix3x3 intrinsics = m_inIntrinsics.get(t);

	cv::Mat destColor, destGray;
	cv::Mat & imgColor = destColor;
	cv::Mat & imgGray = destGray;
	if (imageRGB->origin() == 0) {
		imgColor = imageRGB->Mat();
		imgGray = imageGray->Mat();
	}
	else {
		// the direct show image is rotated 180 degrees
		cv::flip(imageRGB->Mat(), destColor, 0);
		cv::flip(imageGray->Mat(), destGray, 0);
	}

	//cv::imshow("newImage imgRGB", imgRGB);
	//cv::imshow("newImage imgGray", imgGray);
	//cv::waitKey(1);

	// pass the image to OpenFace
	cv::Vec6d pose;
	double confidence = estimateHeadPose(pose, imgColor, imgGray, intrinsics);
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
	Math::Vector3d headTrans = Math::Vector3d(tx, -ty, -tz);
	Math::Pose headPose = Math::Pose(headRot, headTrans);
	Measurement::Pose meaHeadPose = Measurement::Pose(imageRGB.time(), headPose);
	m_outPort.send(meaHeadPose);
}

double OpenFaceTracking::estimateHeadPose(cv::Vec6d & pose, cv::Mat & imageColor, cv::Mat & imageGray, Measurement::Matrix3x3 intrinsics)
{
	//cv::imshow("estimateHeadPose3 image", imageRGB);
	//cv::waitKey(1);

	// The actual facial landmark detection / tracking
	bool detection_success = false;
	if (m_face_model->detection_certainty > 0.7) {
		detection_success = LandmarkDetector::DetectLandmarksInVideo(imageColor, m_face_model->GetBoundingBox(), *m_face_model, m_det_parameters, imageGray);
	}
	else {
		detection_success = LandmarkDetector::DetectLandmarksInVideo(imageColor, *m_face_model, m_det_parameters, imageGray);
	}
		

	double confidence = 0.0;
	if (detection_success)
	{
		float fx = (*intrinsics)(0, 0);
		float fy = (*intrinsics)(1, 1);
		float cx = -(*intrinsics)(0, 2);
		float cy = -(*intrinsics)(1, 2);

		// Work out the pose of the head from the tracked model
		pose = LandmarkDetector::GetPose(*m_face_model, fx, fy, cx, cy);
		confidence = m_face_model->detection_certainty;
	}
	return confidence;
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

