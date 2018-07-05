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

// get a logger
static log4cpp::Category& logger( log4cpp::Category::getInstance( "Ubitrack.Vision.OpenFaceTracking" ) );

using namespace Ubitrack;
using namespace Ubitrack::Vision;
using namespace LandmarkDetectorInterop;
using namespace FaceDetectorInterop;
using namespace FaceAnalyserInterop;

namespace Ubitrack { namespace Drivers {

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
	Dataflow::PushConsumer< Measurement::ImageMeasurement > m_inPortRGB;

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

   // convert from ubitrack enum to visage enum (for image pixel format)
   //int switchPixelFormat(Vision::Image::PixelFormat pf);
};


OpenFaceTracking::OpenFaceTracking( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph )
	: Dataflow::Component( sName )	
	, m_outPort( "Output", *this )
	, m_inPortRGB( "ImageInputRGB", *this, boost::bind(&OpenFaceTracking::newImage, this, _1))
{
   if (subgraph->m_DataflowAttributes.hasAttribute("modelDirectory"))
   {
      std::string modelDir = subgraph->m_DataflowAttributes.getAttributeString("modelDirectory");

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

      m_landmark_detector = new CLNF(*m_face_model_params);

      m_face_analyser = new FaceAnalyserManaged(modelDir, m_DynamicAUModels, m_image_output_size, m_MaskAligned);
   }
   else 
   {
      std::ostringstream os;
      os << "OpenFace Model Directory is required, but was not provided!";
      UBITRACK_THROW(os.str());
   }
}

/*int OpenFaceTracking::switchPixelFormat(Vision::Image::PixelFormat pf)
{
   using Ubitrack::Vision::Image;
   switch (pf)
   {
   case Image::LUMINANCE:
      return VISAGE_FRAMEGRABBER_FMT_LUMINANCE;
   case Image::RGB:
      return VISAGE_FRAMEGRABBER_FMT_RGB;
   case Image::BGR:
      return VISAGE_FRAMEGRABBER_FMT_BGR;
   case Image::RGBA:
      return VISAGE_FRAMEGRABBER_FMT_RGBA;
   case Image::BGRA:
      return VISAGE_FRAMEGRABBER_FMT_BGRA;
   case Image::YUV422:
   case Image::YUV411:
   case Image::RAW:
   case Image::DEPTH:
   case Image::UNKNOWN_PIXELFORMAT:
   default:
      return -1;
   }
}*/


void OpenFaceTracking::newImage(Measurement::ImageMeasurement imageRGB)
{
   /*int visageFormat = switchPixelFormat(imageRGB->pixelFormat());
   if (visageFormat == -1)
   {
      LOG4CPP_ERROR(logger, "YUV422, YUV411, RAW, DEPTH, UNKNOWN_PIXELFORMAT are not supported by Visage");
      LOG4CPP_ERROR(logger, "imageRGB->pixelFormat() == " << imageRGB->pixelFormat());
   }
   else*/
   {
      // the direct show image is rotated 180 degrees
      cv::Mat dest;
      cv::rotate(imageRGB->Mat(), dest, cv::RotateFlags::ROTATE_180);

      // pass the image to OpenFace
      cv::Vec6f pose;
      double confidence = estimateHeadPose(pose, imageRGB);
      
      if (confidence >= 0.1f)
      {
         LOG4CPP_DEBUG(logger, "Head Rotation X Y Z:  " << pose[3] << " " << pose[4] << " " << pose[5]);
         LOG4CPP_DEBUG(logger, "Head Translation X Y Z: " << pose[0] << " " << pose[1] << " " << pose[2]);
         LOG4CPP_DEBUG(logger, "Tracking Confidence: " << confidence);
      }
     
     // output head pose
      Math::Quaternion headRot = Math::Quaternion(-pose[5], -pose[4], pose[3]);
      Math::Vector3d headTrans = Math::Vector3d(pose[0], pose[1], -pose[2]);
      Math::Pose headPose = Math::Pose(headRot, headTrans);
      Measurement::Pose meaHeadPose = Measurement::Pose(imageRGB.time(), headPose);
      m_outPort.send(meaHeadPose);
   }
}

double OpenFaceTracking::estimateHeadPose(cv::Vec6f & pose, Measurement::ImageMeasurement imageRGB)
{
   // Detect faces here and return bounding boxes
   vector<cv::Rect_<float>> face_detections;
   vector<float> confidences;
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

   //for (int i = 0; i < face_detections.size(); ++i)
   if (!face_detections.empty())
   {
      //bool detection_succeeding = m_landmark_detector->DetectFaceLandmarksInImage(imageRGB->Mat(), face_detections[i], *m_face_model_params, imageGray->Mat());
      bool detection_succeeding = m_landmark_detector->DetectFaceLandmarksInImage(imageRGB->Mat(), face_detections[0], *m_face_model_params, imageRGB->getGrayscale()->Mat());

      auto landmarks = m_landmark_detector->CalculateAllLandmarks();

      // Predict action units
      auto au_preds = m_face_analyser->PredictStaticAUsAndComputeFeatures(imageRGB->Mat(), landmarks);

      // Only the final face will contain the details
      //VisualizeFeatures(frame, visualizer_of, landmarks, landmark_detector.GetVisibilities(), detection_succeeding, i == 0, true, reader.GetFx(), reader.GetFy(), reader.GetCx(), reader.GetCy(), progress);
      float fx = -1;
      float fy = -1;
      float cx = -1;
      float cy = -1;
      m_landmark_detector->GetPose(pose, fx, fy, cx, cy);
      return m_landmark_detector->GetConfidence();

      // Record an observation
      //RecordObservation(recorder, visualizer_of.GetVisImage(), i, detection_succeeding, reader.GetFx(), reader.GetFy(), reader.GetCx(), reader.GetCy(), 0, 0);
   }
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

