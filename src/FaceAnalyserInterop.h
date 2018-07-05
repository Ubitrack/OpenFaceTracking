///////////////////////////////////////////////////////////////////////////////
// Ported from C# to C++
// Original file: OpenFace\lib\local\CppInerop\FaceAnalyserInterop.h
//
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

// FaceAnalyser_Interop.h
#ifndef __FACE_ANALYSER_INTEROP_h_
#define __FACE_ANALYSER_INTEROP_h_

#pragma once

// Include all the unmanaged things we need.
//#pragma managed
//#include <msclr\marshal.h>
//#include <msclr\marshal_cppstd.h>
//#pragma unmanaged

// Allows to overcome boost name clash stuff with C++ CLI
#ifdef __cplusplus_cli
#define generic __identifier(generic)
#endif

#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <OpenCVWrappers.h>
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <VisualizationUtils.h>

// Boost stuff
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#ifdef __cplusplus_cli
#undef generic
#endif

//using namespace System::Collections::Generic;
using namespace std;

//#pragma managed

namespace FaceAnalyserInterop {

class FaceAnalyserManaged
{

private:

	FaceAnalysis::FaceAnalyser* face_analyser;

	// The actual descriptors (for visualisation and output)
	cv::Mat_<float>* hog_features;
	cv::Mat* aligned_face;
	int* num_rows;
	int* num_cols;

public:

	FaceAnalyserManaged(string root, bool dynamic, int output_width, bool mask_aligned) 
	{
		FaceAnalysis::FaceAnalyserParameters params(root);
		
		if (!dynamic)
		{
			params.OptimizeForImages();
		}

		params.setAlignedOutput(output_width, -1.0, mask_aligned);
		face_analyser = new FaceAnalysis::FaceAnalyser(params);

		hog_features = new cv::Mat_<float>();

		aligned_face = new cv::Mat();

		num_rows = new int;
		num_cols = new int;

	}
	
	void PostProcessOutputFile(string file)
	{		
		face_analyser->PostprocessOutputFile(file);
	}

	/*void AddNextFrame(cv::Mat & frame, List<System::Tuple<float, float>^>^ landmarks, bool success, bool online) {
			
		// Construct an OpenCV matric from the landmarks
		cv::Mat_<float> landmarks_mat(landmarks->Count * 2, 1, 0.0);
		for (int i = 0; i < landmarks->Count; ++i)
		{
			landmarks_mat.at<float>(i, 0) = landmarks[i]->Item1;
			landmarks_mat.at<float>(i + landmarks->Count, 0) = landmarks[i]->Item2;
		}

		//(captured_image, face_model.detected_landmarks, face_model.detection_success, sequence_reader.time_stamp, sequence_reader.IsWebcam());

		face_analyser->AddNextFrame(frame->Mat, landmarks_mat, success, 0, online);

		cv::Mat_<double> hog_d;
		face_analyser->GetLatestHOG(hog_d, *num_rows, *num_cols);
		hog_d.convertTo(*hog_features, CV_32F);
		
		face_analyser->GetLatestAlignedFace(*aligned_face);
				
	}*/
	
	// Predicting AUs from a single image
   pair<map<string, double>, map<string, double>>
		PredictStaticAUsAndComputeFeatures(cv::Mat & frame, vector<cv::Point2f> & landmarks)
	{
		
		// Construct an OpenCV matric from the landmarks
		cv::Mat_<float> landmarks_mat(landmarks.size() * 2, 1, 0.0);
		for (int i = 0; i < landmarks.size(); ++i)
		{
			landmarks_mat.at<float>(i, 0) = landmarks[i].x;
			landmarks_mat.at<float>(i + landmarks.size(), 0) = landmarks[i].y;
		}

		face_analyser->PredictStaticAUsAndComputeFeatures(frame, landmarks_mat);

		// Set the computed appearance features
		cv::Mat_<double> hog_tmp;
		face_analyser->GetLatestHOG(hog_tmp, *num_rows, *num_cols);
		hog_tmp.convertTo(*hog_features, CV_32F);

		face_analyser->GetLatestAlignedFace(*aligned_face);

		// Set the computed AUs
		auto AU_predictions_intensity = face_analyser->GetCurrentAUsReg();
		auto AU_predictions_occurence = face_analyser->GetCurrentAUsClass();

      map<string, double> au_intensities;
      map<string, double> au_occurences;

		for (auto p : AU_predictions_intensity)
		{
         string s(p.first.c_str());
         au_intensities[s] = p.second;
		}

		for (auto p : AU_predictions_occurence)
		{
         string s(p.first.c_str());
         au_occurences[s] = p.second;
		}

      pair<map<string, double>, map<string, double>> returnPair(au_intensities, au_occurences);
		return returnPair;

	}

	/*List<System::String^>^ GetClassActionUnitsNames()
	{
		auto names = face_analyser->GetAUClassNames();

		auto names_ret = gcnew List<System::String^>();

		for(std::string name : names)
		{
			names_ret->Add(gcnew System::String(name.c_str()));
		}

		return names_ret;

	}*/

   /*List<System::String^>^ GetRegActionUnitsNames()
	{
		auto names = face_analyser->GetAURegNames();

		auto names_ret = gcnew List<System::String^>();

		for(std::string name : names)
		{
			names_ret->Add(gcnew System::String(name.c_str()));
		}

		return names_ret;

	}*/

	/*Dictionary<System::String^, double>^ GetCurrentAUsClass()
	{
		auto classes = face_analyser->GetCurrentAUsClass();
		auto au_classes = gcnew Dictionary<System::String^, double>();

		for(auto p: classes)
		{
			au_classes->Add(gcnew System::String(p.first.c_str()), p.second);
		}
		return au_classes;
	}*/

	/*Dictionary<System::String^, double>^ GetCurrentAUsReg()
	{
		auto preds = face_analyser->GetCurrentAUsReg();
		auto au_preds = gcnew Dictionary<System::String^, double>();

		for(auto p: preds)
		{
			au_preds->Add(gcnew System::String(p.first.c_str()), p.second);
		}
		return au_preds;
	}*/

	/*OpenCVWrappers::RawImage^ GetLatestAlignedFace() {
		OpenCVWrappers::RawImage^ face_aligned_image = gcnew OpenCVWrappers::RawImage(*aligned_face);
		return face_aligned_image;
	}*/
	
	/*OpenCVWrappers::RawImage^ GetLatestHOGFeature() {
		OpenCVWrappers::RawImage^ HOG_feature = gcnew OpenCVWrappers::RawImage(*hog_features);
		return HOG_feature;
	}*/

	// As the number of HOG rows and columns might not be known in advance, have methods for querying them
	int GetHOGRows()
	{
		return *num_rows;
	}

	int GetHOGCols()
	{
		return *num_cols;
	}

	// The number of channels is always the same
	int GetHOGChannels()
	{
		return 31;
	}

	void Reset()
	{
		face_analyser->Reset();
	}

	~FaceAnalyserManaged()
	{
      if (hog_features != nullptr)
      {
         delete hog_features;
      }

      if (aligned_face != nullptr)
      {
         delete aligned_face;
      }

      if (num_cols != nullptr)
      {
         delete num_cols;
      }

      if (num_rows != nullptr)
      {
         delete num_rows;
      }

      if (face_analyser != nullptr)
      {
         delete face_analyser;
      }
   }

};

}

#endif