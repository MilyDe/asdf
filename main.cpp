//input/output stream
#include<iostream> 
#include <cstdlib> // For rand() function
////Headder files from opencv
#include <opencv2/opencv.hpp> // Include the core OpenCV functionalities and convenience functions
#include <opencv2/imgcodecs.hpp> // Include functions for reading from and writing to image files
#include <opencv2/highgui.hpp> // Include functions for creating windows and displaying images, and capturing video
#include <opencv2/imgproc.hpp> // Include image processing functions such as filtering, transformations, and drawing
#include <opencv2/objdetect.hpp> // Include functions for object detection -> Haar Cascade classifiers (used in this code)
#include <opencv2/core.hpp>

#include <string>

//include opencv core
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <direct.h>

//file handling
#include <fstream>
#include <sstream>


//file handling
#include <fstream>
#include <sstream>

using namespace cv;  // Import the entire OpenCV namespace to simplify code to use direct functions and classes without prefixing with cv::
using namespace std; // Import the entire standard library namespace to simplify code to use direct standard library functions and classes without prefixing with std::

// Declaation of class for video capture
class VideoCaptureHandler {
private:
    VideoCapture video; // Initialize video capture object to capture video from a video file or directly from a camera device
    CascadeClassifier faceCascade; // Declare Cascade Classifier object for face detection

public:
    // Constructor for video capture
    VideoCaptureHandler(int cameraIndex, const string& faceCascadePath) {
        video.open(cameraIndex);
        // Check if camera and model is correctly loaded if not show error
        if (!video.isOpened()) { 
            cout << "Error: Could not open the camera." << endl;
            exit(-1);
        }

        if (!faceCascade.load(faceCascadePath)) {
            cout << "Error: Could not load face cascade." << endl;
            exit(-1);
        }
    }
    // Destructor for video capture
    ~VideoCaptureHandler() {
        video.release();
        destroyAllWindows();
    }
    // Function to capture and display video recorded by camera
    void captureAndDisplay() {
        // Declare a matrix (Mat) to hold the current frame captured from the video
        Mat frame;
        // Frames are continuously read from the camera in a while (true) loop
        while (true) {
            video.read(frame);
            // Declare a vector to hold rectangles representing the bounding boxes of detected faces
            vector<Rect> faces;
            // Faces are detected in each frame using faceCascade.detectMultiScale
            faceCascade.detectMultiScale(frame, faces, 1.1, 10, 0, Size(30, 30));
            // Prints out in the console how many faces are found
            cout << faces.size() << endl;

            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            // Rectangles are drawn around the detected faces
            for (size_t i = 0; i < faces.size(); ++i) {
                // Draw rectangles around faces
                rectangle(frame, faces[i], color, 3);
                // Draw a filled rectangle 
                rectangle(frame,Point(0,0), Point(280,70), Scalar(128, 0, 128), FILLED);
                // Write text in the frame
                putText(frame, to_string(faces.size())+" Face Found", Point(20,40), FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,255, 1));
            }
            // Show the frame
            imshow("Camera Feed", frame);
            // Exit the loop if the user presses 'q'
            if (waitKey(1) == 'q') {
                break;
            }
        }
    }
};

int main() {
    // Load model for cascade classifier
    const string faceCascadePath = "haarcascade_frontalface_default.xml";
    // Create a VideoCaptureHandler object with camera index 0 and the specified cascade classifier path
    VideoCaptureHandler captureHandler(0, faceCascadePath);
    // Capture and display video while detecting faces
    captureHandler.captureAndDisplay();
    return 0;
}
// #include <iostream>
// #include <string>
// #include <vector>
// #include <opencv2/core.hpp>
// #include <opencv2/face.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/objdetect.hpp>
// #include <opencv2/opencv.hpp>
// #include <direct.h>
// #include <fstream>
// #include <sstream>

// using namespace std;
// using namespace cv;
// using namespace cv::face;

// class FaceRecognition {
// private:
//     CascadeClassifier face_cascade;
//     string filename;
//     string name;
//     int filenumber = 0;

//     void detectAndDisplay(Mat frame) {
//         vector<Rect> faces;
//         Mat frame_gray;
//         Mat crop;
//         Mat res;
//         Mat gray;
//         string text;
//         stringstream sstm;

//         cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
//         equalizeHist(frame_gray, frame_gray);

//         face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

//         Rect roi_b;
//         Rect roi_c;

//         size_t ic = 0;
//         int ac = 0;

//         size_t ib = 0;
//         int ab = 0;

//         for (ic = 0; ic < faces.size(); ic++) {
//             roi_c.x = faces[ic].x;
//             roi_c.y = faces[ic].y;
//             roi_c.width = faces[ic].width;
//             roi_c.height = faces[ic].height;

//             ac = roi_c.width * roi_c.height;

//             roi_b.x = faces[ib].x;
//             roi_b.y = faces[ib].y;
//             roi_b.width = faces[ib].width;
//             roi_b.height = faces[ib].height;

//             crop = frame(roi_b);
//             resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);
//             cvtColor(crop, gray, COLOR_BGR2GRAY);
//             stringstream ssfn;
//             filename = "C:\\Users\\Asus\\Desktop\\Faces\\";
//             ssfn << filename.c_str() << name << filenumber << ".jpg";
//             filename = ssfn.str();
//             imwrite(filename, res);
//             filenumber++;

//             Point pt1(faces[ic].x, faces[ic].y);
//             Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
//             rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
//         }

//         sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
//         text = sstm.str();

//         if (!crop.empty()) {
//             imshow("detected", crop);
//         } else {
//             destroyWindow("detected");
//         }
//     }

//     static void dbread(vector<Mat>& images, vector<int>& labels) {
//         vector<cv::String> fn;
//         string filename = "C:\\Users\\Asus\\Desktop\\Faces\\";
//         glob(filename, fn, false);

//         size_t count = fn.size();

//         for (size_t i = 0; i < count; i++) {
//             string itsname = "";
//             char sep = '\\';
//             size_t j = fn[i].rfind(sep, fn[i].length());
//             if (j != string::npos) {
//                 itsname = fn[i].substr(j + 1, fn[i].length() - j - 6);
//             }
//             images.push_back(imread(fn[i], 0));
//             labels.push_back(atoi(itsname.c_str()));
//         }
//     }

// public:
//     void addFace() {
//         cout << "\nEnter Your Name: ";
//         cin >> name;

//         VideoCapture capture(0);

//         if (!capture.isOpened())
//             return;

//         if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
//             cout << "error" << endl;
//             return;
//         };

//         Mat frame;
//         cout << "\nCapturing your face 10 times, Press 'C' 10 times keeping your face front of the camera";
//         char key;
//         int i = 0;

//         for (;;) {
//             capture >> frame;
//             detectAndDisplay(frame);
//             i++;
//             if (i == 10) {
//                 cout << "Face Added";
//                 break;
//             }
//             int c = waitKey(10);
//             if (27 == char(c)) {
//                 break;
//             }
//         }
//     }

//     void eigenFaceTrainer() {
//         vector<Mat> images;
//         vector<int> labels;
//         dbread(images, labels);
//         cout << "size of the images is " << images.size() << endl;
//         cout << "size of the labels is " << labels.size() << endl;
//         cout << "Training begins...." << endl;

//         Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();

//         model->train(images, labels);

//         model->save("eigenfaces.yml");

//         cout << "Training finished...." << endl;
//         waitKey(10000);
//     }

//     void faceRecognition() {
//         cout << "start recognizing..." << endl;

//         Ptr<FaceRecognizer> model = FisherFaceRecognizer::create();
//         model->read("C:\\Users\\Asus\\Desktop\\eigenface.yml");

//         Mat testSample = imread("C:\\Users\\Asus\\Desktop\\0.jpg", 0);

//         int img_width = testSample.cols;
//         int img_height = testSample.rows;

//         string window = "Capture - face detection";

//         if (!face_cascade.load("C:\\Users\\Asus\\Downloads\\opencv4.1\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")) {
//             cout << " Error loading file" << endl;
//             return;
//         }

//         VideoCapture cap(0);

//         if (!cap.isOpened()) {
//             cout << "exit" << endl;
//             return;
//         }

//         namedWindow(window, 1);
//         long count = 0;
//         string Pname = "";

//         while (true) {
//             vector<Rect> faces;
//             Mat frame;
//             Mat graySacleFrame;
//             Mat original;

//             cap >> frame;
//             count++;

//             if (!frame.empty()) {
//                 original = frame.clone();
//                 cvtColor(original, graySacleFrame, COLOR_BGR2GRAY);

//                 face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, Size(90, 90));

//                 int width = 0, height = 0;

//                 for (int i = 0; i < faces.size(); i++) {
//                     Rect face_i = faces[i];

//                     Mat face = graySacleFrame(face_i);
//                     Mat face_resized;
//                     resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

//                     int label = -1;
//                     double confidence = 0;
//                     model->predict(face_resized, label, confidence);

//                     cout << " confidence " << confidence << " Label: " << label << endl;

//                     Pname = to_string(label);

//                     rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
//                     string text = Pname;

//                     int pos_x = max(face_i.tl().x - 10, 0);
//                     int pos_y = max(face_i.tl().y - 10, 0);

//                     putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
//                 }

//                 putText(original, "Frames: " + to_string(count), Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
//                 putText(original, "No. of Persons detected: " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
//                 imshow(window, original);
//             }
//             if (waitKey(30) >= 0) break;
//         }
//     }
// };

// int main(){
//     FaceRecognition Fr;
//     int choice;
//     cout << "1. Recognise Face\n";
// 	cout << "2. Add Face\n";
// 	cout << "Choose One: ";
// 	cin >> choice;
// 	switch (choice)
// 	{
// 	case 1:
// 		Fr.faceRecognition();
// 		break;
// 	case 2:
// 		Fr.addFace();
// 		Fr.eigenFaceTrainer();
// 		break;
// 	default:
// 		return 0;
// 	}
//     //system("pause");
// 	return 0;
// }