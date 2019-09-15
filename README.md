# Perceptual-User-Interface-by-Hand-and-Finger-Tips-Tracking-to-AR-Objects
Developed a novel way of using hand gesture to control the interaction of AR objects. Here the movements of the fingers act as an interface for bringing out motion of an AR object. As gesture by human hands has been on of the key communication modes through our human lives this implementation aims at bringing virtual reality experience closer to our perception.



**Current Implementation**
* Hand segmentation:
  * Skin Detection using HSV color space
  * Placing a Contour on the hand
* Finger Detection
  * Convex Hull approach
* Finger Tip as an Input
  * Finger count
* Camera Calibration
* 3D Cube Augmentation on Chessboard

**Future Works**
* Augmentation of stabilized 3d world objects on live image capture
* Augmented Object viewing from different angles
* Placing AR object on any flat surface instead of having chessboard
* Better hand segmentation and optical flow using KLT algorithm with analysis to stabilize hand detection and finger tracking
* In addition to finger count, utilization of finger motion itself as an input
