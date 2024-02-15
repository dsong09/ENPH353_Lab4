#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np


class My_App(QtWidgets.QMainWindow):

	def __init__(self):

		self.sift = cv2.SIFT_create() 
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		# Initialize camera and processing related variables
		self._cam_id = 0
		self._cam_fps = 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		# Connect UI buttons to their respective slots
		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		# Set up the camera device with specified resolution
		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():

			# Load the selected image as the template for feature matching
			self.template_path = dlg.selectedFiles()[0]
			self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)  # Make sure to load as grayscale

			if self.template_img is not None:
				# Detect and compute SIFT features in the template image
				self.template_kp, self.template_des = self.sift.detectAndCompute(self.template_img, None) #twoforone
				self.template_img_with_kp = cv2.drawKeypoints(self.template_img, self.template_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

				# Display the template image in the UI
				pixmap = self.convert_cv_to_pixmap(self.template_img)
				self.template_label.setPixmap(pixmap.scaled(self.template_label.size(), QtCore.Qt.KeepAspectRatio))
				self._is_template_loaded = True
				print("Loaded template image file: " + self.template_path)
			
			else:
				print("Nothing to SIFT")

	def convert_cv_to_pixmap(self, cv_img):

		# Conversion process
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):

		# Capture frame from camera
		ret, frame = self._camera_device.read()

		# Feature Matching:
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		if ret:

			gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

			#sift = cv2.SIFT_create()
			kp, des = self.sift.detectAndCompute(gray,None)
			
			matches = flann.knnMatch(self.template_des, des, k=2)

			good_keypoints = []

			for m, n in matches:
				#ratio test
				if m.distance < 0.6*n.distance:
					good_keypoints.append(m)
	
			frame_with_matches = cv2.drawMatches(self.template_img, self.template_kp, gray, kp, good_keypoints, gray)
			pixmap = self.convert_cv_to_pixmap(frame_with_matches)
					
			# Homography
			
			if len(good_keypoints) > 10:
				query_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_keypoints]).reshape(-1, 1, 2)
				train_pts = np.float32([kp[m.trainIdx].pt for m in good_keypoints]).reshape(-1, 1, 2)

				matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
				if matrix is not None:
					matches_mask = mask.ravel().tolist()

					height, width = self.template_img.shape[:2] 
					pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
					dst = cv2.perspectiveTransform(pts, matrix)

					homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
					pixmap = self.convert_cv_to_pixmap(homography)
					self.live_image_label.setPixmap(pixmap)
			else:
				# Ensure template image is suitable for drawing matches (e.g., converted to color if grayscale)
				frame_with_matches = cv2.drawMatches(self.template_img, self.template_kp, frame, kp, good_keypoints, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
				pixmap = self.convert_cv_to_pixmap(frame_with_matches)
				self.live_image_label.setPixmap(pixmap)

		else:
			print("Failed to SIFT frame")

	def SLOT_toggle_camera(self):

		# Toggle camera functionality based on its current state
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")



if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())