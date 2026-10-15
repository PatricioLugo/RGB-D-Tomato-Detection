'''
Oak-D Pro visualization (rgb & depth frames)
'''

import depthai as dai
import cv2


# Configuration

FPS = 30.0

RGB_SIZE = (1280, 720)
MONO_SIZE = (640, 400)

# DepthAi pipeline

pipeline = dai.Pipeline()

# Cameras

camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Stereo depth and sync

stereo = pipeline.create(dai.node.StereoDepth)

sync = pipeline.create(dai.node.Sync)

stereo.setExtendedDisparity(True)

# Camera outputs

rgbOut = camRgb.requestOutput(size=RGB_SIZE, fps=FPS, enableUndistortion=True)

leftOut = left.requestOutput(size=MONO_SIZE, fps=FPS)

rightOut = right.requestOutput(size=MONO_SIZE, fps=FPS)


# Linking

rgbOut.link(sync.inputs["rgb"])

leftOut.link(stereo.left)
rightOut.link(stereo.right)

# depth align (specific to RV2)

stereo.depth.link(sync.inputs["depth_aligned"])

rgbOut.link(stereo.inputAlignTo)

# Synched output queu

queue = sync.out.createOutputQueue()


rgbWeight = 0.5
depthWeight = 0.5

# Device connection and pipeline start
with pipeline:
    pipeline.start()

    # Window configuration
    windowName = "rgb-depth"

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1280, 720)

    while True:
        messageGroup = queue.get()

        frameRgb = messageGroup["rgb"]
        frameDepth = messageGroup["depth_aligned"]

        # Convert to CV2 arrays

        rgbFrame = frameRgb.getCvFrame()
        depthFrame = frameDepth.getCvFrame()

        cvFrame = frameRgb.getCvFrame()

        # Colorize the aligned depth
        alignedDepthColorized = dai.utility.colorizeDepthFrame(frameDepth).getCvFrame()

        # Blend rgb and depth
        if len(cvFrame.shape) == 2:
            cvFrameUndistorted = cv2.cvtColor(cvFrame, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(
            cvFrame, rgbWeight, alignedDepthColorized, depthWeight, 0
        )

        cv2.imshow(windowName, blended)


        key = cv2.waitKey(1)
        if key == ord("q"):
            break