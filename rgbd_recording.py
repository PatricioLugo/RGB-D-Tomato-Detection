'''
Oak-D pro recording
'''

import depthai as dai
import cv2
import csv

from pathlib import Path
from datetime import timedelta


# Config

FPS = 10.0

RGB_SIZE = (1280, 720)
MONO_SIZE = (640, 400)

OUTPUT_DIR = Path("recording")

RGB_DIR = OUTPUT_DIR / "rgb"   
DEPTH_DIR = OUTPUT_DIR / "depth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RGB_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_DIR.mkdir(parents=True, exist_ok=True)


# DepthAi pipeline
pipeline = dai.Pipeline()

# Cameras

camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Stereo depth and sync

stereo = pipeline.create(dai.node.StereoDepth)

sync = pipeline.create(dai.node.Sync)

# Stereo configuration
stereo.setExtendedDisparity(True)

# Maximum timestamp difference allowed for Sync
sync.setSyncThreshold(
    timedelta(seconds=1 / (2 * FPS))
)


# Camera outputs config

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


# recording

frameIndex = 0

timestampsPath = OUTPUT_DIR / "timestamps.csv"

print("\nStarting RGB-D recording...")

try:

    with pipeline:

        pipeline.start()

        # open csv file for timestamps
        with open(
            timestampsPath,
            "w",
            newline=""
        ) as timestampFile:

            csvWriter = csv.writer(timestampFile)

            csvWriter.writerow([
                "frame_index",
                "rgb_filename",
                "depth_filename",
                "rgb_timestamp_seconds",
                "depth_timestamp_seconds",
                "timestamp_difference_seconds"
            ])

            try:

                while pipeline.isRunning():

                    # get synched group

                    messageGroup = queue.get()

                    frameRgb = messageGroup["rgb"]

                    frameDepth = messageGroup["depth_aligned"]

                    # convert to CV2 arrays

                    rgbFrame = frameRgb.getCvFrame()

                    depthFrame = frameDepth.getCvFrame()

                    # Create file names (same index)

                    rgbFilename = (
                        f"rgb_{frameIndex:06d}.jpg"
                    )

                    depthFilename = (
                        f"depth_{frameIndex:06d}.png"
                    )

                    rgbPath = RGB_DIR / rgbFilename

                    depthPath = DEPTH_DIR / depthFilename

                    # save RGB image

                    rgbSaved = cv2.imwrite(
                        str(rgbPath),
                        rgbFrame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

                    # save depth image

                    depthSaved = cv2.imwrite(
                        str(depthPath),
                        depthFrame
                    )

                    if not rgbSaved or not depthSaved:
                        raise RuntimeError(
                            f"Failed to save frame {frameIndex}"
                        )

                    # get timestamps

                    rgbTimestamp = (
                        frameRgb
                        .getTimestampDevice()
                        .total_seconds()
                    )

                    depthTimestamp = (
                        frameDepth
                        .getTimestampDevice()
                        .total_seconds()
                    )

                    timestampDifference = abs(
                        rgbTimestamp - depthTimestamp
                    )

                    # wite metadata for csv file

                    csvWriter.writerow([
                        frameIndex,
                        rgbFilename,
                        depthFilename,
                        rgbTimestamp,
                        depthTimestamp,
                        timestampDifference
                    ])

                    # counter

                    frameIndex += 1

                    if frameIndex % 100 == 0:

                        print(
                            f"Recorded {frameIndex} RGB-D pairs"
                        )

            except KeyboardInterrupt:

                print("\nStopping recording...")

finally:

    print(f"\nTotal RGB-D pairs saved: {frameIndex}")

    print(f"RGB images: {RGB_DIR}")

    print(f"Depth images: {DEPTH_DIR}")

    print(f"Timestamps: {timestampsPath}")

    print("Recording finished.")