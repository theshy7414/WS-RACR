import logging
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)

class RealSenseCamera:
    def __init__(self, width = 1280, height = 720, fps = 30, serial = None, extrinsics = None):
        # Realsense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial is not None:
            self.config.enable_device(serial)
        # Video info
        self.width = width
        self.height = height
        self.fps = fps
        # Enable stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        color_stream = self.profile.get_stream(rs.stream.color)
        
        # Parameters
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.extrinsics = extrinsics
        # Filters
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        # Colorizer
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.min_distance, 0.5) # (m)
        self.colorizer.set_option(rs.option.max_distance, 2.5) # (m)

        # Frames
        self.frames = None
        self.aligned_frames = None
        # Color
        self.color_frame = None
        self.color_image = None
        # Depth
        self.depth_frame = None
        self.depth_image = None

        logger.info("RealSense camera started.")
    
    def wait_for_frames(self):

        def apply_filter(depth_frame):
            """對depth_frame套用 spatial、temporal 與 hole filling 濾波器"""
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
            return depth_frame.as_depth_frame()

        # Frames
        self.frames = self.pipeline.wait_for_frames()
        # Aligned frames
        self.aligned_frames = self.align.process(self.frames)
        # Color frame & image
        self.color_frame = self.aligned_frames.get_color_frame()
        self.color_image = np.asanyarray(self.color_frame.get_data())
        # Filterd depth frame & image
        self.depth_frame = apply_filter(self.aligned_frames.get_depth_frame())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        
    def get_depth_at(self, u, v):
        if self.depth_frame is None:
            return None
        return self.depth_frame.get_distance(u, v)

    def get_colored_depth(self):
        """colorize depth_frame"""
        colored_depth_frame = self.colorizer.colorize(self.depth_frame)
        colored_depth_image = np.asanyarray(colored_depth_frame.get_data())
        return colored_depth_image

    def deproject_image_to_camera(self, u, v):
        """image points (u, v) 反投影成 world points (X, Y, Z)"""
        z = self.get_depth_at(u, v)
        if z is None or z <= 0:
            return None
        xyz = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z)
        return xyz
    
    def deproject_image_to_world(self, u, v):
        xyz = self.deproject_image_to_camera(u, v)
        if xyz is None or self.extrinsics is None:
            return None

        Rmat = self.extrinsics[:, :3]
        Tvec = self.extrinsics[:, 3]
        XYZ = (Rmat @ xyz + Tvec).tolist()
        return XYZ

    def release(self):
        """停止串流並釋放資源"""
        self.pipeline.stop()
        logger.info("RealSense camera stopped.")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

