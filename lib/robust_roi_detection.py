"""
Robust ROI Detection for Behavioral Arenas (TRAP Box)

This module provides geometric-constraint-based detection of rectangular arenas
in behavioral neuroscience video recordings. It handles common challenges like:
- Rotated boxes
- Lighting artifacts that bisect the arena
- Reflections and spurious edges
- Variable camera angles

Key approach:
1. Hough line detection to find candidate edges
2. Geometric validation using known box dimensions
3. Region merging for split arenas
4. Multi-hypothesis ranking

Author: June (Molas Lab)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RotatedBoxROI:
    """Container for detected rotated box region of interest."""
    corners: np.ndarray  # 4x2 array of corner coordinates (ordered)
    center: Tuple[float, float]
    width: float  # Longer dimension
    height: float  # Shorter dimension
    angle: float  # Rotation angle in degrees
    confidence: float  # 0-1 confidence score
    method: str  # Detection method used
    
    def get_ordered_corners(self) -> np.ndarray:
        """Get corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
        corners = self.corners.copy()
        
        # Sort by y-coordinate to separate top and bottom
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        
        # Top two points (smaller y)
        top_points = sorted_by_y[:2]
        top_left = top_points[np.argmin(top_points[:, 0])]
        top_right = top_points[np.argmax(top_points[:, 0])]
        
        # Bottom two points (larger y)
        bottom_points = sorted_by_y[2:]
        bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
        bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the ROI."""
        corners = self.corners.reshape((-1, 1, 2)).astype(np.int32)
        result = cv2.pointPolygonTest(corners, (float(x), float(y)), False)
        return result >= 0
    
    def get_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a binary mask for this ROI."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        corners_int = self.corners.astype(np.int32)
        cv2.fillPoly(mask, [corners_int], 255)
        return mask
    
    def get_perspective_transform(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get perspective transform matrix to rectify the rotated box.
        
        Args:
            target_size: (width, height) of output. If None, uses detected dimensions.
            
        Returns:
            3x3 perspective transformation matrix
        """
        src = self.get_ordered_corners().astype(np.float32)
        
        if target_size is None:
            w, h = int(self.width), int(self.height)
        else:
            w, h = target_size
            
        dst = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        return cv2.getPerspectiveTransform(src, dst)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'corners': self.corners.tolist(),
            'center': list(self.center),
            'width': float(self.width),
            'height': float(self.height),
            'angle': float(self.angle),
            'confidence': float(self.confidence),
            'method': self.method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RotatedBoxROI':
        """Create from dictionary."""
        return cls(
            corners=np.array(data['corners']),
            center=tuple(data['center']),
            width=data['width'],
            height=data['height'],
            angle=data['angle'],
            confidence=data['confidence'],
            method=data.get('method', 'unknown')
        )


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_for_edges(image: np.ndarray, 
                         clahe_clip: float = 3.0,
                         clahe_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Preprocess image for robust edge detection under varied lighting.
    
    Args:
        image: BGR image
        clahe_clip: CLAHE clip limit
        clahe_grid: CLAHE tile grid size
        
    Returns:
        Enhanced grayscale image
    """
    # Convert to LAB for lighting-invariant processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    l_enhanced = clahe.apply(l_channel)
    
    # Also try individual color channels
    b, g, r = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Select channel with highest edge content (variance of Laplacian)
    channels = [l_enhanced, b, g, r, gray]
    edge_scores = [cv2.Laplacian(ch, cv2.CV_64F).var() for ch in channels]
    best_channel = channels[np.argmax(edge_scores)]
    
    # Apply additional enhancement
    enhanced = clahe.apply(best_channel)
    
    return enhanced


def detect_edges(image: np.ndarray,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 blur_size: int = 5) -> np.ndarray:
    """
    Multi-scale edge detection with morphological enhancement.
    
    Args:
        image: Grayscale image
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        blur_size: Gaussian blur kernel size
        
    Returns:
        Binary edge map
    """
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    
    # Multi-scale Canny
    edges1 = cv2.Canny(blurred, canny_low, canny_high)
    edges2 = cv2.Canny(blurred, canny_low // 2, canny_high // 2)
    
    # Combine
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Morphological closing to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges


# =============================================================================
# LINE DETECTION AND CLUSTERING
# =============================================================================

def detect_hough_lines(edges: np.ndarray,
                       rho: float = 1,
                       theta: float = np.pi / 180,
                       threshold: int = 50,
                       min_line_length: int = 50,
                       max_line_gap: int = 10) -> List[Dict]:
    """
    Detect lines using probabilistic Hough transform.
    
    Returns:
        List of line dictionaries with 'line', 'angle', 'length', 'midpoint' keys
    """
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)
    
    if lines is None:
        return []
    
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle (0-180 degrees)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        # Calculate length
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Midpoint
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        line_data.append({
            'line': (x1, y1, x2, y2),
            'angle': angle,
            'length': length,
            'midpoint': midpoint
        })
    
    return line_data


def cluster_lines_by_angle(lines: List[Dict],
                           angle_tolerance: float = 20.0,
                           min_lines_per_cluster: int = 3) -> Tuple[List[Dict], List[Dict], float]:
    """
    Cluster lines into two perpendicular groups at arbitrary orientations.
    
    This handles rotated boxes by finding the two dominant angle clusters
    that are roughly perpendicular to each other.
    
    Args:
        lines: List of line dictionaries
        angle_tolerance: Maximum angle deviation within a cluster (degrees)
        min_lines_per_cluster: Minimum lines required in each cluster
        
    Returns:
        (cluster1, cluster2, angle_between) - Two line clusters and their angular separation
    """
    if len(lines) < 4:
        return [], [], 0
    
    # Get angles and weights (by length)
    angles = np.array([l['angle'] for l in lines])
    lengths = np.array([l['length'] for l in lines])
    
    # Weight by length squared (longer lines are more reliable)
    weights = (lengths ** 2) / (lengths ** 2).sum()
    
    # Find dominant angles using finer histogram
    n_bins = 72  # 2.5 degree bins
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 180), weights=weights)
    
    # Smooth histogram to find robust peaks
    from scipy.ndimage import gaussian_filter1d
    try:
        hist_smooth = gaussian_filter1d(hist, sigma=2, mode='wrap')
    except ImportError:
        # Fallback: simple moving average
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        hist_smooth = np.convolve(hist, kernel, mode='same')
    
    # Find peaks by looking for local maxima
    peaks = []
    for i in range(n_bins):
        prev_idx = (i - 1) % n_bins
        next_idx = (i + 1) % n_bins
        if hist_smooth[i] > hist_smooth[prev_idx] and hist_smooth[i] > hist_smooth[next_idx]:
            if hist_smooth[i] > 0.01:  # Minimum peak height
                peak_angle = (bin_edges[i] + bin_edges[i + 1]) / 2
                peaks.append((hist_smooth[i], peak_angle))
    
    # Sort peaks by strength
    peaks.sort(reverse=True)
    
    logger.debug(f"Found {len(peaks)} angle peaks: {[(f'{p[1]:.1f}°', f'{p[0]:.3f}') for p in peaks[:5]]}")
    
    # Helper function for angular distance
    def angle_distance(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 180 - diff)
    
    # Find two peaks that are roughly perpendicular
    # Try all pairs in top peaks, not just consecutive
    dominant_angle1 = None
    dominant_angle2 = None
    best_perp_score = 0
    
    for i, (strength1, angle1) in enumerate(peaks[:8]):
        for j, (strength2, angle2) in enumerate(peaks[:8]):
            if i >= j:
                continue
            
            # Check perpendicularity
            angle_diff = angle_distance(angle1, angle2)
            
            # Score: prefer closer to 90° and stronger peaks
            if 75 <= angle_diff <= 105:
                perp_error = abs(angle_diff - 90)
                perp_score = (strength1 + strength2) * (1 - perp_error / 15)
                
                if perp_score > best_perp_score:
                    best_perp_score = perp_score
                    dominant_angle1 = angle1
                    dominant_angle2 = angle2
                    logger.debug(f"Found perpendicular candidate: {angle1:.1f}° and {angle2:.1f}° (diff: {angle_diff:.1f}°, score: {perp_score:.3f})")
    
    # If no perpendicular pair found, use strongest peak and assume perpendicular
    if dominant_angle1 is None and peaks:
        # Assume box edges - second cluster should be 90° from first
        dominant_angle2 = (dominant_angle1 + 90) % 180
        logger.debug(f"Inferred perpendicular angle: {dominant_angle2:.1f}°")
    
    if dominant_angle1 is None or dominant_angle2 is None:
        logger.warning("Could not find two dominant angle clusters")
        return [], [], 0
    
    # Assign lines to clusters with expanded tolerance
    cluster1 = []
    cluster2 = []
    
    for line_dict in lines:
        angle = line_dict['angle']
        
        # Angular distance accounting for wraparound at 180°
        def angle_distance(a1, a2):
            diff = abs(a1 - a2)
            return min(diff, 180 - diff)
        
        dist1 = angle_distance(angle, dominant_angle1)
        dist2 = angle_distance(angle, dominant_angle2)
        
        # Assign to closer cluster if within tolerance
        if dist1 < dist2 and dist1 < angle_tolerance:
            cluster1.append(line_dict)
        elif dist2 <= dist1 and dist2 < angle_tolerance:
            cluster2.append(line_dict)
    
    logger.debug(f"Cluster sizes after assignment: {len(cluster1)}, {len(cluster2)}")
    
    # If one cluster is too small, try to recover by expanding tolerance
    if len(cluster1) < min_lines_per_cluster or len(cluster2) < min_lines_per_cluster:
        expanded_tolerance = angle_tolerance * 1.5
        cluster1 = []
        cluster2 = []
        
        for line_dict in lines:
            angle = line_dict['angle']
            
            def angle_distance(a1, a2):
                diff = abs(a1 - a2)
                return min(diff, 180 - diff)
            
            dist1 = angle_distance(angle, dominant_angle1)
            dist2 = angle_distance(angle, dominant_angle2)
            
            if dist1 < dist2 and dist1 < expanded_tolerance:
                cluster1.append(line_dict)
            elif dist2 <= dist1 and dist2 < expanded_tolerance:
                cluster2.append(line_dict)
        
        logger.debug(f"Cluster sizes after expansion: {len(cluster1)}, {len(cluster2)}")
    
    # Calculate actual angle between clusters
    if cluster1 and cluster2:
        # Use length-weighted mean angle for each cluster
        # Handle 0/180 wraparound for angles near the boundary
        
        def weighted_mean_angle(cluster):
            """Compute weighted mean angle handling 0/180 wraparound"""
            angles = np.array([l['angle'] for l in cluster])
            weights = np.array([l['length'] for l in cluster])
            
            # Check if angles span the 0/180 boundary
            if angles.max() - angles.min() > 90:
                # Shift angles to avoid boundary: map 170-180 to -10-0
                angles = np.where(angles > 90, angles - 180, angles)
            
            mean_angle = np.average(angles, weights=weights)
            
            # Shift back to 0-180 range
            if mean_angle < 0:
                mean_angle += 180
            
            return mean_angle
        
        avg_angle1 = weighted_mean_angle(cluster1)
        avg_angle2 = weighted_mean_angle(cluster2)
        
        def angle_distance(a1, a2):
            diff = abs(a1 - a2)
            return min(diff, 180 - diff)
        
        angle_between = angle_distance(avg_angle1, avg_angle2)
        
        logger.info(f"Cluster angles: {avg_angle1:.1f}° and {avg_angle2:.1f}°, separation: {angle_between:.1f}°")
    else:
        angle_between = 90
    
    return cluster1, cluster2, angle_between


def find_extreme_lines(line_cluster: List[Dict],
                       percentile_low: float = 10,
                       percentile_high: float = 90,
                       min_length_ratio: float = 0.3) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find the two outermost lines in a cluster.
    
    Strategy:
    1. Sort lines by position (y for horizontal, x for vertical)
    2. Use percentiles to reject outliers
    3. From the extreme candidates, pick the longest lines
    
    Args:
        line_cluster: Lines with similar angles
        percentile_low: Lower percentile for outlier rejection
        percentile_high: Upper percentile for outlier rejection
        min_length_ratio: Minimum length as ratio of max length in cluster
        
    Returns:
        (low_extreme_line, high_extreme_line) or (None, None) if insufficient lines
    """
    if len(line_cluster) < 2:
        return None, None
    
    # Determine if cluster is mostly horizontal or vertical
    # Handle 0/180 wraparound
    angles = np.array([l['angle'] for l in line_cluster])
    if angles.max() - angles.min() > 90:
        angles = np.where(angles > 90, angles - 180, angles)
    avg_angle = np.mean(angles)
    if avg_angle < 0:
        avg_angle += 180
    
    is_horizontal = avg_angle < 45 or avg_angle > 135
    
    # Get max length for filtering
    max_length = max(l['length'] for l in line_cluster)
    min_length = max_length * min_length_ratio
    
    # Project lines onto perpendicular axis
    positions = []
    for line_dict in line_cluster:
        # Skip very short lines
        if line_dict['length'] < min_length:
            continue
            
        x1, y1, x2, y2 = line_dict['line']
        
        if is_horizontal:
            pos = (y1 + y2) / 2
        else:
            pos = (x1 + x2) / 2
        
        positions.append((pos, line_dict))
    
    if len(positions) < 2:
        # Fall back to using all lines
        positions = []
        for line_dict in line_cluster:
            x1, y1, x2, y2 = line_dict['line']
            pos = (y1 + y2) / 2 if is_horizontal else (x1 + x2) / 2
            positions.append((pos, line_dict))
    
    positions.sort(key=lambda x: x[0])
    n = len(positions)
    
    # Get position range
    pos_min = positions[0][0]
    pos_max = positions[-1][0]
    pos_range = pos_max - pos_min
    
    logger.debug(f"{'Horizontal' if is_horizontal else 'Vertical'} cluster: "
                 f"{n} lines, position range: {pos_min:.0f} to {pos_max:.0f}")
    
    # Find extreme lines using percentiles
    low_threshold = pos_min + pos_range * (percentile_low / 100)
    high_threshold = pos_min + pos_range * (percentile_high / 100)
    
    # Candidates are lines near the extremes (within threshold)
    low_candidates = [(pos, ld) for pos, ld in positions if pos <= low_threshold]
    high_candidates = [(pos, ld) for pos, ld in positions if pos >= high_threshold]
    
    # If no candidates in threshold region, use the actual extremes
    if not low_candidates:
        low_candidates = positions[:max(1, n // 10)]
    if not high_candidates:
        high_candidates = positions[max(0, n - n // 10):]
    
    # Select by: prefer longer lines, then more extreme position
    def select_best(candidates, prefer_extreme='low'):
        if not candidates:
            return None
        
        # Score: length * position_extremity
        best = None
        best_score = -1
        
        for pos, line_dict in candidates:
            length = line_dict['length']
            
            # Position score (how extreme is it)
            if prefer_extreme == 'low':
                pos_score = (pos_max - pos) / (pos_range + 1)
            else:
                pos_score = (pos - pos_min) / (pos_range + 1)
            
            # Combined score: length matters most, but position breaks ties
            score = length * (0.7 + 0.3 * pos_score)
            
            if score > best_score:
                best_score = score
                best = line_dict
        
        return best
    
    low_extreme = select_best(low_candidates, 'low')
    high_extreme = select_best(high_candidates, 'high')
    
    if low_extreme and high_extreme:
        low_pos = (low_extreme['line'][1] + low_extreme['line'][3]) / 2 if is_horizontal else \
                  (low_extreme['line'][0] + low_extreme['line'][2]) / 2
        high_pos = (high_extreme['line'][1] + high_extreme['line'][3]) / 2 if is_horizontal else \
                   (high_extreme['line'][0] + high_extreme['line'][2]) / 2
        logger.debug(f"Selected extremes: {low_pos:.0f} and {high_pos:.0f}")
    
    return low_extreme, high_extreme


# =============================================================================
# GEOMETRIC UTILITIES
# =============================================================================

def line_intersection(line1: Tuple[int, int, int, int],
                      line2: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Find intersection point of two line segments (extended to infinite lines).
    
    Args:
        line1, line2: Line segments as (x1, y1, x2, y2)
        
    Returns:
        Intersection point as numpy array [x, y], or None if parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    
    return np.array([px, py])


def compute_box_from_lines(h_lines: Tuple[Dict, Dict],
                           v_lines: Tuple[Dict, Dict],
                           image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Compute box corners from two pairs of perpendicular extreme lines.
    
    Args:
        h_lines: (top_line, bottom_line) dictionaries
        v_lines: (left_line, right_line) dictionaries
        image_shape: (height, width) of image for bounds checking
        
    Returns:
        4x2 array of corner points, or None if invalid
    """
    h1, h2 = h_lines
    v1, v2 = v_lines
    
    if any(x is None for x in [h1, h2, v1, v2]):
        return None
    
    # Compute all four intersections
    corners = []
    for h in [h1, h2]:
        for v in [v1, v2]:
            pt = line_intersection(h['line'], v['line'])
            if pt is not None:
                corners.append(pt)
    
    if len(corners) != 4:
        return None
    
    corners = np.array(corners)
    
    # Validate corners are within image bounds (with some margin)
    height, width = image_shape[:2]
    margin = 0.1  # 10% margin outside image is OK
    
    for corner in corners:
        if (corner[0] < -margin * width or corner[0] > (1 + margin) * width or
            corner[1] < -margin * height or corner[1] > (1 + margin) * height):
            return None
    
    # Order corners consistently (convex hull)
    hull = cv2.convexHull(corners.astype(np.float32))
    if len(hull) != 4:
        return None
    
    return hull.reshape(4, 2)


# =============================================================================
# GEOMETRIC VALIDATION
# =============================================================================

def validate_aspect_ratio(detected_width: float,
                          detected_height: float,
                          known_aspect_ratio: float,
                          tolerance: float = 0.20) -> Tuple[bool, float]:
    """
    Validate that detected box matches known aspect ratio.
    
    Args:
        detected_width, detected_height: Detected dimensions
        known_aspect_ratio: Expected width/height ratio
        tolerance: Maximum relative error allowed
        
    Returns:
        (is_valid, error) - Whether valid and the relative error
    """
    # Ensure width >= height for consistent comparison
    if detected_width < detected_height:
        detected_width, detected_height = detected_height, detected_width
    
    detected_ratio = detected_width / (detected_height + 1e-10)
    
    # Also consider inverse (in case known ratio is height/width)
    error1 = abs(detected_ratio - known_aspect_ratio) / (known_aspect_ratio + 1e-10)
    error2 = abs(detected_ratio - 1/known_aspect_ratio) / (1/known_aspect_ratio + 1e-10)
    
    error = min(error1, error2)
    is_valid = error < tolerance
    
    return is_valid, error


def validate_box_geometry(corners: np.ndarray,
                          image_shape: Tuple[int, int],
                          known_aspect_ratio: Optional[float] = None,
                          min_area_ratio: float = 1/6,
                          max_area_ratio: float = 0.85,
                          aspect_tolerance: float = 0.20) -> Tuple[bool, float, Dict]:
    """
    Comprehensive validation of detected box geometry.
    
    Args:
        corners: 4x2 array of corner coordinates
        image_shape: (height, width) of image
        known_aspect_ratio: Expected width/height ratio (if known)
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        
    Returns:
        (is_valid, confidence, details) - Validation result, confidence score, and detail dict
    """
    height, width = image_shape[:2]
    image_area = height * width
    
    details = {}
    
    # 1. Area check
    box_area = cv2.contourArea(corners.astype(np.float32))
    area_ratio = box_area / image_area
    details['area_ratio'] = area_ratio
    
    if area_ratio < min_area_ratio:
        details['failure'] = f'Area too small: {area_ratio:.2%} < {min_area_ratio:.2%}'
        return False, 0.0, details
    
    if area_ratio > max_area_ratio:
        details['failure'] = f'Area too large: {area_ratio:.2%} > {max_area_ratio:.2%}'
        return False, 0.0, details
    
    # 2. Get rotated rectangle properties
    rect = cv2.minAreaRect(corners.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    
    if w < h:
        w, h = h, w
        angle = (angle + 90) % 180
    
    details['width'] = w
    details['height'] = h
    details['angle'] = angle
    details['center'] = (cx, cy)
    
    # 3. Aspect ratio check
    detected_ratio = w / (h + 1e-10)
    details['detected_aspect_ratio'] = detected_ratio
    
    aspect_score = 1.0
    if known_aspect_ratio is not None:
        is_valid_aspect, aspect_error = validate_aspect_ratio(w, h, known_aspect_ratio, aspect_tolerance)
        details['aspect_error'] = aspect_error
        
        if not is_valid_aspect:
            details['failure'] = f'Aspect ratio mismatch: {detected_ratio:.2f} vs expected {known_aspect_ratio:.2f}'
            return False, 0.0, details
        
        aspect_score = 1.0 - aspect_error
    
    # 4. Corner angle check (should be ~90 degrees)
    angles = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(abs(angle_rad - np.pi/2))
    
    avg_angle_error = np.mean(angles)
    angle_score = np.exp(-avg_angle_error * 2)
    details['corner_angle_error'] = np.degrees(avg_angle_error)
    
    # 5. Compute confidence
    area_score = min(1.0, (area_ratio - min_area_ratio) / (0.5 - min_area_ratio + 1e-10))
    
    confidence = 0.35 * area_score + 0.35 * aspect_score + 0.30 * angle_score
    confidence = np.clip(confidence, 0, 1)
    
    details['area_score'] = area_score
    details['aspect_score'] = aspect_score
    details['angle_score'] = angle_score
    
    return True, confidence, details


# =============================================================================
# NEGATIVE SPACE DETECTION
# =============================================================================

def detect_box_from_negative_space(image: np.ndarray,
                                   lines: List[Dict],
                                   known_aspect_ratio: Optional[float] = None,
                                   min_area_ratio: float = 1/6,
                                   max_area_ratio: float = 0.85,
                                   aspect_tolerance: float = 0.25) -> Optional[np.ndarray]:
    """
    Detect box by finding the largest rectangular negative space (area not covered by lines).
    
    This works well when the arena interior is smooth but surrounded by textured areas
    that generate many Hough lines. Also handles split arenas by merging adjacent regions.
    
    Args:
        image: Original BGR image
        lines: Detected Hough lines
        known_aspect_ratio: Expected width/height ratio
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        
    Returns:
        4x2 array of corners, or None
    """
    height, width = image.shape[:2]
    image_area = height * width
    
    # Create a mask of areas covered by lines
    line_mask = np.zeros((height, width), dtype=np.uint8)
    
    for line_dict in lines:
        x1, y1, x2, y2 = line_dict['line']
        # Use line thickness proportional to line length
        thickness = max(3, int(line_dict['length'] / 30))
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness)
    
    # Dilate to connect nearby lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    line_mask = cv2.dilate(line_mask, kernel, iterations=2)
    
    # Invert to get negative space
    negative_space = cv2.bitwise_not(line_mask)
    
    # Clean up with morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    negative_space = cv2.morphologyEx(negative_space, cv2.MORPH_OPEN, kernel_open)
    
    # Find contours in negative space
    contours, _ = cv2.findContours(negative_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find all significant rectangular regions
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        # Need at least 5% to be considered
        if area_ratio < 0.05:
            continue
        
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        
        # Check rectangularity
        rectangularity = area / (box_area + 1e-10)
        
        # Get dimensions
        (cx, cy), (w, h), angle = rect
        if w < h:
            w, h = h, w
        
        aspect = w / (h + 1e-10)
        
        # Filter by shape quality
        if rectangularity < 0.5 or aspect > 5:
            continue
        
        candidates.append({
            'contour': contour,
            'box': box,
            'rect': rect,
            'area': area,
            'area_ratio': area_ratio,
            'rectangularity': rectangularity,
            'aspect': aspect
        })
    
    logger.debug(f"Negative space: found {len(candidates)} candidate regions")
    
    if not candidates:
        return None
    
    # Sort by area
    candidates.sort(key=lambda x: x['area'], reverse=True)
    
    # Check if top candidates should be merged (split arena case)
    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]
        
        # Get bounding boxes
        x1, y1, w1, h1 = cv2.boundingRect(c1['contour'])
        x2, y2, w2, h2 = cv2.boundingRect(c2['contour'])
        
        # Check horizontal overlap
        h_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        min_width = min(w1, w2)
        
        # Check vertical adjacency
        if y1 + h1 < y2:  # c1 above c2
            v_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:  # c2 above c1
            v_gap = y1 - (y2 + h2)
        else:  # overlapping
            v_gap = 0
        
        # Merge if horizontally overlapping and vertically close
        if h_overlap > 0.3 * min_width and v_gap < 100:
            logger.info(f"Merging split regions: h_overlap={h_overlap:.0f}, v_gap={v_gap:.0f}")
            
            merged_points = np.vstack([c1['contour'], c2['contour']])
            merged_hull = cv2.convexHull(merged_points)
            merged_rect = cv2.minAreaRect(merged_hull.astype(np.float32))
            merged_box = cv2.boxPoints(merged_rect)
            merged_area = cv2.contourArea(merged_box)
            
            area_ratio = merged_area / image_area
            
            (cx, cy), (mw, mh), angle = merged_rect
            if mw < mh:
                mw, mh = mh, mw
            
            # Validate merged region
            if min_area_ratio <= area_ratio <= max_area_ratio:
                merged_aspect = mw / (mh + 1e-10)
                
                # Check aspect ratio if known
                if known_aspect_ratio is not None:
                    aspect_error = min(
                        abs(merged_aspect - known_aspect_ratio) / known_aspect_ratio,
                        abs(merged_aspect - 1/known_aspect_ratio) / (1/known_aspect_ratio)
                    )
                    if aspect_error < aspect_tolerance:
                        logger.info(f"Merged region: {mw:.0f}x{mh:.0f}, area={area_ratio:.1%}, aspect={merged_aspect:.2f}")
                        return merged_box
                else:
                    logger.info(f"Merged region: {mw:.0f}x{mh:.0f}, area={area_ratio:.1%}")
                    return merged_box
    
    # Try single best candidate
    best = candidates[0]
    
    if best['area_ratio'] < min_area_ratio or best['area_ratio'] > max_area_ratio:
        return None
    
    # Check aspect ratio if known
    if known_aspect_ratio is not None:
        aspect_error = min(
            abs(best['aspect'] - known_aspect_ratio) / known_aspect_ratio,
            abs(best['aspect'] - 1/known_aspect_ratio) / (1/known_aspect_ratio)
        )
        if aspect_error > aspect_tolerance:
            return None
    
    logger.info(f"Negative space detection: area={best['area_ratio']:.1%}, aspect={best['aspect']:.2f}")
    
    return best['box']


# =============================================================================
# REGION MERGING FOR SPLIT ARENAS
# =============================================================================

def find_smooth_regions(image: np.ndarray,
                        edges: np.ndarray,
                        min_area_ratio: float = 0.05) -> List[np.ndarray]:
    """
    Find smooth (low-edge-density) regions that could be parts of the arena.
    
    Args:
        image: Original image
        edges: Binary edge map
        min_area_ratio: Minimum region size as fraction of image
        
    Returns:
        List of contours representing smooth regions
    """
    height, width = image.shape[:2]
    min_area = min_area_ratio * height * width
    
    # Invert edges to get non-edge regions
    smooth_mask = cv2.bitwise_not(edges)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            valid_regions.append(contour)
    
    return valid_regions


def are_regions_adjacent(rect1: Tuple, rect2: Tuple,
                         threshold: float = 100) -> Tuple[bool, str]:
    """
    Check if two rotated rectangles are adjacent (share an edge).
    
    Args:
        rect1, rect2: Output of cv2.minAreaRect
        threshold: Maximum gap between edges to consider adjacent
        
    Returns:
        (is_adjacent, shared_edge_direction) - 'horizontal' or 'vertical' or ''
    """
    box1 = cv2.boxPoints(rect1).astype(np.float32)
    box2 = cv2.boxPoints(rect2).astype(np.float32)
    
    # Check each edge pair
    for i in range(4):
        edge1_p1, edge1_p2 = box1[i], box1[(i + 1) % 4]
        edge1_mid = (edge1_p1 + edge1_p2) / 2
        edge1_dir = edge1_p2 - edge1_p1
        edge1_len = np.linalg.norm(edge1_dir)
        
        if edge1_len < 1:
            continue
            
        edge1_dir = edge1_dir / edge1_len
        
        for j in range(4):
            edge2_p1, edge2_p2 = box2[j], box2[(j + 1) % 4]
            edge2_mid = (edge2_p1 + edge2_p2) / 2
            edge2_dir = edge2_p2 - edge2_p1
            edge2_len = np.linalg.norm(edge2_dir)
            
            if edge2_len < 1:
                continue
                
            edge2_dir = edge2_dir / edge2_len
            
            # Check if edges are parallel
            cos_angle = abs(np.dot(edge1_dir, edge2_dir))
            if cos_angle < 0.9:  # Not parallel enough
                continue
            
            # Check if edges are close
            dist = np.linalg.norm(edge1_mid - edge2_mid)
            
            # Also check perpendicular distance
            perp_dist = abs(np.cross(edge1_dir, edge2_mid - edge1_mid))
            
            if dist < threshold and perp_dist < threshold:
                # Determine direction
                if abs(edge1_dir[0]) > abs(edge1_dir[1]):
                    return True, 'horizontal'
                else:
                    return True, 'vertical'
    
    return False, ''


def merge_adjacent_regions(regions: List[np.ndarray],
                           known_aspect_ratio: float,
                           image_shape: Tuple[int, int],
                           adjacency_threshold: float = 100,
                           aspect_tolerance: float = 0.20) -> Optional[np.ndarray]:
    """
    Merge adjacent rectangular regions if their combination matches expected geometry.
    
    This handles the case where lighting artifacts split the arena into two regions.
    
    Args:
        regions: List of contours
        known_aspect_ratio: Expected width/height ratio
        image_shape: (height, width) of image
        adjacency_threshold: Maximum gap to consider regions adjacent
        aspect_tolerance: Tolerance for aspect ratio validation
        
    Returns:
        Merged box corners as 4x2 array, or None if no valid merge found
    """
    if len(regions) < 2:
        return None
    
    # Sort by area (descending)
    regions_with_rect = [(r, cv2.minAreaRect(r)) for r in regions]
    regions_with_rect.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
    
    best_merge = None
    best_confidence = 0
    
    # Try merging pairs of largest regions
    for i in range(min(5, len(regions_with_rect))):
        region1, rect1 = regions_with_rect[i]
        
        for j in range(i + 1, min(5, len(regions_with_rect))):
            region2, rect2 = regions_with_rect[j]
            
            # Check adjacency
            is_adjacent, direction = are_regions_adjacent(rect1, rect2, adjacency_threshold)
            
            if not is_adjacent:
                continue
            
            logger.debug(f"Found adjacent regions {i} and {j}, shared edge: {direction}")
            
            # Compute enclosing rectangle
            combined_points = np.vstack([region1.reshape(-1, 2), region2.reshape(-1, 2)])
            enclosing_rect = cv2.minAreaRect(combined_points.astype(np.float32))
            enclosing_corners = cv2.boxPoints(enclosing_rect)
            
            # Validate merged geometry
            is_valid, confidence, details = validate_box_geometry(
                enclosing_corners,
                image_shape,
                known_aspect_ratio=known_aspect_ratio,
                aspect_tolerance=aspect_tolerance
            )
            
            if is_valid and confidence > best_confidence:
                best_merge = enclosing_corners
                best_confidence = confidence
                logger.info(f"Valid merge found with confidence {confidence:.2f}")
    
    return best_merge


# =============================================================================
# EDGE DENSITY DETECTION
# =============================================================================

def detect_box_from_edge_density(image: np.ndarray,
                                  known_aspect_ratio: Optional[float] = None,
                                  min_area_ratio: float = 1/6,
                                  max_area_ratio: float = 0.85,
                                  aspect_tolerance: float = 0.25,
                                  density_threshold: int = 30,
                                  kernel_size: int = 31) -> Optional[np.ndarray]:
    """
    Detect box by finding regions with low edge density.
    
    The arena interior typically has smooth surfaces with few edges,
    while the surrounding area has textures and objects that create many edges.
    This method finds the largest rectangular region of low edge density.
    
    Args:
        image: BGR image
        known_aspect_ratio: Expected width/height ratio
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        density_threshold: Threshold for "low" edge density (0-255)
        kernel_size: Size of the averaging kernel for density computation
        
    Returns:
        4x2 array of corners, or None
    """
    height, width = image.shape[:2]
    image_area = height * width
    
    # Compute edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute local edge density
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
    
    # Normalize
    if edge_density.max() > 0:
        edge_density_norm = (edge_density / edge_density.max() * 255).astype(np.uint8)
    else:
        return None
    
    # Threshold to find low-density regions
    _, low_density_mask = cv2.threshold(edge_density_norm, density_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    low_density_mask = cv2.morphologyEx(low_density_mask, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    low_density_mask = cv2.morphologyEx(low_density_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours
    contours, _ = cv2.findContours(low_density_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find candidate regions
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / image_area
        
        if area_ratio < min_area_ratio * 0.5:  # Allow smaller regions for potential merging
            continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        
        (cx, cy), (rw, rh), angle = rect
        if rw < rh:
            rw, rh = rh, rw
        
        aspect = rw / (rh + 1e-10)
        rect_fill = hull_area / (box_area + 1e-10)
        
        # Filter by shape quality
        if aspect > 5 or rect_fill < 0.5:
            continue
        
        candidates.append({
            'contour': cnt,
            'hull': hull,
            'box': box,
            'rect': rect,
            'area': area,
            'area_ratio': area_ratio,
            'aspect': aspect,
            'rect_fill': rect_fill
        })
    
    logger.debug(f"Edge density: found {len(candidates)} candidate regions")
    
    if not candidates:
        return None
    
    # Sort by area
    candidates.sort(key=lambda x: x['area'], reverse=True)
    
    # Check if top candidates should be merged (split arena case)
    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]
        
        x1, y1, w1, h1 = cv2.boundingRect(c1['hull'])
        x2, y2, w2, h2 = cv2.boundingRect(c2['hull'])
        
        h_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        min_width = min(w1, w2)
        
        # Calculate vertical gap
        if y1 + h1 < y2:
            v_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:
            v_gap = y1 - (y2 + h2)
        else:
            v_gap = 0
        
        # Merge if adjacent
        if h_overlap > 0.3 * min_width and v_gap < 100:
            logger.info(f"Edge density: merging split regions (v_gap={v_gap:.0f})")
            
            merged_points = np.vstack([c1['hull'], c2['hull']])
            merged_hull = cv2.convexHull(merged_points)
            merged_rect = cv2.minAreaRect(merged_hull.astype(np.float32))
            merged_box = cv2.boxPoints(merged_rect)
            merged_area_ratio = cv2.contourArea(merged_box) / image_area
            
            (cx, cy), (mw, mh), angle = merged_rect
            if mw < mh:
                mw, mh = mh, mw
            
            merged_aspect = mw / (mh + 1e-10)
            
            # Validate
            if min_area_ratio <= merged_area_ratio <= max_area_ratio:
                if known_aspect_ratio is not None:
                    aspect_error = min(
                        abs(merged_aspect - known_aspect_ratio) / known_aspect_ratio,
                        abs(merged_aspect - 1/known_aspect_ratio) / (1/known_aspect_ratio)
                    )
                    if aspect_error < aspect_tolerance:
                        logger.info(f"Edge density merged: {mw:.0f}x{mh:.0f}, area={merged_area_ratio:.1%}")
                        return merged_box
                else:
                    logger.info(f"Edge density merged: {mw:.0f}x{mh:.0f}, area={merged_area_ratio:.1%}")
                    return merged_box
    
    # Use best single candidate
    best = candidates[0]
    
    if best['area_ratio'] < min_area_ratio or best['area_ratio'] > max_area_ratio:
        # Still return for potential use
        if best['area_ratio'] >= min_area_ratio * 0.5:
            logger.info(f"Edge density partial: area={best['area_ratio']:.1%}, may need merging")
            return best['box']
        return None
    
    # Check aspect ratio
    if known_aspect_ratio is not None:
        aspect_error = min(
            abs(best['aspect'] - known_aspect_ratio) / known_aspect_ratio,
            abs(best['aspect'] - 1/known_aspect_ratio) / (1/known_aspect_ratio)
        )
        if aspect_error > aspect_tolerance:
            logger.debug(f"Edge density: aspect ratio mismatch ({best['aspect']:.2f} vs {known_aspect_ratio:.2f})")
            return None
    
    logger.info(f"Edge density detection: {best['area_ratio']:.1%}, aspect={best['aspect']:.2f}")
    return best['box']


# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================

def detect_box_from_hough_lines(image: np.ndarray,
                                known_aspect_ratio: Optional[float] = None,
                                min_area_ratio: float = 1/6,
                                max_area_ratio: float = 0.85,
                                aspect_tolerance: float = 0.20,
                                canny_low: int = 50,
                                canny_high: int = 150,
                                hough_threshold: int = 50,
                                min_line_length: int = 50,
                                visualize: bool = False) -> Tuple[Optional[RotatedBoxROI], Optional[np.ndarray]]:
    """
    Detect arena box using Hough line detection with geometric constraints.
    
    This is the primary detection method. It finds the outermost rectangular
    boundary formed by detected lines.
    
    Args:
        image: BGR image
        known_aspect_ratio: Expected width/height ratio (if known)
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        canny_low, canny_high: Canny edge detection thresholds
        hough_threshold: Hough transform accumulator threshold
        min_line_length: Minimum line length to detect
        visualize: Whether to generate visualization image
        
    Returns:
        (RotatedBoxROI or None, visualization_image or None)
    """
    height, width = image.shape[:2]
    
    # Preprocess
    gray = preprocess_for_edges(image)
    edges = detect_edges(gray, canny_low, canny_high)
    
    # Detect lines
    lines = detect_hough_lines(edges, threshold=hough_threshold, min_line_length=min_line_length)
    
    if len(lines) < 8:
        logger.warning(f"Insufficient lines detected: {len(lines)}")
        return None, None
    
    logger.info(f"Detected {len(lines)} Hough lines")
    
    # Cluster lines
    cluster1, cluster2, angle_between = cluster_lines_by_angle(lines)
    
    logger.info(f"Clustered into {len(cluster1)} and {len(cluster2)} lines, angle: {angle_between:.1f}°")
    
    if len(cluster1) < 2 or len(cluster2) < 2:
        logger.warning("Insufficient lines in clusters")
        return None, None
    
    # Check perpendicularity
    if not (65 <= angle_between <= 115):
        logger.warning(f"Line clusters not perpendicular: {angle_between:.1f}°")
        return None, None
    
    # Find extreme lines in each cluster
    extreme1 = find_extreme_lines(cluster1)
    extreme2 = find_extreme_lines(cluster2)
    
    if None in extreme1 or None in extreme2:
        logger.warning("Could not find extreme lines")
        return None, None
    
    # Compute box from line intersections
    corners = compute_box_from_lines(extreme1, extreme2, (height, width))
    
    if corners is None:
        logger.warning("Could not compute valid box from lines")
        return None, None
    
    # Validate geometry
    is_valid, confidence, details = validate_box_geometry(
        corners, (height, width),
        known_aspect_ratio=known_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        aspect_tolerance=aspect_tolerance
    )
    
    if not is_valid:
        logger.warning(f"Box validation failed: {details.get('failure', 'unknown')}")
        return None, None
    
    # Create ROI object
    roi = RotatedBoxROI(
        corners=corners,
        center=details['center'],
        width=details['width'],
        height=details['height'],
        angle=details['angle'],
        confidence=confidence,
        method='hough_lines'
    )
    
    logger.info(f"Detected box: {roi.width:.0f}x{roi.height:.0f}, "
                f"angle={roi.angle:.1f}°, confidence={roi.confidence:.2f}")
    
    # Generate visualization
    vis_image = None
    if visualize:
        vis_image = create_visualization(
            image, roi, cluster1, cluster2, extreme1, extreme2, details
        )
    
    return roi, vis_image


def detect_box_with_region_merging(image: np.ndarray,
                                   known_aspect_ratio: float,
                                   min_area_ratio: float = 1/6,
                                   max_area_ratio: float = 0.85,
                                   aspect_tolerance: float = 0.20,
                                   adjacency_threshold: float = 100,
                                   visualize: bool = False) -> Tuple[Optional[RotatedBoxROI], Optional[np.ndarray]]:
    """
    Fallback detection using region merging for split arenas.
    
    This handles cases where lighting artifacts bisect the arena into
    two separate regions.
    
    Args:
        image: BGR image
        known_aspect_ratio: Expected width/height ratio (required for merging)
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        adjacency_threshold: Maximum gap between regions to consider adjacent
        visualize: Whether to generate visualization image
        
    Returns:
        (RotatedBoxROI or None, visualization_image or None)
    """
    height, width = image.shape[:2]
    
    # Preprocess and detect edges
    gray = preprocess_for_edges(image)
    edges = detect_edges(gray)
    
    # Find smooth regions
    regions = find_smooth_regions(image, edges, min_area_ratio=0.05)
    
    logger.info(f"Found {len(regions)} smooth regions")
    
    if len(regions) < 1:
        return None, None
    
    # First, check if any single region is valid
    for region in regions:
        rect = cv2.minAreaRect(region.astype(np.float32))
        corners = cv2.boxPoints(rect)
        
        is_valid, confidence, details = validate_box_geometry(
            corners, (height, width),
            known_aspect_ratio=known_aspect_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            aspect_tolerance=aspect_tolerance
        )
        
        if is_valid and confidence > 0.7:
            roi = RotatedBoxROI(
                corners=corners,
                center=details['center'],
                width=details['width'],
                height=details['height'],
                angle=details['angle'],
                confidence=confidence,
                method='single_region'
            )
            
            vis_image = None
            if visualize:
                vis_image = image.copy()
                cv2.polylines(vis_image, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            
            return roi, vis_image
    
    # Try merging adjacent regions
    if len(regions) >= 2:
        merged_corners = merge_adjacent_regions(
            regions, known_aspect_ratio, (height, width),
            adjacency_threshold=adjacency_threshold,
            aspect_tolerance=aspect_tolerance
        )
        
        if merged_corners is not None:
            is_valid, confidence, details = validate_box_geometry(
                merged_corners, (height, width),
                known_aspect_ratio=known_aspect_ratio,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                aspect_tolerance=aspect_tolerance
            )
            
            if is_valid:
                roi = RotatedBoxROI(
                    corners=merged_corners,
                    center=details['center'],
                    width=details['width'],
                    height=details['height'],
                    angle=details['angle'],
                    confidence=confidence,
                    method='merged_regions'
                )
                
                vis_image = None
                if visualize:
                    vis_image = image.copy()
                    cv2.polylines(vis_image, [merged_corners.astype(np.int32)], True, (0, 255, 0), 3)
                    # Draw individual regions
                    for region in regions[:2]:
                        cv2.drawContours(vis_image, [region], -1, (255, 255, 0), 2)
                
                return roi, vis_image
    
    return None, None


def detect_trap_box_robust(image: np.ndarray,
                           known_width_cm: Optional[float] = None,
                           known_height_cm: Optional[float] = None,
                           min_area_ratio: float = 1/6,
                           max_area_ratio: float = 0.85,
                           aspect_tolerance: float = 0.20,
                           visualize: bool = False) -> Tuple[Optional[RotatedBoxROI], Optional[np.ndarray]]:
    """
    Robust TRAP box detection combining multiple methods.
    
    Detection pipeline:
    1. Try Hough line detection (primary method)
    2. If that fails, try negative space detection
    3. If that fails, try region merging (fallback)
    4. Return best result
    
    Args:
        image: BGR image
        known_width_cm: Physical box width in cm (optional but recommended)
        known_height_cm: Physical box height in cm (optional but recommended)
        min_area_ratio: Minimum box area as fraction of image
        max_area_ratio: Maximum box area as fraction of image
        aspect_tolerance: Tolerance for aspect ratio validation
        visualize: Whether to generate visualization image
        
    Returns:
        (RotatedBoxROI or None, visualization_image or None)
    """
    height, width = image.shape[:2]
    
    # Calculate known aspect ratio if dimensions provided
    known_aspect_ratio = None
    if known_width_cm is not None and known_height_cm is not None:
        known_aspect_ratio = max(known_width_cm, known_height_cm) / min(known_width_cm, known_height_cm)
        logger.info(f"Using known aspect ratio: {known_aspect_ratio:.2f}")
    
    best_roi = None
    best_vis = None
    
    # Preprocess once for all methods
    gray = preprocess_for_edges(image)
    edges = detect_edges(gray)
    lines = detect_hough_lines(edges, threshold=50, min_line_length=50)
    
    logger.info(f"Detected {len(lines)} Hough lines")
    
    # Method 1: Hough line detection with clustering
    logger.info("Attempting Hough line detection...")
    roi_hough, vis_hough = detect_box_from_hough_lines(
        image,
        known_aspect_ratio=known_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        aspect_tolerance=aspect_tolerance,
        visualize=visualize
    )
    
    if roi_hough is not None:
        logger.info(f"Hough detection: confidence={roi_hough.confidence:.2f}")
        if roi_hough.confidence > 0.7:
            return roi_hough, vis_hough
        best_roi = roi_hough
        best_vis = vis_hough
    
    # Method 2: Edge density detection (finds smooth arena interior)
    logger.info("Attempting edge density detection...")
    edge_corners = detect_box_from_edge_density(
        image,
        known_aspect_ratio=known_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        aspect_tolerance=aspect_tolerance
    )
    
    if edge_corners is not None:
        is_valid, confidence, details = validate_box_geometry(
            edge_corners, (height, width),
            known_aspect_ratio=known_aspect_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            aspect_tolerance=aspect_tolerance
        )
        
        if is_valid:
            roi_edge = RotatedBoxROI(
                corners=edge_corners,
                center=details['center'],
                width=details['width'],
                height=details['height'],
                angle=details['angle'],
                confidence=confidence,
                method='edge_density'
            )
            logger.info(f"Edge density detection: confidence={confidence:.2f}")
            
            if best_roi is None or confidence > best_roi.confidence:
                best_roi = roi_edge
                if visualize:
                    best_vis = image.copy()
                    cv2.polylines(best_vis, [edge_corners.astype(np.int32)], True, (0, 255, 0), 3)
                    cv2.putText(best_vis, f"Edge Density: {confidence:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Method 3: Negative space detection
    if len(lines) > 50:  # Only try if there are many lines
        logger.info("Attempting negative space detection...")
        neg_space_corners = detect_box_from_negative_space(
            image, lines,
            known_aspect_ratio=known_aspect_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            aspect_tolerance=aspect_tolerance
        )
        
        if neg_space_corners is not None:
            is_valid, confidence, details = validate_box_geometry(
                neg_space_corners, (height, width),
                known_aspect_ratio=known_aspect_ratio,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                aspect_tolerance=aspect_tolerance
            )
            
            if is_valid:
                roi_neg = RotatedBoxROI(
                    corners=neg_space_corners,
                    center=details['center'],
                    width=details['width'],
                    height=details['height'],
                    angle=details['angle'],
                    confidence=confidence,
                    method='negative_space'
                )
                logger.info(f"Negative space detection: confidence={confidence:.2f}")
                
                if best_roi is None or confidence > best_roi.confidence:
                    best_roi = roi_neg
                    if visualize:
                        best_vis = image.copy()
                        cv2.polylines(best_vis, [neg_space_corners.astype(np.int32)], True, (0, 255, 0), 3)
                        cv2.putText(best_vis, f"Negative Space: {confidence:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Method 3: Region merging (requires known aspect ratio)
    if known_aspect_ratio is not None and (best_roi is None or best_roi.confidence < 0.7):
        logger.info("Attempting region merging...")
        roi_merged, vis_merged = detect_box_with_region_merging(
            image,
            known_aspect_ratio=known_aspect_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            aspect_tolerance=aspect_tolerance,
            visualize=visualize
        )
        
        if roi_merged is not None:
            logger.info(f"Region merging: confidence={roi_merged.confidence:.2f}")
            if best_roi is None or roi_merged.confidence > best_roi.confidence:
                best_roi = roi_merged
                best_vis = vis_merged
    
    if best_roi is not None:
        logger.info(f"Best result: {best_roi.method} with confidence {best_roi.confidence:.2f}")
        return best_roi, best_vis
    
    logger.warning("All detection methods failed")
    return None, None


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(image: np.ndarray,
                         roi: RotatedBoxROI,
                         cluster1: List[Dict],
                         cluster2: List[Dict],
                         extreme1: Tuple[Optional[Dict], Optional[Dict]],
                         extreme2: Tuple[Optional[Dict], Optional[Dict]],
                         details: Dict) -> np.ndarray:
    """Create detailed visualization of detection results."""
    vis = image.copy()
    
    # Draw all lines in clusters (faded)
    for line_dict in cluster1:
        x1, y1, x2, y2 = line_dict['line']
        cv2.line(vis, (x1, y1), (x2, y2), (100, 100, 255), 1)
    
    for line_dict in cluster2:
        x1, y1, x2, y2 = line_dict['line']
        cv2.line(vis, (x1, y1), (x2, y2), (255, 100, 100), 1)
    
    # Draw extreme lines (bright)
    for line_dict in extreme1:
        if line_dict is not None:
            x1, y1, x2, y2 = line_dict['line']
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    for line_dict in extreme2:
        if line_dict is not None:
            x1, y1, x2, y2 = line_dict['line']
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    # Draw detected box
    corners = roi.get_ordered_corners().astype(np.int32)
    cv2.polylines(vis, [corners], True, (0, 255, 0), 3)
    
    # Draw corners with labels
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (corner, label) in enumerate(zip(corners, labels)):
        cv2.circle(vis, tuple(corner), 10, (0, 0, 255), -1)
        cv2.putText(vis, label, tuple(corner + np.array([15, 5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add info text
    info_lines = [
        f"Confidence: {roi.confidence:.2f}",
        f"Angle: {roi.angle:.1f} deg",
        f"Size: {roi.width:.0f} x {roi.height:.0f}",
        f"Area: {details['area_ratio']*100:.1f}% of frame",
        f"Method: {roi.method}"
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
    
    return vis


# =============================================================================
# CONVENIENCE FUNCTIONS FOR PIPELINE INTEGRATION
# =============================================================================

def detect_roi_from_video(video_path: str,
                          frame_index: int = 0,
                          known_width_cm: Optional[float] = None,
                          known_height_cm: Optional[float] = None,
                          **kwargs) -> Optional[RotatedBoxROI]:
    """
    Detect ROI from a video file.
    
    Args:
        video_path: Path to video file
        frame_index: Which frame to use for detection
        known_width_cm, known_height_cm: Physical box dimensions
        **kwargs: Additional arguments passed to detect_trap_box_robust
        
    Returns:
        RotatedBoxROI or None
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_index} from video")
    
    roi, _ = detect_trap_box_robust(
        frame,
        known_width_cm=known_width_cm,
        known_height_cm=known_height_cm,
        **kwargs
    )
    
    return roi


def detect_roi_from_background(background: np.ndarray,
                               known_width_cm: Optional[float] = None,
                               known_height_cm: Optional[float] = None,
                               **kwargs) -> Optional[RotatedBoxROI]:
    """
    Detect ROI from a pre-computed background image.
    
    Args:
        background: Background image (BGR)
        known_width_cm, known_height_cm: Physical box dimensions
        **kwargs: Additional arguments passed to detect_trap_box_robust
        
    Returns:
        RotatedBoxROI or None
    """
    roi, _ = detect_trap_box_robust(
        background,
        known_width_cm=known_width_cm,
        known_height_cm=known_height_cm,
        **kwargs
    )
    
    return roi


# =============================================================================
# WRAPPER CLASS FOR PIPELINE COMPATIBILITY
# =============================================================================

class TrapBoxDetector:
    """
    Wrapper class for ROI detection, providing a clean interface for the pipeline.

    Usage:
        detector = TrapBoxDetector(min_area_ratio=0.1)
        result = detector.detect_from_image(background_image)
    """

    def __init__(self,
                 min_area_ratio: float = 1/6,
                 canny_low: int = 30,
                 canny_high: int = 100,
                 known_width_cm: Optional[float] = None,
                 known_height_cm: Optional[float] = None):
        """
        Initialize detector with parameters.

        Args:
            min_area_ratio: Minimum ROI area as fraction of total image area
            canny_low: Canny edge detection low threshold (stored but not currently used)
            canny_high: Canny edge detection high threshold (stored but not currently used)
            known_width_cm: Known physical width of box in cm (optional)
            known_height_cm: Known physical height of box in cm (optional)
        """
        self.min_area_ratio = min_area_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.known_width_cm = known_width_cm
        self.known_height_cm = known_height_cm

    def detect_from_image(self, image: np.ndarray) -> Optional[RotatedBoxROI]:
        """
        Detect ROI from a single image (typically a background frame).

        Args:
            image: Input image (BGR format)

        Returns:
            RotatedBoxROI object or None if detection fails
        """
        return detect_roi_from_background(
            image,
            known_width_cm=self.known_width_cm,
            known_height_cm=self.known_height_cm,
            min_area_ratio=self.min_area_ratio
        )

    def detect_from_video(self, video_path: str,
                          sample_frames: int = 30) -> Optional[RotatedBoxROI]:
        """
        Detect ROI from a video file by sampling frames.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample for background generation

        Returns:
            RotatedBoxROI object or None if detection fails
        """
        return detect_roi_from_video(
            video_path,
            known_width_cm=self.known_width_cm,
            known_height_cm=self.known_height_cm,
            min_area_ratio=self.min_area_ratio,
            canny_low=self.canny_low,
            canny_high=self.canny_high
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect TRAP box ROI in behavioral video")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--width-cm", type=float, help="Known box width in cm")
    parser.add_argument("--height-cm", type=float, help="Known box height in cm")
    parser.add_argument("--min-area", type=float, default=1/6, help="Minimum area ratio")
    parser.add_argument("--output", help="Output visualization path")
    parser.add_argument("--json", help="Output JSON path for ROI data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load image
    if args.input.endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(args.input)
        ret, image = cap.read()
        cap.release()
    else:
        image = cv2.imread(args.input)
    
    if image is None:
        print(f"Error: Could not load {args.input}")
        exit(1)
    
    # Detect
    roi, vis = detect_trap_box_robust(
        image,
        known_width_cm=args.width_cm,
        known_height_cm=args.height_cm,
        min_area_ratio=args.min_area,
        visualize=True
    )
    
    if roi is None:
        print("Detection failed")
        exit(1)
    
    print(f"Detected box:")
    print(f"  Center: {roi.center}")
    print(f"  Size: {roi.width:.1f} x {roi.height:.1f}")
    print(f"  Angle: {roi.angle:.1f}°")
    print(f"  Confidence: {roi.confidence:.2f}")
    print(f"  Method: {roi.method}")
    
    if args.output and vis is not None:
        cv2.imwrite(args.output, vis)
        print(f"Saved visualization to {args.output}")
    
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(roi.to_dict(), f, indent=2)
        print(f"Saved ROI data to {args.json}")
