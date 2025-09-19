"""
Feature extractors for different observation types.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


def estimate_lane_from_position(y_position: float, lane_width: float = 4.0, num_lanes: int = 4) -> int:
    """
    Estimate lane number from lateral position.
    
    Args:
        y_position: Lateral position of vehicle
        lane_width: Width of each lane in meters
        num_lanes: Total number of lanes
        
    Returns:
        Estimated lane number (0-indexed from rightmost lane)
    """
    # Assume lanes are centered around y=0
    # For 4 lanes with 4m width: lane boundaries at [-8, -4, 0, 4, 8]
    # Lane 0 (rightmost): y in [-8, -4), Lane 1: y in [-4, 0), etc.
    
    # Calculate total road width and boundaries
    total_width = num_lanes * lane_width
    right_boundary = -total_width / 2
    
    # Shift coordinate system so rightmost lane starts at y=0
    shifted_y = y_position - right_boundary
    
    # Calculate lane index
    lane = int(shifted_y / lane_width)
    
    # Clamp to valid lane range
    return max(0, min(lane, num_lanes - 1))


def detect_lead_vehicle(ego_state: np.ndarray, other_vehicles: np.ndarray, 
                       same_lane_threshold: float = 2.0) -> Tuple[Optional[int], float]:
    """
    Detect lead vehicle in the same lane and calculate gap.
    
    Args:
        ego_state: Ego vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
        other_vehicles: Array of other vehicle states, shape (n_vehicles, 7)
        same_lane_threshold: Threshold for considering vehicles in same lane (meters)
        
    Returns:
        Tuple of (lead_vehicle_index, gap_distance). Returns (None, float('inf')) if no lead vehicle
    """
    if len(other_vehicles) == 0:
        return None, float('inf')
    
    ego_x, ego_y = ego_state[1], ego_state[2]
    
    # Filter vehicles that are ahead and in similar lane
    ahead_vehicles = []
    for i, vehicle in enumerate(other_vehicles):
        if vehicle[0] == 0:  # Skip if vehicle not present
            continue
            
        v_x, v_y = vehicle[1], vehicle[2]
        
        # Check if vehicle is ahead (positive x direction)
        if v_x > ego_x:
            # Check if in same lane (similar y position)
            if abs(v_y - ego_y) <= same_lane_threshold:
                distance = v_x - ego_x
                ahead_vehicles.append((i, distance))
    
    if not ahead_vehicles:
        return None, float('inf')
    
    # Find closest vehicle ahead
    lead_idx, min_distance = min(ahead_vehicles, key=lambda x: x[1])
    return lead_idx, min_distance


def calculate_time_to_collision(ego_state: np.ndarray, other_state: np.ndarray) -> float:
    """
    Calculate Time-to-Collision between ego vehicle and another vehicle.
    
    Args:
        ego_state: Ego vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
        other_state: Other vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
        
    Returns:
        Time to collision in seconds, or float('inf') if no collision predicted
    """
    if ego_state[0] == 0 or other_state[0] == 0:  # Either vehicle not present
        return float('inf')
    
    # Extract positions and velocities
    ego_pos = ego_state[1:3]  # [x, y]
    ego_vel = ego_state[3:5]  # [vx, vy]
    other_pos = other_state[1:3]  # [x, y]
    other_vel = other_state[3:5]  # [vx, vy]
    
    # Relative position and velocity
    rel_pos = other_pos - ego_pos
    rel_vel = other_vel - ego_vel
    
    # If relative velocity is zero or vehicles are moving apart, no collision
    rel_speed_squared = np.dot(rel_vel, rel_vel)
    if rel_speed_squared < 1e-6:  # Essentially zero relative velocity
        return float('inf')
    
    # Check if vehicles are approaching each other
    if np.dot(rel_pos, rel_vel) >= 0:  # Moving apart or parallel
        return float('inf')
    
    # Calculate time to closest approach
    t_closest = -np.dot(rel_pos, rel_vel) / rel_speed_squared
    
    if t_closest <= 0:  # Closest approach is in the past
        return float('inf')
    
    # Calculate minimum distance at closest approach
    closest_pos = rel_pos + rel_vel * t_closest
    min_distance = np.linalg.norm(closest_pos)
    
    # Assume collision if minimum distance is less than vehicle width (approximately 2 meters)
    collision_threshold = 2.0
    if min_distance > collision_threshold:
        return float('inf')
    
    return t_closest


def calculate_traffic_density(observations: np.ndarray, ego_position: np.ndarray, 
                            radius: float = 50.0) -> float:
    """
    Calculate local traffic density around ego vehicle.
    
    Args:
        observations: Full observation array, shape (n_vehicles, 7)
        ego_position: Ego vehicle position [x, y]
        radius: Radius around ego vehicle to consider (meters)
        
    Returns:
        Traffic density as vehicles per square kilometer
    """
    if len(observations) == 0:
        return 0.0
    
    # Count vehicles within radius
    vehicle_count = 0
    for vehicle in observations:
        if vehicle[0] == 0:  # Skip if not present
            continue
            
        vehicle_pos = vehicle[1:3]
        distance = np.linalg.norm(vehicle_pos - ego_position)
        if distance <= radius:
            vehicle_count += 1
    
    # Calculate density (vehicles per km²)
    area_km2 = np.pi * (radius / 1000.0) ** 2  # Convert to km²
    density = vehicle_count / area_km2 if area_km2 > 0 else 0.0
    
    return density


class KinematicsExtractor:
    """Extract features from kinematics observations."""
    
    def __init__(self, lane_width: float = 4.0, num_lanes: int = 4):
        """
        Initialize kinematics extractor.
        
        Args:
            lane_width: Width of each lane in meters
            num_lanes: Total number of lanes
        """
        self.lane_width = lane_width
        self.num_lanes = num_lanes
    
    def extract_features(self, obs: np.ndarray, context: Dict = None) -> Dict[str, Any]:
        """
        Extract kinematics-based features.
        
        Args:
            obs: Kinematics observation array, shape (n_vehicles, 7)
                 Format: [presence, x, y, vx, vy, cos_h, sin_h]
            context: Additional context information
            
        Returns:
            Dictionary of extracted features
        """
        if len(obs) == 0:
            return {
                'lane_position': 0,
                'speed': 0.0,
                'lead_vehicle_gap': float('inf'),
                'min_ttc': float('inf'),
                'traffic_density': 0.0
            }
        
        # Assume first vehicle is ego vehicle
        ego_state = obs[0]
        other_vehicles = obs[1:] if len(obs) > 1 else np.array([])
        
        # Extract basic ego vehicle info
        ego_pos = ego_state[1:3]  # [x, y]
        ego_vel = ego_state[3:5]  # [vx, vy]
        speed = np.linalg.norm(ego_vel)
        
        # Estimate lane position
        lane_position = estimate_lane_from_position(ego_pos[1], self.lane_width, self.num_lanes)
        
        # Detect lead vehicle and gap
        lead_idx, lead_gap = detect_lead_vehicle(ego_state, other_vehicles)
        
        # Calculate minimum TTC with all other vehicles
        min_ttc = float('inf')
        if len(other_vehicles) > 0:
            ttc_values = []
            for other_vehicle in other_vehicles:
                ttc = calculate_time_to_collision(ego_state, other_vehicle)
                if ttc != float('inf'):
                    ttc_values.append(ttc)
            
            if ttc_values:
                min_ttc = min(ttc_values)
        
        # Calculate traffic density
        traffic_density = calculate_traffic_density(obs, ego_pos)
        
        return {
            'lane_position': lane_position,
            'speed': float(speed),
            'lead_vehicle_gap': float(lead_gap),
            'min_ttc': float(min_ttc),
            'traffic_density': float(traffic_density),
            'ego_x': float(ego_pos[0]),
            'ego_y': float(ego_pos[1]),
            'ego_vx': float(ego_vel[0]),
            'ego_vy': float(ego_vel[1])
        }


class TrafficMetricsExtractor:
    """Extract traffic flow and density metrics."""
    
    def __init__(self, analysis_radius: float = 100.0):
        """
        Initialize traffic metrics extractor.
        
        Args:
            analysis_radius: Radius for traffic analysis in meters
        """
        self.analysis_radius = analysis_radius
    
    def extract_features(self, obs: np.ndarray, context: Dict = None) -> Dict[str, Any]:
        """
        Extract traffic metrics.
        
        Args:
            obs: Full observation array, shape (n_vehicles, 7)
            context: Additional context information
            
        Returns:
            Dictionary of traffic metrics
        """
        if len(obs) == 0:
            return {
                'vehicle_count': 0,
                'average_speed': 0.0,
                'speed_variance': 0.0,
                'lane_change_opportunities': 0
            }
        
        # Count present vehicles
        present_vehicles = obs[obs[:, 0] == 1]  # Filter by presence flag
        vehicle_count = len(present_vehicles)
        
        if vehicle_count == 0:
            return {
                'vehicle_count': 0,
                'average_speed': 0.0,
                'speed_variance': 0.0,
                'lane_change_opportunities': 0
            }
        
        # Calculate speed statistics
        velocities = present_vehicles[:, 3:5]  # [vx, vy] for all vehicles
        speeds = np.linalg.norm(velocities, axis=1)
        average_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        
        # Estimate lane change opportunities (simplified)
        # Count gaps between vehicles that could allow lane changes
        lane_change_opportunities = self._count_lane_change_opportunities(present_vehicles)
        
        return {
            'vehicle_count': int(vehicle_count),
            'average_speed': float(average_speed),
            'speed_variance': float(speed_variance),
            'lane_change_opportunities': int(lane_change_opportunities)
        }
    
    def _count_lane_change_opportunities(self, vehicles: np.ndarray, 
                                       min_gap: float = 15.0) -> int:
        """
        Count potential lane change opportunities.
        
        Args:
            vehicles: Array of present vehicles
            min_gap: Minimum gap required for safe lane change
            
        Returns:
            Number of potential lane change opportunities
        """
        if len(vehicles) < 2:
            return 0
        
        opportunities = 0
        
        # Simple heuristic: count gaps between consecutive vehicles in x-direction
        x_positions = vehicles[:, 1]  # x positions
        sorted_indices = np.argsort(x_positions)
        sorted_x = x_positions[sorted_indices]
        
        # Count gaps larger than minimum required
        for i in range(len(sorted_x) - 1):
            gap = sorted_x[i + 1] - sorted_x[i]
            if gap > min_gap:
                opportunities += 1
        
        return opportunities