"""
Natural language summarizer for driving contexts.
"""

from typing import Dict, Any, List
import numpy as np
from .extractors import (
    estimate_lane_from_position,
    detect_lead_vehicle,
    calculate_time_to_collision,
    calculate_traffic_density
)


class LanguageSummarizer:
    """Generate natural language summaries of driving contexts."""
    
    def __init__(self, lane_width: float = 4.0, num_lanes: int = 4):
        """
        Initialize language summarizer.
        
        Args:
            lane_width: Width of each lane in meters
            num_lanes: Total number of lanes
        """
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self._scenario_templates = {
            'free_flow': self._free_flow_template,
            'dense_commuting': self._dense_commuting_template,
            'stop_and_go': self._stop_and_go_template,
            'aggressive_neighbors': self._aggressive_neighbors_template,
            'lane_closure': self._lane_closure_template,
            'time_budget': self._time_budget_template,
            'default': self._default_template
        }
    
    def summarize(self, ego: np.ndarray, others: np.ndarray, context: Dict = None) -> str:
        """
        Generate natural language summary of driving context.
        
        Args:
            ego: Ego vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
            others: Other vehicles states array
            context: Additional context including scenario type
            
        Returns:
            Natural language description of the driving situation
        """
        if context is None:
            context = {}
        
        # Extract basic features
        features = self._extract_summary_features(ego, others)
        
        # Get scenario-specific template
        scenario = context.get('scenario', 'default')
        template_func = self._scenario_templates.get(scenario, self._default_template)
        
        # Generate summary using template
        summary = template_func(features, context)
        
        return summary
    
    def _extract_summary_features(self, ego: np.ndarray, others: np.ndarray) -> Dict[str, Any]:
        """Extract features needed for summary generation."""
        if ego[0] == 0:  # Ego vehicle not present
            return {
                'lane_position': 0,
                'speed_kmh': 0.0,
                'speed_description': 'stationary',
                'lead_gap': float('inf'),
                'min_ttc': float('inf'),
                'traffic_density': 0.0,
                'num_vehicles': 0,
                'lane_description': 'unknown lane',
                'gap_description': 'no vehicles ahead',
                'ttc_description': 'no collision risk',
                'traffic_description': 'empty road',
                'maneuver_opportunities': ['vehicle not active']
            }
        
        # Basic ego vehicle info
        ego_pos = ego[1:3]
        ego_vel = ego[3:5]
        speed_ms = np.linalg.norm(ego_vel)
        speed_kmh = speed_ms * 3.6
        
        # Lane position
        lane_position = estimate_lane_from_position(ego_pos[1], self.lane_width, self.num_lanes)
        
        # Lead vehicle detection
        lead_idx, lead_gap = detect_lead_vehicle(ego, others)
        
        # Minimum TTC calculation
        min_ttc = float('inf')
        if len(others) > 0:
            for other_vehicle in others:
                if other_vehicle[0] == 1:  # Vehicle present
                    ttc = calculate_time_to_collision(ego, other_vehicle)
                    if ttc < min_ttc:
                        min_ttc = ttc
        
        # Traffic density (only count other vehicles, not ego)
        if len(others) > 0:
            traffic_density = calculate_traffic_density(others, ego_pos)
        else:
            traffic_density = 0.0
        
        # Count present vehicles
        num_vehicles = sum(1 for v in others if v[0] == 1)
        
        # Generate descriptions
        features = {
            'lane_position': lane_position,
            'speed_kmh': speed_kmh,
            'speed_description': self._describe_speed(speed_kmh),
            'lead_gap': lead_gap,
            'min_ttc': min_ttc,
            'traffic_density': traffic_density,
            'num_vehicles': num_vehicles,
            'lane_description': self._describe_lane(lane_position),
            'gap_description': self._describe_gap(lead_gap),
            'ttc_description': self._describe_ttc(min_ttc),
            'traffic_description': self._describe_traffic_density(traffic_density),
            'maneuver_opportunities': self._assess_maneuver_opportunities(ego, others)
        }
        
        return features
    
    def _describe_speed(self, speed_kmh: float) -> str:
        """Describe speed in natural language."""
        if speed_kmh < 5:
            return "stationary"
        elif speed_kmh < 30:
            return "moving slowly"
        elif speed_kmh < 60:
            return "moving at moderate speed"
        elif speed_kmh < 90:
            return "moving at highway speed"
        else:
            return "moving at high speed"
    
    def _describe_lane(self, lane_position: int) -> str:
        """Describe lane position in natural language."""
        if self.num_lanes == 4:
            lane_names = ["rightmost lane", "right lane", "left lane", "leftmost lane"]
        elif self.num_lanes == 3:
            lane_names = ["right lane", "middle lane", "left lane"]
        elif self.num_lanes == 2:
            lane_names = ["right lane", "left lane"]
        else:
            return f"lane {lane_position + 1}"
        
        return lane_names[min(lane_position, len(lane_names) - 1)]
    
    def _describe_gap(self, gap: float) -> str:
        """Describe gap to lead vehicle."""
        if gap == float('inf'):
            return "no vehicles ahead"
        elif gap < 10:
            return "very close to lead vehicle"
        elif gap < 25:
            return "close to lead vehicle"
        elif gap < 50:
            return "moderate gap to lead vehicle"
        else:
            return "large gap to lead vehicle"
    
    def _describe_ttc(self, ttc: float) -> str:
        """Describe time to collision risk."""
        if ttc == float('inf'):
            return "no collision risk"
        elif ttc < 2:
            return "immediate collision risk"
        elif ttc < 5:
            return "high collision risk"
        elif ttc < 10:
            return "moderate collision risk"
        else:
            return "low collision risk"
    
    def _describe_traffic_density(self, density: float) -> str:
        """Describe traffic density."""
        if density < 100:
            return "light traffic"
        elif density < 300:
            return "moderate traffic"
        elif density < 600:
            return "heavy traffic"
        else:
            return "very heavy traffic"
    
    def _assess_maneuver_opportunities(self, ego: np.ndarray, others: np.ndarray) -> List[str]:
        """Assess available maneuver opportunities."""
        opportunities = []
        
        if len(others) == 0:
            opportunities.append("free to change lanes")
            opportunities.append("clear road ahead")
            return opportunities
        
        ego_lane = estimate_lane_from_position(ego[2], self.lane_width, self.num_lanes)
        
        # Check lane change opportunities
        left_clear = self._is_lane_clear(ego, others, ego_lane + 1)
        right_clear = self._is_lane_clear(ego, others, ego_lane - 1)
        
        if left_clear and ego_lane < self.num_lanes - 1:
            opportunities.append("left lane change available")
        if right_clear and ego_lane > 0:
            opportunities.append("right lane change available")
        
        # Check overtaking opportunity
        lead_idx, lead_gap = detect_lead_vehicle(ego, others)
        if lead_idx is not None and lead_gap < 30 and (left_clear or right_clear):
            opportunities.append("overtaking opportunity")
        
        if not opportunities:
            opportunities.append("limited maneuver options")
        
        return opportunities
    
    def _is_lane_clear(self, ego: np.ndarray, others: np.ndarray, target_lane: int) -> bool:
        """Check if target lane is clear for lane change."""
        if target_lane < 0 or target_lane >= self.num_lanes:
            return False
        
        ego_x = ego[1]
        safe_distance = 20.0  # Minimum safe distance for lane change
        
        for vehicle in others:
            if vehicle[0] == 0:  # Not present
                continue
            
            v_lane = estimate_lane_from_position(vehicle[2], self.lane_width, self.num_lanes)
            if v_lane == target_lane:
                v_x = vehicle[1]
                if abs(v_x - ego_x) < safe_distance:
                    return False
        
        return True
    
    # Scenario-specific templates
    def _free_flow_template(self, features: Dict, context: Dict) -> str:
        """Template for free flow scenarios."""
        base = f"Vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        if features['num_vehicles'] == 0:
            return f"{base} The highway is clear with no other vehicles visible."
        
        traffic_info = f" Traffic is {features['traffic_description']} with {features['num_vehicles']} other vehicles nearby."
        gap_info = f" {features['gap_description'].capitalize()}."
        
        maneuvers = ", ".join(features['maneuver_opportunities'])
        maneuver_info = f" Maneuver assessment: {maneuvers}."
        
        return f"{base}{traffic_info}{gap_info}{maneuver_info}"
    
    def _dense_commuting_template(self, features: Dict, context: Dict) -> str:
        """Template for dense commuting scenarios."""
        base = f"In heavy commuter traffic, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        gap_info = f" {features['gap_description'].capitalize()}."
        ttc_info = f" Collision assessment: {features['ttc_description']}."
        
        if features['min_ttc'] < 5:
            urgency = " Caution required due to close proximity of other vehicles."
        else:
            urgency = ""
        
        maneuvers = ", ".join(features['maneuver_opportunities'])
        maneuver_info = f" Available maneuvers: {maneuvers}."
        
        return f"{base}{gap_info}{ttc_info}{urgency}{maneuver_info}"
    
    def _stop_and_go_template(self, features: Dict, context: Dict) -> str:
        """Template for stop and go scenarios."""
        if features['speed_kmh'] < 10:
            base = f"In stop-and-go traffic, vehicle is nearly stationary at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        else:
            base = f"In stop-and-go traffic, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        gap_info = f" {features['gap_description'].capitalize()}."
        
        if features['lead_gap'] < 15:
            following_info = " Following closely due to congested conditions."
        else:
            following_info = " Maintaining safe following distance despite congestion."
        
        return f"{base}{gap_info}{following_info}"
    
    def _aggressive_neighbors_template(self, features: Dict, context: Dict) -> str:
        """Template for aggressive neighbors scenarios."""
        base = f"Surrounded by aggressive drivers, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        ttc_info = f" Collision risk: {features['ttc_description']}."
        
        if features['min_ttc'] < 3:
            warning = " High alert required due to aggressive nearby vehicles."
        else:
            warning = " Monitoring nearby vehicles for sudden maneuvers."
        
        defensive_info = " Maintaining defensive driving posture."
        
        return f"{base}{ttc_info}{warning}{defensive_info}"
    
    def _lane_closure_template(self, features: Dict, context: Dict) -> str:
        """Template for lane closure scenarios."""
        base = f"Approaching lane closure, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        if features['lane_position'] == self.num_lanes - 1:  # Leftmost lane
            merge_info = " In the continuing lane, monitoring merging traffic."
        else:
            merge_info = " May need to merge due to lane closure ahead."
        
        maneuvers = ", ".join(features['maneuver_opportunities'])
        maneuver_info = f" Merge opportunities: {maneuvers}."
        
        return f"{base}{merge_info}{maneuver_info}"
    
    def _time_budget_template(self, features: Dict, context: Dict) -> str:
        """Template for time budget scenarios."""
        base = f"Under time pressure, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        if features['speed_kmh'] > 80:
            urgency_info = " Maintaining high speed to meet time constraints."
        elif features['speed_kmh'] < 40:
            urgency_info = " Progress impeded by traffic conditions."
        else:
            urgency_info = " Balancing speed with safety considerations."
        
        maneuvers = ", ".join(features['maneuver_opportunities'])
        if "overtaking opportunity" in maneuvers:
            tactical_info = " Evaluating overtaking to improve progress."
        else:
            tactical_info = " Limited options for improving travel time."
        
        return f"{base}{urgency_info}{tactical_info}"
    
    def _default_template(self, features: Dict, context: Dict) -> str:
        """Default template for general scenarios."""
        base = f"Vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}."
        
        if features['num_vehicles'] > 0:
            traffic_info = f" {features['traffic_description'].capitalize()} with {features['num_vehicles']} nearby vehicles."
            gap_info = f" {features['gap_description'].capitalize()}."
            
            if features['min_ttc'] < 10:
                safety_info = f" {features['ttc_description'].capitalize()}."
            else:
                safety_info = ""
            
            maneuvers = ", ".join(features['maneuver_opportunities'])
            maneuver_info = f" Maneuver options: {maneuvers}."
            
            return f"{base}{traffic_info}{gap_info}{safety_info}{maneuver_info}"
        else:
            maneuvers = ", ".join(features['maneuver_opportunities'])
            maneuver_info = f" {maneuvers.capitalize()}."
            return f"{base} Clear road conditions with no other vehicles nearby.{maneuver_info}"