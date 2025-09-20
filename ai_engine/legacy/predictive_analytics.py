
"""
CRONOS AI Engine - Legacy System Predictive Analytics

Advanced predictive analytics for legacy system maintenance and failure prevention.
Combines time series analysis, machine learning, and LLM-powered insights.
"""

import logging
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.metrics import AIEngineMetrics

from .models import (
    SystemFailurePrediction,
    MaintenanceRecommendation,
    SystemBehaviorPattern,
    LegacySystemContext,
    SystemMetrics,
    HistoricalPatternDatabase,
    FailureType,
    SeverityLevel,
    MaintenanceType
)


class PredictionHorizon(str, Enum):
    """Time horizons for predictions."""
    IMMEDIATE = "immediate"  # 0-4 hours
    SHORT_TERM = "short_term"  # 4-24 hours
    MEDIUM_TERM = "medium_term"  # 1-7 days
    LONG_TERM = "long_term"  # 7-30 days
    STRATEGIC = "strategic"  # 30+ days


@dataclass
class TimeSeriesData:
    """Time series data container."""
    timestamps: List[datetime]
    values: List[float]
    metric_name: str
    system_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        }).set_index('timestamp')
    
    def get_latest_window(self, hours: int = 24) -> 'TimeSeriesData':
        """Get data from the latest N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_timestamps = []
        filtered_values = []
        
        for ts, val in zip(self.timestamps, self.values):
            if ts >= cutoff_time:
                filtered_timestamps.append(ts)
                filtered_values.append(val)
        
        return TimeSeriesData(
            timestamps=filtered_timestamps,
            values=filtered_values,
            metric_name=self.metric_name,
            system_id=self.system_id,
            metadata=self.metadata
        )


@dataclass
class PredictionResult:
    """Result of a predictive analysis."""
    prediction_id: str
    system_id: str
    metric_name: str
    prediction_horizon: PredictionHorizon
    predicted_values: List[float]
    prediction_timestamps: List[datetime]
    confidence_interval: Tuple[List[float], List[float]]  # (lower, upper)
    model_confidence: float
    anomaly_probability: float
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    seasonal_pattern: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class FailurePredictor:
    """
    Advanced failure prediction using time series analysis and LLM insights.
    
    Combines statistical models, machine learning, and domain expertise
    to predict system failures with high accuracy.
    """
    
    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize failure predictor."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
        # Time series models
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        
        # Model state
        self.models_trained = False
        self.training_data: Dict[str, List[TimeSeriesData]] = {}
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        # Analysis parameters
        self.min_data_points = 100
        self.prediction_horizons = {
            PredictionHorizon.IMMEDIATE: 4,  # hours
            PredictionHorizon.SHORT_TERM: 24,
            PredictionHorizon.MEDIUM_TERM: 168,  # 7 days
            PredictionHorizon.LONG_TERM: 720,  # 30 days
        }
        
        self.logger.info("FailurePredictor initialized")
    
    async def predict_failure_probability(
        self,
        system_id: str,
        time_series_data: List[TimeSeriesData],
        system_context: Optional[LegacySystemContext] = None,
        prediction_horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM
    ) -> Dict[str, Any]:
        """
        Predict failure probability using time series analysis.
        
        Args:
            system_id: System identifier
            time_series_data: Historical metric data
            system_context: System context information
            prediction_horizon: Prediction time horizon
            
        Returns:
            Comprehensive failure prediction analysis
        """
        
        start_time = time.time()
        
        try:
            # Validate input data
            if not time_series_data:
                raise CronosAIException("No time series data provided")
            
            # Prepare data for analysis
            combined_features = self._prepare_features(time_series_data)
            
            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(
                time_series_data, prediction_horizon
            )
            
            # Anomaly detection
            anomaly_analysis = self._detect_anomalies(combined_features)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(time_series_data)
            
            # Pattern analysis
            pattern_analysis = self._analyze_patterns(time_series_data)
            
            # LLM-enhanced prediction
            llm_insights = {}
            if self.llm_service:
                llm_insights = await self._get_llm_failure_insights(
                    system_id=system_id,
                    statistical_analysis=statistical_analysis,
                    anomaly_analysis=anomaly_analysis,
                    trend_analysis=trend_analysis,
                    system_context=system_context
                )
            
            # Synthesize final prediction
            failure_prediction = self._synthesize_failure_prediction(
                system_id=system_id,
                statistical_analysis=statistical_analysis,
                anomaly_analysis=anomaly_analysis,
                trend_analysis=trend_analysis,
                pattern_analysis=pattern_analysis,
                llm_insights=llm_insights,
                prediction_horizon=prediction_horizon,
                processing_time=time.time() - start_time
            )
            
            return failure_prediction
            
        except Exception as e:
            self.logger.error(f"Failure prediction failed for system {system_id}: {e}")
            raise CronosAIException(f"Failure prediction error: {e}")
    
    def _prepare_features(self, time_series_data: List[TimeSeriesData]) -> np.ndarray:
        """Prepare features from time series data."""
        
        features = []
        
        for ts_data in time_series_data:
            if len(ts_data.values) < self.min_data_points:
                self.logger.warning(f"Insufficient data points for {ts_data.metric_name}")
                continue
            
            # Basic statistical features
            values = np.array(ts_data.values)
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
                len(values)
            ])
            
            # Trend features
            if len(values) > 1:
                # Linear trend slope
                x = np.arange(len(values))
                trend_slope = np.polyfit(x, values, 1)[0]
                features.append(trend_slope)
                
                # Volatility
                volatility = np.std(np.diff(values))
                features.append(volatility)
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features).reshape(1, -1) if features else np.array([]).reshape(1, -1)
    
    def _perform_statistical_analysis(
        self, 
        time_series_data: List[TimeSeriesData], 
        prediction_horizon: PredictionHorizon
    ) -> Dict[str, Any]:
        """Perform statistical analysis on time series data."""
        
        analysis = {
            "metrics_analyzed": len(time_series_data),
            "total_data_points": sum(len(ts.values) for ts in time_series_data),
            "time_range_hours": 0,
            "metric_correlations": {},
            "trend_indicators": {},
            "seasonality_detected": False,
            "data_quality_score": 1.0
        }
        
        if not time_series_data:
            return analysis
        
        # Calculate time range
        all_timestamps = []
        for ts_data in time_series_data:
            all_timestamps.extend(ts_data.timestamps)
        
        if all_timestamps:
            time_range = max(all_timestamps) - min(all_timestamps)
            analysis["time_range_hours"] = time_range.total_seconds() / 3600
        
        # Analyze each metric
        for ts_data in time_series_data:
            values = np.array(ts_data.values)
            
            if len(values) < 2:
                continue
            
            # Trend analysis
            x = np.arange(len(values))
            trend_slope = np.polyfit(x, values, 1)[0]
            
            analysis["trend_indicators"][ts_data.metric_name] = {
                "slope": float(trend_slope),
                "direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable",
                "volatility": float(np.std(np.diff(values))),
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
            
            # Basic seasonality detection (simplified)
            if len(values) > 24:  # Need at least 24 data points
                # Check for daily patterns (simplified)
                try:
                    if len(values) >= 48:  # 2 days of hourly data
                        first_half = values[:len(values)//2]
                        second_half = values[len(values)//2:]
                        correlation = np.corrcoef(first_half[:len(second_half)], second_half)[0, 1]
                        if not np.isnan(correlation) and correlation > 0.5:
                            analysis["seasonality_detected"] = True
                except Exception:
                    pass
        
        # Data quality assessment
        quality_factors = []
        
        for ts_data in time_series_data:
            # Check for missing values (gaps in timestamps)
            if len(ts_data.timestamps) > 1:
                time_diffs = [
                    (ts_data.timestamps[i+1] - ts_data.timestamps[i]).total_seconds()
                    for i in range(len(ts_data.timestamps)-1)
                ]
                expected_interval = np.median(time_diffs)
                gaps = sum(1 for diff in time_diffs if diff > expected_interval * 2)
                completeness = 1.0 - (gaps / len(time_diffs))
                quality_factors.append(completeness)
            
            # Check for outliers
            values = np.array(ts_data.values)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            outliers = sum(1 for v in values if v < q1 - 1.5*iqr or v > q3 + 1.5*iqr)
            outlier_ratio = outliers / len(values)
            quality_factors.append(1.0 - min(outlier_ratio, 0.3) / 0.3 * 0.3)
        
        if quality_factors:
            analysis["data_quality_score"] = np.mean(quality_factors)
        
        return analysis
    
    def _detect_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in the feature space."""
        
        if features.size == 0:
            return {"anomaly_detected": False, "anomaly_score": 0.0}
        
        try:
            # Use isolation forest for anomaly detection
            anomaly_score = self.isolation_forest.fit_predict(features)[0]
            anomaly_probability = self.isolation_forest.score_samples(features)[0]
            
            return {
                "anomaly_detected": anomaly_score == -1,
                "anomaly_score": float(-anomaly_probability),  # Convert to positive score
                "isolation_score": float(anomaly_score),
                "confidence": 0.7 if anomaly_score == -1 else 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {"anomaly_detected": False, "anomaly_score": 0.0, "error": str(e)}
    
    def _analyze_trends(self, time_series_data: List[TimeSeriesData]) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        
        trends = {}
        overall_trend = "stable"
        trend_strength = 0.0
        
        for ts_data in time_series_data:
            if len(ts_data.values) < 3:
                continue
            
            values = np.array(ts_data.values)
            x = np.arange(len(values))
            
            # Linear regression for trend
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Trend classification
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0.01:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Trend strength (R-squared)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            trends[ts_data.metric_name] = {
                "trend": trend,
                "slope": float(slope),
                "strength": float(r_squared),
                "confidence": min(r_squared * 2, 1.0)  # Scale confidence
            }
            
            # Update overall trend
            if r_squared > trend_strength:
                overall_trend = trend
                trend_strength = r_squared
        
        return {
            "metric_trends": trends,
            "overall_trend": overall_trend,
            "trend_strength": trend_strength,
            "trend_confidence": min(trend_strength * 1.5, 1.0)
        }
    
    def _analyze_patterns(self, time_series_data: List[TimeSeriesData]) -> Dict[str, Any]:
        """Analyze patterns in time series data."""
        
        patterns = {
            "cyclic_patterns": [],
            "spike_patterns": [],
            "drift_patterns": [],
            "correlation_patterns": {}
        }
        
        for ts_data in time_series_data:
            values = np.array(ts_data.values)
            
            if len(values) < 10:
                continue
            
            # Detect spikes (values > 2 standard deviations)
            mean_val = np.mean(values)
            std_val = np.std(values)
            spikes = [i for i, v in enumerate(values) if abs(v - mean_val) > 2 * std_val]
            
            if spikes:
                patterns["spike_patterns"].append({
                    "metric": ts_data.metric_name,
                    "spike_count": len(spikes),
                    "spike_ratio": len(spikes) / len(values),
                    "max_spike_magnitude": float(max(abs(values[i] - mean_val) for i in spikes))
                })
            
            # Detect drift (gradual change in baseline)
            if len(values) > 20:
                first_quarter = values[:len(values)//4]
                last_quarter = values[-len(values)//4:]
                
                mean_diff = np.mean(last_quarter) - np.mean(first_quarter)
                std_diff = np.sqrt(np.var(first_quarter) + np.var(last_quarter))
                
                if abs(mean_diff) > std_diff:
                    patterns["drift_patterns"].append({
                        "metric": ts_data.metric_name,
                        "drift_magnitude": float(mean_diff),
                        "drift_direction": "upward" if mean_diff > 0 else "downward",
                        "drift_significance": float(abs(mean_diff) / std_diff)
                    })
        
        # Cross-correlation analysis between metrics
        if len(time_series_data) > 1:
            for i, ts1 in enumerate(time_series_data):
                for j, ts2 in enumerate(time_series_data[i+1:], i+1):
                    if len(ts1.values) == len(ts2.values) and len(ts1.values) > 5:
                        try:
                            correlation = np.corrcoef(ts1.values, ts2.values)[0, 1]
                            if not np.isnan(correlation) and abs(correlation) > 0.5:
                                patterns["correlation_patterns"][f"{ts1.metric_name}_vs_{ts2.metric_name}"] = {
                                    "correlation": float(correlation),
                                    "strength": "strong" if abs(correlation) > 0.8 else "moderate"
                                }
                        except Exception:
                            pass
        
        return patterns
    
    async def _get_llm_failure_insights(
        self,
        system_id: str,
        statistical_analysis: Dict[str, Any],
        anomaly_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        system_context: Optional[LegacySystemContext]
    ) -> Dict[str, Any]:
        """Get LLM insights for failure prediction."""
        
        if not self.llm_service:
            return {}
        
        try:
            # Prepare context for LLM
            context_info = ""
            if system_context:
                context_info = f"""
System: {system_context.system_name} ({system_context.system_type.value})
Criticality: {system_context.criticality.value}
Business Function: {system_context.business_function or 'Unknown'}
Age: {(datetime.now() - system_context.installation_date).days // 365 if system_context.installation_date else 'Unknown'} years
Uptime Requirement: {system_context.uptime_requirement}%
"""
            
            prompt = f"""
You are analyzing a legacy system for failure prediction based on statistical analysis.

{context_info}

Statistical Analysis Results:
- Data Quality Score: {statistical_analysis.get('data_quality_score', 0):.2f}
- Time Range: {statistical_analysis.get('time_range_hours', 0):.1f} hours
- Seasonality Detected: {statistical_analysis.get('seasonality_detected', False)}
- Trend Indicators: {statistical_analysis.get('trend_indicators', {})}

Anomaly Analysis:
- Anomaly Detected: {anomaly_analysis.get('anomaly_detected', False)}
- Anomaly Score: {anomaly_analysis.get('anomaly_score', 0):.3f}

Trend Analysis:
- Overall Trend: {trend_analysis.get('overall_trend', 'unknown')}
- Trend Strength: {trend_analysis.get('trend_strength', 0):.3f}

Based on this analysis, provide expert insights on:

1. FAILURE RISK ASSESSMENT:
   - Probability of failure in next 7 days (0-100%)
   - Most likely failure mode
   - Critical warning signs identified

2. ROOT CAUSE ANALYSIS:
   - Primary factors contributing to risk
   - Historical patterns that match current behavior
   - System-specific vulnerabilities

3. PREDICTIVE INDICATORS:
   - Key metrics to monitor closely
   - Thresholds that indicate imminent failure
   - Early warning timeline (hours/days before failure)

4. RECOMMENDED ACTIONS:
   - Immediate steps to reduce failure risk
   - Monitoring adjustments needed
   - Preventive measures to implement

Focus on actionable insights specific to this type of legacy system.
Consider the system's age, criticality, and business impact in your assessment.
"""
            
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="legacy_whisperer",
                max_tokens=2000,
                temperature=0.1
            )
            
            response = await self.llm_service.process_request(llm_request)
            
            # Parse LLM response
            insights = self._parse_llm_insights(response.content)
            insights["llm_confidence"] = response.confidence
            insights["processing_time"] = response.processing_time
            
            return insights
            
        except Exception as e:
            self.logger.error(f"LLM failure insights failed: {e}")
            return {"error": str(e)}
    
    def _parse_llm_insights(self, content: str) -> Dict[str, Any]:
        """Parse LLM insights response."""
        
        # Simplified parsing - in production, use more sophisticated NLP
        insights = {
            "failure_probability": 0.0,
            "likely_failure_mode": "unknown",
            "warning_signs": [],
            "contributing_factors": [],
            "monitoring_recommendations": [],
            "recommended_actions": [],
            "time_to_failure": "unknown"
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract failure probability
            if "probability" in line.lower() and any(char.isdigit() for char in line):
                import re
                numbers = re.findall(r'(\d+(?:\.\d+)?)%?', line)
                if numbers:
                    try:
                        prob = float(numbers[0])
                        if prob <= 100:  # Assuming percentage
                            insights["failure_probability"] = prob / 100.0
                        else:
                            insights["failure_probability"] = min(prob, 1.0)
                    except ValueError:
                        pass
            
            # Extract list items
            if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                item = re.sub(r'^[-•*\d\.\s]+', '', line).strip()
                if item:
                    if current_section == "warning_signs":
                        insights["warning_signs"].append(item)
                    elif current_section == "contributing_factors":
                        insights["contributing_factors"].append(item)
                    elif current_section == "monitoring":
                        insights["monitoring_recommendations"].append(item)
                    elif current_section == "actions":
                        insights["recommended_actions"].append(item)
            
            # Detect sections
            if "warning signs" in line.lower():
                current_section = "warning_signs"
            elif "contributing factors" in line.lower() or "root cause" in line.lower():
                current_section = "contributing_factors"
            elif "monitoring" in line.lower():
                current_section = "monitoring"
            elif "recommended actions" in line.lower() or "actions" in line.lower():
                current_section = "actions"
        
        return insights
    
    def _synthesize_failure_prediction(
        self,
        system_id: str,
        statistical_analysis: Dict[str, Any],
        anomaly_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        llm_insights: Dict[str, Any],
        prediction_horizon: PredictionHorizon,
        processing_time: float
    ) -> Dict[str, Any]:
        """Synthesize final failure prediction from all analysis components."""
        
        # Calculate combined failure probability
        base_probability = 0.1  # Base 10% failure probability
        
        # Adjust based on anomaly detection
        if anomaly_analysis.get("anomaly_detected", False):
            base_probability += anomaly_analysis.get("anomaly_score", 0) * 0.5
        
        # Adjust based on trend analysis
        trend_strength = trend_analysis.get("trend_strength", 0)
        if trend_analysis.get("overall_trend") in ["increasing", "decreasing"]:
            base_probability += trend_strength * 0.3
        
        # Adjust based on data quality
        data_quality = statistical_analysis.get("data_quality_score", 1.0)
        base_probability *= data_quality  # Lower quality data = less confident prediction
        
        # Incorporate LLM insights
        llm_probability = llm_insights.get("failure_probability", 0)
        if llm_probability > 0:
            # Weight LLM and statistical predictions
            combined_probability = (base_probability + llm_probability) / 2
        else:
            combined_probability = base_probability
        
        # Cap probability at reasonable maximum
        final_probability = min(combined_probability, 0.95)
        
        # Determine failure severity based on probability and trend
        if final_probability > 0.8:
            severity = SeverityLevel.CRITICAL
        elif final_probability > 0.6:
            severity = SeverityLevel.HIGH
        elif final_probability > 0.4:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        # Calculate confidence score
        confidence_factors = [
            statistical_analysis.get("data_quality_score", 0.5),
            trend_analysis.get("trend_confidence", 0.5),
            llm_insights.get("llm_confidence", 0.5) if llm_insights else 0.5,
            0.8 if anomaly_analysis.get("anomaly_detected") else 0.6
        ]
        overall_confidence = np.mean(confidence_factors)
        
        return {
            "system_id": system_id,
            "prediction_horizon": prediction_horizon.value,
            "failure_probability": final_probability,
            "severity": severity.value,
            "confidence": overall_confidence,
            "statistical_analysis": statistical_analysis,
            "anomaly_analysis": anomaly_analysis,
            "trend_analysis": trend_analysis,
            "pattern_analysis": pattern_analysis,
            "llm_insights": llm_insights,
            "processing_time_seconds": processing_time,
            "prediction_timestamp": datetime.now().isoformat(),
            "recommended_actions": llm_insights.get("recommended_actions", []),
            "monitoring_recommendations": llm_insights.get("monitoring_recommendations", []),
            "key_indicators": llm_insights.get("warning_signs", [])
        }


class PerformanceMonitor:
    """
    Enhanced performance monitoring for legacy systems.
    
    Provides real-time performance analysis with predictive capabilities
    and anomaly detection specifically tuned for legacy system characteristics.
    """
    
    def __init__(self, config: Config, metrics_collector: Optional[AIEngineMetrics] = None):
        """Initialize performance monitor."""
        self.config = config
        self.metrics = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds (configurable)
        self.performance_thresholds = {
            "cpu_utilization": {"warning": 80.0, "critical": 95.0},
            "memory_utilization": {"warning": 85.0, "critical": 95.0},
            "disk_utilization": {"warning": 90.0, "critical": 98.0},
            "response_time_ms": {"warning": 5000, "critical": 15000},
            "error_rate": {"warning": 0.05, "critical": 0.15},
            "transaction_rate": {"warning_low": 0.5, "critical_low": 0.2}  # Relative to baseline
        }
        
        # Performance history
        self.performance_history: Dict[str, List[SystemMetrics]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Anomaly detection models per system
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        self.logger.info("PerformanceMonitor initialized")
    
    async def analyze_performance(
        self,
        system_id: str,
        current_metrics: SystemMetrics,
        historical_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze current system performance against historical baselines.
        
        Args:
            system_id: System identifier
            current_metrics: Current system metrics
            historical_window_hours: Historical comparison window
            
        Returns:
            Comprehensive performance analysis
        """
        
        try:
            # Store current metrics
            if system_id not in self.performance_history:
                self.performance_history[system_id] = []
            
            self.performance_history[system_id].append(current_metrics)
            
            # Limit history size
            max_history = 1000
            if len(self.performance_history[system_id]) > max_history:
                self.performance_history[system_id] = self.performance_history[system_id][-max_history:]
            
            # Get historical data for comparison
            historical_metrics = self._get_historical_metrics(
                system_id, historical_window_hours
            )
            
            # Calculate performance indicators
            performance_indicators = self._calculate_performance_indicators(
                current_metrics, historical_metrics
            )
            
            # Threshold analysis
            threshold_analysis = self._analyze_thresholds(current_metrics)
            
            # Anomaly detection
            anomaly_analysis = await self._detect_performance_anomalies(
                system_id, current_metrics, historical_metrics
            )
            
            # Trend analysis
            trend_analysis = self._analyze_performance_trends(historical_metrics)
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                performance_indicators, threshold_analysis, anomaly_analysis
            )
            
            analysis_result = {
                "system_id": system_id,
                "timestamp": current_metrics.timestamp.isoformat(),
                "performance_score": performance_score,
                "performance_indicators": performance_indicators,
                "threshold_analysis": threshold_analysis,
                "anomaly_analysis": anomaly_analysis,
                "trend_analysis": trend_analysis,
                "recommendations": self._generate_performance_recommendations(
                    performance_indicators, threshold_analysis, anomaly_analysis
                ),
                "alert_level": self._determine_alert_level(threshold_analysis, anomaly_analysis)
            }
            
            # Record metrics
            if self.metrics:
                self._record_performance_metrics(system_id, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed for system {system_id}: {e}")
            raise CronosAIException(f"Performance analysis error: {e}")
    
    def _get_historical_metrics(
        self, system_id: str, window_hours: int
    ) -> List[SystemMetrics]:
        """Get historical metrics within the specified time window."""
        
        if system_id not in self.performance_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        return [
            metric for metric in self.performance_history[system_id]
            if metric.timestamp >= cutoff_time
        ]
    
    def _calculate_performance_indicators(
        self,
        current_metrics: SystemMetrics,
        historical_metrics: List[SystemMetrics]
    ) -> Dict[str, Any]:
        """Calculate performance indicators comparing current vs historical."""
        
        indicators = {
            "cpu_utilization": {
                "current": current_metrics.cpu_utilization,
                "baseline": 0.0,
                "deviation": 0.0,
                "percentile_rank": 0.0
            },
            "memory_utilization": {
                "current": current_metrics.memory_utilization,
                "baseline": 0.0,
                "deviation": 0.0,
                "percentile_rank": 0.0
            },
            "disk_utilization": {
                "current": current_metrics.disk_utilization,
                "baseline": 0.0,
                "deviation": 0.0,
                "percentile_rank": 0.0
            }
        }
        
        if not historical_metrics:
            return indicators
        
        # Calculate baselines and deviations
        for metric_name in indicators:
            values = []
            for hist_metric in historical_metrics:
                if hasattr(hist_metric, metric_name):
                    value = getattr(hist_metric, metric_name)
                    if value is not None:
                        values.append(value)
            
            if values:
                baseline = np.mean(values)
                std_dev = np.std(values)
                current_value = indicators[metric_name]["current"]
                
                indicators[metric_name]["baseline"] = baseline
                indicators[metric_name]["deviation"] = (
                    (current_value - baseline) / std_dev if std_dev > 0 else 0.0
                )
                
                # Calculate percentile rank
                if current_value is not None:
                    percentile_rank = (
                        sum(1 for v in values if v <= current_value) / len(values)
                    )
                    indicators[metric_name]["percentile_rank"] = percentile_rank
        
        # Add response time and error rate if available
        if current_metrics.response_time_ms is not None:
            response_times = [
                m.response_time_ms for m in historical_metrics
                if m.response_time_ms is not None
            ]
            if response_times:
                indicators["response_time"] = {
                    "current": current_metrics.response_time_ms,
                    "baseline": np.mean(response_times),
                    "deviation": (
                        (current_metrics.response_time_ms - np.mean(response_times)) /
                        np.std(response_times) if np.std(response_times) > 0 else 0.0
                    ),
                    "percentile_rank": (
                        sum(1 for t in response_times if t <= current_metrics.response_time_ms) /
                        len(response_times)
                    )
                }
        
        return indicators
    
    def _analyze_thresholds(self, current_metrics: SystemMetrics) -> Dict[str, Any]:
        """Analyze current metrics against defined thresholds."""
        
        threshold_analysis = {
            "violations": [],
            "warnings": [],
            "overall_status": "normal"
        }
        
        # Check each metric against thresholds
        metrics_to_check = {
            "cpu_utilization": current_metrics.cpu_utilization,
            "memory_utilization": current_metrics.memory_utilization,
            "disk_utilization": current_metrics.disk_utilization,
            "response_time_ms": current_metrics.response_time_ms,
            "error_rate": current_metrics.error_rate
        }
        
        for metric_name, value in metrics_to_check.items():
            if value is None or metric_name not in self.performance_thresholds:
                continue
            
            thresholds = self.performance_thresholds[metric_name]
            
            if "critical" in thresholds and value >= thresholds["critical"]:
                threshold_analysis["violations"].append({
                    "metric": metric_name,
                    "value": value,
                    "threshold": thresholds["critical"],
                    "severity": "critical"
                })
                threshold_analysis["overall_status"] = "critical"
            elif "warning" in thresholds and value >= thresholds["warning"]:
                threshold_analysis["warnings"].append({
                    "metric": metric_name,
                    "value": value,
                    "threshold": thresholds["warning"],
                    "severity": "warning"
                })
                if threshold_analysis["overall_status"] == "normal":
                    threshold_analysis["overall_status"] = "warning"
        
        return threshold_analysis
    
    async def _detect_performance_anomalies(
        self,
        system_id: str,
        current_metrics: SystemMetrics,
        historical_metrics: List[SystemMetrics]
    ) -> Dict[str, Any]:
        """Detect performance anomalies using machine learning."""
        
        anomaly_analysis = {
            "anomaly_detected": False,
            "anomaly_score": 0.0,
            "anomalous_metrics": [],
            "confidence": 0.0
        }
        
        if len(historical_metrics) < 20:  # Need sufficient historical data
            return anomaly_analysis
        
        try:
            # Prepare feature matrix
            features = []
            for metric in historical_metrics:
                feature_vector = [
                    metric.cpu_utilization,
                    metric.memory_utilization,
                    metric.disk_utilization,
                    metric.response_time_ms or 0.0,
                    metric.error_rate or 0.0,
                    metric.transaction_rate or 0.0
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Train or use existing anomaly detector
            if system_id not in self.anomaly_detectors:
                self.anomaly_detectors[system_id] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.anomaly_detectors[system_id].fit(features)
            
            # Test current metrics
            current_features = np.array([[
                current_metrics.cpu_utilization,
                current_metrics.memory_utilization,
                current_metrics.disk_utilization,
                current_metrics.response_time_ms or 0.0,
                current_metrics.error_rate or 0.0,
                current_metrics.transaction_rate or 0.0
            ]])
            
            detector = self.anomaly_detectors[system_id]
            anomaly_prediction = detector.predict(current_features)[0]
            anomaly_score = detector.score_samples(current_features)[0]
            
            anomaly_analysis["anomaly_detected"] = anomaly_prediction == -1
            anomaly_analysis["anomaly_score"] = float(-anomaly_score)  # Convert to positive
            anomaly_analysis["confidence"] = 0.8 if anomaly_prediction == -1 else 0.3
            
            # Identify which specific metrics are anomalous
            if anomaly_analysis["anomaly_detected"]:
                metric_names = [
                    "cpu_utilization", "memory_utilization", "disk_utilization",
                    "response_time_ms", "error_rate", "transaction_rate"
                ]
                
                for i, (metric_name, current_value) in enumerate(zip(
                    metric_names, current_features[0]
                )):
                    historical_values = features[:, i]
                    if len(historical_values) > 0:
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)
                        
                        if std_val > 0 and abs(current_value - mean_val) > 2 * std_val:
                            anomaly_analysis["anomalous_metrics"].append({
                                "metric": metric_name,
                                "current_value": float(current_value),
                                "expected_range": [
                                    float(mean_val - 2 * std_val),
                                    float(mean_val + 2 * std_val)
                                ],
                                "deviation_magnitude": float(
                                    abs(current_value - mean_val) / std_val
                                )
                            })
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            anomaly_analysis["error"] = str(e)
        
        return anomaly_analysis
    
    def _analyze_performance_trends(
        self, historical_metrics: List[SystemMetrics]
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        if len(historical_metrics) < 5:
            return {"insufficient_data": True}
        
        # Sort by timestamp
        sorted_metrics = sorted(historical_metrics, key=lambda m: m.timestamp)
        
        trends = {}
        metric_names = ["cpu_utilization", "memory_utilization", "disk_utilization"]
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in sorted_metrics]
            
            if all(v is not None for v in values):
                # Calculate linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                # Determine trend direction
                if abs(slope) < 0.1:
                    direction = "stable"
                elif slope > 0.1:
                    direction = "increasing"
                else:
                    direction = "decreasing"
                
                # Calculate trend strength (R-squared)
                y_pred = np.polyval([slope, np.mean(values) - slope * np.mean(x)], x)
                ss_res = np.sum((values - y_pred) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                trends[metric_name] = {
                    "direction": direction,
                    "slope": float(slope),
                    "strength": float(r_squared),
                    "confidence": min(float(r_squared * 1.5), 1.0)
                }
        
        return {"metric_trends": trends}
    
    def _calculate_performance_score(
        self,
        performance_indicators: Dict[str, Any],
        threshold_analysis: Dict[str, Any],
        anomaly_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall performance score (0-100)."""
        
        base_score = 100.0
        
        # Deduct for threshold violations
        for violation in threshold_analysis.get("violations", []):
            if violation["severity"] == "critical":
                base_score -= 25
            elif violation["severity"] == "warning":
                base_score -= 10
        
        for warning in threshold_analysis.get("warnings", []):
            base_score -= 5
        
        # Deduct for anomalies
        if anomaly_analysis.get("anomaly_detected", False):
            anomaly_score = anomaly_analysis.get("anomaly_score", 0)
            base_score -= min(anomaly_score * 30, 30)  # Max 30 points for anomalies
        
        # Adjust for performance deviations
        for metric_name, indicator in performance_indicators.items():
            if isinstance(indicator, dict) and "deviation" in indicator:
                deviation = abs(indicator["deviation"])
                if deviation > 2:  # More than 2 standard deviations
                    base_score -= min(deviation * 5, 20)  # Max 20 points per metric
        
        return max(min(base_score, 100.0), 0.0)
    
    def _generate_performance_recommendations(
        self,
        performance_indicators: Dict[str, Any],
        threshold_analysis: Dict[str, Any],
        anomaly_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        
        # Threshold-based recommendations
        for violation in threshold_analysis.get("violations", []):
            metric = violation["metric"]
            if metric == "cpu_utilization":
                recommendations.append(
                    "CPU utilization is critically high. Consider scaling resources or "
                    "optimizing CPU-intensive processes."
                )
            elif metric == "memory_utilization":
                recommendations.append(
                    "Memory utilization is critically high. Check for memory leaks and "
                    "consider increasing available memory."
                )
            elif metric == "disk_utilization":
                recommendations.append(
                    "Disk utilization is critically high. Clean up unnecessary files and "
                    "consider expanding storage capacity."
                )
            elif metric == "response_time_ms":
                recommendations.append(
                    "Response times are critically high. Investigate performance bottlenecks "
                    "and optimize slow operations."
                )
        
        # Anomaly-based recommendations
        if anomaly_analysis.get("anomaly_detected", False):
            anomalous_metrics = anomaly_analysis.get("anomalous_metrics", [])
            if anomalous_metrics:
                metrics_list = ", ".join([m["metric"] for m in anomalous_metrics])
                recommendations.append(
                    f"Anomalous behavior detected in: {metrics_list}. "
                    "Review recent changes and investigate root causes."
                )
        
        # Performance deviation recommendations
        for metric_name, indicator in performance_indicators.items():
            if isinstance(indicator, dict) and "deviation" in indicator:
                if indicator["deviation"] > 3:  # Very high deviation
                    recommendations.append(
                        f"Unusual {metric_name} pattern detected. "
                        f"Current value is {indicator['deviation']:.1f} standard deviations "
                        "from normal. Investigate potential causes."
                    )
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations
    
    def _determine_alert_level(
        self,
        threshold_analysis: Dict[str, Any],
        anomaly_analysis: Dict[str, Any]
    ) -> str:
        """Determine overall alert level."""
        
        if threshold_analysis.get("violations"):
            for violation in threshold_analysis["violations"]:
                if violation["severity"] == "critical":
                    return "critical"
            return "warning"
        
        if anomaly_analysis.get("anomaly_detected", False):
            anomaly_score = anomaly_analysis.get("anomaly_score", 0)
            if anomaly_score > 0.8:
                return "warning"
        
        if threshold_analysis.get("warnings"):
            return "info"
        
        return "normal"
    
    def _record_performance_metrics(
        self, system_id: str, analysis_result: Dict[str, Any]
    ) -> None:
        """Record performance metrics for monitoring."""
        
        if not self.metrics:
            return
        
        try:
            # Record performance score
            self.metrics.custom.set_custom_gauge(
                f"performance_score_{system_id}",
                analysis_result["performance_score"]
            )
            
            # Record alert level as numeric value
            alert_levels = {"normal": 0, "info": 1, "warning": 2, "critical": 3}
            alert_numeric = alert_levels.get(analysis_result["alert_level"], 0)
            self.metrics.custom.set_custom_gauge(
                f"alert_level_{system_id}",
                alert_numeric
            )
            
            # Record anomaly detection
            if analysis_result["anomaly_analysis"].get("anomaly_detected"):
                self.metrics.custom.increment_custom_counter(
                    f"anomalies_detected_{system_id}"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")
    
    def update_performance_thresholds(
        self, metric_name: str, warning: float, critical: float
    ) -> None:
        """Update performance thresholds for a metric."""
        
        if metric_name not in self.performance_thresholds:
            self.performance_thresholds[metric_name] = {}
        
        self.performance_thresholds[metric_name]["warning"] = warning
        self.performance_thresholds[metric_name]["critical"] = critical
        
        self.logger.info(f"Updated thresholds for {metric_name}: warning={warning}, critical={critical}")
    
    def get_system_performance_summary(self, system_id: str) -> Dict[str, Any]:
        """Get performance summary for a system."""
        
        if system_id not in self.performance_history:
            return {"system_id": system_id, "status": "no_data"}
        
        recent_metrics = self.performance_history[system_id][-10:]  # Last 10 measurements
        
        if not recent_metrics:
            return {"system_id": system_id, "status": "no_recent_data"}
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = np.mean([m.memory_utilization for m in recent_metrics])
        avg_disk = np.mean([m.disk_utilization for m in recent_metrics])
        
        latest_metric = recent_metrics[-1]
        
        return {
            "system_id": system_id,
            "status": "active",
            "latest_timestamp": latest_metric.timestamp.isoformat(),
            "recent_averages": {
                "cpu_utilization": avg_cpu,
                "memory_utilization": avg_memory,
                "disk_utilization": avg_disk
            },
            "current_values": {
                "cpu_utilization": latest_metric.cpu_utilization,
                "memory_utilization": latest_metric.memory_utilization,
                "disk_utilization": latest_metric.disk_utilization,
                "response_time_ms": latest_metric.response_time_ms,
                "error_rate": latest_metric.error_rate
            },
            "data_points": len(self.performance_history[system_id]),
            "monitoring_duration_hours": (
                latest_metric.timestamp - self.performance_history[system_id][0].timestamp
            ).total_seconds() / 3600
        }


class MaintenanceScheduler:
    """
    Intelligent maintenance scheduling optimization engine.
    
    Optimizes maintenance schedules based on system health, business requirements,
    resource availability, and predictive analytics.
    """
    
    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize maintenance scheduler."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
        # Scheduling parameters
        self.scheduling_constraints = {
            "business_hours": {"start": 9, "end": 17},  # 9 AM - 5 PM
            "blackout_days": ["saturday", "sunday"],
            "max_concurrent_maintenance": 3,
            "min_lead_time_hours": 24,
            "max_schedule_horizon_days": 90
        }
        
        # Maintenance cost models
        self.cost_models = {
            MaintenanceType.PREVENTIVE: {"base_cost": 1000, "urgency_multiplier": 1.0},
            MaintenanceType.PREDICTIVE: {"base_cost": 1500, "urgency_multiplier": 1.2},
            MaintenanceType.CORRECTIVE: {"base_cost": 2500, "urgency_multiplier": 2.0},
            MaintenanceType.EMERGENCY: {"base_cost": 5000, "urgency_multiplier": 5.0}
        }
        
        # Active maintenance schedules
        self.scheduled_maintenance: Dict[str, MaintenanceRecommendation] = {}
        self.maintenance_history: List[MaintenanceRecommendation] = []
        
        self.logger.info("MaintenanceScheduler initialized")
    
    async def optimize_maintenance_schedule(
        self,
        maintenance_requests: List[MaintenanceRecommendation],
        resource_constraints: Optional[Dict[str, Any]] = None,
        business_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize maintenance schedule considering all constraints.
        
        Args:
            maintenance_requests: List of maintenance requests to schedule
            resource_constraints: Available resources and constraints
            business_constraints: Business-specific scheduling constraints
            
        Returns:
            Optimized maintenance schedule with recommendations
        """
        
        try:
            # Filter and prioritize requests
            prioritized_requests = self._prioritize_maintenance_requests(maintenance_requests)
            
            # Check scheduling constraints
            scheduling_analysis = self._analyze_scheduling_constraints(
                prioritized_requests, resource_constraints, business_constraints
            )
            
            # Generate optimal schedule
            optimized_schedule = await self._generate_optimal_schedule(
                prioritized_requests, scheduling_analysis
            )
            
            # Calculate resource utilization
            resource_utilization = self._calculate_resource_utilization(optimized_schedule)
            
            # Get LLM recommendations for schedule optimization
            llm_recommendations = {}
            if self.llm_service:
                llm_recommendations = await self._get_llm_scheduling_recommendations(
                    optimized_schedule, scheduling_analysis, resource_utilization
                )
            
            return {
                "optimized_schedule": optimized_schedule,
                "scheduling_analysis": scheduling_analysis,
                "resource_utilization": resource_utilization,
                "llm_recommendations": llm_recommendations,
                "optimization_metrics": {
                    "total_requests": len(maintenance_requests),
                    "scheduled_requests": len(optimized_schedule),
                    "deferred_requests": len(maintenance_requests) - len(optimized_schedule),
                    "average_lead_time_hours": self._calculate_average_lead_time(optimized_schedule),
                    "estimated_total_cost": sum(r.estimated_cost or 0 for r in optimized_schedule),
                    "schedule_efficiency_score": self._calculate_schedule_efficiency(
                        optimized_schedule, resource_utilization
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Maintenance schedule optimization failed: {e}")
            raise CronosAIException(f"Schedule optimization error: {e}")
    
    def _prioritize_maintenance_requests(
        self, requests: List[MaintenanceRecommendation]
    ) -> List[MaintenanceRecommendation]:
        """Prioritize maintenance requests based on multiple factors."""
        
        def calculate_priority_score(request: MaintenanceRecommendation) -> float:
            score = 0.0
            
            # Priority based on maintenance type
            type_weights = {
                MaintenanceType.EMERGENCY: 100,
                MaintenanceType.CORRECTIVE: 80,
                MaintenanceType.PREDICTIVE: 60,
                MaintenanceType.PREVENTIVE: 40,
                MaintenanceType.ROUTINE: 20
            }
            score += type_weights.get(request.maintenance_type, 20)
            
            # Priority based on severity
            severity_weights = {
                SeverityLevel.CRITICAL: 50,
                SeverityLevel.HIGH: 40,
                SeverityLevel.MEDIUM: 25,
                SeverityLevel.LOW: 10,
                SeverityLevel.INFO: 5
            }
            score += severity_weights.get(request.priority, 10)
            
            # Risk-based priority
            risk_weights = {
                SeverityLevel.CRITICAL: 30,
                SeverityLevel.HIGH: 20,
                SeverityLevel.MEDIUM: 10,
                SeverityLevel.LOW: 5,
                SeverityLevel.INFO: 2
            }
            score += risk_weights.get(request.risk_level, 5)
            
            # Urgency based on recommended start time
            if request.recommended_start_time:
                time_to_start = (request.recommended_start_time - datetime.now()).total_seconds() / 3600
                if time_to_start < 24:  # Less than 24 hours
                    score += 40
                elif time_to_start < 72:  # Less than 72 hours
                    score += 20
                elif time_to_start < 168:  # Less than 1 week
                    score += 10
            
            # Business impact consideration
            if hasattr(request, 'business_impact_score') and request.metadata.get('business_impact_score'):
                score += min(request.metadata['business_impact_score'] * 5, 25)
            
            return score
        
        # Sort by priority score (descending)
        prioritized = sorted(requests, key=calculate_priority_score, reverse=True)
        
        # Add priority scores to metadata for reference
        for i, request in enumerate(prioritized):
            if not request.metadata:
                request.metadata = {}
            request.metadata['priority_score'] = calculate_priority_score(request)
            request.metadata['priority_rank'] = i + 1
        
        return prioritized
    
    def _analyze_scheduling_constraints(
        self,
        requests: List[MaintenanceRecommendation],
        resource_constraints: Optional[Dict[str, Any]],
        business_constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze scheduling constraints and availability."""
        
        # Merge constraints
        constraints = self.scheduling_constraints.copy()
        if business_constraints:
            constraints.update(business_constraints)
        
        # Analyze time windows
        available_windows = []
        start_date = datetime.now()
        end_date = start_date + timedelta(days=constraints.get("max_schedule_horizon_days", 90))
        
        current_date = start_date
        while current_date < end_date:
            # Check if day is not blackout
            day_name = current_date.strftime("%A").lower()
            if day_name not in constraints.get("blackout_days", []):
                
                # Determine available hours
                if constraints.get("allow_after_hours", True):
                    # 24/7 availability
                    start_hour = 0
                    end_hour = 24
                else:
                    # Business hours only
                    start_hour = constraints.get("business_hours", {}).get("start", 9)
                    end_hour = constraints.get("business_hours", {}).get("end", 17)
                
                available_windows.append({
                    "date": current_date.date(),
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "available_hours": end_hour - start_hour
                })
            
            current_date += timedelta(days=1)
        
        # Analyze resource constraints
        resource_analysis = {
            "available_technicians": resource_constraints.get("technicians", 5) if resource_constraints else 5,
            "available_tools": resource_constraints.get("tools", []) if resource_constraints else [],
            "budget_limit": resource_constraints.get("budget", 100000) if resource_constraints else 100000,
            "concurrent_capacity": constraints.get("max_concurrent_maintenance", 3)
        }
        
        # Analyze request requirements
        request_analysis = {
            "total_requests": len(requests),
            "emergency_requests": sum(1 for r in requests if r.maintenance_type == MaintenanceType.EMERGENCY),
            "critical_requests": sum(1 for r in requests if r.priority == SeverityLevel.CRITICAL),
            "total_estimated_duration": sum(r.expected_duration_hours for r in requests),
            "total_estimated_cost": sum(r.estimated_cost or 0 for r in requests),
            "required_expertise": list(set(
                expertise for r in requests 
                for expertise in r.required_expertise
            ))
        }
        
        return {
            "time_constraints": constraints,
            "available_windows": available_windows,
            "resource_analysis": resource_analysis,
            "request_analysis": request_analysis,
            "scheduling_feasibility": self._assess_scheduling_feasibility(
                requests, available_windows, resource_analysis
            )
        }
    
    def _assess_scheduling_feasibility(
        self,
        requests: List[MaintenanceRecommendation],
        available_windows: List[Dict[str, Any]],
        resource_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the feasibility of scheduling all requests."""
        
        total_available_hours = sum(window["available_hours"] for window in available_windows)
        total_required_hours = sum(request.expected_duration_hours for request in requests)
        
        # Account for concurrent capacity
        concurrent_capacity = resource_analysis.get("concurrent_capacity", 1)
        effective_available_hours = total_available_hours * concurrent_capacity
        
        utilization_ratio = total_required_hours / effective_available_hours if effective_available_hours > 0 else float('inf')
        
        return {
            "total_available_hours": total_available_hours,
            "effective_available_hours": effective_available_hours,
            "total_required_hours": total_required_hours,
            "utilization_ratio": utilization_ratio,
            "feasible": utilization_ratio <= 0.9,  # 90% utilization threshold
            "over_capacity_hours": max(0, total_required_hours - effective_available_hours),
            "capacity_recommendations": self._generate_capacity_recommendations(utilization_ratio)
        }
    
    def _generate_capacity_recommendations(self, utilization_ratio: float) -> List[str]:
        """Generate recommendations for capacity management."""
        
        recommendations = []
        
        if utilization_ratio > 1.0:
            recommendations.append(
                f"Schedule is over capacity by {(utilization_ratio - 1.0) * 100:.1f}%. "
                "Consider deferring lower priority maintenance or increasing resources."
            )
        elif utilization_ratio > 0.9:
            
            recommendations.append(
                "Schedule utilization is high (>90%). Monitor for potential bottlenecks "
                "and consider buffer time for unexpected issues."
            )
        elif utilization_ratio < 0.5:
            recommendations.append(
                "Schedule utilization is low (<50%). Consider accelerating lower priority "
                "maintenance or optimizing resource allocation."
            )
        else:
            recommendations.append("Schedule utilization is within optimal range (50-90%).")
        
        return recommendations
    
    async def _generate_optimal_schedule(
        self,
        prioritized_requests: List[MaintenanceRecommendation],
        scheduling_analysis: Dict[str, Any]
    ) -> List[MaintenanceRecommendation]:
        """Generate optimal maintenance schedule."""
        
        optimized_schedule = []
        available_windows = scheduling_analysis["available_windows"]
        resource_analysis = scheduling_analysis["resource_analysis"]
        
        # Track resource usage across time windows
        resource_usage = {}  # date -> {concurrent_tasks: int, allocated_hours: float}
        
        for request in prioritized_requests:
            # Find best time slot for this request
            best_slot = self._find_best_time_slot(
                request, available_windows, resource_usage, resource_analysis
            )
            
            if best_slot:
                # Schedule the request
                scheduled_request = self._schedule_request(request, best_slot)
                optimized_schedule.append(scheduled_request)
                
                # Update resource usage tracking
                self._update_resource_usage(resource_usage, scheduled_request)
            else:
                # Could not schedule - add to deferred list
                request.status = "deferred"
                request.metadata = request.metadata or {}
                request.metadata["deferral_reason"] = "No available time slots or resources"
        
        # Sort schedule by start time
        optimized_schedule.sort(key=lambda r: r.recommended_start_time or datetime.max)
        
        return optimized_schedule
    
    def _find_best_time_slot(
        self,
        request: MaintenanceRecommendation,
        available_windows: List[Dict[str, Any]],
        resource_usage: Dict[str, Dict[str, Any]],
        resource_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the best time slot for a maintenance request."""
        
        min_start_time = datetime.now() + timedelta(
            hours=self.scheduling_constraints.get("min_lead_time_hours", 24)
        )
        
        # Consider preferred start time if specified
        preferred_start = request.recommended_start_time or min_start_time
        
        best_slot = None
        best_score = -1
        
        for window in available_windows:
            window_date = datetime.combine(window["date"], datetime.min.time())
            
            # Check multiple time slots within the window
            for hour in range(window["start_hour"], window["end_hour"] - int(request.expected_duration_hours)):
                slot_start = window_date.replace(hour=hour)
                slot_end = slot_start + timedelta(hours=request.expected_duration_hours)
                
                # Check if slot meets minimum lead time
                if slot_start < min_start_time:
                    continue
                
                # Check resource availability
                if not self._check_resource_availability(
                    slot_start, slot_end, resource_usage, resource_analysis, request
                ):
                    continue
                
                # Calculate slot score
                slot_score = self._calculate_slot_score(
                    slot_start, preferred_start, request, resource_usage
                )
                
                if slot_score > best_score:
                    best_score = slot_score
                    best_slot = {
                        "start_time": slot_start,
                        "end_time": slot_end,
                        "window": window,
                        "score": slot_score
                    }
        
        return best_slot
    
    def _check_resource_availability(
        self,
        slot_start: datetime,
        slot_end: datetime,
        resource_usage: Dict[str, Dict[str, Any]],
        resource_analysis: Dict[str, Any],
        request: MaintenanceRecommendation
    ) -> bool:
        """Check if resources are available for the time slot."""
        
        slot_date = slot_start.date()
        concurrent_capacity = resource_analysis.get("concurrent_capacity", 3)
        
        # Check concurrent task limit
        current_usage = resource_usage.get(str(slot_date), {"concurrent_tasks": 0})
        if current_usage["concurrent_tasks"] >= concurrent_capacity:
            return False
        
        # Check technician availability (simplified)
        available_technicians = resource_analysis.get("available_technicians", 5)
        required_technicians = len(request.required_expertise) or 1
        
        if required_technicians > available_technicians:
            return False
        
        # Check for expertise requirements
        if request.required_expertise:
            # In production, this would check against actual technician skills
            pass
        
        return True
    
    def _calculate_slot_score(
        self,
        slot_start: datetime,
        preferred_start: datetime,
        request: MaintenanceRecommendation,
        resource_usage: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate a score for a time slot."""
        
        score = 100.0
        
        # Prefer slots close to preferred start time
        time_diff = abs((slot_start - preferred_start).total_seconds() / 3600)  # hours
        time_penalty = min(time_diff * 0.5, 50)  # Max 50 points penalty
        score -= time_penalty
        
        # Prefer earlier slots for high priority requests
        if request.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            days_from_now = (slot_start - datetime.now()).days
            urgency_bonus = max(30 - days_from_now, 0)
            score += urgency_bonus
        
        # Prefer slots with lower resource contention
        slot_date = slot_start.date()
        current_usage = resource_usage.get(str(slot_date), {"concurrent_tasks": 0})
        contention_penalty = current_usage["concurrent_tasks"] * 5
        score -= contention_penalty
        
        # Prefer business hours for routine maintenance
        if request.maintenance_type == MaintenanceType.ROUTINE:
            hour = slot_start.hour
            if 9 <= hour <= 17:  # Business hours
                score += 10
        
        # Prefer off-hours for critical maintenance
        if request.maintenance_type in [MaintenanceType.EMERGENCY, MaintenanceType.CORRECTIVE]:
            hour = slot_start.hour
            if hour < 9 or hour > 17:  # Off-hours
                score += 15
        
        return max(score, 0)
    
    def _schedule_request(
        self,
        request: MaintenanceRecommendation,
        time_slot: Dict[str, Any]
    ) -> MaintenanceRecommendation:
        """Schedule a maintenance request in the specified time slot."""
        
        scheduled_request = request
        scheduled_request.recommended_start_time = time_slot["start_time"]
        scheduled_request.recommended_completion_time = time_slot["end_time"]
        scheduled_request.status = "scheduled"
        
        # Add scheduling metadata
        if not scheduled_request.metadata:
            scheduled_request.metadata = {}
        
        scheduled_request.metadata.update({
            "scheduled_at": datetime.now().isoformat(),
            "scheduling_score": time_slot["score"],
            "window_utilization": time_slot.get("utilization", 0)
        })
        
        return scheduled_request
    
    def _update_resource_usage(
        self,
        resource_usage: Dict[str, Dict[str, Any]],
        scheduled_request: MaintenanceRecommendation
    ) -> None:
        """Update resource usage tracking."""
        
        if not scheduled_request.recommended_start_time:
            return
        
        start_date = scheduled_request.recommended_start_time.date()
        date_key = str(start_date)
        
        if date_key not in resource_usage:
            resource_usage[date_key] = {
                "concurrent_tasks": 0,
                "allocated_hours": 0.0,
                "scheduled_requests": []
            }
        
        resource_usage[date_key]["concurrent_tasks"] += 1
        resource_usage[date_key]["allocated_hours"] += scheduled_request.expected_duration_hours
        resource_usage[date_key]["scheduled_requests"].append(scheduled_request.recommendation_id)
    
    def _calculate_resource_utilization(
        self, scheduled_requests: List[MaintenanceRecommendation]
    ) -> Dict[str, Any]:
        """Calculate resource utilization metrics."""
        
        if not scheduled_requests:
            return {"utilization": 0.0, "efficiency": 0.0, "distribution": {}}
        
        # Calculate daily utilization
        daily_utilization = {}
        total_hours = 0
        
        for request in scheduled_requests:
            if request.recommended_start_time:
                date_key = request.recommended_start_time.date().isoformat()
                if date_key not in daily_utilization:
                    daily_utilization[date_key] = 0.0
                
                daily_utilization[date_key] += request.expected_duration_hours
                total_hours += request.expected_duration_hours
        
        # Calculate metrics
        avg_daily_utilization = np.mean(list(daily_utilization.values())) if daily_utilization else 0.0
        max_daily_utilization = max(daily_utilization.values()) if daily_utilization else 0.0
        
        # Efficiency based on even distribution
        utilization_variance = np.var(list(daily_utilization.values())) if len(daily_utilization) > 1 else 0.0
        efficiency = max(0, 100 - utilization_variance)  # Lower variance = higher efficiency
        
        return {
            "total_scheduled_hours": total_hours,
            "daily_utilization": daily_utilization,
            "average_daily_hours": avg_daily_utilization,
            "peak_daily_hours": max_daily_utilization,
            "efficiency_score": efficiency,
            "schedule_span_days": len(daily_utilization),
            "utilization_distribution": self._calculate_utilization_distribution(daily_utilization)
        }
    
    def _calculate_utilization_distribution(
        self, daily_utilization: Dict[str, float]
    ) -> Dict[str, int]:
        """Calculate distribution of daily utilization levels."""
        
        if not daily_utilization:
            return {"low": 0, "medium": 0, "high": 0, "peak": 0}
        
        distribution = {"low": 0, "medium": 0, "high": 0, "peak": 0}
        
        for hours in daily_utilization.values():
            if hours < 4:
                distribution["low"] += 1
            elif hours < 8:
                distribution["medium"] += 1
            elif hours < 12:
                distribution["high"] += 1
            else:
                distribution["peak"] += 1
        
        return distribution
    
    async def _get_llm_scheduling_recommendations(
        self,
        optimized_schedule: List[MaintenanceRecommendation],
        scheduling_analysis: Dict[str, Any],
        resource_utilization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get LLM recommendations for schedule optimization."""
        
        if not self.llm_service:
            return {}
        
        try:
            # Prepare schedule summary
            schedule_summary = {
                "total_scheduled": len(optimized_schedule),
                "schedule_span_days": resource_utilization.get("schedule_span_days", 0),
                "total_hours": resource_utilization.get("total_scheduled_hours", 0),
                "efficiency_score": resource_utilization.get("efficiency_score", 0),
                "peak_daily_hours": resource_utilization.get("peak_daily_hours", 0)
            }
            
            # Maintenance type distribution
            type_distribution = {}
            priority_distribution = {}
            
            for request in optimized_schedule:
                mtype = request.maintenance_type.value
                type_distribution[mtype] = type_distribution.get(mtype, 0) + 1
                
                priority = request.priority.value
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            prompt = f"""
You are optimizing a maintenance schedule for legacy systems.

Schedule Summary:
- Total Scheduled Tasks: {schedule_summary['total_scheduled']}
- Schedule Span: {schedule_summary['schedule_span_days']} days
- Total Hours: {schedule_summary['total_hours']:.1f}
- Efficiency Score: {schedule_summary['efficiency_score']:.1f}%
- Peak Daily Load: {schedule_summary['peak_daily_hours']:.1f} hours

Maintenance Type Distribution:
{type_distribution}

Priority Distribution:
{priority_distribution}

Resource Utilization:
{resource_utilization.get('utilization_distribution', {})}

Scheduling Constraints:
- Feasible: {scheduling_analysis.get('scheduling_feasibility', {}).get('feasible', False)}
- Capacity Utilization: {scheduling_analysis.get('scheduling_feasibility', {}).get('utilization_ratio', 0):.1f}

Provide optimization recommendations for:

1. SCHEDULE OPTIMIZATION:
   - Load balancing improvements
   - Resource allocation efficiency
   - Timeline optimization opportunities

2. RISK MITIGATION:
   - Schedule conflicts to watch for
   - Resource bottlenecks identified
   - Contingency planning recommendations

3. OPERATIONAL IMPROVEMENTS:
   - Maintenance sequencing optimization
   - Dependencies and coordination needs
   - Communication and coordination requirements

4. COST OPTIMIZATION:
   - Resource utilization improvements
   - Economies of scale opportunities
   - Budget allocation recommendations

Focus on practical, actionable recommendations for legacy system maintenance.
"""
            
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="legacy_whisperer",
                max_tokens=2000,
                temperature=0.2
            )
            
            response = await self.llm_service.process_request(llm_request)
            
            # Parse recommendations
            recommendations = self._parse_scheduling_recommendations(response.content)
            recommendations["llm_confidence"] = response.confidence
            recommendations["processing_time"] = response.processing_time
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"LLM scheduling recommendations failed: {e}")
            return {"error": str(e)}
    
    def _parse_scheduling_recommendations(self, content: str) -> Dict[str, Any]:
        """Parse LLM scheduling recommendations."""
        
        recommendations = {
            "schedule_optimization": [],
            "risk_mitigation": [],
            "operational_improvements": [],
            "cost_optimization": [],
            "overall_assessment": ""
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "schedule optimization" in line.lower():
                current_section = "schedule_optimization"
            elif "risk mitigation" in line.lower():
                current_section = "risk_mitigation"
            elif "operational improvements" in line.lower():
                current_section = "operational_improvements"
            elif "cost optimization" in line.lower():
                current_section = "cost_optimization"
            elif current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                # Extract recommendation item
                item = line.lstrip('-•* ').strip()
                if item and current_section in recommendations:
                    recommendations[current_section].append(item)
        
        # Extract overall assessment from beginning of response
        first_paragraph = content.split('\n\n')[0] if content else ""
        if len(first_paragraph) > 50 and "optimization" not in first_paragraph.lower():
            recommendations["overall_assessment"] = first_paragraph
        
        return recommendations
    
    def _calculate_average_lead_time(
        self, scheduled_requests: List[MaintenanceRecommendation]
    ) -> float:
        """Calculate average lead time for scheduled requests."""
        
        if not scheduled_requests:
            return 0.0
        
        now = datetime.now()
        lead_times = []
        
        for request in scheduled_requests:
            if request.recommended_start_time:
                lead_time_hours = (request.recommended_start_time - now).total_seconds() / 3600
                lead_times.append(max(0, lead_time_hours))
        
        return np.mean(lead_times) if lead_times else 0.0
    
    def _calculate_schedule_efficiency(
        self,
        scheduled_requests: List[MaintenanceRecommendation],
        resource_utilization: Dict[str, Any]
    ) -> float:
        """Calculate overall schedule efficiency score."""
        
        if not scheduled_requests:
            return 0.0
        
        efficiency_factors = []
        
        # Resource utilization efficiency
        efficiency_score = resource_utilization.get("efficiency_score", 0)
        efficiency_factors.append(efficiency_score / 100.0)
        
        # Priority scheduling efficiency (high priority items scheduled earlier)
        priority_scores = []
        for i, request in enumerate(scheduled_requests):
            priority_weight = {
                SeverityLevel.CRITICAL: 1.0,
                SeverityLevel.HIGH: 0.8,
                SeverityLevel.MEDIUM: 0.6,
                SeverityLevel.LOW: 0.4,
                SeverityLevel.INFO: 0.2
            }.get(request.priority, 0.5)
            
            # Earlier positions get higher scores for high priority items
            position_score = max(0, 1.0 - (i / len(scheduled_requests)))
            priority_scores.append(priority_weight * position_score)
        
        if priority_scores:
            efficiency_factors.append(np.mean(priority_scores))
        
        # Time utilization efficiency
        total_hours = resource_utilization.get("total_scheduled_hours", 0)
        span_days = resource_utilization.get("schedule_span_days", 1)
        if span_days > 0:
            hours_per_day = total_hours / span_days
            optimal_hours_per_day = 8  # Assuming 8-hour work days
            utilization_efficiency = min(hours_per_day / optimal_hours_per_day, 1.0)
            efficiency_factors.append(utilization_efficiency)
        
        return np.mean(efficiency_factors) * 100 if efficiency_factors else 0.0
    
    def create_maintenance_recommendation(
        self,
        system_id: str,
        maintenance_type: MaintenanceType,
        title: str,
        description: str,
        priority: SeverityLevel = SeverityLevel.MEDIUM,
        expected_duration_hours: float = 4.0,
        estimated_cost: Optional[float] = None,
        required_expertise: Optional[List[str]] = None,
        recommended_start_time: Optional[datetime] = None
    ) -> MaintenanceRecommendation:
        """Create a new maintenance recommendation."""
        
        # Calculate estimated cost if not provided
        if estimated_cost is None:
            cost_model = self.cost_models.get(maintenance_type, {"base_cost": 1000, "urgency_multiplier": 1.0})
            base_cost = cost_model["base_cost"]
            urgency_multiplier = cost_model["urgency_multiplier"]
            
            # Adjust for duration
            duration_multiplier = expected_duration_hours / 4.0  # 4 hours as baseline
            
            # Adjust for priority
            priority_multiplier = {
                SeverityLevel.CRITICAL: 1.5,
                SeverityLevel.HIGH: 1.2,
                SeverityLevel.MEDIUM: 1.0,
                SeverityLevel.LOW: 0.8,
                SeverityLevel.INFO: 0.6
            }.get(priority, 1.0)
            
            estimated_cost = base_cost * urgency_multiplier * duration_multiplier * priority_multiplier
        
        recommendation = MaintenanceRecommendation(
            recommendation_id=str(uuid.uuid4()),
            system_id=system_id,
            maintenance_type=maintenance_type,
            priority=priority,
            title=title,
            description=description,
            rationale=f"Generated maintenance recommendation for {system_id}",
            expected_duration_hours=expected_duration_hours,
            estimated_cost=estimated_cost,
            required_expertise=required_expertise or [],
            recommended_start_time=recommended_start_time,
            status="pending"
        )
        
        self.logger.info(f"Created maintenance recommendation {recommendation.recommendation_id} for system {system_id}")
        
        return recommendation
    
    def get_maintenance_calendar(
        self, start_date: Optional[datetime] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get maintenance calendar view."""
        
        if start_date is None:
            start_date = datetime.now()
        
        end_date = start_date + timedelta(days=days)
        
        # Get scheduled maintenance in the date range
        scheduled_maintenance = [
            maintenance for maintenance in self.scheduled_maintenance.values()
            if maintenance.recommended_start_time and 
            start_date <= maintenance.recommended_start_time <= end_date
        ]
        
        # Group by date
        calendar_view = {}
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            date_key = current_date.isoformat()
            calendar_view[date_key] = {
                "date": date_key,
                "maintenance_items": [],
                "total_hours": 0.0,
                "risk_level": "low"
            }
            current_date += timedelta(days=1)
        
        # Populate calendar with maintenance items
        for maintenance in scheduled_maintenance:
            if maintenance.recommended_start_time:
                date_key = maintenance.recommended_start_time.date().isoformat()
                if date_key in calendar_view:
                    calendar_view[date_key]["maintenance_items"].append({
                        "id": maintenance.recommendation_id,
                        "title": maintenance.title,
                        "type": maintenance.maintenance_type.value,
                        "priority": maintenance.priority.value,
                        "duration_hours": maintenance.expected_duration_hours,
                        "start_time": maintenance.recommended_start_time.isoformat(),
                        "system_id": maintenance.system_id
                    })
                    calendar_view[date_key]["total_hours"] += maintenance.expected_duration_hours
                    
                    # Update risk level based on maintenance priority
                    if maintenance.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                        calendar_view[date_key]["risk_level"] = "high"
                    elif (maintenance.priority == SeverityLevel.MEDIUM and 
                          calendar_view[date_key]["risk_level"] == "low"):
                        calendar_view[date_key]["risk_level"] = "medium"
        
        return {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "total_days": days,
            "calendar": calendar_view,
            "summary": {
                "total_maintenance_items": len(scheduled_maintenance),
                "high_risk_days": sum(1 for day in calendar_view.values() if day["risk_level"] == "high"),
                "busiest_day": max(calendar_view.values(), key=lambda d: d["total_hours"]) if calendar_view else None,
                "total_scheduled_hours": sum(m.expected_duration_hours for m in scheduled_maintenance)
            }
        }
    
    def add_scheduled_maintenance(self, maintenance: MaintenanceRecommendation) -> None:
        """Add a maintenance item to the schedule."""
        maintenance.status = "scheduled"
        self.scheduled_maintenance[maintenance.recommendation_id] = maintenance
        self.logger.info(f"Added maintenance {maintenance.recommendation_id} to schedule")
    
    def remove_scheduled_maintenance(self, recommendation_id: str) -> bool:
        """Remove a maintenance item from the schedule."""
        if recommendation_id in self.scheduled_maintenance:
            maintenance = self.scheduled_maintenance.pop(recommendation_id)
            maintenance.status = "cancelled"
            self.maintenance_history.append(maintenance)
            self.logger.info(f"Removed maintenance {recommendation_id} from schedule")
            return True
        return False
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics and metrics."""
        
        active_maintenance = list(self.scheduled_maintenance.values())
        
        if not active_maintenance:
            return {
                "total_scheduled": 0,
                "by_type": {},
                "by_priority": {},
                "by_system": {},
                "timeline": {},
                "resource_requirements": {}
            }
        
        # Statistics by maintenance type
        by_type = {}
        for maintenance in active_maintenance:
            mtype = maintenance.maintenance_type.value
            by_type[mtype] = by_type.get(mtype, 0) + 1
        
        # Statistics by priority
        by_priority = {}
        for maintenance in active_maintenance:
            priority = maintenance.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        # Statistics by system
        by_system = {}
        for maintenance in active_maintenance:
            system_id = maintenance.system_id
            by_system[system_id] = by_system.get(system_id, 0) + 1
        
        # Timeline analysis
        now = datetime.now()
        timeline = {
            "overdue": 0,
            "today": 0,
            "this_week": 0,
            "this_month": 0,
            "future": 0
        }
        
        for maintenance in active_maintenance:
            if maintenance.recommended_start_time:
                days_diff = (maintenance.recommended_start_time - now).days
                
                if days_diff < 0:
                    timeline["overdue"] += 1
                elif days_diff == 0:
                    timeline["today"] += 1
                elif days_diff <= 7:
                    timeline["this_week"] += 1
                elif days_diff <= 30:
                    timeline["this_month"] += 1
                else:
                    timeline["future"] += 1
        
        # Resource requirements
        required_expertise = set()
        total_cost = 0.0
        total_hours = 0.0
        
        for maintenance in active_maintenance:
            required_expertise.update(maintenance.required_expertise)
            total_cost += maintenance.estimated_cost or 0
            total_hours += maintenance.expected_duration_hours
        
        return {
            "total_scheduled": len(active_maintenance),
            "by_type": by_type,
            "by_priority": by_priority,
            "by_system": by_system,
            "timeline": timeline,
            "resource_requirements": {
                "required_expertise": list(required_expertise),
                "total_estimated_cost": total_cost,
                "total_estimated_hours": total_hours,
                "average_duration_hours": total_hours / len(active_maintenance) if active_maintenance else 0
            },
            "history": {
                "completed_maintenance": len(self.maintenance_history),
                "success_rate": self._calculate_success_rate()
            }
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate maintenance success rate from history."""
        
        if not self.maintenance_history:
            return 1.0
        
        successful = sum(
            1 for m in self.maintenance_history
            if m.success_score and m.success_score > 0.7
        )
        
        return successful / len(self.maintenance_history) if self.maintenance_history else 1.0