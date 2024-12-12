from typing import List, Dict, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from queue import Queue
import threading

from .unified_analytics import UnifiedAnalytics, AnalysisResult
from .pdf_analytics_report import PDFAnalyticsReport

@dataclass
class BatchConfig:
    parallel_jobs: int = 4
    timeout_seconds: int = 3600
    retry_attempts: int = 3
    export_format: List[str] = ('json', 'pdf')
    error_handling: str = 'continue'  # 'continue' or 'fail'

@dataclass
class BatchResult:
    successful_analyses: List[AnalysisResult]
    failed_analyses: List[Dict]
    execution_time: float
    total_kpis: int
    metadata: Dict

class BatchAnalyzer:
    def __init__(self,
                 analytics: UnifiedAnalytics,
                 config: Optional[BatchConfig] = None,
                 output_dir: str = './batch_output'):
        """Initialize the batch analyzer"""
        self.analytics = analytics
        self.config = config or BatchConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pdf_reporter = PDFAnalyticsReport(str(self.output_dir / 'reports'))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Progress tracking
        self.progress_queue = Queue()
        self.progress_thread = None

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / 'batch_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def analyze_batch(self,
                     kpi_data: Dict[str, Dict],
                     metadata: Optional[Dict] = None) -> BatchResult:
        """
        Analyze multiple KPIs in parallel
        
        kpi_data format:
        {
            'kpi_name': {
                'values': List[float],
                'timestamps': List[datetime],
                'metadata': Dict
            }
        }
        """
        start_time = datetime.now()
        self.logger.info(f"Starting batch analysis of {len(kpi_data)} KPIs")
        
        # Start progress monitoring
        self._start_progress_monitoring(len(kpi_data))
        
        # Initialize results
        successful_analyses = []
        failed_analyses = []
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.config.parallel_jobs) as executor:
            # Submit all analysis jobs
            future_to_kpi = {
                executor.submit(
                    self._analyze_single_kpi,
                    kpi_name,
                    kpi_info['values'],
                    kpi_info['timestamps'],
                    kpi_info.get('metadata', {})
                ): kpi_name
                for kpi_name, kpi_info in kpi_data.items()
            }
            
            # Process completed analyses
            for future in as_completed(future_to_kpi):
                kpi_name = future_to_kpi[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    successful_analyses.append(result)
                    self.logger.info(f"Successfully analyzed KPI: {kpi_name}")
                except Exception as e:
                    self.logger.error(f"Failed to analyze KPI {kpi_name}: {str(e)}")
                    failed_analyses.append({
                        'kpi_name': kpi_name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    if self.config.error_handling == 'fail':
                        raise
                
                # Update progress
                self.progress_queue.put(1)
        
        # Stop progress monitoring
        self._stop_progress_monitoring()
        
        # Generate batch report
        if successful_analyses:
            self._generate_batch_report(successful_analyses)
        
        # Create batch result
        execution_time = (datetime.now() - start_time).total_seconds()
        batch_result = BatchResult(
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            execution_time=execution_time,
            total_kpis=len(kpi_data),
            metadata={
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'config': {
                    'parallel_jobs': self.config.parallel_jobs,
                    'timeout_seconds': self.config.timeout_seconds,
                    'retry_attempts': self.config.retry_attempts,
                    'error_handling': self.config.error_handling
                },
                'user_metadata': metadata or {}
            }
        )
        
        # Export batch results
        self._export_batch_result(batch_result)
        
        return batch_result

    def _analyze_single_kpi(self,
                           kpi_name: str,
                           values: List[float],
                           timestamps: List[datetime],
                           metadata: Dict) -> AnalysisResult:
        """Analyze a single KPI with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                return self.analytics.analyze_kpi(
                    kpi_name=kpi_name,
                    values=values,
                    timestamps=timestamps,
                    metadata=metadata
                )
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for KPI {kpi_name}: {str(e)}. Retrying..."
                )

    def _start_progress_monitoring(self, total_kpis: int):
        """Start progress monitoring in a separate thread"""
        self.progress_queue = Queue()
        self.progress_thread = threading.Thread(
            target=self._monitor_progress,
            args=(total_kpis,)
        )
        self.progress_thread.daemon = True
        self.progress_thread.start()

    def _stop_progress_monitoring(self):
        """Stop progress monitoring"""
        if self.progress_thread:
            self.progress_queue.put(None)
            self.progress_thread.join()

    def _monitor_progress(self, total_kpis: int):
        """Monitor and log analysis progress"""
        completed = 0
        while True:
            item = self.progress_queue.get()
            if item is None:
                break
            completed += item
            progress = (completed / total_kpis) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({completed}/{total_kpis} KPIs)")

    def _generate_batch_report(self, results: List[AnalysisResult]):
        """Generate comprehensive batch analysis report"""
        # Generate individual PDF reports
        if 'pdf' in self.config.export_format:
            self.pdf_reporter.batch_generate_reports(results)
        
        # Generate batch summary
        summary = {
            'total_kpis': len(results),
            'total_alerts': sum(len(r.alerts) for r in results),
            'alerts_by_severity': {},
            'alerts_by_type': {},
            'performance_summary': self._calculate_performance_summary(results)
        }
        
        # Export summary
        summary_file = self.output_dir / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _calculate_performance_summary(self,
                                    results: List[AnalysisResult]) -> Dict:
        """Calculate performance metrics across all KPIs"""
        all_metrics = []
        for result in results:
            metrics = result.accuracy_metrics.get('ensemble', {})
            if metrics:
                all_metrics.append({
                    'kpi_name': result.kpi_name,
                    'mse': metrics.get('mse', float('nan')),
                    'mae': metrics.get('mae', float('nan')),
                    'mape': metrics.get('mape', float('nan'))
                })
        
        df = pd.DataFrame(all_metrics)
        return {
            'average_mse': float(df['mse'].mean()),
            'average_mae': float(df['mae'].mean()),
            'average_mape': float(df['mape'].mean()),
            'best_performing_kpi': df.loc[df['mape'].idxmin(), 'kpi_name'],
            'worst_performing_kpi': df.loc[df['mape'].idxmax(), 'kpi_name']
        }

    def _export_batch_result(self, result: BatchResult):
        """Export batch analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to JSON
        if 'json' in self.config.export_format:
            results_file = self.output_dir / f'batch_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'successful_analyses': [
                        r.to_dict() for r in result.successful_analyses
                    ],
                    'failed_analyses': result.failed_analyses,
                    'execution_time': result.execution_time,
                    'total_kpis': result.total_kpis,
                    'metadata': result.metadata
                }, f, indent=2)

    def get_batch_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """Retrieve historical batch analysis results"""
        results = []
        for file in self.output_dir.glob('batch_results_*.json'):
            try:
                # Parse timestamp from filename
                timestamp = datetime.strptime(
                    file.stem.split('_')[2],
                    "%Y%m%d%H%M%S"
                )
                
                # Apply time filters
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                # Load results
                with open(file, 'r') as f:
                    results.append(json.load(f))
            
            except Exception as e:
                self.logger.error(
                    f"Error loading batch results from {file}: {str(e)}",
                    exc_info=True
                )
        
        return sorted(results, key=lambda x: x['metadata']['start_time'])
