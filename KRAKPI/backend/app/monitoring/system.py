"""System monitoring and metrics collection."""
from typing import Dict, Optional
from datetime import datetime
import psutil
import prometheus_client as prom
from fastapi import Request
import time
import asyncio
from ..utils.logging import monitoring_logger
from ..config import settings

# Prometheus metrics
REQUEST_COUNT = prom.Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = prom.Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = prom.Gauge(
    'http_requests_active',
    'Active HTTP requests'
)

DB_QUERY_LATENCY = prom.Histogram(
    'db_query_duration_seconds',
    'Database query latency',
    ['operation', 'collection']
)

CACHE_HITS = prom.Counter(
    'cache_hits_total',
    'Cache hit count',
    ['cache_type']
)

CACHE_MISSES = prom.Counter(
    'cache_misses_total',
    'Cache miss count',
    ['cache_type']
)

ERROR_COUNT = prom.Counter(
    'error_count_total',
    'Total error count',
    ['error_type', 'endpoint']
)

class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self._setup_system_metrics()
    
    def _setup_system_metrics(self):
        """Setup system-level metrics."""
        self.system_memory = prom.Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )
        
        self.system_cpu = prom.Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.disk_usage = prom.Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['mount_point']
        )
        
        # Start background monitoring
        asyncio.create_task(self._monitor_system())
    
    async def _monitor_system(self):
        """Continuously monitor system metrics."""
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_memory.set(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu.set(cpu_percent)
                
                # Disk usage
                for partition in psutil.disk_partitions():
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage.labels(
                        mount_point=partition.mountpoint
                    ).set(usage.used)
                
                # Log system status
                if cpu_percent > 80 or memory.percent > 80:
                    monitoring_logger.warning(
                        "High resource usage detected",
                        extra={
                            "cpu_percent": cpu_percent,
                            "memory_percent": memory.percent,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
            except Exception as e:
                monitoring_logger.error(
                    f"System monitoring error: {str(e)}",
                    extra={"timestamp": datetime.now().isoformat()}
                )
            
            await asyncio.sleep(60)  # Monitor every minute

class RequestMonitor:
    """Request monitoring and metrics collection."""
    
    def __init__(self):
        self.active_requests: Dict[str, float] = {}
    
    async def track_request(
        self,
        request: Request,
        call_next,
        request_id: Optional[str] = None
    ):
        """Track request metrics."""
        start_time = time.time()
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        self.active_requests[request_id] = start_time
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log slow requests
            if duration > settings.SLOW_REQUEST_THRESHOLD:
                monitoring_logger.warning(
                    "Slow request detected",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "duration": duration,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return response
            
        except Exception as e:
            # Record error
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint=request.url.path
            ).inc()
            
            monitoring_logger.error(
                f"Request error: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise
            
        finally:
            # Cleanup
            ACTIVE_REQUESTS.dec()
            self.active_requests.pop(request_id, None)

class DatabaseMonitor:
    """Database monitoring and metrics collection."""
    
    @staticmethod
    async def track_query(operation: str, collection: str, duration: float):
        """Track database query metrics."""
        DB_QUERY_LATENCY.labels(
            operation=operation,
            collection=collection
        ).observe(duration)
        
        # Log slow queries
        if duration > settings.SLOW_QUERY_THRESHOLD:
            monitoring_logger.warning(
                "Slow database query detected",
                extra={
                    "operation": operation,
                    "collection": collection,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
            )

class CacheMonitor:
    """Cache monitoring and metrics collection."""
    
    @staticmethod
    def track_cache_hit(cache_type: str):
        """Track cache hit."""
        CACHE_HITS.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def track_cache_miss(cache_type: str):
        """Track cache miss."""
        CACHE_MISSES.labels(cache_type=cache_type).inc()

# Initialize monitors
system_monitor = SystemMonitor()
request_monitor = RequestMonitor()
db_monitor = DatabaseMonitor()
cache_monitor = CacheMonitor()
