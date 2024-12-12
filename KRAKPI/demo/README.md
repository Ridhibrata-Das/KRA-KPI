# KPI Analytics Platform - Demo Environment

This demo environment provides a fully functional instance of the KPI Analytics Platform with sample data and pre-configured dashboards.

## Quick Start

```bash
# Start the demo environment
docker-compose -f docker-compose.demo.yml up -d

# Check status
docker-compose -f docker-compose.demo.yml ps
```

## Demo Access Points

1. **Main Application**
   - URL: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Admin Panel: http://localhost:8000/admin

2. **Monitoring**
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

## Demo Accounts

```yaml
Admin User:
  Email: demo.admin@kpi-system.com
  Password: demo_admin_123

Regular User:
  Email: demo.user@kpi-system.com
  Password: demo_user_123
```

## Sample Data

The demo environment comes with pre-configured:

1. **KPIs**
   - Monthly Sales
   - Customer Satisfaction
   - Website Conversion Rate

2. **Alerts**
   - Sales below $90,000
   - Customer satisfaction below 4.0

3. **Dashboards**
   - KPI Overview
   - Performance Metrics
   - Alert Dashboard

## Features Available in Demo

1. **KPI Management**
   - Create/Edit KPIs
   - View historical data
   - Set targets and thresholds

2. **Analytics**
   - Forecasting
   - Anomaly detection
   - Statistical analysis

3. **Alerts**
   - Create alert rules
   - View notifications
   - Configure channels

4. **Monitoring**
   - System metrics
   - API performance
   - Resource usage

## Demo Limitations

1. **Data Reset**
   - Demo data resets every 24 hours
   - Custom data is not persistent

2. **Features**
   - Email notifications are disabled
   - External integrations are simulated
   - File uploads are limited to 1MB

3. **Performance**
   - Rate limiting is stricter
   - Some operations are throttled

## Support

For demo support:
- Email: demo.support@kpi-system.com
- Documentation: http://localhost:8000/docs
- Issues: https://github.com/your-repo/issues
