# End-User Testing Guide

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git
- Postman (optional, for API testing)

## 1. Quick Setup

```bash
# 1. Clone the repository
git clone [repository-url]
cd KRAKPI

# 2. Create test environment
cp .env.example .env

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps
```

## 2. Test Environment URLs

- Main Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Admin Interface: http://localhost:8000/admin
- Monitoring: http://localhost:9090

## 3. Test Accounts

```yaml
Admin User:
  Email: admin@test.com
  Password: admin123!
  API Key: test_admin_key_123

Regular User:
  Email: user@test.com
  Password: user123!
  API Key: test_user_key_456
```

## 4. Testing Scenarios

### 4.1 KPI Management

1. **Create KPI**
```http
POST /api/v1/kpis
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Monthly Sales",
  "description": "Monthly sales target",
  "target": 100000,
  "unit": "USD",
  "frequency": "monthly"
}
```

2. **Update KPI Data**
```http
PUT /api/v1/kpis/{kpi_id}/data
Content-Type: application/json
Authorization: Bearer {token}

{
  "value": 95000,
  "timestamp": "2024-12-12T00:00:00Z"
}
```

### 4.2 Analytics Testing

1. **Generate Forecast**
```http
GET /api/v1/analytics/forecast/{kpi_id}
Authorization: Bearer {token}
```

2. **Check Anomalies**
```http
GET /api/v1/analytics/anomalies/{kpi_id}
Authorization: Bearer {token}
```

### 4.3 Alert System

1. **Create Alert Rule**
```http
POST /api/v1/alerts
Content-Type: application/json
Authorization: Bearer {token}

{
  "kpi_id": "{kpi_id}",
  "condition": "threshold",
  "threshold": 90000,
  "operator": "less_than",
  "notification_channel": "email"
}
```

## 5. Common Test Cases

### 5.1 Authentication
- [ ] Login with valid credentials
- [ ] Login with invalid credentials
- [ ] Password reset flow
- [ ] MFA setup and validation
- [ ] API key authentication

### 5.2 KPI Operations
- [ ] Create new KPI
- [ ] Update KPI details
- [ ] Delete KPI
- [ ] Add KPI data points
- [ ] View KPI history

### 5.3 Analytics
- [ ] Generate forecasts
- [ ] Detect anomalies
- [ ] View statistical validations
- [ ] Export reports
- [ ] Test visualization features

### 5.4 Alerts
- [ ] Create alert rules
- [ ] Trigger test alerts
- [ ] Update alert settings
- [ ] Check notification delivery

## 6. Performance Testing

### 6.1 Load Testing Scenarios
```bash
# Using k6 for load testing
k6 run load-tests/api-test.js

# Basic load test configuration
export const options = {
  vus: 10,
  duration: '30s',
};
```

### 6.2 Stress Testing
```bash
# High load scenario
k6 run --vus 50 --duration 60s load-tests/stress-test.js
```

## 7. Error Handling

Test these error scenarios:
1. Invalid API keys
2. Malformed requests
3. Rate limit exceeded
4. Network timeouts
5. Database connection issues

## 8. Reporting Issues

### Template
```markdown
## Issue Description
[Clear description of the issue]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happened]

## Environment
- OS: [e.g., Windows 10]
- Browser: [e.g., Chrome 96]
- API Version: [e.g., v1.0.0]

## Additional Context
[Screenshots, logs, etc.]
```

## 9. Monitoring During Testing

1. **Check Metrics**
   - http://localhost:9090/metrics
   - Monitor system resources
   - Check API response times

2. **View Logs**
   - `docker-compose logs -f`
   - Check for errors and warnings
   - Monitor performance metrics

## 10. Security Testing

1. **Authentication Tests**
   - Invalid tokens
   - Expired tokens
   - Wrong permissions

2. **API Security**
   - SQL injection attempts
   - XSS attempts
   - CSRF protection
   - Rate limiting

## Support Contacts

- Technical Issues: tech@kpi-system.com
- User Support: support@kpi-system.com
- Security Issues: security@kpi-system.com
