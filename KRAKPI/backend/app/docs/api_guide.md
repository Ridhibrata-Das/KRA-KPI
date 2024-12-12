# KPI Management System API Guide

## Introduction

The KPI Management System API provides a comprehensive set of endpoints for managing Key Performance Indicators (KPIs) in an enterprise environment. This guide covers authentication, common operations, and best practices.

## Authentication

All API endpoints require JWT (JSON Web Token) authentication.

### Getting a Token

```http
POST /api/auth/token
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

### Using the Token

Include the token in the Authorization header:

```http
Authorization: Bearer your_jwt_token
```

## Common Operations

### 1. KPI Management

#### Create a KPI

```http
POST /api/kpis
Content-Type: application/json

{
    "name": "Quarterly Sales Target",
    "description": "Track quarterly sales performance",
    "type": "CURRENCY",
    "thresholds": {
        "min_value": 0,
        "max_value": 1000000,
        "target_value": 500000,
        "warning_threshold": 0.8,
        "critical_threshold": 0.6
    },
    "time_config": {
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-03-31T23:59:59"
    }
}
```

#### Update a KPI

```http
PUT /api/kpis/{kpi_id}
Content-Type: application/json

{
    "thresholds": {
        "target_value": 600000
    }
}
```

#### List KPIs

```http
GET /api/kpis?type=CURRENCY&status=active
```

### 2. Assignments

#### Create Assignment

```http
POST /api/kpis/{kpi_id}/assignments
Content-Type: application/json

{
    "type": "TEAM",
    "target_id": "sales_team_1",
    "role": "OWNER",
    "permissions": {
        "can_view": true,
        "can_edit": true,
        "can_assign": true
    },
    "notifications": {
        "preference": "DAILY",
        "threshold_alerts": true
    }
}
```

#### Update Assignment

```http
PUT /api/kpis/{kpi_id}/assignments/{assignment_id}
Content-Type: application/json

{
    "role": "CONTRIBUTOR",
    "permissions": {
        "can_edit": false
    }
}
```

#### List Assignments

```http
GET /api/kpis/{kpi_id}/assignments
```

### 3. Analytics

#### Get KPI Performance

```http
GET /api/kpis/{kpi_id}/analytics
```

#### Get Predictions

```http
GET /api/kpis/{kpi_id}/predictions?horizon=30
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```json
{
    "detail": "Error description",
    "code": "ERROR_CODE",
    "timestamp": "2024-12-12T22:35:53+05:30"
}
```

Common error codes:
- `INVALID_INPUT`: Invalid request data
- `KPI_NOT_FOUND`: KPI doesn't exist
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `INVALID_TOKEN`: Authentication failed

## Pagination

List endpoints support pagination using `skip` and `limit` parameters:

```http
GET /api/kpis?skip=0&limit=10
```

Response includes pagination metadata:

```json
{
    "items": [...],
    "total": 100,
    "page": 1,
    "pages": 10
}
```

## Filtering and Sorting

### Filtering

Use query parameters for filtering:

```http
GET /api/kpis?type=CURRENCY&status=active&team_id=sales_team_1
```

### Sorting

Use `sort_by` and `ascending` parameters:

```http
GET /api/kpis?sort_by=created_at&ascending=false
```

## Best Practices

1. **Rate Limiting**
   - Implement client-side rate limiting
   - Use exponential backoff for retries

2. **Error Handling**
   - Always check response status codes
   - Implement proper error handling
   - Log errors with context

3. **Performance**
   - Use pagination for large datasets
   - Request only needed fields
   - Cache responses when appropriate

4. **Security**
   - Store tokens securely
   - Refresh tokens before expiry
   - Validate input data

## Webhooks

Subscribe to KPI events:

```http
POST /api/webhooks
Content-Type: application/json

{
    "url": "https://your-domain.com/webhook",
    "events": ["kpi.created", "kpi.updated", "threshold.breached"],
    "secret": "your_webhook_secret"
}
```

## Support

For API support:
- Email: api-support@yourdomain.com
- Documentation: https://api.yourdomain.com/docs
- Status: https://status.yourdomain.com
